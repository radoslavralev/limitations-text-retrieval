"""
Lightweight Verifiers for Semantic Cache Reranking.

Implements a spectrum of verifiers (F0-F4) that operate on token similarity matrices
after ANN candidate generation. These verifiers increase in expressivity and cost
while remaining far cheaper than running a full LLM.

Verifiers:
- F0: Bag-of-similarities (mean of similarity matrix)
- F1: Late interaction / MaxSim (sum of row-wise max)
- F2: Monotone alignment with positional bias
- F3: Tiny CNN over similarity matrix
- F4: Tiny cross-encoder (2-6 layer transformer)

Conditional Penalty Mode:
When enabled, verifiers act as structural mismatch detectors rather than similarity scorers.
The final score is: S(q,c) = S_base(q,c) - β * 1[S_base > τ] * P_struct(q,c)
This preserves retrieval performance while penalizing structural mismatches only for high-similarity candidates.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


class BaseVerifier(nn.Module, ABC):
    """Abstract base class for all verifiers.
    
    Supports two modes:
    1. Standard mode: Verifier outputs a similarity score directly
    2. Conditional penalty mode: Verifier acts as a structural penalty applied only 
       when base similarity exceeds a threshold τ
       
    Args:
        name: Name identifier for the verifier
        use_conditional_penalty: If True, use conditional penalty mode
        penalty_beta: Strength of the structural penalty (default: 0.5, aggressive)
        penalty_tau: Similarity threshold above which penalty is applied (default: 0.9)
        base_score_type: Type of base score to use ("maxsim" or "cosine")
    """
    
    def __init__(
        self,
        name: str,
        use_conditional_penalty: bool = False,
        penalty_beta: float = 0.5,
        penalty_tau: float = 0.9,
        base_score_type: str = "maxsim",
    ):
        super().__init__()
        self._name = name
        self.use_conditional_penalty = use_conditional_penalty
        self.penalty_beta = penalty_beta
        self.penalty_tau = penalty_tau
        self.base_score_type = base_score_type
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def num_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @staticmethod
    def compute_maxsim_score(
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute MaxSim (F1-style) base score.
        
        For each query token, find max similarity with any candidate token,
        then average over query tokens. This is retrieval-friendly.
        
        Args:
            query_embeddings: (batch, m, d)
            candidate_embeddings: (batch, n, d)
            query_mask: (batch, m)
            candidate_mask: (batch, n)
            
        Returns:
            scores: (batch,)
        """
        # Normalize embeddings
        query_norm = F.normalize(query_embeddings, p=2, dim=-1)
        cand_norm = F.normalize(candidate_embeddings, p=2, dim=-1)
        
        # Compute similarity matrix: (batch, m, n)
        M = torch.bmm(query_norm, cand_norm.transpose(1, 2))
        
        if candidate_mask is not None:
            # Mask out invalid candidate tokens
            mask = candidate_mask.unsqueeze(1)  # (batch, 1, n)
            M = M.masked_fill(~mask.bool(), float('-inf'))
        
        # Max over candidate dimension: (batch, m)
        max_sims, _ = M.max(dim=-1)
        
        # Handle -inf from fully masked rows
        max_sims = max_sims.masked_fill(max_sims == float('-inf'), 0.0)
        
        if query_mask is not None:
            max_sims = max_sims * query_mask
            num_valid = query_mask.sum(dim=-1).clamp(min=1)
            scores = max_sims.sum(dim=-1) / num_valid
        else:
            scores = max_sims.mean(dim=-1)
        
        return scores
    
    @staticmethod
    def compute_cosine_score(
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute mean-pooled cosine similarity (NoneVerifier-style).
        
        Args:
            query_embeddings: (batch, m, d)
            candidate_embeddings: (batch, n, d)
            query_mask: (batch, m)
            candidate_mask: (batch, n)
            
        Returns:
            scores: (batch,)
        """
        # Mean pool with mask
        if query_mask is not None:
            query_mask_expanded = query_mask.unsqueeze(-1)
            query_pooled = (query_embeddings * query_mask_expanded).sum(dim=1)
            query_pooled = query_pooled / query_mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            query_pooled = query_embeddings.mean(dim=1)
        
        if candidate_mask is not None:
            candidate_mask_expanded = candidate_mask.unsqueeze(-1)
            candidate_pooled = (candidate_embeddings * candidate_mask_expanded).sum(dim=1)
            candidate_pooled = candidate_pooled / candidate_mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            candidate_pooled = candidate_embeddings.mean(dim=1)
        
        # Normalize and compute cosine
        query_pooled = F.normalize(query_pooled, p=2, dim=-1)
        candidate_pooled = F.normalize(candidate_pooled, p=2, dim=-1)
        
        return (query_pooled * candidate_pooled).sum(dim=-1)
    
    def compute_base_score(
        self,
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the base retrieval score based on configured type."""
        if self.base_score_type == "maxsim":
            return self.compute_maxsim_score(
                query_embeddings, candidate_embeddings, query_mask, candidate_mask
            )
        else:
            return self.compute_cosine_score(
                query_embeddings, candidate_embeddings, query_mask, candidate_mask
            )
    
    @abstractmethod
    def compute_raw_score(
        self,
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the raw verifier score (before conditional penalty logic).
        
        This is what each verifier subclass implements.
        
        Args:
            query_embeddings: Token embeddings for queries (batch, m, d)
            candidate_embeddings: Token embeddings for candidates (batch, n, d)
            query_mask: Attention mask for queries (batch, m), 1 for valid tokens
            candidate_mask: Attention mask for candidates (batch, n), 1 for valid tokens
        
        Returns:
            Scores tensor of shape (batch,)
        """
        pass
    
    def forward(
        self,
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute verifier scores for query-candidate pairs.
        
        In standard mode, returns the raw verifier score.
        In conditional penalty mode, returns:
            S(q,c) = S_base(q,c) - β * 1[S_base > τ] * P_struct(q,c)
        where P_struct = 1 - raw_score (structural mismatch penalty).
        
        Args:
            query_embeddings: Token embeddings for queries (batch, m, d)
            candidate_embeddings: Token embeddings for candidates (batch, n, d)
            query_mask: Attention mask for queries (batch, m), 1 for valid tokens
            candidate_mask: Attention mask for candidates (batch, n), 1 for valid tokens
        
        Returns:
            Scores tensor of shape (batch,)
        """
        raw_score = self.compute_raw_score(
            query_embeddings, candidate_embeddings, query_mask, candidate_mask
        )
        
        if not self.use_conditional_penalty:
            return raw_score
        
        # Conditional penalty mode
        base_score = self.compute_base_score(
            query_embeddings, candidate_embeddings, query_mask, candidate_mask
        )
        
        # Structural penalty: higher when raw_score is low (mismatch detected)
        # raw_score should be in [-1, 1] or [0, 1] range after calibration
        # P_struct = 1 - raw_score means: low similarity = high penalty
        p_struct = 1.0 - raw_score.clamp(-1, 1)
        
        # Apply penalty only when base similarity exceeds threshold
        # Using soft threshold with sigmoid for differentiability
        # gate ≈ 1 when base_score > tau, ≈ 0 otherwise
        gate = torch.sigmoid((base_score - self.penalty_tau) * 20.0)
        
        # Final score: base score minus conditional penalty
        final_score = base_score - self.penalty_beta * gate * p_struct
        
        return final_score
    
    def score(
        self,
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Alias for forward() for cleaner API."""
        return self.forward(query_embeddings, candidate_embeddings, query_mask, candidate_mask)
    
    def set_conditional_penalty_mode(
        self,
        enabled: bool,
        beta: Optional[float] = None,
        tau: Optional[float] = None,
    ):
        """Enable or disable conditional penalty mode at runtime."""
        self.use_conditional_penalty = enabled
        if beta is not None:
            self.penalty_beta = beta
        if tau is not None:
            self.penalty_tau = tau


class TokenSimilarityMatrix(nn.Module):
    """
    Compute token-level similarity matrix M between query and candidate embeddings.
    
    M[i,j] = similarity(query_token_i, candidate_token_j)
    """
    
    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
    
    def forward(
        self,
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute similarity matrix.
        
        Args:
            query_embeddings: (batch, m, d)
            candidate_embeddings: (batch, n, d)
        
        Returns:
            Similarity matrix of shape (batch, m, n)
        """
        if self.normalize:
            query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
            candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=-1)
        
        # Batched matrix multiplication: (batch, m, d) @ (batch, d, n) -> (batch, m, n)
        similarity_matrix = torch.bmm(query_embeddings, candidate_embeddings.transpose(1, 2))
        return similarity_matrix


class NoneVerifier(BaseVerifier):
    """
    None/Identity verifier: Baseline that uses only mean-pooled embedding similarity.
    
    F_none(q, c) = cosine_similarity(mean_pool(q), mean_pool(c))
    
    This is the "no verifier" baseline - it simply computes the cosine similarity
    between the mean-pooled query and candidate embeddings, which is what a 
    standard embedding model would produce.
    
    Use this to compare verifier performance against just using the embedding model.
    No learnable parameters.
    """
    
    def __init__(
        self,
        use_conditional_penalty: bool = False,
        penalty_beta: float = 0.5,
        penalty_tau: float = 0.9,
    ):
        super().__init__(
            name="None_Identity",
            use_conditional_penalty=use_conditional_penalty,
            penalty_beta=penalty_beta,
            penalty_tau=penalty_tau,
        )
    
    def compute_raw_score(
        self,
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Use the static method from base class
        return self.compute_cosine_score(
            query_embeddings, candidate_embeddings, query_mask, candidate_mask
        )


class F0Verifier(BaseVerifier):
    """
    F0: Bag-of-similarities verifier.
    
    F0(q, c) = (1/mn) * sum_{i,j} M_{ij}
    
    This is the simplest verifier - just the mean of all pairwise similarities.
    No learnable parameters.
    """
    
    def __init__(
        self,
        use_conditional_penalty: bool = False,
        penalty_beta: float = 0.5,
        penalty_tau: float = 0.9,
    ):
        super().__init__(
            name="F0_BagOfSimilarities",
            use_conditional_penalty=use_conditional_penalty,
            penalty_beta=penalty_beta,
            penalty_tau=penalty_tau,
        )
        self.similarity = TokenSimilarityMatrix(normalize=True)
    
    def compute_raw_score(
        self,
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Compute similarity matrix: (batch, m, n)
        M = self.similarity(query_embeddings, candidate_embeddings)
        
        if query_mask is not None and candidate_mask is not None:
            # Create mask for valid token pairs: (batch, m, n)
            mask = query_mask.unsqueeze(-1) * candidate_mask.unsqueeze(-2)
            # Use masked_fill instead of multiplication to preserve gradients better
            M = M.masked_fill(~mask.bool(), 0.0)
            # Compute mean over valid positions
            num_valid = mask.sum(dim=(1, 2)).clamp(min=1)
            scores = M.sum(dim=(1, 2)) / num_valid
        else:
            # Simple mean over all positions
            scores = M.mean(dim=(1, 2))
        
        return scores


class F1Verifier(BaseVerifier):
    """
    F1: Late interaction / MaxSim verifier (ColBERT-style).
    
    F1(q, c) = sum_i max_j M_{ij}
    
    For each query token, find the maximum similarity with any candidate token,
    then sum these maximum values. No learnable parameters.
    """
    
    def __init__(
        self,
        use_conditional_penalty: bool = False,
        penalty_beta: float = 0.5,
        penalty_tau: float = 0.9,
    ):
        super().__init__(
            name="F1_MaxSim",
            use_conditional_penalty=use_conditional_penalty,
            penalty_beta=penalty_beta,
            penalty_tau=penalty_tau,
        )
        self.similarity = TokenSimilarityMatrix(normalize=True)
    
    def compute_raw_score(
        self,
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Compute similarity matrix: (batch, m, n)
        M = self.similarity(query_embeddings, candidate_embeddings)
        
        if candidate_mask is not None:
            # Mask out invalid candidate tokens with large negative value
            mask = candidate_mask.unsqueeze(1)  # (batch, 1, n)
            M = M.masked_fill(~mask.bool(), float('-inf'))
        
        # Max over candidate dimension: (batch, m)
        max_sims, _ = M.max(dim=-1)
        
        # Handle -inf from fully masked rows
        max_sims = max_sims.masked_fill(max_sims == float('-inf'), 0.0)
        
        if query_mask is not None:
            # Only average over valid query tokens (normalize to [-1, 1] range)
            max_sims = max_sims * query_mask
            num_valid = query_mask.sum(dim=-1).clamp(min=1)
            scores = max_sims.sum(dim=-1) / num_valid
        else:
            # Average over all query tokens
            scores = max_sims.mean(dim=-1)
        
        return scores


class F2SoftAttnVerifier(BaseVerifier):
    """
    Differentiable, no-DP relaxation of monotone alignment:
      - Row-wise soft alignment with positional bias (|i-j| penalty)
      - Optional monotonicity regularizer on expected aligned positions

    Score:
      A_{ij} = softmax_j((M_{ij} - lambda*|i-j|)/tau)
      s = mean_i sum_j A_{ij} * M_{ij}

    Optional training-time regularization:
      p_i = sum_j A_{ij} * j
      R = sum_i ReLU(p_i - p_{i+1})^2
      return s - gamma * R
    """

    def __init__(
        self,
        lambda_penalty: float = 0.1,
        tau: float = 0.1,
        bandwidth: Optional[int] = None,   # if set, restrict attention to |i-j| <= bandwidth
        gamma_mono: float = 0.0,           # if >0, subtract gamma*monotone_penalty from score
        eps: float = 1e-9,
        use_conditional_penalty: bool = False,
        penalty_beta: float = 0.5,
        penalty_tau: float = 0.9,
    ):
        super().__init__(
            name="F2_SoftAttn",
            use_conditional_penalty=use_conditional_penalty,
            penalty_beta=penalty_beta,
            penalty_tau=penalty_tau,
        )
        self.lambda_penalty = float(lambda_penalty)
        self.tau = float(tau)
        self.bandwidth = bandwidth
        self.gamma_mono = float(gamma_mono)
        self.eps = float(eps)
        self.similarity = TokenSimilarityMatrix(normalize=True)

        # For logging/inspection (e.g., during training)
        self.last_penalty: Optional[torch.Tensor] = None

    @staticmethod
    def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1, eps: float = 1e-9):
        """
        logits: (..., n)
        mask:   same shape, bool or {0,1}; True = keep
        Returns probabilities with rows that are fully masked -> all zeros (not NaNs).
        """
        mask = mask.bool()
        neg = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~mask, neg)

        # Stable softmax
        logits = logits - logits.max(dim=dim, keepdim=True).values
        exp = torch.exp(logits) * mask.to(logits.dtype)
        denom = exp.sum(dim=dim, keepdim=True).clamp(min=eps)
        return exp / denom

    def compute_raw_score(
        self,
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns: scores (batch,)
        If gamma_mono>0, returns score - gamma_mono * monotonic_penalty.
        """
        M = self.similarity(query_embeddings, candidate_embeddings)  # (B, m, n)
        B, m, n = M.shape
        device = M.device
        dtype = M.dtype

        # Build positional penalty |i-j|
        i = torch.arange(m, device=device, dtype=dtype).unsqueeze(1)  # (m,1)
        j = torch.arange(n, device=device, dtype=dtype).unsqueeze(0)  # (1,n)
        pos_pen = (i - j).abs() * self.lambda_penalty                 # (m,n)

        # Logits with positional bias
        logits = (M - pos_pen.unsqueeze(0)) / max(self.tau, self.eps)  # (B,m,n)

        # Candidate mask (padding)
        if candidate_mask is not None:
            cand_keep = candidate_mask.unsqueeze(1).expand(B, m, n) > 0  # (B,m,n)
        else:
            cand_keep = torch.ones((B, m, n), device=device, dtype=torch.bool)

        # Optional band mask: only allow |i-j| <= bandwidth
        if self.bandwidth is not None:
            band_keep = (pos_pen <= float(self.bandwidth)).unsqueeze(0).expand(B, m, n)
            keep = cand_keep & band_keep
        else:
            keep = cand_keep

        # Row-wise alignment weights A (B,m,n)
        A = self._masked_softmax(logits, keep, dim=-1, eps=self.eps)

        # Expected similarity per query token: (B,m)
        token_scores = (A * M).sum(dim=-1)

        # Query mask (padding)
        if query_mask is not None:
            q_keep = (query_mask > 0).to(dtype)  # (B,m)
            token_scores = token_scores * q_keep
            denom = q_keep.sum(dim=-1).clamp(min=1.0)
        else:
            denom = torch.full((B,), float(m), device=device, dtype=dtype)

        score = token_scores.sum(dim=-1) / denom  # (B,)

        # Optional monotonicity regularizer on expected aligned position p_i
        if self.gamma_mono > 0.0:
            # p_i = sum_j A_{ij} * j
            j_pos = torch.arange(n, device=device, dtype=dtype).view(1, 1, n)  # (1,1,n)
            p = (A * j_pos).sum(dim=-1)  # (B,m)

            if query_mask is not None:
                # Only compare consecutive *valid* tokens; simplest: keep mask and compute diffs
                q_bool = (query_mask > 0)
            else:
                q_bool = torch.ones((B, m), device=device, dtype=torch.bool)

            # diffs for i=0..m-2: relu(p_i - p_{i+1})^2
            diffs = F.relu(p[:, :-1] - p[:, 1:]) ** 2  # (B,m-1)

            # mask out comparisons where either token is padding
            pair_keep = q_bool[:, :-1] & q_bool[:, 1:]
            diffs = diffs * pair_keep.to(dtype)

            penalty = diffs.sum(dim=-1) / pair_keep.to(dtype).sum(dim=-1).clamp(min=1.0)  # (B,)
            self.last_penalty = penalty.detach()
            score = score - self.gamma_mono * penalty
        else:
            self.last_penalty = None

        return score


class F3Verifier(BaseVerifier):
    """
    F3: Tiny CNN over similarity matrix.
    
    F3(q, c) = tanh(MLP(CNN_{k×k}(φ(M))))
    
    Applies a small CNN over the similarity matrix (optionally augmented with
    additional features like row/column maxes) followed by an MLP.
    Output is calibrated to [-1, 1] range using tanh.
    
    Learnable parameters: ~100K-500K depending on configuration.
    """
    
    def __init__(
        self,
        in_channels: int = 3,  # M, row_max, col_max
        hidden_channels: int = 32,
        kernel_size: int = 3,
        num_conv_layers: int = 3,
        mlp_hidden_dim: int = 128,
        max_seq_length: int = 128,
        use_augmented_features: bool = True,
        use_conditional_penalty: bool = False,
        penalty_beta: float = 0.5,
        penalty_tau: float = 0.9,
    ):
        super().__init__(
            name="F3_CNN",
            use_conditional_penalty=use_conditional_penalty,
            penalty_beta=penalty_beta,
            penalty_tau=penalty_tau,
        )
        self.similarity = TokenSimilarityMatrix(normalize=True)
        self.use_augmented_features = use_augmented_features
        self.max_seq_length = max_seq_length
        
        # Determine input channels based on feature augmentation
        actual_in_channels = in_channels if use_augmented_features else 1
        
        # Build CNN layers
        conv_layers = []
        current_channels = actual_in_channels
        for i in range(num_conv_layers):
            out_channels = hidden_channels * (2 ** i)
            conv_layers.extend([
                nn.Conv2d(current_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            ])
            current_channels = out_channels
        
        self.conv = nn.Sequential(*conv_layers)
        
        # Calculate output size after convolutions
        # After num_conv_layers pooling operations, size is reduced by 2^num_conv_layers
        final_size = max_seq_length // (2 ** num_conv_layers)
        final_channels = hidden_channels * (2 ** (num_conv_layers - 1))
        
        # Global average pooling + MLP
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(final_channels, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, 1),
        )
    
    def _augment_matrix(
        self,
        M: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Augment similarity matrix with additional features.
        
        φ(M) includes:
        - Original similarity matrix M
        - Row-wise max values (broadcast to matrix shape)
        - Column-wise max values (broadcast to matrix shape)
        
        Uses masked max when mask is provided.
        
        Returns: (batch, 3, m, n)
        """
        batch_size, m, n = M.shape
        
        if mask is not None:
            # Use masked_fill for proper max computation
            M_masked = M.masked_fill(~mask.bool(), float('-inf'))
            row_max = M_masked.max(dim=-1, keepdim=True)[0]
            col_max = M_masked.max(dim=-2, keepdim=True)[0]
            # Replace -inf with 0 for the feature maps
            row_max = row_max.masked_fill(row_max == float('-inf'), 0.0)
            col_max = col_max.masked_fill(col_max == float('-inf'), 0.0)
        else:
            row_max = M.max(dim=-1, keepdim=True)[0]
            col_max = M.max(dim=-2, keepdim=True)[0]
        
        # Broadcast to matrix shape
        row_max = row_max.expand_as(M)
        col_max = col_max.expand_as(M)
        
        # Stack as channels: (batch, 3, m, n)
        augmented = torch.stack([M, row_max, col_max], dim=1)
        return augmented
    
    def _pad_to_size(self, x: torch.Tensor, target_size: int, pad_value: float = 0.0) -> torch.Tensor:
        """Pad input to target size for consistent CNN processing."""
        batch, channels, h, w = x.shape
        if h < target_size or w < target_size:
            pad_h = max(0, target_size - h)
            pad_w = max(0, target_size - w)
            x = F.pad(x, (0, pad_w, 0, pad_h), value=pad_value)
        return x[:, :, :target_size, :target_size]
    
    def compute_raw_score(
        self,
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Compute similarity matrix: (batch, m, n)
        M = self.similarity(query_embeddings, candidate_embeddings)
        
        # Create mask for valid token pairs
        if query_mask is not None and candidate_mask is not None:
            mask = query_mask.unsqueeze(-1) * candidate_mask.unsqueeze(-2)
            # Use masked_fill instead of multiplication for proper handling
            M_for_cnn = M.masked_fill(~mask.bool(), 0.0)
        else:
            mask = None
            M_for_cnn = M
        
        # Augment with additional features if enabled
        if self.use_augmented_features:
            x = self._augment_matrix(M_for_cnn, mask)  # (batch, 3, m, n)
        else:
            x = M_for_cnn.unsqueeze(1)  # (batch, 1, m, n)
        
        # Pad to consistent size
        x = self._pad_to_size(x, self.max_seq_length, pad_value=0.0)
        
        # CNN forward pass
        x = self.conv(x)  # (batch, final_channels, h', w')
        
        # Global pooling and MLP
        x = self.global_pool(x)  # (batch, final_channels, 1, 1)
        x = x.flatten(1)  # (batch, final_channels)
        raw_scores = self.mlp(x).squeeze(-1)  # (batch,)
        
        # Score calibration: bound output to [-1, 1] using tanh
        scores = torch.tanh(raw_scores)
        
        return scores


class F4Verifier(BaseVerifier):
    """
    F4 (matrix variant): Tiny Transformer over a (possibly augmented) similarity matrix.

    Pipeline:
      1) M = sim(q, c)  -> (B, m, n)
      2) Optional augmentation φ(M): [M, row_max, col_max] -> (B, C, m, n)
      3) Pad/trim to fixed (max_q_len, max_c_len)
      4) Patch-embed via Conv2d(stride=patch_size) -> tokens (B, P, d)
      5) Transformer over tokens (+ [CLS]) -> score via MLP -> tanh to [-1, 1]

    Notes:
      - This keeps the strong inductive bias of alignment patterns (like F3),
        but gives you a learned global aggregator (Transformer).
      - Patch masking is derived from the token-pair mask: if a patch contains
        no valid pairs, it’s masked out in attention.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_q_len: int = 128,
        max_c_len: int = 128,
        patch_size: int = 8,
        use_augmented_features: bool = True,  # M, row_max, col_max
        use_conditional_penalty: bool = False,
        penalty_beta: float = 0.5,
        penalty_tau: float = 0.9,
    ):
        super().__init__(
            name=f"F4_SimTransformer_{num_layers}L",
            use_conditional_penalty=use_conditional_penalty,
            penalty_beta=penalty_beta,
            penalty_tau=penalty_tau,
        )

        if max_q_len % patch_size != 0 or max_c_len % patch_size != 0:
            raise ValueError(
                f"max_q_len ({max_q_len}) and max_c_len ({max_c_len}) must be divisible "
                f"by patch_size ({patch_size})."
            )

        self.similarity = TokenSimilarityMatrix(normalize=True)
        self.use_augmented_features = use_augmented_features

        self.max_q_len = max_q_len
        self.max_c_len = max_c_len
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

        in_channels = 3 if use_augmented_features else 1

        # Patch embedding like ViT: (B, C, H, W) -> (B, D, H/P, W/P)
        self.patch_embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

        # Token counts are fixed due to fixed pad/trim and fixed patching
        self.grid_h = max_q_len // patch_size
        self.grid_w = max_c_len // patch_size
        self.num_patches = self.grid_h * self.grid_w

        # Learnable [CLS] token and positional embeddings (CLS + patches)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embedding_dim))

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )

        # Initialize like common transformer practice
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def _pad_or_trim_2d(self, x: torch.Tensor, target_h: int, target_w: int, pad_value: float = 0.0) -> torch.Tensor:
        """
        x: (B, C, H, W) -> pad/trim to (B, C, target_h, target_w)
        """
        b, c, h, w = x.shape
        # Pad if needed
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), value=pad_value)
        # Trim if needed
        return x[:, :, :target_h, :target_w]

    def _augment_matrix(self, M: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        M: (B, H, W), mask: (B, H, W) boolean or None
        Returns: (B, 3, H, W) with [M, row_max, col_max]
        """
        if mask is not None:
            M_masked = M.masked_fill(~mask, float("-inf"))
            row_max = M_masked.max(dim=-1, keepdim=True)[0]  # (B, H, 1)
            col_max = M_masked.max(dim=-2, keepdim=True)[0]  # (B, 1, W)
            row_max = row_max.masked_fill(row_max == float("-inf"), 0.0)
            col_max = col_max.masked_fill(col_max == float("-inf"), 0.0)
        else:
            row_max = M.max(dim=-1, keepdim=True)[0]
            col_max = M.max(dim=-2, keepdim=True)[0]

        row_max = row_max.expand_as(M)
        col_max = col_max.expand_as(M)
        return torch.stack([M, row_max, col_max], dim=1)

    def compute_raw_score(
        self,
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        query_embeddings: (B, m, d)
        candidate_embeddings: (B, n, d)
        query_mask: (B, m) with 1 for valid, 0 for pad (optional)
        candidate_mask: (B, n) with 1 for valid, 0 for pad (optional)
        """
        # 1) Similarity matrix
        # M: (B, m, n)
        M = self.similarity(query_embeddings, candidate_embeddings)

        # 2) Pairwise mask (valid token pairs)
        pair_mask = None
        if query_mask is not None and candidate_mask is not None:
            pair_mask = (query_mask.unsqueeze(-1) * candidate_mask.unsqueeze(-2)).bool()  # (B, m, n)

        # 3) Prepare channels (M or augmented)
        if self.use_augmented_features:
            x = self._augment_matrix(M, pair_mask)  # (B, 3, m, n)
        else:
            x = M.unsqueeze(1)  # (B, 1, m, n)

        # For the actual input tensor, set invalid pairs to 0 (don’t leak pad patterns)
        if pair_mask is not None:
            # pair_mask: (B, m, n) -> broadcast to channels
            x = x.masked_fill(~pair_mask.unsqueeze(1), 0.0)

        # 4) Pad/trim to fixed (max_q_len, max_c_len)
        x = self._pad_or_trim_2d(x, self.max_q_len, self.max_c_len, pad_value=0.0)  # (B, C, H, W)

        # 5) Patch mask (mask out patches that contain no valid pairs)
        src_key_padding_mask = None
        if pair_mask is not None:
            pm = pair_mask.unsqueeze(1).float()  # (B, 1, m, n)
            pm = self._pad_or_trim_2d(pm, self.max_q_len, self.max_c_len, pad_value=0.0)  # (B, 1, H, W)

            # Downsample to patch grid: valid if ANY valid pair exists in patch
            patch_valid = F.max_pool2d(pm, kernel_size=self.patch_size, stride=self.patch_size)  # (B, 1, Gh, Gw)
            patch_valid = patch_valid.squeeze(1)  # (B, Gh, Gw)
            patch_valid = patch_valid.reshape(patch_valid.size(0), -1)  # (B, P)

            # Transformer wants True for "ignore/mask"
            src_key_padding_mask = (patch_valid == 0)

        # 6) Patch embed -> tokens
        # (B, D, Gh, Gw)
        tokens = self.patch_embed(x)
        # (B, P, D)
        tokens = tokens.flatten(2).transpose(1, 2)

        # 7) Add [CLS] and positional embeddings
        bsz = tokens.size(0)
        cls = self.cls_token.expand(bsz, -1, -1)  # (B, 1, D)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, 1+P, D)

        # Build padding mask including CLS (never masked)
        if src_key_padding_mask is not None:
            cls_pad = torch.zeros(bsz, 1, dtype=torch.bool, device=tokens.device)
            src_key_padding_mask = torch.cat([cls_pad, src_key_padding_mask], dim=1)  # (B, 1+P)

        tokens = tokens + self.pos_embed
        tokens = self.layer_norm(tokens)
        tokens = self.dropout(tokens)

        # 8) Transformer
        out = self.transformer(tokens, src_key_padding_mask=src_key_padding_mask)

        # 9) Score from CLS
        cls_out = out[:, 0, :]
        raw = self.mlp(cls_out).squeeze(-1)
        return torch.tanh(raw)



class TokenEmbeddingExtractor:
    """
    Extract token-level embeddings from a frozen SentenceTransformer.
    
    This is used to get the token embeddings Q and C that are fed into the verifiers.
    """
    
    def __init__(
        self,
        model: SentenceTransformer,
        device: str = "cuda",
    ):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Freeze the model
        for param in self.model.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def extract(
        self,
        texts: List[str],
        max_length: int = 128,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract token embeddings for a batch of texts.
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length
        
        Returns:
            token_embeddings: (batch, seq_len, hidden_dim)
            attention_mask: (batch, seq_len)
        """
        # Tokenize
        tokenizer = self.model.tokenizer
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        # Get token embeddings from the transformer
        transformer = self.model[0]  # First module is the Transformer
        outputs = transformer.auto_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Use last hidden state as token embeddings
        token_embeddings = outputs.last_hidden_state
        
        return token_embeddings, attention_mask.float()


def create_verifier(
    verifier_type: str,
    use_conditional_penalty: bool = False,
    penalty_beta: float = 0.5,
    penalty_tau: float = 0.9,
    **kwargs,
) -> BaseVerifier:
    """
    Factory function to create verifiers.
    
    Args:
        verifier_type: One of "None", "F0", "F1", "F2", "F3", "F4"
        use_conditional_penalty: If True, use conditional penalty mode where
            verifier acts as structural mismatch detector applied only when
            base similarity > tau
        penalty_beta: Strength of structural penalty (default: 0.5, aggressive)
        penalty_tau: Similarity threshold for applying penalty (default: 0.9)
        **kwargs: Additional arguments for the specific verifier
    
    Returns:
        Initialized verifier instance
    """
    # Common conditional penalty args
    cp_kwargs = {
        "use_conditional_penalty": use_conditional_penalty,
        "penalty_beta": penalty_beta,
        "penalty_tau": penalty_tau,
    }
    
    verifier_map = {
        "None": NoneVerifier,
        "F0": F0Verifier,
        "F1": F1Verifier,
        "F2": F2SoftAttnVerifier,
        "F3": F3Verifier,
        "F4": F4Verifier,
    }
    
    if verifier_type not in verifier_map:
        raise ValueError(f"Unknown verifier type: {verifier_type}. Choose from {list(verifier_map.keys())}")
    
    # Merge conditional penalty kwargs with user kwargs
    all_kwargs = {**cp_kwargs, **kwargs}
    
    return verifier_map[verifier_type](**all_kwargs)


def get_all_verifiers(
    f2_lambda: float = 0.1,
    f3_kwargs: Optional[dict] = None,
    f4_kwargs: Optional[dict] = None,
    use_conditional_penalty: bool = False,
    penalty_beta: float = 0.5,
    penalty_tau: float = 0.9,
) -> List[BaseVerifier]:
    """
    Create all verifiers with default configurations.
    
    Args:
        f2_lambda: Lambda penalty for F2 verifier
        f3_kwargs: Additional kwargs for F3 verifier
        f4_kwargs: Additional kwargs for F4 verifier
        use_conditional_penalty: Enable conditional penalty mode for all verifiers
        penalty_beta: Strength of structural penalty
        penalty_tau: Similarity threshold for applying penalty
    
    Returns:
        List of all verifier instances
    """
    f3_kwargs = f3_kwargs or {}
    f4_kwargs = f4_kwargs or {}
    
    cp_kwargs = {
        "use_conditional_penalty": use_conditional_penalty,
        "penalty_beta": penalty_beta,
        "penalty_tau": penalty_tau,
    }
    
    return [
        NoneVerifier(**cp_kwargs),
        F0Verifier(**cp_kwargs),
        F1Verifier(**cp_kwargs),
        F2SoftAttnVerifier(lambda_penalty=f2_lambda, **cp_kwargs),
        F3Verifier(**f3_kwargs, **cp_kwargs),
        F4Verifier(**f4_kwargs, **cp_kwargs),
    ]


# Backwards compatibility alias
F2Verifier = F2SoftAttnVerifier


if __name__ == "__main__":
    # Quick test
    print("Testing verifiers...")
    
    batch_size = 4
    query_len = 16
    cand_len = 32
    hidden_dim = 384
    
    # Random embeddings for testing
    query_emb = torch.randn(batch_size, query_len, hidden_dim)
    cand_emb = torch.randn(batch_size, cand_len, hidden_dim)
    query_mask = torch.ones(batch_size, query_len)
    cand_mask = torch.ones(batch_size, cand_len)
    
    print("\n=== Standard Mode ===")
    # Test each verifier in standard mode
    for verifier_type in ["None", "F0", "F1", "F2", "F3", "F4"]:
        if verifier_type == "F4":
            verifier = create_verifier(verifier_type, embedding_dim=hidden_dim)
        else:
            verifier = create_verifier(verifier_type)
        
        scores = verifier(query_emb, cand_emb, query_mask, cand_mask)
        print(f"{verifier.name}: scores shape = {scores.shape}, "
              f"range = [{scores.min().item():.3f}, {scores.max().item():.3f}], "
              f"params = {verifier.num_parameters:,}")
    
    print("\n=== Conditional Penalty Mode (beta=0.5, tau=0.9) ===")
    # Test each verifier in conditional penalty mode
    for verifier_type in ["None", "F0", "F1", "F2", "F3", "F4"]:
        if verifier_type == "F4":
            verifier = create_verifier(
                verifier_type, 
                embedding_dim=hidden_dim,
                use_conditional_penalty=True,
                penalty_beta=0.5,
                penalty_tau=0.9,
            )
        else:
            verifier = create_verifier(
                verifier_type,
                use_conditional_penalty=True,
                penalty_beta=0.5,
                penalty_tau=0.9,
            )
        
        scores = verifier(query_emb, cand_emb, query_mask, cand_mask)
        print(f"{verifier.name}: scores shape = {scores.shape}, "
              f"range = [{scores.min().item():.3f}, {scores.max().item():.3f}], "
              f"params = {verifier.num_parameters:,}")
    
    print("\nAll verifiers working correctly!")
