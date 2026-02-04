#!/usr/bin/env python3
"""
Comprehensive Verifier Training and Evaluation Experiment.

This script runs two experiments:
1. Train all verifiers (F0-F4) with a FROZEN encoder on data_full/train_structured.json
2. Train all verifiers END-TO-END (encoder unfrozen) on data_full/train_structured.json

Then evaluates on:
- NanoBEIR datasets (NDCG, MRR, ACC@1, ACC@10)
- MRPC unified negative pairs (cosine similarity analysis)

The goal is to show:
- Training on structural data doesn't degrade retrieval performance
- End-to-end training improves performance on structural perturbations
- F0/F1 should perform worse than F2-F4 on structural tasks

Usage:
    python train_and_eval_verifiers_experiment.py \
        --encoder sentence-transformers/all-MiniLM-L6-v2 \
        --output-dir ./verifier_experiment_results
"""

import json
import argparse
import os
import random
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from verifiers import (
    BaseVerifier,
    NoneVerifier,
    F0Verifier,
    F1Verifier,
    F2SoftAttnVerifier,
    F3Verifier,
    F4Verifier,
)

# Import shared utilities
from training_utils import (
    TripletDataset,
    load_triplets,
    collate_triplets,
    TrainableTokenEmbeddingExtractor,
)
from evaluation_utils import (
    NanoBEIRDataset,
    compute_retrieval_metrics,
)

# Optimize CUDA settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class VerifierResult:
    """Results for a single verifier configuration."""
    verifier_type: str
    training_mode: str  # "frozen" or "end_to_end"
    num_parameters: int
    training_time_seconds: float
    nanobeir_encoder_only: Dict  # Encoder-only cosine similarity metrics
    nanobeir_with_verifier: Dict  # Two-stage retrieval with verifier reranking
    mrpc_encoder_only: Dict  # Encoder-only cosine similarity on MRPC negatives
    mrpc_with_verifier: Dict  # Verifier scores on MRPC negatives


# Verifiers that support end-to-end training (gradient flow through encoder)
E2E_COMPATIBLE_VERIFIERS = ["None", "F0", "F1", "F2", "F3", "F4"]


# =============================================================================
# Verifier Trainer (supports frozen and end-to-end training)
# =============================================================================

class UnifiedVerifierTrainer:
    """
    Trainer for verifiers that supports both frozen and end-to-end training.
    
    For end-to-end training, gradients flow through the encoder as well.
    
    Supports two loss functions:
    1. Pairwise loss: Standard triplet-style loss with explicit negative
    2. MNRL loss: Multiple Negatives Ranking Loss with in-batch negatives
    """
    
    def __init__(
        self,
        verifier: BaseVerifier,
        encoder: SentenceTransformer,
        freeze_encoder: bool = True,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        encoder_learning_rate: float = 1e-5,
        weight_decay: float = 0.0001,
        encoder_weight_decay: float = 0.01,
        max_seq_length: int = 128,
        use_mnrl_loss: bool = True,
        mnrl_temperature: float = 0.05,
        warmup_ratio: float = 0.1,
        lr_scheduler_type: str = "linear",
    ):
        self.device = device
        self.verifier = verifier.to(device)
        self.freeze_encoder = freeze_encoder
        self.max_seq_length = max_seq_length
        self.weight_decay = weight_decay
        self.encoder_weight_decay = encoder_weight_decay
        self.use_mnrl_loss = use_mnrl_loss
        self.mnrl_temperature = mnrl_temperature
        self.warmup_ratio = warmup_ratio
        self.lr_scheduler_type = lr_scheduler_type
        
        # Create token embedding extractor (using shared TrainableTokenEmbeddingExtractor)
        self.token_extractor = TrainableTokenEmbeddingExtractor(
            encoder,
            freeze_encoder=freeze_encoder,
            device=device,
        )
        
        # Build optimizer
        # For end-to-end training, we use different LRs and weight decays for encoder vs verifier
        verifier_params = list(self.verifier.parameters())
        has_verifier_params = len(verifier_params) > 0
        
        if freeze_encoder and not has_verifier_params:
            # No trainable parameters at all (e.g., F0/F1/F2 with frozen encoder)
            self.optimizer = None
            self.trainable = False
        elif freeze_encoder:
            # Only verifier parameters
            self.optimizer = AdamW(
                verifier_params,
                lr=learning_rate,
                weight_decay=weight_decay,
            )
            self.trainable = True
        else:
            # End-to-end: encoder + optionally verifier with separate weight decays
            encoder_params = list(self.token_extractor.model.parameters())
            
            param_groups = []
            if has_verifier_params:
                param_groups.append({
                    "params": verifier_params,
                    "lr": learning_rate,
                    "weight_decay": weight_decay,
                })
            param_groups.append({
                "params": encoder_params,
                "lr": encoder_learning_rate,
                "weight_decay": encoder_weight_decay,
            })
            
            self.optimizer = AdamW(param_groups)
            self.trainable = True
        
        self.best_loss = float('inf')
        self.best_verifier_state = None
        self.best_encoder_state = None
    
    def compute_pairwise_loss(
        self,
        anchor_emb: torch.Tensor,
        anchor_mask: torch.Tensor,
        pos_emb: torch.Tensor,
        pos_mask: torch.Tensor,
        neg_emb: torch.Tensor,
        neg_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute pairwise softmax cross-entropy loss (ColBERT-style).
        
        L = -log(exp(S(q, d+)) / (exp(S(q, d+)) + exp(S(q, d-))))
        
        This is equivalent to binary cross-entropy over the softmax of scores.
        """
        pos_scores = self.verifier(anchor_emb, pos_emb, anchor_mask, pos_mask)
        neg_scores = self.verifier(anchor_emb, neg_emb, anchor_mask, neg_mask)
        
        # Pairwise softmax cross-entropy: -log(exp(pos) / (exp(pos) + exp(neg)))
        scores = torch.stack([pos_scores, neg_scores], dim=1)  # (batch, 2)
        log_sum_exp = torch.logsumexp(scores, dim=1)  # (batch,)
        loss = (log_sum_exp - pos_scores).mean()
        
        accuracy = (pos_scores > neg_scores).float().mean()
        
        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "pos_score_mean": pos_scores.mean().item(),
            "neg_score_mean": neg_scores.mean().item(),
            "score_margin": (pos_scores - neg_scores).mean().item(),
        }
        
        return loss, metrics
    
    def compute_mnrl_loss(
        self,
        anchor_emb: torch.Tensor,
        anchor_mask: torch.Tensor,
        pos_emb: torch.Tensor,
        pos_mask: torch.Tensor,
        neg_emb: torch.Tensor,
        neg_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute Multiple Negatives Ranking Loss (MNRL) with in-batch negatives.
        
        For each anchor, treats all other positives in the batch as additional
        negatives. This provides a richer contrastive signal.
        
        L = -log(exp(S(a_i, p_i)/τ) / (sum_j exp(S(a_i, p_j)/τ) + exp(S(a_i, n_i)/τ)))
        
        where τ is temperature, p_i is the positive for anchor i, p_j are in-batch
        negatives (other positives), and n_i is the explicit negative.
        """
        B = anchor_emb.shape[0]
        device = anchor_emb.device
        
        # Compute all pairwise scores between anchors and positives: (B, B)
        # scores_matrix[i, j] = verifier(anchor_i, positive_j)
        scores_list = []
        for i in range(B):
            # Expand anchor_i to score against all positives
            anchor_i_emb = anchor_emb[i:i+1].expand(B, -1, -1)  # (B, m, d)
            anchor_i_mask = anchor_mask[i:i+1].expand(B, -1)    # (B, m)
            
            # Score anchor_i against all positives
            row_scores = self.verifier(anchor_i_emb, pos_emb, anchor_i_mask, pos_mask)  # (B,)
            scores_list.append(row_scores)
        
        scores_matrix = torch.stack(scores_list)  # (B, B)
        
        # Compute scores for explicit negatives: (B,)
        neg_scores = self.verifier(anchor_emb, neg_emb, anchor_mask, neg_mask)
        
        # Append explicit negatives as additional column: (B, B+1)
        scores_with_neg = torch.cat([scores_matrix, neg_scores.unsqueeze(1)], dim=1)
        
        # Apply temperature scaling
        scores_scaled = scores_with_neg / self.mnrl_temperature
        
        # Labels: for anchor i, the positive is at index i (diagonal)
        labels = torch.arange(B, device=device)
        
        # Cross-entropy loss (InfoNCE)
        loss = F.cross_entropy(scores_scaled, labels)
        
        # Compute metrics
        with torch.no_grad():
            # Accuracy: does the model rank the true positive highest?
            preds = scores_scaled.argmax(dim=1)
            accuracy = (preds == labels).float().mean()
            
            # Diagonal scores are positive scores
            pos_scores_diag = scores_matrix[torch.arange(B), torch.arange(B)]
            
            # Mean of off-diagonal (in-batch negatives)
            mask = ~torch.eye(B, dtype=torch.bool, device=device)
            in_batch_neg_mean = scores_matrix[mask].mean() if B > 1 else torch.tensor(0.0)
        
        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "pos_score_mean": pos_scores_diag.mean().item(),
            "neg_score_mean": neg_scores.mean().item(),
            "in_batch_neg_mean": in_batch_neg_mean.item(),
            "score_margin": (pos_scores_diag - neg_scores).mean().item(),
        }
        
        return loss, metrics
    
    def compute_loss(
        self,
        anchor_emb: torch.Tensor,
        anchor_mask: torch.Tensor,
        pos_emb: torch.Tensor,
        pos_mask: torch.Tensor,
        neg_emb: torch.Tensor,
        neg_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute loss using the configured loss function."""
        if self.use_mnrl_loss:
            return self.compute_mnrl_loss(
                anchor_emb, anchor_mask,
                pos_emb, pos_mask,
                neg_emb, neg_mask,
            )
        else:
            return self.compute_pairwise_loss(
                anchor_emb, anchor_mask,
                pos_emb, pos_mask,
                neg_emb, neg_mask,
            )
    
    def train_step(
        self,
        anchors: List[str],
        positives: List[str],
        negatives: List[str],
    ) -> Dict:
        """Execute a single training step."""
        if not self.trainable:
            # No trainable parameters, just evaluate
            return self.eval_step(anchors, positives, negatives)
        
        self.verifier.train()
        if not self.freeze_encoder:
            self.token_extractor.model.train()
        
        anchor_emb, anchor_mask = self.token_extractor.extract(anchors, self.max_seq_length)
        pos_emb, pos_mask = self.token_extractor.extract(positives, self.max_seq_length)
        neg_emb, neg_mask = self.token_extractor.extract(negatives, self.max_seq_length)
        
        loss, metrics = self.compute_loss(
            anchor_emb, anchor_mask,
            pos_emb, pos_mask,
            neg_emb, neg_mask,
        )
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.verifier.num_parameters > 0:
            torch.nn.utils.clip_grad_norm_(self.verifier.parameters(), max_norm=1.0)
        if not self.freeze_encoder:
            torch.nn.utils.clip_grad_norm_(self.token_extractor.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return metrics
    
    @torch.no_grad()
    def eval_step(
        self,
        anchors: List[str],
        positives: List[str],
        negatives: List[str],
    ) -> Dict:
        """Execute a single evaluation step."""
        self.verifier.eval()
        self.token_extractor.model.eval()
        
        anchor_emb, anchor_mask = self.token_extractor.extract(anchors, self.max_seq_length)
        pos_emb, pos_mask = self.token_extractor.extract(positives, self.max_seq_length)
        neg_emb, neg_mask = self.token_extractor.extract(negatives, self.max_seq_length)
        
        _, metrics = self.compute_loss(
            anchor_emb, anchor_mask,
            pos_emb, pos_mask,
            neg_emb, neg_mask,
        )
        
        return metrics
    
    def _create_scheduler(self, max_steps: int) -> LambdaLR:
        """Create learning rate scheduler with warmup.
        
        Supports:
        - "linear": Linear decay after warmup (matches HuggingFace Trainer default)
        - "cosine": Cosine annealing after warmup
        """
        warmup_steps = int(max_steps * self.warmup_ratio)
        
        def lr_lambda(current_step: int) -> float:
            # Warmup phase
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            
            # Decay phase
            progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            
            if self.lr_scheduler_type == "cosine":
                import math
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            else:  # linear
                return max(0.0, 1.0 - progress)
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        max_steps: int = 5000,
        eval_steps: int = 500,
        logging_steps: int = 100,
        save_path: Optional[Path] = None,
        early_stopping_patience: int = 5,
        early_stopping_threshold: float = 0.001,
        use_wandb: bool = False,
        # Retrieval validation parameters
        retrieval_val_datasets: Optional[List[str]] = None,
        retrieval_val_steps: Optional[int] = None,
        use_retrieval_early_stopping: bool = False,
        retrieval_val_batch_size: int = 32,
    ) -> Dict:
        """Train the verifier (and optionally encoder) with early stopping.
        
        Args:
            train_dataloader: DataLoader for training triplets
            eval_dataloader: Optional DataLoader for triplet-based evaluation
            max_steps: Maximum number of training steps
            eval_steps: Evaluate every N steps
            logging_steps: Log metrics every N steps
            save_path: Path to save checkpoints
            early_stopping_patience: Number of eval steps without improvement before stopping
            early_stopping_threshold: Minimum improvement required to reset patience
            use_wandb: Whether to log to Weights & Biases
            retrieval_val_datasets: List of NanoBEIR dataset names for retrieval validation
                                    (e.g., ["nq", "fiqa"]). If None, no retrieval validation.
            retrieval_val_steps: Evaluate retrieval every N steps (defaults to eval_steps)
            use_retrieval_early_stopping: Use retrieval metrics (NDCG@10) for early stopping
                                          instead of triplet accuracy
            retrieval_val_batch_size: Batch size for retrieval validation encoding
        
        Returns:
            Dict with training results including best metrics
        """
        # Set default retrieval validation steps
        if retrieval_val_steps is None:
            retrieval_val_steps = eval_steps
        
        if not self.trainable:
            print("  No trainable parameters, skipping training.")
            return {"best_loss": 0.0}
        
        scheduler = self._create_scheduler(max_steps)
        warmup_steps = int(max_steps * self.warmup_ratio)
        
        global_step = 0
        running_loss = 0.0
        running_acc = 0.0
        running_pos_score = 0.0
        running_neg_score = 0.0
        running_margin = 0.0
        
        # Early stopping tracking
        patience_counter = 0
        best_eval_acc = -float('inf')
        best_retrieval_ndcg = -float('inf')
        
        mode = "frozen" if self.freeze_encoder else "end-to-end"
        print(f"\nTraining {self.verifier.name} ({mode}) for up to {max_steps} iterations")
        print(f"Verifier params: {self.verifier.num_parameters:,}")
        print(f"LR scheduler: {self.lr_scheduler_type} with {self.warmup_ratio*100:.0f}% warmup ({warmup_steps} steps)")
        
        # Log early stopping mode
        if use_retrieval_early_stopping and retrieval_val_datasets:
            es_metric = "retrieval NDCG@10"
        else:
            es_metric = "triplet accuracy"
        print(f"Early stopping: patience={early_stopping_patience}, threshold={early_stopping_threshold}, metric={es_metric}")
        
        # Log retrieval validation info
        if retrieval_val_datasets:
            print(f"Retrieval validation: {retrieval_val_datasets} every {retrieval_val_steps} steps")
        
        if not self.freeze_encoder:
            encoder_params = sum(p.numel() for p in self.token_extractor.model.parameters())
            print(f"Encoder params: {encoder_params:,}")
        
        data_iter = iter(train_dataloader)
        pbar = tqdm(total=max_steps, desc=f"Training ({mode})")
        
        while global_step < max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_dataloader)
                batch = next(data_iter)
            
            anchors, positives, negatives = batch
            
            metrics = self.train_step(anchors, positives, negatives)
            scheduler.step()
            
            running_loss += metrics["loss"]
            running_acc += metrics["accuracy"]
            running_pos_score += metrics["pos_score_mean"]
            running_neg_score += metrics["neg_score_mean"]
            running_margin += metrics["score_margin"]
            global_step += 1
            pbar.update(1)
            
            if global_step % logging_steps == 0:
                avg_loss = running_loss / logging_steps
                avg_acc = running_acc / logging_steps
                avg_pos_score = running_pos_score / logging_steps
                avg_neg_score = running_neg_score / logging_steps
                avg_margin = running_margin / logging_steps
                
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "acc": f"{avg_acc:.4f}",
                    "patience": f"{patience_counter}/{early_stopping_patience}",
                })
                
                # Log to wandb
                if use_wandb and WANDB_AVAILABLE:
                    current_lr = scheduler.get_last_lr()[0]
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/accuracy": avg_acc,
                        "train/pos_score_mean": avg_pos_score,
                        "train/neg_score_mean": avg_neg_score,
                        "train/score_margin": avg_margin,
                        "train/learning_rate": current_lr,
                        "train/step": global_step,
                    }, step=global_step)
                
                running_loss = 0.0
                running_acc = 0.0
                running_pos_score = 0.0
                running_neg_score = 0.0
                running_margin = 0.0
            
            # Triplet-based evaluation
            if eval_dataloader is not None and global_step % eval_steps == 0:
                eval_metrics = self.evaluate(eval_dataloader)
                eval_loss = eval_metrics["loss"]
                eval_acc = eval_metrics["accuracy"]
                
                # Log eval metrics to wandb
                if use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        "eval/loss": eval_loss,
                        "eval/accuracy": eval_acc,
                        "eval/best_accuracy": best_eval_acc if best_eval_acc > -float('inf') else eval_acc,
                    }, step=global_step)
                
                # Update best triplet accuracy (for tracking, may not be used for early stopping)
                if eval_acc > best_eval_acc:
                    best_eval_acc = eval_acc
                
                # Early stopping based on triplet accuracy (if not using retrieval)
                if not use_retrieval_early_stopping:
                    if eval_acc > (best_eval_acc - early_stopping_threshold):
                        # Improvement - reset counter and save
                        if eval_acc > best_eval_acc:
                            patience_counter = 0
                            self._save_best_model(save_path)
                    else:
                        patience_counter += 1
                    
                    if use_wandb and WANDB_AVAILABLE:
                        wandb.log({"eval/patience_counter": patience_counter}, step=global_step)
                    
                    if patience_counter >= early_stopping_patience:
                        print(f"\n  Early stopping triggered at step {global_step} (patience={early_stopping_patience})")
                        break
            
            # Retrieval-based validation
            if retrieval_val_datasets and global_step % retrieval_val_steps == 0:
                # Run retrieval evaluation on validation datasets
                retrieval_results = self._evaluate_retrieval(
                    datasets=retrieval_val_datasets,
                    batch_size=retrieval_val_batch_size,
                )
                
                current_ndcg = retrieval_results["mean_ndcg@10"]
                
                # Log retrieval metrics to wandb
                if use_wandb and WANDB_AVAILABLE:
                    wandb_log = {
                        "retrieval_val/mean_ndcg@10": current_ndcg,
                        "retrieval_val/mean_mrr@10": retrieval_results["mean_mrr@10"],
                        "retrieval_val/mean_acc@1": retrieval_results["mean_accuracy@1"],
                        "retrieval_val/mean_acc@10": retrieval_results["mean_accuracy@10"],
                        "retrieval_val/best_ndcg@10": best_retrieval_ndcg if best_retrieval_ndcg > -float('inf') else current_ndcg,
                    }
                    # Log per-dataset metrics
                    for dataset_name in retrieval_val_datasets:
                        for metric in ["ndcg@10", "mrr@10", "accuracy@1", "accuracy@10"]:
                            key = f"{dataset_name}_{metric}"
                            if key in retrieval_results:
                                wandb_log[f"retrieval_val/{key}"] = retrieval_results[key]
                    wandb.log(wandb_log, step=global_step)
                
                # Early stopping based on retrieval metrics
                if use_retrieval_early_stopping:
                    if current_ndcg > (best_retrieval_ndcg + early_stopping_threshold):
                        best_retrieval_ndcg = current_ndcg
                        patience_counter = 0
                        self._save_best_model(save_path)
                        print(f"\n    New best NDCG@10: {current_ndcg:.4f}")
                    else:
                        patience_counter += 1
                    
                    if use_wandb and WANDB_AVAILABLE:
                        wandb.log({"retrieval_val/patience_counter": patience_counter}, step=global_step)
                    
                    if patience_counter >= early_stopping_patience:
                        print(f"\n  Early stopping triggered at step {global_step} (patience={early_stopping_patience}, metric=NDCG@10)")
                        break
        
        pbar.close()
        
        # Load best model
        if self.best_verifier_state is not None:
            self.verifier.load_state_dict(self.best_verifier_state)
        if self.best_encoder_state is not None and not self.freeze_encoder:
            self.token_extractor.model.load_state_dict(self.best_encoder_state)
        
        # Log final best metrics to wandb
        if use_wandb and WANDB_AVAILABLE:
            final_metrics = {
                "final/best_eval_accuracy": best_eval_acc,
                "final/steps_trained": global_step,
            }
            if retrieval_val_datasets:
                final_metrics["final/best_retrieval_ndcg@10"] = best_retrieval_ndcg
            wandb.log(final_metrics)
        
        return {
            "best_eval_acc": best_eval_acc,
            "best_retrieval_ndcg": best_retrieval_ndcg if retrieval_val_datasets else None,
            "steps_trained": global_step,
        }
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict:
        """Evaluate the verifier on a dataset."""
        self.verifier.eval()
        self.token_extractor.model.eval()
        
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        for batch in dataloader:
            anchors, positives, negatives = batch
            metrics = self.eval_step(anchors, positives, negatives)
            total_loss += metrics["loss"]
            total_acc += metrics["accuracy"]
            num_batches += 1
            if num_batches >= 50:  # Limit eval batches for speed
                break
        
        return {
            "loss": total_loss / num_batches,
            "accuracy": total_acc / num_batches,
        }
    
    def _save_best_model(self, save_path: Optional[Path] = None):
        """Save the current model as the best model."""
        self.best_verifier_state = {
            k: v.cpu().clone() for k, v in self.verifier.state_dict().items()
        }
        if not self.freeze_encoder:
            self.best_encoder_state = {
                k: v.cpu().clone() 
                for k, v in self.token_extractor.model.state_dict().items()
            }
        if save_path:
            self.save_checkpoint(save_path / "best_model.pt")
    
    @torch.no_grad()
    def _evaluate_retrieval(
        self,
        datasets: List[str],
        batch_size: int = 32,
        top_k: int = 100,
    ) -> Dict:
        """
        Evaluate retrieval performance on NanoBEIR validation datasets.
        
        Uses encoder-only cosine similarity (faster than full verifier reranking).
        
        Args:
            datasets: List of NanoBEIR dataset names (e.g., ["nq", "fiqa"])
            batch_size: Batch size for encoding
            top_k: Number of top candidates to retrieve
        
        Returns:
            Dict with per-dataset and mean metrics
        """
        self.verifier.eval()
        self.token_extractor.model.eval()
        
        encoder = self.token_extractor.model
        
        all_metrics = {}
        
        for dataset_name in datasets:
            # Load dataset
            dataset = NanoBEIRDataset(dataset_name)
            
            corpus_texts = dataset.get_corpus_texts()
            query_texts = dataset.get_query_texts()
            
            # Encode corpus and queries
            corpus_embeddings = encoder.encode(
                corpus_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            query_embeddings = encoder.encode(
                query_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            
            # Compute similarities and get top-K
            similarities = query_embeddings @ corpus_embeddings.T
            top_k_indices = np.argsort(-similarities, axis=1)[:, :top_k]
            
            # Build rankings
            rankings = {}
            for i, query_id in enumerate(dataset.query_ids):
                ranked_doc_ids = [dataset.corpus_ids[idx] for idx in top_k_indices[i]]
                rankings[query_id] = ranked_doc_ids
            
            # Compute metrics
            metrics = compute_retrieval_metrics(rankings, dataset.qrels)
            
            # Store per-dataset metrics
            for metric, value in metrics.items():
                all_metrics[f"{dataset_name}_{metric}"] = value
        
        # Compute mean metrics across datasets
        metric_names = ["ndcg@10", "mrr@10", "accuracy@1", "accuracy@10"]
        for metric in metric_names:
            values = [v for k, v in all_metrics.items() if metric in k]
            all_metrics[f"mean_{metric}"] = np.mean(values) if values else 0.0
        
        return all_metrics
    
    def save_checkpoint(self, path: Path):
        """Save checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "verifier_state": self.verifier.state_dict(),
            "verifier_name": self.verifier.name,
            "freeze_encoder": self.freeze_encoder,
        }
        if not self.freeze_encoder:
            checkpoint["encoder_state"] = self.token_extractor.model.state_dict()
        torch.save(checkpoint, path)
    
    def get_encoder(self) -> SentenceTransformer:
        """Get the (possibly trained) encoder."""
        return self.token_extractor.model


# =============================================================================
# Evaluation Functions
# =============================================================================

@torch.no_grad()
def evaluate_on_nanobeir(
    encoder: SentenceTransformer,
    token_extractor: Optional[TrainableTokenEmbeddingExtractor] = None,
    verifier: Optional[BaseVerifier] = None,
    datasets: Optional[List[str]] = None,
    top_k: int = 100,
    batch_size: int = 32,
    max_seq_length: int = 128,
    use_verifier: bool = False,
) -> Dict:
    """
    Evaluate encoder on NanoBEIR using two-stage retrieval.
    
    Stage 1: Cosine similarity retrieval to get top-K candidates
    Stage 2 (optional): Verifier reranking of top-K candidates (if use_verifier=True)
    
    This ensures both encoder-only and verifier evaluations use the exact same
    retrieval pipeline, eliminating any discrepancies in metric computation.
    
    Args:
        encoder: The sentence transformer encoder
        token_extractor: Token embedding extractor (required if use_verifier=True)
        verifier: The verifier model (required if use_verifier=True)
        datasets: List of NanoBEIR dataset names to evaluate on
        top_k: Number of top candidates to retrieve/rerank
        batch_size: Batch size for encoding
        max_seq_length: Maximum sequence length for token extraction
        use_verifier: Whether to apply verifier reranking
    
    Returns:
        Dict with metrics before reranking (encoder-only) and after reranking (with verifier if enabled)
    """
    if datasets is None:
        datasets = ["msmarco", "nq"]
    
    if use_verifier and (verifier is None or token_extractor is None):
        raise ValueError("verifier and token_extractor are required when use_verifier=True")
    
    encoder.eval()
    if verifier is not None:
        verifier.eval()
    
    all_metrics_before = {}
    all_metrics_after = {}
    
    for dataset_name in datasets:
        mode_str = "with verifier reranking" if use_verifier else "encoder-only"
        print(f"    Evaluating {dataset_name} ({mode_str})...")
        
        # Load dataset
        dataset = NanoBEIRDataset(dataset_name)
        
        corpus_texts = dataset.get_corpus_texts()
        query_texts = dataset.get_query_texts()
        
        # Stage 1: Encode and retrieve with cosine similarity
        corpus_embeddings = encoder.encode(
            corpus_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        query_embeddings = encoder.encode(
            query_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        
        # Compute similarities and get top-K
        similarities = query_embeddings @ corpus_embeddings.T
        top_k_indices = np.argsort(-similarities, axis=1)[:, :top_k]
        
        # Build rankings before reranking
        rankings_before = {}
        for i, query_id in enumerate(dataset.query_ids):
            ranked_doc_ids = [dataset.corpus_ids[idx] for idx in top_k_indices[i]]
            rankings_before[query_id] = ranked_doc_ids
        
        # Compute metrics before reranking
        metrics_before = compute_retrieval_metrics(rankings_before, dataset.qrels)
        
        # Stage 2: Optionally rerank with verifier
        if use_verifier:
            rankings_after = {}
            for q_idx, query_id in enumerate(dataset.query_ids):
                query = query_texts[q_idx]
                candidate_indices = top_k_indices[q_idx]
                candidates = [corpus_texts[idx] for idx in candidate_indices]
                
                if len(candidates) == 0:
                    rankings_after[query_id] = []
                    continue
                
                # Compute verifier scores in batches
                scores = []
                for i in range(0, len(candidates), batch_size):
                    batch_candidates = candidates[i:i + batch_size]
                    batch_queries = [query] * len(batch_candidates)
                    
                    # Extract token embeddings
                    query_emb, query_mask = token_extractor.extract(batch_queries, max_seq_length)
                    cand_emb, cand_mask = token_extractor.extract(batch_candidates, max_seq_length)
                    
                    # Get verifier scores
                    batch_scores = verifier(query_emb, cand_emb, query_mask, cand_mask)
                    scores.extend(batch_scores.cpu().numpy().tolist())
                
                # Sort by verifier scores (descending)
                scores = np.array(scores)
                sorted_indices = np.argsort(-scores)
                reranked_doc_ids = [dataset.corpus_ids[candidate_indices[idx]] for idx in sorted_indices]
                rankings_after[query_id] = reranked_doc_ids
            
            # Compute metrics after reranking
            metrics_after = compute_retrieval_metrics(rankings_after, dataset.qrels)
        else:
            # No verifier - after metrics are same as before
            metrics_after = metrics_before
        
        # Store per-dataset metrics
        for metric, value in metrics_before.items():
            key = f"{dataset_name}_{metric}_before"
            all_metrics_before[key] = value
        for metric, value in metrics_after.items():
            key = f"{dataset_name}_{metric}_after"
            all_metrics_after[key] = value
    
    # Compute mean metrics across datasets
    metric_names = ["ndcg@10", "mrr@10", "accuracy@1", "accuracy@10"]
    mean_before = {}
    mean_after = {}
    
    for metric in metric_names:
        before_values = [v for k, v in all_metrics_before.items() if metric in k]
        after_values = [v for k, v in all_metrics_after.items() if metric in k]
        mean_before[metric] = np.mean(before_values) if before_values else 0.0
        mean_after[metric] = np.mean(after_values) if after_values else 0.0
    
    return {
        "before_rerank": mean_before,
        "after_rerank": mean_after,
        "improvement": {
            metric: mean_after[metric] - mean_before[metric]
            for metric in metric_names
        },
        "full_results": {
            "before": all_metrics_before,
            "after": all_metrics_after,
        },
    }


@torch.no_grad()
def evaluate_on_mrpc_negatives(
    encoder: SentenceTransformer,
    mrpc_path: Path,
    verifier: Optional[BaseVerifier] = None,
    token_extractor: Optional[TrainableTokenEmbeddingExtractor] = None,
    batch_size: int = 32,
    max_seq_length: int = 128,
) -> Dict:
    """
    Evaluate on MRPC negative pairs dataset using verifier scores.
    
    If verifier and token_extractor are provided, uses verifier similarity scores.
    Otherwise, falls back to encoder cosine similarity.
    
    Lower similarity between original and perturbed sentences = better.
    
    Returns metrics similar to the paper table:
    - Mean similarity per perturbation type
    - Standard deviation
    """
    df = pd.read_csv(mrpc_path)
    
    # Filter invalid rows
    invalid_patterns = ['negatives_placeholder', 'negatives_fix', 'negatives are barred']
    mask = ~df['sentence2'].str.contains('|'.join(invalid_patterns), case=False, na=False)
    mask &= df['sentence2'].str.len() > 20
    df = df[mask].reset_index(drop=True)
    
    encoder.eval()
    if verifier is not None:
        verifier.eval()
    
    sent1_list = df['sentence1'].tolist()
    sent2_list = df['sentence2'].tolist()
    
    use_verifier = verifier is not None and token_extractor is not None
    
    if not use_verifier:
        # Fallback to encoder cosine similarity
        emb1 = encoder.encode(sent1_list, batch_size=batch_size, show_progress_bar=False, convert_to_tensor=True)
        emb2 = encoder.encode(sent2_list, batch_size=batch_size, show_progress_bar=False, convert_to_tensor=True)
        sims = torch.nn.functional.cosine_similarity(emb1, emb2).cpu().numpy()
    else:
        # Use verifier scores
        sims = []
        for i in range(0, len(sent1_list), batch_size):
            batch_sent1 = sent1_list[i:i+batch_size]
            batch_sent2 = sent2_list[i:i+batch_size]
            
            # Get token embeddings
            emb1, mask1 = token_extractor.extract(batch_sent1, max_seq_length)
            emb2, mask2 = token_extractor.extract(batch_sent2, max_seq_length)
            
            # Compute verifier scores for each pair
            for j in range(len(batch_sent1)):
                score = verifier(
                    emb1[j:j+1],
                    emb2[j:j+1],
                    mask1[j:j+1],
                    mask2[j:j+1],
                )
                sims.append(score.item())
        
        sims = np.array(sims)
    
    # Group by category (map to display names)
    category_map = {
        'cannot_negation': 'negation',
        'binding_negation': 'binding',
        'spatial': 'spatial',
    }
    
    results = {}
    for category, display_name in category_map.items():
        category_mask = df['category'] == category
        category_sims = sims[category_mask]
        if len(category_sims) > 0:
            results[display_name] = {
                "count": int(category_mask.sum()),
                "mean": float(np.mean(category_sims)),
                "std": float(np.std(category_sims)),
            }
    
    # Overall
    results['overall'] = {
        "count": len(sims),
        "mean": float(np.mean(sims)),
        "std": float(np.std(sims)),
    }
    
    return results


# =============================================================================
# Main Experiment
# =============================================================================

def create_verifier_with_config(
    verifier_type: str,
    embedding_dim: int = 384,
    max_seq_length: int = 128,
    use_conditional_penalty: bool = False,
    penalty_beta: float = 0.5,
    penalty_tau: float = 0.9,
) -> BaseVerifier:
    """Create a verifier with default configuration.
    
    Args:
        verifier_type: One of "None", "F0", "F1", "F2", "F3", "F4"
        embedding_dim: Embedding dimension (needed for F4)
        max_seq_length: Maximum sequence length
        use_conditional_penalty: Enable conditional penalty mode where verifier
            acts as structural mismatch detector applied only when base similarity > tau
        penalty_beta: Strength of structural penalty (default: 0.5, aggressive)
        penalty_tau: Similarity threshold for applying penalty (default: 0.9)
    """
    cp_kwargs = {
        "use_conditional_penalty": use_conditional_penalty,
        "penalty_beta": penalty_beta,
        "penalty_tau": penalty_tau,
    }
    
    if verifier_type == "None":
        return NoneVerifier(**cp_kwargs)
    elif verifier_type == "F0":
        return F0Verifier(**cp_kwargs)
    elif verifier_type == "F1":
        return F1Verifier(**cp_kwargs)
    elif verifier_type == "F2":
        return F2SoftAttnVerifier(lambda_penalty=0.1, tau=0.1, **cp_kwargs)
    elif verifier_type == "F3":
        return F3Verifier(
            hidden_channels=128,
            kernel_size=3,
            num_conv_layers=2,
            max_seq_length=max_seq_length,
            **cp_kwargs,
        )
    elif verifier_type == "F4":
        return F4Verifier(
            embedding_dim=embedding_dim,
            num_layers=1,
            num_heads=1,
            ff_dim=32,
            max_q_len=max_seq_length,
            max_c_len=max_seq_length,
            **cp_kwargs,
        )
    else:
        raise ValueError(f"Unknown verifier type: {verifier_type}")


def run_single_experiment(
    verifier_type: str,
    encoder_name: str,
    train_triplets: List[Dict],
    eval_triplets: List[Dict],
    mrpc_path: Path,
    freeze_encoder: bool,
    device: str = "cuda",
    batch_size: int = 32,
    max_steps: int = 5000,
    eval_steps: int = 250,
    learning_rate: float = 1e-4,
    encoder_learning_rate: float = 1e-5,
    weight_decay: float = 0.0001,
    encoder_weight_decay: float = 0.01,
    max_seq_length: int = 128,
    early_stopping_patience: int = 50,
    output_path: Optional[Path] = None,
    use_wandb: bool = False,
    wandb_project: str = "verifier-training",
    wandb_entity: Optional[str] = None,
    use_mnrl_loss: bool = True,
    mnrl_temperature: float = 0.05,
    warmup_ratio: float = 0.1,
    lr_scheduler_type: str = "linear",
    use_conditional_penalty: bool = False,
    penalty_beta: float = 0.5,
    penalty_tau: float = 0.9,
    # Retrieval validation parameters
    retrieval_val_datasets: Optional[List[str]] = None,
    retrieval_val_steps: Optional[int] = None,
    use_retrieval_early_stopping: bool = False,
) -> VerifierResult:
    """Run a single training + evaluation experiment.
    
    Args:
        verifier_type: Type of verifier to train
        encoder_name: Name/path of the sentence transformer encoder
        train_triplets: Training triplet data
        eval_triplets: Evaluation triplet data
        mrpc_path: Path to MRPC negative pairs dataset
        freeze_encoder: If True, freeze encoder during training
        device: Device to use
        batch_size: Training batch size
        max_steps: Maximum training steps
        eval_steps: Evaluation frequency
        learning_rate: Verifier learning rate
        encoder_learning_rate: Encoder learning rate (E2E only)
        weight_decay: Verifier weight decay
        encoder_weight_decay: Encoder weight decay (E2E only)
        max_seq_length: Maximum sequence length
        early_stopping_patience: Early stopping patience
        output_path: Path to save checkpoints
        use_wandb: Enable W&B logging
        wandb_project: W&B project name
        wandb_entity: W&B entity
        use_mnrl_loss: Use MNRL loss with in-batch negatives (default: True)
        mnrl_temperature: Temperature for MNRL loss (default: 0.05)
        warmup_ratio: Warmup ratio for LR scheduler (default: 0.1)
        lr_scheduler_type: LR scheduler type - "linear" or "cosine" (default: "linear")
        use_conditional_penalty: Enable conditional penalty mode for verifier
        penalty_beta: Structural penalty strength (default: 0.5)
        penalty_tau: Similarity threshold for penalty (default: 0.9)
        retrieval_val_datasets: List of NanoBEIR dataset names for retrieval validation
                                (e.g., ["nq", "fiqa"]). If None, no retrieval validation.
        retrieval_val_steps: Evaluate retrieval every N steps (defaults to eval_steps)
        use_retrieval_early_stopping: Use retrieval metrics (NDCG@10) for early stopping
    """
    
    mode = "frozen" if freeze_encoder else "end_to_end"
    loss_type = "MNRL" if use_mnrl_loss else "pairwise"
    cp_mode = "conditional_penalty" if use_conditional_penalty else "standard"
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {verifier_type} ({mode})")
    print(f"Loss: {loss_type}, Mode: {cp_mode}")
    if use_conditional_penalty:
        print(f"Penalty: beta={penalty_beta}, tau={penalty_tau}")
    print(f"{'='*60}")
    
    # Initialize wandb for this experiment
    if use_wandb and WANDB_AVAILABLE:
        encoder_short_name = Path(encoder_name).name
        run_name = f"{verifier_type}_{mode}_{encoder_short_name}"
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            config={
                "verifier_type": verifier_type,
                "training_mode": mode,
                "encoder": encoder_name,
                "batch_size": batch_size,
                "max_steps": max_steps,
                "eval_steps": eval_steps,
                "learning_rate": learning_rate,
                "encoder_learning_rate": encoder_learning_rate,
                "weight_decay": weight_decay,
                "encoder_weight_decay": encoder_weight_decay,
                "max_seq_length": max_seq_length,
                "early_stopping_patience": early_stopping_patience,
                "freeze_encoder": freeze_encoder,
                "use_mnrl_loss": use_mnrl_loss,
                "mnrl_temperature": mnrl_temperature,
                "warmup_ratio": warmup_ratio,
                "lr_scheduler_type": lr_scheduler_type,
                "use_conditional_penalty": use_conditional_penalty,
                "penalty_beta": penalty_beta,
                "penalty_tau": penalty_tau,
                "retrieval_val_datasets": retrieval_val_datasets,
                "retrieval_val_steps": retrieval_val_steps,
                "use_retrieval_early_stopping": use_retrieval_early_stopping,
            },
            tags=[verifier_type, mode, encoder_short_name, loss_type, cp_mode],
            reinit=True,
        )
    
    # Load fresh encoder
    encoder = SentenceTransformer(encoder_name, device=device)
    # ModernBert doesn't support token_type_ids, so remove it from tokenizer inputs
    if hasattr(encoder.tokenizer, 'model_input_names') and 'token_type_ids' in encoder.tokenizer.model_input_names:
        encoder.tokenizer.model_input_names = [n for n in encoder.tokenizer.model_input_names if n != 'token_type_ids']
    embedding_dim = encoder.get_sentence_embedding_dimension()
    
    # Create verifier with conditional penalty settings
    verifier = create_verifier_with_config(
        verifier_type, 
        embedding_dim, 
        max_seq_length,
        use_conditional_penalty=use_conditional_penalty,
        penalty_beta=penalty_beta,
        penalty_tau=penalty_tau,
    )
    print(f"Verifier parameters: {verifier.num_parameters:,}")
    
    # Create datasets
    train_dataset = TripletDataset(train_triplets)
    eval_dataset = TripletDataset(eval_triplets)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_triplets,
        pin_memory=True,
        drop_last=True,
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_triplets,
        pin_memory=True,
    )
    
    # Create trainer
    trainer = UnifiedVerifierTrainer(
        verifier=verifier,
        encoder=encoder,
        freeze_encoder=freeze_encoder,
        device=device,
        learning_rate=learning_rate,
        encoder_learning_rate=encoder_learning_rate,
        weight_decay=weight_decay,
        encoder_weight_decay=encoder_weight_decay,
        max_seq_length=max_seq_length,
        use_mnrl_loss=use_mnrl_loss,
        mnrl_temperature=mnrl_temperature,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
    )
    
    # Train
    start_time = time.time()
    
    if output_path:
        save_path = output_path / f"{verifier_type}_{mode}"
    else:
        save_path = None
    
    # Train (the trainer handles the case of no trainable parameters)
    trainer.train(
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        max_steps=max_steps,
        eval_steps=eval_steps,
        save_path=save_path,
        early_stopping_patience=early_stopping_patience,
        use_wandb=use_wandb,
        # Retrieval validation
        retrieval_val_datasets=retrieval_val_datasets,
        retrieval_val_steps=retrieval_val_steps,
        use_retrieval_early_stopping=use_retrieval_early_stopping,
        retrieval_val_batch_size=batch_size,
    )
    training_time = time.time() - start_time
    
    print(f"  Training time: {training_time/60:.1f} minutes")
    
    # Get the trained encoder and verifier
    trained_encoder = trainer.get_encoder()
    trained_verifier = trainer.verifier
    token_extractor = trainer.token_extractor
    
    # Evaluate encoder-only on NanoBEIR (cosine similarity, no verifier reranking)
    print("  Evaluating on NanoBEIR (encoder-only)...")
    nanobeir_encoder_only = evaluate_on_nanobeir(
        encoder=trained_encoder,
        token_extractor=token_extractor,
        verifier=trained_verifier,
        top_k=100,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        use_verifier=False,  # Encoder-only: no reranking
    )
    
    # Evaluate with verifier reranking on NanoBEIR
    print("  Evaluating on NanoBEIR (encoder+verifier)...")
    nanobeir_with_verifier = evaluate_on_nanobeir(
        encoder=trained_encoder,
        token_extractor=token_extractor,
        verifier=trained_verifier,
        top_k=100,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        use_verifier=True,  # Apply verifier reranking
    )
    
    # Evaluate on MRPC negatives - encoder-only (cosine similarity)
    print("  Evaluating on MRPC negatives (encoder-only)...")
    mrpc_encoder_only = evaluate_on_mrpc_negatives(
        encoder=trained_encoder,
        mrpc_path=mrpc_path,
        batch_size=batch_size,
    )
    
    # Evaluate on MRPC negatives - with verifier scores
    print("  Evaluating on MRPC negatives (with verifier)...")
    mrpc_with_verifier = evaluate_on_mrpc_negatives(
        encoder=trained_encoder,
        mrpc_path=mrpc_path,
        verifier=trained_verifier,
        token_extractor=token_extractor,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
    )
    
    result = VerifierResult(
        verifier_type=verifier_type,
        training_mode=mode,
        num_parameters=verifier.num_parameters,
        training_time_seconds=training_time,
        nanobeir_encoder_only=nanobeir_encoder_only,
        nanobeir_with_verifier=nanobeir_with_verifier,
        mrpc_encoder_only=mrpc_encoder_only,
        mrpc_with_verifier=mrpc_with_verifier,
    )
    
    # Print summary
    print(f"\n  NanoBEIR Results (Encoder-Only):")
    print(f"    NDCG@10:  {nanobeir_encoder_only.get('ndcg@10', 0):.4f}")
    print(f"    MRR@10:   {nanobeir_encoder_only.get('mrr@10', 0):.4f}")
    print(f"    ACC@1:    {nanobeir_encoder_only.get('accuracy@1', 0):.4f}")
    print(f"    ACC@10:   {nanobeir_encoder_only.get('accuracy@10', 0):.4f}")
    
    print(f"\n  NanoBEIR Results (With Verifier Reranking):")
    after_rerank = nanobeir_with_verifier.get('after_rerank', {})
    print(f"    NDCG@10:  {after_rerank.get('ndcg@10', 0):.4f}")
    print(f"    MRR@10:   {after_rerank.get('mrr@10', 0):.4f}")
    print(f"    ACC@1:    {after_rerank.get('accuracy@1', 0):.4f}")
    print(f"    ACC@10:   {after_rerank.get('accuracy@10', 0):.4f}")
    
    print(f"\n  MRPC Negative Pairs - Encoder Only (lower = better):")
    for mode_name in ['negation', 'binding', 'spatial', 'overall']:
        m = mrpc_encoder_only.get(mode_name, {})
        print(f"    {mode_name.capitalize()}: {m.get('mean', 0):.3f} ± {m.get('std', 0):.3f} (N={m.get('count', 0)})")
    
    print(f"\n  MRPC Negative Pairs - With Verifier (lower = better):")
    for mode_name in ['negation', 'binding', 'spatial', 'overall']:
        m = mrpc_with_verifier.get(mode_name, {})
        print(f"    {mode_name.capitalize()}: {m.get('mean', 0):.3f} ± {m.get('std', 0):.3f} (N={m.get('count', 0)})")
    
    # Log final evaluation results to wandb
    if use_wandb and WANDB_AVAILABLE:
        # NanoBEIR encoder-only metrics
        wandb.log({
            "nanobeir_encoder/ndcg@10": nanobeir_encoder_only.get("ndcg@10", 0),
            "nanobeir_encoder/mrr@10": nanobeir_encoder_only.get("mrr@10", 0),
            "nanobeir_encoder/accuracy@1": nanobeir_encoder_only.get("accuracy@1", 0),
            "nanobeir_encoder/accuracy@10": nanobeir_encoder_only.get("accuracy@10", 0),
        })
        
        # NanoBEIR with verifier reranking metrics
        after_rerank = nanobeir_with_verifier.get("after_rerank", {})
        improvement = nanobeir_with_verifier.get("improvement", {})
        wandb.log({
            "nanobeir_verifier/ndcg@10": after_rerank.get("ndcg@10", 0),
            "nanobeir_verifier/mrr@10": after_rerank.get("mrr@10", 0),
            "nanobeir_verifier/accuracy@1": after_rerank.get("accuracy@1", 0),
            "nanobeir_verifier/accuracy@10": after_rerank.get("accuracy@10", 0),
            "nanobeir_improvement/ndcg@10": improvement.get("ndcg@10", 0),
            "nanobeir_improvement/mrr@10": improvement.get("mrr@10", 0),
            "nanobeir_improvement/accuracy@1": improvement.get("accuracy@1", 0),
            "nanobeir_improvement/accuracy@10": improvement.get("accuracy@10", 0),
        })
        
        # MRPC encoder-only metrics
        for category in ['negation', 'binding', 'spatial', 'overall']:
            m = mrpc_encoder_only.get(category, {})
            wandb.log({
                f"mrpc_encoder/{category}_mean": m.get("mean", 0),
                f"mrpc_encoder/{category}_std": m.get("std", 0),
            })
        
        # MRPC with verifier metrics
        for category in ['negation', 'binding', 'spatial', 'overall']:
            m = mrpc_with_verifier.get(category, {})
            wandb.log({
                f"mrpc_verifier/{category}_mean": m.get("mean", 0),
                f"mrpc_verifier/{category}_std": m.get("std", 0),
            })
        
        # Log summary metrics
        wandb.summary["training_time_minutes"] = training_time / 60
        wandb.summary["verifier_parameters"] = verifier.num_parameters
        
        # Finish wandb run
        wandb.finish()
    
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run comprehensive verifier training and evaluation experiments"
    )
    
    parser.add_argument(
        "--encoder", type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Base encoder model"
    )
    parser.add_argument(
        "--data-path", type=str,
        default="./data/train_structured.csv",
        help="Path to training data (JSON or CSV)"
    )
    parser.add_argument(
        "--mrpc-path", type=str,
        default="./data/mrpc_unified_negative_pairs.csv",
        help="Path to MRPC negative pairs"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="./verifier_experiment_results",
        help="Output directory"
    )
    parser.add_argument(
        "--no-timestamp-subdir", action="store_true",
        help="Use output-dir directly without creating a timestamped subdirectory"
    )
    parser.add_argument(
        "--verifiers", type=str, nargs="+",
        default=["None", "F0", "F1", "F2", "F3", "F4"],
        help="Verifiers to evaluate (None = embedding-only baseline)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--max-steps", type=int, default=5000,
        help="Maximum training steps"
    )
    parser.add_argument(
        "--early-stopping-patience", type=int, default=50,
        help="Early stopping patience (number of eval steps without improvement)"
    )
    parser.add_argument(
        "--eval-steps", type=int, default=250,
        help="Evaluation frequency in steps"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4,
        help="Verifier learning rate"
    )
    parser.add_argument(
        "--encoder-learning-rate", type=float, default=1e-5,
        help="Encoder learning rate (for end-to-end)"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.0001,
        help="Weight decay for verifier optimizer"
    )
    parser.add_argument(
        "--encoder-weight-decay", type=float, default=0.01,
        help="Weight decay for encoder optimizer (for end-to-end)"
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=128,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device"
    )
    parser.add_argument(
        "--skip-frozen", action="store_true",
        help="Skip frozen encoder experiments"
    )
    parser.add_argument(
        "--skip-end-to-end", action="store_true",
        help="Skip end-to-end experiments"
    )
    
    # Loss function arguments
    parser.add_argument(
        "--use-mnrl-loss", action="store_true", default=True,
        help="Use MNRL loss with in-batch negatives (default: True)"
    )
    parser.add_argument(
        "--no-mnrl-loss", action="store_true",
        help="Disable MNRL loss, use pairwise loss instead"
    )
    parser.add_argument(
        "--mnrl-temperature", type=float, default=0.05,
        help="Temperature for MNRL loss (default: 0.05)"
    )
    
    # Learning rate scheduler arguments
    parser.add_argument(
        "--warmup-ratio", type=float, default=0.1,
        help="Warmup ratio for learning rate scheduler (default: 0.1)"
    )
    parser.add_argument(
        "--lr-scheduler-type", type=str, default="linear",
        choices=["linear", "cosine"],
        help="LR scheduler type: linear or cosine (default: linear)"
    )
    
    # Conditional penalty arguments
    parser.add_argument(
        "--use-conditional-penalty", action="store_true",
        help="Enable conditional penalty mode for verifiers"
    )
    parser.add_argument(
        "--penalty-beta", type=float, default=0.5,
        help="Structural penalty strength (default: 0.5, aggressive)"
    )
    parser.add_argument(
        "--penalty-tau", type=float, default=0.9,
        help="Similarity threshold for applying penalty (default: 0.9)"
    )
    
    # Wandb arguments
    parser.add_argument(
        "--wandb-project", type=str, default="verifier-training",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=None,
        help="Weights & Biases entity (team or username)"
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable Weights & Biases logging"
    )
    
    # Retrieval validation arguments
    parser.add_argument(
        "--retrieval-val-datasets", type=str, nargs="+", default=None,
        help="NanoBEIR datasets for retrieval validation during training (e.g., nq fiqa). "
             "If not specified, only triplet-based validation is used."
    )
    parser.add_argument(
        "--retrieval-val-steps", type=int, default=None,
        help="Evaluate retrieval every N steps (defaults to --eval-steps)"
    )
    parser.add_argument(
        "--use-retrieval-early-stopping", action="store_true",
        help="Use retrieval NDCG@10 for early stopping instead of triplet accuracy"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create output directory
    if args.no_timestamp_subdir:
        # Use output_dir directly (for when shell script manages the directory)
        output_path = Path(args.output_dir)
    else:
        # Create timestamped subdirectory (default behavior)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        encoder_name = Path(args.encoder).name
        output_path = Path(args.output_dir) / f"{encoder_name}_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Verifier Training and Evaluation Experiment")
    print("="*70)
    print(f"Encoder: {args.encoder}")
    print(f"Data: {args.data_path}")
    print(f"MRPC: {args.mrpc_path}")
    print(f"Output: {output_path}")
    print(f"Verifiers: {args.verifiers}")
    print("="*70)
    
    # Load training data
    print("\nLoading training data...")
    all_triplets = load_triplets(Path(args.data_path))
    print(f"Loaded {len(all_triplets)} triplets")
    
    # Split
    random.shuffle(all_triplets)
    split_idx = int(len(all_triplets) * 0.9)
    train_triplets = all_triplets[:split_idx]
    eval_triplets = all_triplets[split_idx:]
    print(f"Train: {len(train_triplets)}, Eval: {len(eval_triplets)}")
    
    mrpc_path = Path(args.mrpc_path)
    
    # Load existing results if available (for incremental updates)
    results_file = output_path / "all_results.json"
    if results_file.exists():
        with open(results_file, "r") as f:
            all_results = json.load(f)
        print(f"Loaded existing results from: {results_file}")
        # Ensure both keys exist
        if "frozen" not in all_results:
            all_results["frozen"] = {}
        if "end_to_end" not in all_results:
            all_results["end_to_end"] = {}
    else:
        # Initialize fresh results
        all_results = {
            "frozen": {},
            "end_to_end": {},
        }
    
    # Determine if wandb should be used
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        print(f"\nWeights & Biases logging enabled (project: {args.wandb_project})")
    else:
        if not WANDB_AVAILABLE:
            print("\nWeights & Biases not available (wandb not installed)")
        else:
            print("\nWeights & Biases logging disabled (--no-wandb)")
    
    # Determine loss function
    use_mnrl_loss = args.use_mnrl_loss and not args.no_mnrl_loss
    loss_type = "MNRL" if use_mnrl_loss else "pairwise"
    print(f"\nLoss function: {loss_type}")
    if use_mnrl_loss:
        print(f"MNRL temperature: {args.mnrl_temperature}")
    
    # Conditional penalty settings
    if args.use_conditional_penalty:
        print(f"\nConditional penalty mode enabled:")
        print(f"  beta (penalty strength): {args.penalty_beta}")
        print(f"  tau (threshold): {args.penalty_tau}")
    
    # Run frozen encoder experiments
    if not args.skip_frozen:
        print("\n" + "="*70)
        print("PHASE 1: FROZEN ENCODER EXPERIMENTS")
        print("="*70)
        
        for verifier_type in args.verifiers:
            result = run_single_experiment(
                verifier_type=verifier_type,
                encoder_name=args.encoder,
                train_triplets=train_triplets,
                eval_triplets=eval_triplets,
                mrpc_path=mrpc_path,
                freeze_encoder=True,
                device=args.device,
                batch_size=args.batch_size,
                max_steps=args.max_steps,
                eval_steps=args.eval_steps,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                encoder_weight_decay=args.encoder_weight_decay,
                max_seq_length=args.max_seq_length,
                early_stopping_patience=args.early_stopping_patience,
                output_path=output_path,
                use_wandb=use_wandb,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                use_mnrl_loss=use_mnrl_loss,
                mnrl_temperature=args.mnrl_temperature,
                warmup_ratio=args.warmup_ratio,
                lr_scheduler_type=args.lr_scheduler_type,
                use_conditional_penalty=args.use_conditional_penalty,
                penalty_beta=args.penalty_beta,
                penalty_tau=args.penalty_tau,
                # Retrieval validation
                retrieval_val_datasets=args.retrieval_val_datasets,
                retrieval_val_steps=args.retrieval_val_steps,
                use_retrieval_early_stopping=args.use_retrieval_early_stopping,
            )
            all_results["frozen"][verifier_type] = asdict(result)
    
    # Run end-to-end experiments
    if not args.skip_end_to_end:
        print("\n" + "="*70)
        print("PHASE 2: END-TO-END TRAINING EXPERIMENTS")
        print("="*70)
        
        for verifier_type in args.verifiers:
            # Skip verifiers that don't support end-to-end training
            if verifier_type not in E2E_COMPATIBLE_VERIFIERS:
                print(f"\n  Skipping {verifier_type} (not compatible with end-to-end training)")
                continue
            
            result = run_single_experiment(
                verifier_type=verifier_type,
                encoder_name=args.encoder,
                train_triplets=train_triplets,
                eval_triplets=eval_triplets,
                mrpc_path=mrpc_path,
                freeze_encoder=False,
                device=args.device,
                batch_size=args.batch_size,
                max_steps=args.max_steps,
                eval_steps=args.eval_steps,
                learning_rate=args.learning_rate,
                encoder_learning_rate=args.encoder_learning_rate,
                weight_decay=args.weight_decay,
                encoder_weight_decay=args.encoder_weight_decay,
                max_seq_length=args.max_seq_length,
                early_stopping_patience=args.early_stopping_patience,
                output_path=output_path,
                use_wandb=use_wandb,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                use_mnrl_loss=use_mnrl_loss,
                mnrl_temperature=args.mnrl_temperature,
                warmup_ratio=args.warmup_ratio,
                lr_scheduler_type=args.lr_scheduler_type,
                use_conditional_penalty=args.use_conditional_penalty,
                penalty_beta=args.penalty_beta,
                penalty_tau=args.penalty_tau,
                # Retrieval validation
                retrieval_val_datasets=args.retrieval_val_datasets,
                retrieval_val_steps=args.retrieval_val_steps,
                use_retrieval_early_stopping=args.use_retrieval_early_stopping,
            )
            all_results["end_to_end"][verifier_type] = asdict(result)
    
    # Save results
    results_file = output_path / "all_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Print summary table - Encoder-Only Performance
    print("\n" + "="*110)
    print("SUMMARY: NanoBEIR Performance - Encoder Only (cosine similarity)")
    print("="*110)
    print(f"{'Verifier':<10} {'Mode':<15} {'NDCG@10':>10} {'MRR@10':>10} {'ACC@1':>10} {'ACC@10':>10}")
    print("-"*70)
    
    for mode in ["frozen", "end_to_end"]:
        for vtype, result in all_results.get(mode, {}).items():
            nb = result.get("nanobeir_encoder_only", {})
            ndcg = nb.get("ndcg@10", 0)
            mrr = nb.get("mrr@10", 0)
            acc1 = nb.get("accuracy@1", 0)
            acc10 = nb.get("accuracy@10", 0)
            print(f"{vtype:<10} {mode:<15} {ndcg:>10.4f} {mrr:>10.4f} {acc1:>10.4f} {acc10:>10.4f}")
        print("-"*70)
    
    # Print summary table - With Verifier Reranking
    print("\n" + "="*110)
    print("SUMMARY: NanoBEIR Performance - With Verifier Reranking (encoder + verifier)")
    print("="*110)
    print(f"{'Verifier':<10} {'Mode':<15} {'NDCG@10':>10} {'MRR@10':>10} {'ACC@1':>10} {'ACC@10':>10}")
    print("-"*70)
    
    for mode in ["frozen", "end_to_end"]:
        for vtype, result in all_results.get(mode, {}).items():
            nb = result.get("nanobeir_with_verifier", {})
            after = nb.get("after_rerank", {})
            ndcg = after.get("ndcg@10", 0)
            mrr = after.get("mrr@10", 0)
            acc1 = after.get("accuracy@1", 0)
            acc10 = after.get("accuracy@10", 0)
            print(f"{vtype:<10} {mode:<15} {ndcg:>10.4f} {mrr:>10.4f} {acc1:>10.4f} {acc10:>10.4f}")
        print("-"*70)
    
    print("\n" + "="*130)
    print("SUMMARY: MRPC Negative Pairs (lower similarity = better)")
    print("="*130)
    print(f"{'Verifier':<10} {'Mode':<15} {'Eval':<12} {'Negation':>12} {'Binding':>12} {'Spatial':>12} {'Overall':>12}")
    print("-"*100)
    
    for mode in ["frozen", "end_to_end"]:
        for vtype, result in all_results.get(mode, {}).items():
            # Encoder-only row
            mrpc_enc = result.get("mrpc_encoder_only", {})
            neg = mrpc_enc.get("negation", {}).get("mean", 0)
            bind = mrpc_enc.get("binding", {}).get("mean", 0)
            spat = mrpc_enc.get("spatial", {}).get("mean", 0)
            ovr = mrpc_enc.get("overall", {}).get("mean", 0)
            print(f"{vtype:<10} {mode:<15} {'Encoder':<12} {neg:>12.3f} {bind:>12.3f} {spat:>12.3f} {ovr:>12.3f}")
            
            # With verifier row
            mrpc_v = result.get("mrpc_with_verifier", {})
            neg = mrpc_v.get("negation", {}).get("mean", 0)
            bind = mrpc_v.get("binding", {}).get("mean", 0)
            spat = mrpc_v.get("spatial", {}).get("mean", 0)
            ovr = mrpc_v.get("overall", {}).get("mean", 0)
            print(f"{'':<10} {'':<15} {'+Verifier':<12} {neg:>12.3f} {bind:>12.3f} {spat:>12.3f} {ovr:>12.3f}")
        print("-"*100)
    
    # Generate LaTeX table - comprehensive with both encoder-only and with-verifier
    latex_lines = []
    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\small")
    latex_lines.append(r"\begin{tabular}{llccccc}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"Verifier & Mode & Enc. NDCG@10 & +V NDCG@10 & MRR@10 & ACC@1 & ACC@10 \\")
    latex_lines.append(r"\midrule")
    
    for mode in ["frozen", "end_to_end"]:
        mode_label = "Frozen" if mode == "frozen" else "E2E"
        for vtype, result in all_results.get(mode, {}).items():
            enc_only = result.get("nanobeir_encoder_only", {})
            with_v = result.get("nanobeir_with_verifier", {})
            after = with_v.get("after_rerank", {})
            
            enc_ndcg = enc_only.get("ndcg@10", 0)
            v_ndcg = after.get("ndcg@10", 0)
            mrr = after.get("mrr@10", 0)
            acc1 = after.get("accuracy@1", 0)
            acc10 = after.get("accuracy@10", 0)
            
            latex_lines.append(f"{vtype} & {mode_label} & {enc_ndcg:.3f} & {v_ndcg:.3f} & {mrr:.3f} & {acc1:.3f} & {acc10:.3f} \\\\")
        latex_lines.append(r"\midrule")
    
    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\caption{Verifier performance on NanoBEIR benchmarks. Enc. NDCG@10 shows encoder-only retrieval, +V NDCG@10 shows retrieval with verifier reranking.}")
    latex_lines.append(r"\label{tab:verifier_nanobeir}")
    latex_lines.append(r"\end{table}")
    
    latex_file = output_path / "nanobeir_table.tex"
    with open(latex_file, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"\nLaTeX table saved to: {latex_file}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
