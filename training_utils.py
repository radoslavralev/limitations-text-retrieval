"""
Shared Training Utilities for Verifier Training.

This module contains common classes and functions used by both:
- train_verifiers.py (frozen encoder training)
- train_and_eval_verifiers_experiment.py (frozen + E2E training)

Components:
- TripletDataset: Dataset class for triplet-based training
- load_triplets(): Load triplets from JSON or CSV files
- collate_triplets(): Batch collation function
- TrainableTokenEmbeddingExtractor: Token embedding extraction with optional gradient flow
"""

import csv
import json
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer


class TripletDataset(Dataset):
    """Dataset for triplet-based verifier training."""
    
    def __init__(self, triplets: List[Dict]):
        self.triplets = triplets
    
    def __len__(self) -> int:
        return len(self.triplets)
    
    def __getitem__(self, idx: int) -> Tuple[str, str, str]:
        t = self.triplets[idx]
        return t["anchor"], t["positive"], t["negative"]


def load_triplets(data_path: Path) -> List[Dict]:
    """Load triplets from JSON or CSV file, filtering out any with null values.
    
    Supports multiple formats:
    - JSON: List of dicts with 'anchor', 'positive', 'negative' keys
    - CSV (triplet format): Columns 'anchor', 'positive', 'negative', 'category'
    - CSV (pair format): Columns 'sentence1', 'sentence2', 'category' where:
        - sentence1 is used as both anchor and positive (identity)
        - sentence2 is the hard negative
    
    Args:
        data_path: Path to JSON or CSV file containing triplets
        
    Returns:
        List of valid triplet dictionaries with 'anchor', 'positive', 'negative' keys
    """
    data_path = Path(data_path)
    
    if data_path.suffix.lower() == '.csv':
        # Load from CSV format
        triplets = []
        with open(data_path, "r", newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            
            # Detect format based on column names
            is_triplet_format = 'anchor' in fieldnames and 'positive' in fieldnames and 'negative' in fieldnames
            
            for row in reader:
                if is_triplet_format:
                    # Triplet format: anchor, positive, negative, category
                    anchor = row.get("anchor", "").strip()
                    positive = row.get("positive", "").strip()
                    negative = row.get("negative", "").strip()
                    category = row.get("category", "unknown").strip()
                    
                    if anchor and positive and negative:
                        triplets.append({
                            "anchor": anchor,
                            "positive": positive,
                            "negative": negative,
                            "negative_type": category,
                        })
                else:
                    # Pair format: sentence1, sentence2, category (identity pairs)
                    sentence1 = row.get("sentence1", "").strip()
                    sentence2 = row.get("sentence2", "").strip()
                    category = row.get("category", "unknown").strip()
                    
                    if sentence1 and sentence2:
                        triplets.append({
                            "anchor": sentence1,
                            "positive": sentence1,  # Identity: anchor == positive
                            "negative": sentence2,
                            "negative_type": category,
                        })
        
        format_type = "triplet" if is_triplet_format else "identity pairs"
        print(f"Loaded {len(triplets)} triplets from CSV ({format_type} format)")
        return triplets
    else:
        # Load from JSON format (original behavior)
        with open(data_path, "r") as f:
            triplets = json.load(f)
        
        # Filter out triplets with null values
        valid_triplets = [
            t for t in triplets
            if t.get("anchor") is not None 
            and t.get("positive") is not None 
            and t.get("negative") is not None
        ]
        
        if len(valid_triplets) < len(triplets):
            print(f"Warning: Filtered out {len(triplets) - len(valid_triplets)} triplets with null values")
        
        return valid_triplets


def collate_triplets(batch: List[Tuple[str, str, str]]) -> Tuple[List[str], List[str], List[str]]:
    """
    Collate triplets into batches of anchors, positives, and negatives.
    
    Args:
        batch: List of (anchor, positive, negative) tuples
        
    Returns:
        Tuple of (anchors, positives, negatives) as lists of strings
    """
    anchors = [b[0] for b in batch]
    positives = [b[1] for b in batch]
    negatives = [b[2] for b in batch]
    return anchors, positives, negatives


class TrainableTokenEmbeddingExtractor(nn.Module):
    """
    Extract token-level embeddings from a SentenceTransformer.
    
    This extractor supports both frozen and trainable (end-to-end) modes:
    - freeze_encoder=True: No gradients flow through the encoder (for frozen training)
    - freeze_encoder=False: Gradients flow through for end-to-end fine-tuning
    
    Attributes:
        model: The SentenceTransformer model
        device: Device to use for computation
        freeze_encoder: Whether to freeze the encoder parameters
    """
    
    def __init__(
        self,
        model: SentenceTransformer,
        freeze_encoder: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        self.model = model
        self.device = device
        self.model.to(device)
        self.freeze_encoder = freeze_encoder
        
        # ModernBert doesn't support token_type_ids, so remove it from tokenizer inputs
        if hasattr(self.model.tokenizer, 'model_input_names') and 'token_type_ids' in self.model.tokenizer.model_input_names:
            self.model.tokenizer.model_input_names = [
                n for n in self.model.tokenizer.model_input_names if n != 'token_type_ids'
            ]
        
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        else:
            for param in self.model.parameters():
                param.requires_grad = True
    
    def extract(
        self,
        texts: List[str],
        max_length: int = 128,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract token embeddings for a batch of texts.
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length for tokenization
            
        Returns:
            token_embeddings: (batch, seq_len, hidden_dim)
            attention_mask: (batch, seq_len) as float tensor
        """
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
        
        transformer = self.model[0]
        
        # Build kwargs for the model, excluding token_type_ids for ModernBert compatibility
        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "return_dict": True,
        }
        
        if self.freeze_encoder:
            with torch.no_grad():
                outputs = transformer.auto_model(**model_kwargs)
        else:
            outputs = transformer.auto_model(**model_kwargs)
        
        token_embeddings = outputs.last_hidden_state
        
        return token_embeddings, attention_mask.float()
