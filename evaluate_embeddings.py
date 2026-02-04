#!/usr/bin/env python3
"""
Evaluate model-a-baseline vs model-b-structured on negative pairs datasets.

This script evaluates how well each model distinguishes between:
1. Synthetic negative pairs (negation, binding, spatial perturbations)
2. Retrieval triplets from MSMARCO or NQ (comparing positive vs negative pair similarities)

For negatives: Lower similarity = model correctly distinguishes the perturbation
For positives: Higher similarity = model correctly recognizes semantic equivalence

Supports multi-seed results:
- Can save evaluation results to JSON for later aggregation
- Can aggregate results from multiple seed directories
- Displays error bars in tables and plots when aggregated
"""

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
from typing import List, Dict, Tuple, Optional, Union, Any
import argparse
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import os
import matplotlib.pyplot as plt

from aggregate_utils import (
    find_seed_dirs,
    aggregate_nested_dict,
    load_json_safe,
    get_mean,
    get_std,
    is_aggregated_value,
    format_with_std_latex,
)


@dataclass
class EvalResult:
    """Result for a single pair evaluation."""
    pair_id: str
    mode: str
    sentence1: str
    sentence2: str
    sim_baseline: Optional[float]  # Baseline model similarity (if provided)
    sim_a: float
    sim_b: float
    delta: float  # sim_b - sim_a (negative means B is better)
    pair_type: str = "negative"  # "positive" or "negative"


def load_single_model(model_name: str, label: str) -> SentenceTransformer:
    """Load a single model from HuggingFace or local path."""
    import os
    print(f"Loading {label}: {model_name}")
    
    # Check if it's a local path
    is_local = os.path.isdir(model_name)
    
    if is_local:
        print(f"  Loading from local directory: {model_name}")
        model = SentenceTransformer(model_name, local_files_only=True)
    else:
        # Force download fresh copy from HuggingFace
        print(f"  Downloading from HuggingFace: {model_name}")
        snapshot_download(model_name, force_download=True)
        model = SentenceTransformer(model_name, local_files_only=False)
    
    # ModernBert doesn't support token_type_ids, so remove it from tokenizer inputs
    if hasattr(model.tokenizer, 'model_input_names') and 'token_type_ids' in model.tokenizer.model_input_names:
        model.tokenizer.model_input_names = [n for n in model.tokenizer.model_input_names if n != 'token_type_ids']
    return model


def load_models(
    model_a_name: str, 
    model_b_name: str,
    baseline_name: Optional[str] = None
) -> Tuple[Optional[SentenceTransformer], SentenceTransformer, SentenceTransformer]:
    """Load models from HuggingFace."""
    baseline = None
    if baseline_name:
        baseline = load_single_model(baseline_name, "Baseline")
    
    model_a = load_single_model(model_a_name, "Model A")
    model_b = load_single_model(model_b_name, "Model B")
    
    return baseline, model_a, model_b


def load_dataset(path: str) -> pd.DataFrame:
    """Load the MRPC negative pairs dataset."""
    df = pd.read_csv(path)
    original_len = len(df)
    
    # Filter out invalid placeholder rows
    invalid_patterns = ['negatives_placeholder', 'negatives_fix', 'negatives are barred']
    mask = ~df['sentence2'].str.contains('|'.join(invalid_patterns), case=False, na=False)
    # Also filter rows where sent2 is just "negatives" or very short
    mask &= df['sentence2'].str.len() > 20
    df = df[mask].reset_index(drop=True)
    
    print(f"Loaded {original_len} pairs from {path}")
    print(f"  Filtered out {original_len - len(df)} invalid rows")
    print(f"  Valid pairs: {len(df)}")
    print(f"    - Negation: {len(df[df['category'] == 'cannot_negation'])} pairs")
    print(f"    - Binding: {len(df[df['category'] == 'binding_negation'])} pairs")
    print(f"    - Spatial: {len(df[df['category'] == 'spatial'])} pairs")
    return df


def load_triplets_from_csv(path: str, max_samples: Optional[int] = None) -> pd.DataFrame:
    """Load triplets dataset from a local CSV file.
    
    Args:
        path: Path to local CSV file
        max_samples: Maximum number of samples to load
    
    Format: anchor, positive, negative, category
    Returns DataFrame with columns: anchor, positive, negative, category
    """
    if path is None:
        raise ValueError("path must be provided")
    
    df = pd.read_csv(path)
    original_len = len(df)
    
    # Filter out rows with missing values
    df = df.dropna(subset=['anchor', 'positive', 'negative']).reset_index(drop=True)
    
    # Optionally subsample
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    
    print(f"\nLoaded triplets from {path}")
    print(f"  Original rows: {original_len}")
    print(f"  After filtering: {len(df)}")
    if max_samples:
        print(f"  Subsampled to: {len(df)}")
    
    return df


def load_triplets_from_huggingface(dataset: str, max_samples: Optional[int] = None) -> pd.DataFrame:
    """Download and load triplets from HuggingFace.
    
    Downloads triplets from the sentence-transformers embedding-training-data repository.
    
    Args:
        dataset: Either "msmarco" or "nq"
        max_samples: Maximum number of samples to load
    
    Returns DataFrame with columns: anchor, positive, negative, category
    """
    import gzip
    import json
    import tempfile
    import urllib.request
    import random
    
    DATASET_URLS = {
        "msmarco": "https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/msmarco-triplets.jsonl.gz",
        "nq": "https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/NQ-train_pairs.jsonl.gz",
    }
    
    if dataset not in DATASET_URLS:
        raise ValueError(f"Unknown dataset: {dataset}. Must be one of: {list(DATASET_URLS.keys())}")
    
    url = DATASET_URLS[dataset]
    is_pairs = dataset == "nq"  # NQ is pairs format, MSMARCO is triplets
    
    print(f"\nDownloading {dataset.upper()} {'pairs' if is_pairs else 'triplets'} from HuggingFace...")
    print(f"  URL: {url}")
    
    # Download to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.jsonl.gz', delete=False) as tmp_file:
        tmp_path = tmp_file.name
        print(f"  Downloading to temporary file...")
        urllib.request.urlretrieve(url, tmp_path)
    
    # Load from the gzipped JSONL file
    records = []
    print(f"  Parsing {'pairs' if is_pairs else 'triplets'}...")
    
    with gzip.open(tmp_path, 'rt', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if line_num % 100000 == 0 and line_num > 0:
                print(f"    Processed {line_num} lines...")
            
            data = json.loads(line.strip())
            
            if is_pairs:
                # NQ format: ["query", "answer"] list
                if isinstance(data, list) and len(data) >= 2:
                    query = str(data[0]).strip()
                    answer = str(data[1]).strip()
                    if query and answer:
                        records.append({"query": query, "answer": answer})
            else:
                # MSMARCO format: {"query": "...", "pos": [...], "neg": [...]}
                if isinstance(data, dict) and "query" in data:
                    query = data.get("query", "").strip()
                    positives = data.get("pos", [])
                    negatives = data.get("neg", [])
                    
                    if query and positives and negatives:
                        positive = positives[0].strip() if positives else ""
                        negative = negatives[0].strip() if negatives else ""
                        
                        if positive and negative:
                            records.append({
                                "anchor": query,
                                "positive": positive,
                                "negative": negative,
                                "category": dataset
                            })
    
    print(f"  Total records available: {len(records)}")
    
    # Clean up temporary file
    import os
    os.unlink(tmp_path)
    
    # For NQ pairs, convert to triplets with random negatives
    if is_pairs:
        print(f"  Converting pairs to triplets with random negatives...")
        all_answers = [r["answer"] for r in records]
        n = len(all_answers)
        random.seed(42)
        triplets = []
        for i, record in enumerate(records):
            neg_idx = random.randint(0, n - 2)
            if neg_idx >= i:
                neg_idx += 1
            triplets.append({
                "anchor": record["query"],
                "positive": record["answer"],
                "negative": all_answers[neg_idx],
                "category": dataset
            })
        records = triplets
    
    # Subsample if needed
    if max_samples and len(records) > max_samples:
        random.seed(42)
        records = random.sample(records, max_samples)
        print(f"  Subsampled to: {len(records)}")
    
    df = pd.DataFrame(records)
    print(f"  Final triplets loaded: {len(df)}")
    
    return df


def load_msmarco_from_huggingface(max_samples: Optional[int] = None) -> pd.DataFrame:
    """Legacy wrapper for backward compatibility."""
    return load_triplets_from_huggingface("msmarco", max_samples)


def evaluate_pairs(
    model_a: SentenceTransformer,
    model_b: SentenceTransformer,
    df: pd.DataFrame,
    batch_size: int = 32,
    baseline: Optional[SentenceTransformer] = None,
    pair_type: str = "negative"
) -> List[EvalResult]:
    """Evaluate all pairs and compute similarities."""
    results = []
    
    # Get all sentences
    sent1_list = df['sentence1'].tolist()
    sent2_list = df['sentence2'].tolist()
    
    print(f"\nEncoding {len(sent1_list)} sentence pairs ({pair_type})...")
    
    # Encode with Baseline (if provided)
    sims_baseline = None
    if baseline is not None:
        print("  Encoding with Baseline...")
        emb1_base = baseline.encode(sent1_list, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)
        emb2_base = baseline.encode(sent2_list, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)
        sims_baseline = torch.nn.functional.cosine_similarity(emb1_base, emb2_base).cpu().numpy()
    
    # Encode with Model A
    print("  Encoding with Model A...")
    emb1_a = model_a.encode(sent1_list, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)
    emb2_a = model_a.encode(sent2_list, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)
    
    # Encode with Model B
    print("  Encoding with Model B...")
    emb1_b = model_b.encode(sent1_list, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)
    emb2_b = model_b.encode(sent2_list, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)
    
    # Compute cosine similarities
    print("  Computing similarities...")
    sims_a = torch.nn.functional.cosine_similarity(emb1_a, emb2_a).cpu().numpy()
    sims_b = torch.nn.functional.cosine_similarity(emb1_b, emb2_b).cpu().numpy()
    
    # Create results
    for idx, row in df.iterrows():
        sim_a = float(sims_a[idx])
        sim_b = float(sims_b[idx])
        sim_base = float(sims_baseline[idx]) if sims_baseline is not None else None
        results.append(EvalResult(
            pair_id=str(idx),
            mode=row['category'],
            sentence1=row['sentence1'],
            sentence2=row['sentence2'],
            sim_baseline=sim_base,
            sim_a=sim_a,
            sim_b=sim_b,
            delta=sim_b - sim_a,
            pair_type=pair_type
        ))
    
    return results


def evaluate_triplets(
    model_a: SentenceTransformer,
    model_b: SentenceTransformer,
    df: pd.DataFrame,
    batch_size: int = 32,
    baseline: Optional[SentenceTransformer] = None
) -> Tuple[List[EvalResult], List[EvalResult]]:
    """Evaluate triplets dataset, returning both positive and negative pair results.
    
    Returns:
        positive_results: Similarity between anchor and positive (should be HIGH)
        negative_results: Similarity between anchor and negative (should be LOW)
    """
    anchor_list = df['anchor'].tolist()
    positive_list = df['positive'].tolist()
    negative_list = df['negative'].tolist()
    categories = df['category'].tolist() if 'category' in df.columns else ['unknown'] * len(df)
    
    print(f"\nEncoding {len(anchor_list)} triplets...")
    
    # Encode all sentences
    all_sentences = anchor_list + positive_list + negative_list
    n = len(anchor_list)
    
    # Encode with Baseline (if provided)
    emb_baseline = None
    if baseline is not None:
        print("  Encoding with Baseline...")
        emb_baseline = baseline.encode(all_sentences, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)
    
    # Encode with Model A
    print("  Encoding with Model A...")
    emb_a = model_a.encode(all_sentences, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)
    
    # Encode with Model B
    print("  Encoding with Model B...")
    emb_b = model_b.encode(all_sentences, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)
    
    # Split embeddings
    anchor_emb_a, pos_emb_a, neg_emb_a = emb_a[:n], emb_a[n:2*n], emb_a[2*n:]
    anchor_emb_b, pos_emb_b, neg_emb_b = emb_b[:n], emb_b[n:2*n], emb_b[2*n:]
    
    if emb_baseline is not None:
        anchor_emb_base, pos_emb_base, neg_emb_base = emb_baseline[:n], emb_baseline[n:2*n], emb_baseline[2*n:]
    
    # Compute similarities
    print("  Computing similarities...")
    
    # Positive pairs: anchor vs positive
    pos_sims_a = torch.nn.functional.cosine_similarity(anchor_emb_a, pos_emb_a).cpu().numpy()
    pos_sims_b = torch.nn.functional.cosine_similarity(anchor_emb_b, pos_emb_b).cpu().numpy()
    pos_sims_base = None
    if emb_baseline is not None:
        pos_sims_base = torch.nn.functional.cosine_similarity(anchor_emb_base, pos_emb_base).cpu().numpy()
    
    # Negative pairs: anchor vs negative
    neg_sims_a = torch.nn.functional.cosine_similarity(anchor_emb_a, neg_emb_a).cpu().numpy()
    neg_sims_b = torch.nn.functional.cosine_similarity(anchor_emb_b, neg_emb_b).cpu().numpy()
    neg_sims_base = None
    if emb_baseline is not None:
        neg_sims_base = torch.nn.functional.cosine_similarity(anchor_emb_base, neg_emb_base).cpu().numpy()
    
    # Create results
    positive_results = []
    negative_results = []
    
    for idx in range(n):
        cat = categories[idx] if isinstance(categories[idx], str) else 'unknown'
        
        # Positive pair result
        positive_results.append(EvalResult(
            pair_id=f"{cat}_pos_{idx}",
            mode=f"{cat}_positive",
            sentence1=anchor_list[idx],
            sentence2=positive_list[idx],
            sim_baseline=float(pos_sims_base[idx]) if pos_sims_base is not None else None,
            sim_a=float(pos_sims_a[idx]),
            sim_b=float(pos_sims_b[idx]),
            delta=float(pos_sims_b[idx]) - float(pos_sims_a[idx]),
            pair_type="positive"
        ))
        
        # Negative pair result
        negative_results.append(EvalResult(
            pair_id=f"{cat}_neg_{idx}",
            mode=f"{cat}_negative",
            sentence1=anchor_list[idx],
            sentence2=negative_list[idx],
            sim_baseline=float(neg_sims_base[idx]) if neg_sims_base is not None else None,
            sim_a=float(neg_sims_a[idx]),
            sim_b=float(neg_sims_b[idx]),
            delta=float(neg_sims_b[idx]) - float(neg_sims_a[idx]),
            pair_type="negative"
        ))
    
    return positive_results, negative_results


def compute_statistics(results: List[EvalResult], has_baseline: bool = False) -> Dict:
    """Compute aggregate statistics."""
    stats = {}
    
    # Group by mode
    by_mode = defaultdict(list)
    for r in results:
        by_mode[r.mode].append(r)
    
    for mode, mode_results in by_mode.items():
        sims_a = [r.sim_a for r in mode_results]
        sims_b = [r.sim_b for r in mode_results]
        deltas = [r.delta for r in mode_results]
        
        # Count how often each model "wins" (lower similarity = better)
        b_wins = sum(1 for d in deltas if d < -0.01)  # B has notably lower similarity
        a_wins = sum(1 for d in deltas if d > 0.01)   # A has notably lower similarity
        ties = len(deltas) - b_wins - a_wins
        
        mode_stats = {
            'count': len(mode_results),
            'sim_a_mean': np.mean(sims_a),
            'sim_a_std': np.std(sims_a),
            'sim_b_mean': np.mean(sims_b),
            'sim_b_std': np.std(sims_b),
            'delta_mean': np.mean(deltas),
            'delta_std': np.std(deltas),
            'b_wins': b_wins,
            'a_wins': a_wins,
            'ties': ties,
            'b_win_rate': b_wins / len(mode_results) * 100,
        }
        
        # Add baseline stats if available
        if has_baseline:
            sims_baseline = [r.sim_baseline for r in mode_results if r.sim_baseline is not None]
            if sims_baseline:
                mode_stats['sim_baseline_mean'] = np.mean(sims_baseline)
                mode_stats['sim_baseline_std'] = np.std(sims_baseline)
        
        stats[mode] = mode_stats
    
    # Overall stats
    all_sims_a = [r.sim_a for r in results]
    all_sims_b = [r.sim_b for r in results]
    all_deltas = [r.delta for r in results]
    
    b_wins_all = sum(1 for d in all_deltas if d < -0.01)
    a_wins_all = sum(1 for d in all_deltas if d > 0.01)
    
    overall_stats = {
        'count': len(results),
        'sim_a_mean': np.mean(all_sims_a),
        'sim_a_std': np.std(all_sims_a),
        'sim_b_mean': np.mean(all_sims_b),
        'sim_b_std': np.std(all_sims_b),
        'delta_mean': np.mean(all_deltas),
        'delta_std': np.std(all_deltas),
        'b_wins': b_wins_all,
        'a_wins': a_wins_all,
        'ties': len(results) - b_wins_all - a_wins_all,
        'b_win_rate': b_wins_all / len(results) * 100,
    }
    
    if has_baseline:
        all_sims_baseline = [r.sim_baseline for r in results if r.sim_baseline is not None]
        if all_sims_baseline:
            overall_stats['sim_baseline_mean'] = np.mean(all_sims_baseline)
            overall_stats['sim_baseline_std'] = np.std(all_sims_baseline)
    
    stats['overall'] = overall_stats
    
    return stats


def save_stats_to_json(stats: Dict, output_path: Path, metadata: Dict = None):
    """
    Save statistics to a JSON file for later aggregation.
    
    Args:
        stats: Statistics dictionary from compute_statistics
        output_path: Path to save JSON file
        metadata: Optional metadata to include (model names, etc.)
    """
    output_data = {
        "stats": stats,
        "metadata": metadata or {},
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved statistics to {output_path}")


def load_and_aggregate_seed_stats(parent_dir: Path, json_filename: str = "eval_stats.json") -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Load statistics from multiple seed directories and aggregate them.
    
    Args:
        parent_dir: Directory containing seed_* subdirectories
        json_filename: Name of the JSON file in each seed directory
        
    Returns:
        Tuple of (aggregated_stats, metadata) or (None, None) if not found
    """
    seed_dirs = find_seed_dirs(parent_dir)
    
    if not seed_dirs:
        # Try loading directly from parent_dir
        json_path = parent_dir / json_filename
        if json_path.exists():
            data = load_json_safe(json_path)
            if data:
                return data.get("stats"), data.get("metadata")
        return None, None
    
    print(f"Found {len(seed_dirs)} seeds: {[d.name for d in seed_dirs]}")
    
    all_stats = []
    metadata = None
    
    for seed_dir in seed_dirs:
        json_path = seed_dir / json_filename
        data = load_json_safe(json_path)
        if data:
            all_stats.append(data.get("stats", {}))
            if metadata is None:
                metadata = data.get("metadata")
    
    if not all_stats:
        return None, None
    
    print(f"Aggregating statistics from {len(all_stats)} seeds...")
    aggregated = aggregate_nested_dict(all_stats)
    
    return aggregated, metadata


def format_stat_value(value: Union[float, Dict], precision: int = 3, show_std: bool = True) -> str:
    """Format a statistic value that may be aggregated (with mean/std)."""
    mean = get_mean(value)
    std = get_std(value)
    
    if show_std and std > 0:
        return f"{mean:.{precision}f} +/- {std:.{precision}f}"
    return f"{mean:.{precision}f}"


def print_summary(stats: Dict, model_a_name: str, model_b_name: str, baseline_name: Optional[str] = None):
    """
    Print summary statistics table.
    
    Supports both simple stats and aggregated stats (with mean/std from multiple seeds).
    """
    print("\n" + "="*120)
    print(" AGGREGATE RESULTS: Similarity to Perturbed Sentences")
    print(" (Lower similarity = model correctly distinguishes the perturbation)")
    print("="*120)
    
    has_baseline = baseline_name is not None and 'sim_baseline_mean' in stats.get('overall', {})
    
    if has_baseline:
        print(f"\n{'category':<15} {'N':>6} {baseline_name:>22} {model_a_name:>22} {model_b_name:>22} {'Δ (B-A)':>15} {'B Wins':>12}")
        print("-" * 130)
    else:
        print(f"\n{'category':<15} {'N':>6} {model_a_name:>24} {model_b_name:>24} {'Δ (B-A)':>18} {'B Wins':>14}")
        print("-" * 110)
    
    mode_order = ['cannot_negation', 'binding_negation', 'spatial', 'overall']
    for mode in mode_order:
        if mode not in stats:
            continue
        s = stats[mode]
        
        # Handle both simple and aggregated stats
        sim_a_mean = get_mean(s['sim_a_mean'])
        sim_a_std_inner = get_mean(s['sim_a_std'])  # Within-seed std
        sim_a_std_outer = get_std(s['sim_a_mean'])  # Across-seed std
        
        sim_b_mean = get_mean(s['sim_b_mean'])
        sim_b_std_inner = get_mean(s['sim_b_std'])
        sim_b_std_outer = get_std(s['sim_b_mean'])
        
        delta_mean = get_mean(s['delta_mean'])
        delta_std_outer = get_std(s['delta_mean'])
        
        b_win_rate = get_mean(s['b_win_rate'])
        b_win_std = get_std(s['b_win_rate'])
        
        count = get_mean(s['count'])
        
        # Format strings - show cross-seed std if available
        if sim_a_std_outer > 0:
            a_str = f"{sim_a_mean:.3f} ± {sim_a_std_outer:.3f}"
            b_str = f"{sim_b_mean:.3f} ± {sim_b_std_outer:.3f}"
            delta_str = f"{delta_mean:+.3f} ± {delta_std_outer:.3f}"
            win_str = f"{b_win_rate:.1f}% ± {b_win_std:.1f}%"
        else:
            a_str = f"{sim_a_mean:.3f} ± {sim_a_std_inner:.3f}"
            b_str = f"{sim_b_mean:.3f} ± {sim_b_std_inner:.3f}"
            delta_str = f"{delta_mean:+.3f}"
            win_str = f"{b_win_rate:.1f}%"
        
        mode_display = mode.upper() if mode == 'overall' else mode.capitalize()
        
        if has_baseline:
            base_mean = get_mean(s['sim_baseline_mean'])
            base_std_inner = get_mean(s['sim_baseline_std'])
            base_std_outer = get_std(s['sim_baseline_mean'])
            if base_std_outer > 0:
                base_str = f"{base_mean:.3f} ± {base_std_outer:.3f}"
            else:
                base_str = f"{base_mean:.3f} ± {base_std_inner:.3f}"
            print(f"{mode_display:<15} {int(count):>6} {base_str:>22} {a_str:>22} {b_str:>22} {delta_str:>15} {win_str:>12}")
        else:
            print(f"{mode_display:<15} {int(count):>6} {a_str:>24} {b_str:>24} {delta_str:>18} {win_str:>14}")
        
        if mode == 'spatial':
            if has_baseline:
                print("-" * 130)
            else:
                print("-" * 110)
    
    print()


def compute_simple_stats(results: List[EvalResult], has_baseline: bool = False) -> Dict:
    """Compute simple aggregate statistics for a list of results."""
    if not results:
        return {}
    
    sims_a = [r.sim_a for r in results]
    sims_b = [r.sim_b for r in results]
    deltas = [r.delta for r in results]
    
    stats = {
        'count': len(results),
        'sim_a_mean': np.mean(sims_a),
        'sim_a_std': np.std(sims_a),
        'sim_b_mean': np.mean(sims_b),
        'sim_b_std': np.std(sims_b),
        'delta_mean': np.mean(deltas),
        'delta_std': np.std(deltas),
    }
    
    if has_baseline:
        sims_baseline = [r.sim_baseline for r in results if r.sim_baseline is not None]
        if sims_baseline:
            stats['sim_baseline_mean'] = np.mean(sims_baseline)
            stats['sim_baseline_std'] = np.std(sims_baseline)
    
    return stats


def print_comparison_summary(
    unified_neg_results: List[EvalResult],
    comparison_pos_results: List[EvalResult],
    comparison_neg_results: List[EvalResult],
    model_a_name: str,
    model_b_name: str,
    baseline_name: Optional[str] = None,
    comparison_dataset_name: str = "MSMARCO"
):
    """Print comparison of unified negatives vs comparison dataset positives/negatives."""
    
    has_baseline = baseline_name is not None
    
    print("\n" + "="*140)
    print(f" COMPARISON: Unified Negatives vs {comparison_dataset_name} Triplets")
    print(" This compares how models handle different types of semantic similarity/dissimilarity")
    print("="*140)
    
    # Compute stats for each group
    unified_stats = compute_simple_stats(unified_neg_results, has_baseline)
    comparison_pos_stats = compute_simple_stats(comparison_pos_results, has_baseline)
    comparison_neg_stats = compute_simple_stats(comparison_neg_results, has_baseline)
    
    # Print header
    if has_baseline:
        print(f"\n{'Dataset':<30} {'Type':<10} {'N':>7} {baseline_name:>15} {model_a_name:>15} {model_b_name:>15} {'Δ (B-A)':>10}")
        print("-" * 115)
    else:
        print(f"\n{'Dataset':<30} {'Type':<10} {'N':>7} {model_a_name:>18} {model_b_name:>18} {'Δ (B-A)':>10}")
        print("-" * 100)
    
    # Print each row
    datasets = [
        (f"{comparison_dataset_name} (anchor-positive)", "positive", comparison_pos_stats),
        (f"{comparison_dataset_name} (anchor-negative)", "negative", comparison_neg_stats),
        ("Unified Negatives Test", "negative", unified_stats),
    ]
    
    for name, pair_type, stats in datasets:
        if not stats:
            continue
        
        a_str = f"{stats['sim_a_mean']:.4f} ± {stats['sim_a_std']:.3f}"
        b_str = f"{stats['sim_b_mean']:.4f} ± {stats['sim_b_std']:.3f}"
        delta_str = f"{stats['delta_mean']:+.4f}"
        
        if has_baseline:
            base_str = f"{stats.get('sim_baseline_mean', 0):.4f} ± {stats.get('sim_baseline_std', 0):.3f}"
            print(f"{name:<30} {pair_type:<10} {stats['count']:>7} {base_str:>15} {a_str:>15} {b_str:>15} {delta_str:>10}")
        else:
            print(f"{name:<30} {pair_type:<10} {stats['count']:>7} {a_str:>18} {b_str:>18} {delta_str:>10}")
    
    print()
    
    # Compute and print separation metrics
    print("\n" + "-"*80)
    print(" SEPARATION ANALYSIS")
    print(" (Good embeddings should maximize positive-negative gap)")
    print("-"*80)
    
    if comparison_pos_stats and comparison_neg_stats:
        # For comparison dataset
        comparison_gap_a = comparison_pos_stats['sim_a_mean'] - comparison_neg_stats['sim_a_mean']
        comparison_gap_b = comparison_pos_stats['sim_b_mean'] - comparison_neg_stats['sim_b_mean']
        
        print(f"\n{comparison_dataset_name} Positive-Negative Gap:")
        print(f"  {model_a_name}: {comparison_gap_a:.4f} (pos: {comparison_pos_stats['sim_a_mean']:.4f}, neg: {comparison_neg_stats['sim_a_mean']:.4f})")
        print(f"  {model_b_name}: {comparison_gap_b:.4f} (pos: {comparison_pos_stats['sim_b_mean']:.4f}, neg: {comparison_neg_stats['sim_b_mean']:.4f})")
        print(f"  Δ Gap (B-A): {comparison_gap_b - comparison_gap_a:+.4f}")
        
        if has_baseline:
            comparison_gap_base = comparison_pos_stats['sim_baseline_mean'] - comparison_neg_stats['sim_baseline_mean']
            print(f"  {baseline_name}: {comparison_gap_base:.4f}")
    
    if unified_stats and comparison_neg_stats:
        # Compare unified negatives to comparison dataset negatives
        print(f"\nNegative Similarity Comparison:")
        print(f"  Unified Negatives ({model_a_name}): {unified_stats['sim_a_mean']:.4f}")
        print(f"  Unified Negatives ({model_b_name}): {unified_stats['sim_b_mean']:.4f}")
        print(f"  {comparison_dataset_name} Negatives ({model_a_name}): {comparison_neg_stats['sim_a_mean']:.4f}")
        print(f"  {comparison_dataset_name} Negatives ({model_b_name}): {comparison_neg_stats['sim_b_mean']:.4f}")
        
        # The unified negatives are "harder" if they have higher similarity
        unified_harder_a = unified_stats['sim_a_mean'] > comparison_neg_stats['sim_a_mean']
        unified_harder_b = unified_stats['sim_b_mean'] > comparison_neg_stats['sim_b_mean']
        
        print(f"\n  Unified negatives are {'HARDER' if unified_harder_a else 'EASIER'} than {comparison_dataset_name} negatives for {model_a_name}")
        print(f"  Unified negatives are {'HARDER' if unified_harder_b else 'EASIER'} than {comparison_dataset_name} negatives for {model_b_name}")
        
        diff_a = unified_stats['sim_a_mean'] - comparison_neg_stats['sim_a_mean']
        diff_b = unified_stats['sim_b_mean'] - comparison_neg_stats['sim_b_mean']
        print(f"  Hardness difference ({model_a_name}): {diff_a:+.4f}")
        print(f"  Hardness difference ({model_b_name}): {diff_b:+.4f}")
    
    print()


def plot_similarity_distributions(
    unified_neg_results: List[EvalResult],
    comparison_pos_results: List[EvalResult],
    comparison_neg_results: List[EvalResult],
    model_a_name: str,
    model_b_name: str,
    output_dir: str = "figures",
    baseline_name: Optional[str] = None,
    comparison_dataset_name: str = "MSMARCO"
):
    """
    Create histogram plots of similarity distributions for each model.
    
    Creates separate plots for each model (Baseline, Model A, Model B),
    each showing distributions for: comparison_pos, comparison_neg, negation, binding, spatial
    
    Saves plots as both PNG and PDF formats.
    
    Args:
        unified_neg_results: Results from unified negatives evaluation
        comparison_pos_results: Results from comparison dataset positive pairs
        comparison_neg_results: Results from comparison dataset negative pairs
        model_a_name: Short name for model A
        model_b_name: Short name for model B
        output_dir: Base directory to save figures
        baseline_name: Optional baseline model name
    """
    # Group unified negatives by category
    negation_results = [r for r in unified_neg_results if r.mode == 'cannot_negation']
    binding_results = [r for r in unified_neg_results if r.mode == 'binding_negation']
    spatial_results = [r for r in unified_neg_results if r.mode == 'spatial']
    
    # Define categories and their data
    categories = [
        ('comparison_pos', comparison_pos_results, f'{comparison_dataset_name} Positive', '#2ecc71'),  # green
        ('comparison_neg', comparison_neg_results, f'{comparison_dataset_name} Negative', '#3498db'),  # blue
        ('negation', negation_results, 'Negation', '#e74c3c'),  # red
        ('binding', binding_results, 'Binding', '#9b59b6'),  # purple
        ('spatial', spatial_results, 'Spatial', '#f39c12'),  # orange
    ]
    
    # Filter out empty categories
    categories = [(name, results, label, color) for name, results, label, color in categories if results]
    
    if not categories:
        print("No data available for plotting.")
        return
    
    # Models to plot
    models = [
        ('sim_a', model_a_name),
        ('sim_b', model_b_name),
    ]
    if baseline_name and any(r.sim_baseline is not None for r in unified_neg_results):
        models.insert(0, ('sim_baseline', baseline_name))
    
    # Helper function to save figure in multiple formats
    def save_figure(fig, base_path):
        """Save figure as both PNG and PDF."""
        png_path = f"{base_path}.png"
        pdf_path = f"{base_path}.pdf"
        fig.savefig(png_path, dpi=150, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        print(f"  Saved: {png_path}")
        print(f"  Saved: {pdf_path}")
    
    # Create separate plots for each model
    for sim_attr, model_name in models:
        # Create output directory for this model
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        print(f"\nGenerating plots for {model_name}...")
        
        # Get similarities for this model
        def get_sims(results):
            if sim_attr == 'sim_baseline':
                return [r.sim_baseline for r in results if r.sim_baseline is not None]
            elif sim_attr == 'sim_a':
                return [r.sim_a for r in results]
            else:
                return [r.sim_b for r in results]
        
        bins = np.linspace(0, 1, 51)  # 50 bins from 0 to 1
        
        # ============================================
        # Plot 1: Combined histogram (all categories)
        # ============================================
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for cat_name, results, label, color in categories:
            sims = get_sims(results)
            if sims:
                ax.hist(sims, bins=bins, alpha=0.5,
                       label=f"{label} (n={len(sims)}, μ={np.mean(sims):.3f})", 
                       color=color, edgecolor='white', linewidth=0.5)
        
        ax.set_xlabel('Cosine Similarity', fontsize=14)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Similarity Distribution - {model_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add vertical lines for means
        for cat_name, results, label, color in categories:
            sims = get_sims(results)
            if sims:
                ax.axvline(x=np.mean(sims), color=color, linestyle='--', linewidth=1.5, alpha=0.8)
        
        plt.tight_layout()
        save_figure(fig, os.path.join(model_dir, 'similarity_distribution_combined'))
        plt.close()
        
        # ============================================
        # Plot 2: Normalized histogram (density)
        # ============================================
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for cat_name, results, label, color in categories:
            sims = get_sims(results)
            if sims and len(sims) > 1:
                ax.hist(sims, bins=bins, alpha=0.4, 
                       label=f"{label} (n={len(sims)}, μ={np.mean(sims):.3f})", 
                       color=color, edgecolor='white', linewidth=0.5, density=True)
        
        ax.set_xlabel('Cosine Similarity', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Similarity Distribution (Normalized) - {model_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add vertical lines for means
        for cat_name, results, label, color in categories:
            sims = get_sims(results)
            if sims:
                ax.axvline(x=np.mean(sims), color=color, linestyle='--', linewidth=1.5, alpha=0.8)
        
        plt.tight_layout()
        save_figure(fig, os.path.join(model_dir, 'similarity_distribution_normalized'))
        plt.close()
        
        # ============================================
        # Plot 3: Separate subplot for each category
        # ============================================
        n_cats = len(categories)
        fig, axes = plt.subplots(n_cats, 1, figsize=(12, 3 * n_cats), sharex=True)
        if n_cats == 1:
            axes = [axes]
        
        for ax, (cat_name, results, label, color) in zip(axes, categories):
            sims = get_sims(results)
            if sims:
                ax.hist(sims, bins=bins, alpha=0.7, color=color, edgecolor='white', linewidth=0.5)
                ax.axvline(x=np.mean(sims), color='black', linestyle='--', linewidth=2, 
                          label=f'Mean: {np.mean(sims):.3f}')
                ax.set_ylabel('Count', fontsize=10)
                ax.set_title(f'{label} (n={len(sims)})', fontsize=11, fontweight='bold')
                ax.legend(loc='upper right', fontsize=9)
                ax.set_xlim(0, 1)
                ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Cosine Similarity', fontsize=12)
        plt.suptitle(f'Similarity Distributions by Category - {model_name}', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_figure(fig, os.path.join(model_dir, 'similarity_distribution_by_category'))
        plt.close()
        
        # ============================================
        # Plot 4: Box plot comparison
        # ============================================
        fig, ax = plt.subplots(figsize=(10, 6))
        
        box_data = []
        box_labels = []
        box_colors = []
        
        for cat_name, results, label, color in categories:
            sims = get_sims(results)
            if sims:
                box_data.append(sims)
                box_labels.append(label)
                box_colors.append(color)
        
        if box_data:
            bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            ax.set_ylabel('Cosine Similarity', fontsize=12)
            ax.set_title(f'Similarity Distribution Box Plot - {model_name}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1)
            
            plt.tight_layout()
            save_figure(fig, os.path.join(model_dir, 'similarity_distribution_boxplot'))
            plt.close()
    
    # ============================================
    # Create a combined comparison plot (all models side by side)
    # ============================================
    print(f"\nGenerating comparison plots...")
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6), sharey=True)
    if len(models) == 1:
        axes = [axes]
    
    bins = np.linspace(0, 1, 41)  # 40 bins
    
    for ax, (sim_attr, model_name) in zip(axes, models):
        for cat_name, results, label, color in categories:
            if sim_attr == 'sim_baseline':
                sims = [r.sim_baseline for r in results if r.sim_baseline is not None]
            elif sim_attr == 'sim_a':
                sims = [r.sim_a for r in results]
            else:
                sims = [r.sim_b for r in results]
            
            if sims and len(sims) > 1:
                # Include mean in the label like the separate plots
                ax.hist(sims, bins=bins, alpha=0.5, 
                       label=f"{label} (n={len(sims)}, μ={np.mean(sims):.3f})", 
                       color=color, edgecolor='white', linewidth=0.5, density=True)
        
        ax.set_xlabel('Cosine Similarity', fontsize=11)
        ax.set_title(model_name, fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        # Add legend to each subplot
        ax.legend(loc='upper left', fontsize=12)
    
    axes[0].set_ylabel('Density', fontsize=11)
    
    plt.suptitle('Similarity Distributions Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'similarity_distribution_comparison'))
    plt.close()
    
    # ============================================
    # Box plot comparison across all models
    # ============================================
    fig, ax = plt.subplots(figsize=(14, 8))
    
    positions = []
    all_data = []
    all_colors = []
    tick_positions = []
    tick_labels = []
    
    group_width = len(models) + 1
    
    for i, (cat_name, results, label, color) in enumerate(categories):
        group_start = i * group_width
        tick_positions.append(group_start + len(models) / 2)
        tick_labels.append(label)
        
        for j, (sim_attr, model_name) in enumerate(models):
            if sim_attr == 'sim_baseline':
                sims = [r.sim_baseline for r in results if r.sim_baseline is not None]
            elif sim_attr == 'sim_a':
                sims = [r.sim_a for r in results]
            else:
                sims = [r.sim_b for r in results]
            
            if sims:
                all_data.append(sims)
                positions.append(group_start + j)
                all_colors.append(color)
    
    if all_data:
        bp = ax.boxplot(all_data, positions=positions, widths=0.7, patch_artist=True)
        for patch, color in zip(bp['boxes'], all_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=11)
        ax.set_ylabel('Cosine Similarity', fontsize=12)
        ax.set_title('Similarity Distribution Comparison Across Models', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1)
        
        # Add legend for models
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='gray', alpha=0.6, label=model_name) 
                          for _, model_name in models]
        ax.legend(legend_elements, [m[1] for m in models], loc='lower right', fontsize=10)
        
        plt.tight_layout()
        save_figure(fig, os.path.join(output_dir, 'similarity_distribution_boxplot_comparison'))
        plt.close()
    
    print(f"\nAll plots saved to: {output_dir}/")


def print_distribution_analysis(results: List[EvalResult], model_a_name: str, model_b_name: str):
    """Print similarity distribution analysis."""
    print("\n" + "="*100)
    print(" SIMILARITY DISTRIBUTION ANALYSIS")
    print("="*100)
    
    # Define similarity buckets
    buckets = [
        (0.0, 0.5, "Very Low (0.0-0.5)"),
        (0.5, 0.7, "Low (0.5-0.7)"),
        (0.7, 0.85, "Medium (0.7-0.85)"),
        (0.85, 0.95, "High (0.85-0.95)"),
        (0.95, 1.01, "Very High (0.95-1.0)"),
    ]
    
    for mode in ['cannot_negation', 'binding_negation', 'spatial']:
        mode_results = [r for r in results if r.mode == mode]
        if not mode_results:
            continue
            
        print(f"\n{mode.upper()}:")
        print(f"{'Similarity Range':<25} {model_a_name:>15} {model_b_name:>15}")
        print("-" * 60)
        
        for low, high, label in buckets:
            count_a = sum(1 for r in mode_results if low <= r.sim_a < high)
            count_b = sum(1 for r in mode_results if low <= r.sim_b < high)
            pct_a = count_a / len(mode_results) * 100
            pct_b = count_b / len(mode_results) * 100
            print(f"{label:<25} {count_a:>6} ({pct_a:>5.1f}%) {count_b:>6} ({pct_b:>5.1f}%)")
        
        # Check for negative similarities (unusual)
        neg_a = sum(1 for r in mode_results if r.sim_a < 0)
        neg_b = sum(1 for r in mode_results if r.sim_b < 0)
        if neg_a > 0 or neg_b > 0:
            print(f"{'Negative (<0)':<25} {neg_a:>6} ({neg_a/len(mode_results)*100:>5.1f}%) {neg_b:>6} ({neg_b/len(mode_results)*100:>5.1f}%)")


def find_interesting_cases(
    results: List[EvalResult],
    n_examples: int = 5
) -> Dict[str, Dict[str, List[EvalResult]]]:
    """
    Find interesting cases for each mode:
    - b_success: Model B correctly distinguishes (low sim) but Model A fails (high sim)
    - b_failure: Model B fails (high sim) but Model A succeeds (low sim)  
    - both_fail: Both models have high similarity
    - both_succeed: Both models have low similarity
    """
    cases = defaultdict(lambda: defaultdict(list))
    
    # Thresholds for "high" and "low" similarity
    HIGH_SIM = 0.85
    LOW_SIM = 0.75
    
    for r in results:
        mode = r.mode
        
        # B success: B has notably lower similarity than A
        if r.sim_b < r.sim_a - 0.05 and r.sim_b < HIGH_SIM:
            cases[mode]['b_success'].append(r)
        
        # B failure: B has notably higher similarity than A (A is better)
        if r.sim_b > r.sim_a + 0.05 and r.sim_b > HIGH_SIM:
            cases[mode]['b_failure'].append(r)
        
        # Both fail: Both have high similarity
        if r.sim_a > HIGH_SIM and r.sim_b > HIGH_SIM:
            cases[mode]['both_fail'].append(r)
        
        # Both succeed: Both have low similarity
        if r.sim_a < LOW_SIM and r.sim_b < LOW_SIM:
            cases[mode]['both_succeed'].append(r)
    
    # Sort and limit
    for mode in cases:
        for case_type in cases[mode]:
            if case_type == 'b_success':
                # Sort by largest improvement (most negative delta)
                cases[mode][case_type].sort(key=lambda x: x.delta)
            elif case_type == 'b_failure':
                # Sort by largest regression (most positive delta)
                cases[mode][case_type].sort(key=lambda x: -x.delta)
            elif case_type == 'both_fail':
                # Sort by highest similarity (worst cases)
                cases[mode][case_type].sort(key=lambda x: -(x.sim_a + x.sim_b))
            else:
                # Sort by lowest similarity (best cases)
                cases[mode][case_type].sort(key=lambda x: x.sim_a + x.sim_b)
            
            # Limit to n_examples
            cases[mode][case_type] = cases[mode][case_type][:n_examples]
    
    return cases


def print_example(r: EvalResult, model_a_name: str, model_b_name: str):
    """Print a single example."""
    print(f"  Pair ID: {r.pair_id}")
    print(f"  Original:  {r.sentence1[:100]}{'...' if len(r.sentence1) > 100 else ''}")
    print(f"  Perturbed: {r.sentence2[:100]}{'...' if len(r.sentence2) > 100 else ''}")
    print(f"  {model_a_name}: {r.sim_a:.3f}  |  {model_b_name}: {r.sim_b:.3f}  |  Δ: {r.delta:+.3f}")
    print()


def print_interesting_cases(
    cases: Dict,
    model_a_name: str,
    model_b_name: str
):
    """Print interesting cases for analysis."""
    
    mode_names = {
        'cannot_negation': 'cannot_negation',
        'binding_negation': 'BINDING (Modifier Rebinding)',
        'spatial': 'SPATIAL (Directional Relations)'
    }
    
    for mode in ['cannot_negation', 'binding_negation', 'spatial']:
        if mode not in cases:
            continue
            
        print("\n" + "="*100)
        print(f" {mode_names.get(mode, mode.upper())} - Detailed Examples")
        print("="*100)
        
        mode_cases = cases[mode]
        
        # B Success cases
        if mode_cases.get('b_success'):
            print(f"\n--- {model_b_name} SUCCEEDS (correctly distinguishes perturbation) ---")
            print(f"    (Lower similarity = model recognizes sentences are different)\n")
            for r in mode_cases['b_success']:
                print_example(r, model_a_name, model_b_name)
        
        # B Failure cases  
        if mode_cases.get('b_failure'):
            print(f"\n--- {model_b_name} FAILS (assigns high similarity to different sentences) ---")
            print(f"    ({model_a_name} performs better on these examples)\n")
            for r in mode_cases['b_failure']:
                print_example(r, model_a_name, model_b_name)
        
        # Both fail
        if mode_cases.get('both_fail'):
            print(f"\n--- BOTH MODELS FAIL (high similarity despite semantic difference) ---")
            print(f"    (These are the hardest cases for embeddings)\n")
            for r in mode_cases['both_fail']:
                print_example(r, model_a_name, model_b_name)
        
        # Both succeed
        if mode_cases.get('both_succeed'):
            print(f"\n--- BOTH MODELS SUCCEED (correctly distinguish perturbation) ---")
            print(f"    (These perturbations are well-handled by both models)\n")
            for r in mode_cases['both_succeed']:
                print_example(r, model_a_name, model_b_name)


def generate_latex_table(stats: Dict, model_a_name: str, model_b_name: str, baseline_name: Optional[str] = None) -> str:
    """
    Generate LaTeX table for paper.
    
    Supports both simple stats and aggregated stats (with mean/std from multiple seeds).
    """
    has_baseline = baseline_name is not None and 'sim_baseline_mean' in stats.get('overall', {})
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    
    if has_baseline:
        lines.append(r"\begin{tabular}{lrccccc}")
        lines.append(r"\toprule")
        lines.append(f"Perturbation & N & {baseline_name} & {model_a_name} & {model_b_name} & $\\Delta$ & B Wins \\\\")
    else:
        lines.append(r"\begin{tabular}{lrcccc}")
        lines.append(r"\toprule")
        lines.append(f"Perturbation & N & {model_a_name} & {model_b_name} & $\\Delta$ & B Wins \\\\")
    lines.append(r"\midrule")
    
    mode_order = ['cannot_negation', 'binding_negation', 'spatial']
    for mode in mode_order:
        if mode not in stats:
            continue
        s = stats[mode]
        mode_display = mode.capitalize()
        
        # Handle both simple and aggregated stats
        sim_a_mean = get_mean(s['sim_a_mean'])
        sim_a_std = get_std(s['sim_a_mean'])
        sim_b_mean = get_mean(s['sim_b_mean'])
        sim_b_std = get_std(s['sim_b_mean'])
        delta_mean = get_mean(s['delta_mean'])
        delta_std = get_std(s['delta_mean'])
        b_win_rate = get_mean(s['b_win_rate'])
        b_win_std = get_std(s['b_win_rate'])
        count = int(get_mean(s['count']))
        
        # Format with std if available
        a_str = format_with_std_latex(sim_a_mean, sim_a_std, precision=3, show_std=(sim_a_std > 0))
        b_str = format_with_std_latex(sim_b_mean, sim_b_std, precision=3, show_std=(sim_b_std > 0))
        if delta_std > 0:
            sign = "+" if delta_mean >= 0 else ""
            delta_str = f"{sign}{delta_mean:.3f}$\\pm${delta_std:.3f}"
        else:
            delta_str = f"{delta_mean:+.3f}"
        if b_win_std > 0:
            win_str = f"{b_win_rate:.1f}$\\pm${b_win_std:.1f}\\%"
        else:
            win_str = f"{b_win_rate:.1f}\\%"
        
        if has_baseline:
            base_mean = get_mean(s['sim_baseline_mean'])
            base_std = get_std(s['sim_baseline_mean'])
            base_str = format_with_std_latex(base_mean, base_std, precision=3, show_std=(base_std > 0))
            lines.append(f"{mode_display} & {count} & {base_str} & {a_str} & {b_str} & {delta_str} & {win_str} \\\\")
        else:
            lines.append(f"{mode_display} & {count} & {a_str} & {b_str} & {delta_str} & {win_str} \\\\")
    
    lines.append(r"\midrule")
    s = stats['overall']
    
    # Handle aggregated overall stats
    sim_a_mean = get_mean(s['sim_a_mean'])
    sim_a_std = get_std(s['sim_a_mean'])
    sim_b_mean = get_mean(s['sim_b_mean'])
    sim_b_std = get_std(s['sim_b_mean'])
    delta_mean = get_mean(s['delta_mean'])
    delta_std = get_std(s['delta_mean'])
    b_win_rate = get_mean(s['b_win_rate'])
    b_win_std = get_std(s['b_win_rate'])
    count = int(get_mean(s['count']))
    
    a_str = format_with_std_latex(sim_a_mean, sim_a_std, precision=3, show_std=(sim_a_std > 0))
    b_str = format_with_std_latex(sim_b_mean, sim_b_std, precision=3, show_std=(sim_b_std > 0))
    if delta_std > 0:
        sign = "+" if delta_mean >= 0 else ""
        delta_str = f"{sign}{delta_mean:.3f}$\\pm${delta_std:.3f}"
    else:
        delta_str = f"{delta_mean:+.3f}"
    if b_win_std > 0:
        win_str = f"{b_win_rate:.1f}$\\pm${b_win_std:.1f}\\%"
    else:
        win_str = f"{b_win_rate:.1f}\\%"
    
    if has_baseline:
        base_mean = get_mean(s['sim_baseline_mean'])
        base_std = get_std(s['sim_baseline_mean'])
        base_str = format_with_std_latex(base_mean, base_std, precision=3, show_std=(base_std > 0))
        lines.append(f"Overall & {count} & {base_str} & {a_str} & {b_str} & {delta_str} & {win_str} \\\\")
    else:
        lines.append(f"Overall & {count} & {a_str} & {b_str} & {delta_str} & {win_str} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Average cosine similarity between original and perturbed sentences. Lower values indicate the model correctly distinguishes structural perturbations. $\Delta$ shows the difference (negative = structured model is better). ``B Wins'' shows the percentage of pairs where the structured model has notably lower similarity.}")
    lines.append(r"\label{tab:mrpc_perturbation_eval}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def generate_examples_latex(
    cases: Dict,
    model_a_name: str,
    model_b_name: str,
    n_per_category: int = 2
) -> str:
    """Generate LaTeX table with example pairs."""
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{p{1.2cm}p{5.5cm}p{5.5cm}cc}")
    lines.append(r"\toprule")
    lines.append(f"Type & Original & Perturbed & {model_a_name} & {model_b_name} \\\\")
    lines.append(r"\midrule")
    
    def escape_latex(s):
        return s.replace("&", r"\&").replace("%", r"\%").replace("$", r"\$").replace("_", r"\_")
    
    def truncate(s, max_len=120):
        if len(s) > max_len:
            return s[:max_len-3] + "..."
        return s
    
    for mode in ['cannot_negation', 'binding_negation', 'spatial']:
        if mode not in cases:
            continue
        
        mode_display = mode.capitalize()
        
        # Show B success examples
        for i, r in enumerate(cases[mode].get('b_success', [])[:n_per_category]):
            sent1 = escape_latex(truncate(r.sentence1))
            sent2 = escape_latex(truncate(r.sentence2))
            if i == 0:
                lines.append(f"\\multirow{{{n_per_category}}}{{*}}{{{mode_display}}} & {sent1} & {sent2} & {r.sim_a:.2f} & \\textbf{{{r.sim_b:.2f}}} \\\\")
            else:
                lines.append(f" & {sent1} & {sent2} & {r.sim_a:.2f} & \\textbf{{{r.sim_b:.2f}}} \\\\")
        
        lines.append(r"\midrule")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Example pairs where the structured model correctly assigns lower similarity to perturbed sentences. Bold values indicate the better (lower) similarity score.}")
    lines.append(r"\label{tab:mrpc_examples}")
    lines.append(r"\end{table*}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on MRPC negative pairs and compare with retrieval datasets (MSMARCO/NQ)")
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Optional HuggingFace model ID for baseline (shown in first column). Overrides --baseline-dataset."
    )
    parser.add_argument(
        "--baseline-dataset",
        type=str,
        choices=["msmarco", "nq"],
        default=None,
        help="Use model trained on this dataset as baseline. Constructs path from --backbone. Ignored if --baseline is set."
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="thenlper/gte-small",
        help="Backbone model name, used to construct baseline path when using --baseline-dataset"
    )
    parser.add_argument(
        "--model-a", 
        type=str, 
        default="redis/model-a-baseline",
        help="HuggingFace model ID for Model A"
    )
    parser.add_argument(
        "--model-b", 
        type=str, 
        default="redis/model-b-structured",
        help="HuggingFace model ID for Model B (structured)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/mrpc_unified_negative_pairs.csv",
        help="Path to the MRPC negative pairs CSV"
    )
    parser.add_argument(
        "--comparison-dataset",
        type=str,
        choices=["msmarco", "nq"],
        default=None,
        help="Dataset to use for comparison: 'msmarco' or 'nq'. Downloads from HuggingFace."
    )
    parser.add_argument(
        "--comparison-path",
        type=str,
        default=None,
        help="Path to local triplets CSV for comparison (overrides --comparison-dataset)"
    )
    parser.add_argument(
        "--comparison-samples",
        type=int,
        default=5000,
        help="Number of samples to use for comparison (default: 5000)"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Directory for local model checkpoints (e.g., 'models/' for msmarco, 'models_nq/' for nq). "
             "If set, --model-a and --model-b are treated as subdirectory names within this directory."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding"
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=5,
        help="Number of examples to show per category"
    )
    parser.add_argument(
        "--output-latex",
        type=str,
        default=None,
        help="Path to save LaTeX output"
    )
    parser.add_argument(
        "--comparison-only",
        action="store_true",
        help="Only run the comparison analysis (skip detailed per-category analysis)"
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="figures",
        help="Directory to save distribution plots (default: figures)"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating distribution plots"
    )
    parser.add_argument(
        "--save-stats",
        type=str,
        default=None,
        help="Path to save statistics JSON for later aggregation (e.g., results/seed_42/eval_stats.json)"
    )
    parser.add_argument(
        "--aggregate-seeds",
        type=str,
        default=None,
        help="Directory containing seed_* subdirectories with eval_stats.json. "
             "Aggregates results and displays mean +/- std. Skips model evaluation."
    )
    args = parser.parse_args()
    
    # Handle --aggregate-seeds mode: load and aggregate pre-computed stats
    if args.aggregate_seeds:
        parent_dir = Path(args.aggregate_seeds)
        aggregated_stats, metadata = load_and_aggregate_seed_stats(parent_dir)
        
        if aggregated_stats is None:
            print(f"Error: Could not load statistics from {parent_dir}")
            return
        
        # Get model names from metadata or use defaults
        model_a_name = metadata.get("model_a_name", "Model A") if metadata else "Model A"
        model_b_name = metadata.get("model_b_name", "Model B") if metadata else "Model B"
        baseline_name = metadata.get("baseline_name") if metadata else None
        
        print(f"\n=== Aggregated Multi-Seed Results ===")
        print(f"Source: {parent_dir}")
        
        # Print summary with aggregated stats
        print_summary(aggregated_stats, model_a_name, model_b_name, baseline_name)
        
        # Generate LaTeX if requested
        if args.output_latex:
            latex = generate_latex_table(aggregated_stats, model_a_name, model_b_name, baseline_name)
            with open(args.output_latex, "w") as f:
                f.write(latex)
            print(f"\nLaTeX output saved to {args.output_latex}")
        
        return
    
    # Resolve model paths with --models-dir if specified
    model_a_path = args.model_a
    model_b_path = args.model_b
    
    if args.models_dir:
        # Treat model-a and model-b as subdirectory names within models_dir
        import os
        model_a_path = os.path.join(args.models_dir, args.model_a)
        model_b_path = os.path.join(args.models_dir, args.model_b)
        print(f"Loading models from {args.models_dir}/")
        print(f"  Model A: {model_a_path}")
        print(f"  Model B: {model_b_path}")
    
    # Resolve baseline model path
    baseline_model = args.baseline
    if baseline_model is None and args.baseline_dataset:
        # Construct baseline path from backbone and dataset
        # Format: results/<backbone_short>/model-a-<dataset>/
        backbone_short = args.backbone.split("/")[-1]
        baseline_model = f"results/{backbone_short}/model-a-{args.baseline_dataset}"
        print(f"Using baseline model from {args.baseline_dataset} dataset: {baseline_model}")
    
    # Load models
    baseline, model_a, model_b = load_models(model_a_path, model_b_path, baseline_model)
    
    # Extract short names
    if baseline_model:
        # For local paths like results/gte-small/model-a-msmarco, extract meaningful name
        if baseline_model.startswith("results/"):
            parts = baseline_model.rstrip("/").split("/")
            baseline_short = f"{parts[-1]}" if len(parts) >= 2 else baseline_model
        else:
            baseline_short = baseline_model.split("/")[-1]
    else:
        baseline_short = None
    model_a_short = args.model_a.split("/")[-1]
    model_b_short = args.model_b.split("/")[-1]
    
    # Load unified negatives dataset
    df = load_dataset(args.data_path)
    
    # Evaluate unified negatives
    results = evaluate_pairs(model_a, model_b, df, batch_size=args.batch_size, baseline=baseline, pair_type="negative")
    
    # Compute statistics
    has_baseline = baseline_model is not None
    stats = compute_statistics(results, has_baseline=has_baseline)
    
    # Comparison dataset evaluation (MSMARCO or NQ)
    comparison_pos_results = []
    comparison_neg_results = []
    comparison_dataset_name = None
    
    if args.comparison_path or args.comparison_dataset:
        if args.comparison_path:
            # Load from local path
            comparison_df = load_triplets_from_csv(args.comparison_path, max_samples=args.comparison_samples)
            comparison_dataset_name = "Local"
        else:
            # Download from HuggingFace
            comparison_df = load_triplets_from_huggingface(args.comparison_dataset, max_samples=args.comparison_samples)
            comparison_dataset_name = args.comparison_dataset.upper()
        
        comparison_pos_results, comparison_neg_results = evaluate_triplets(
            model_a, model_b, comparison_df, 
            batch_size=args.batch_size, 
            baseline=baseline
        )
        
        # Print comparison summary
        print_comparison_summary(
            results, 
            comparison_pos_results, 
            comparison_neg_results,
            model_a_short, 
            model_b_short, 
            baseline_name=baseline_short,
            comparison_dataset_name=comparison_dataset_name
        )
        
        # Generate distribution plots
        if not args.no_plots:
            print("\n" + "="*100)
            print(f" GENERATING DISTRIBUTION PLOTS ({comparison_dataset_name})")
            print("="*100)
            plot_similarity_distributions(
                unified_neg_results=results,
                comparison_pos_results=comparison_pos_results,
                comparison_neg_results=comparison_neg_results,
                model_a_name=model_a_short,
                model_b_name=model_b_short,
                output_dir=args.figures_dir,
                baseline_name=baseline_short,
                comparison_dataset_name=comparison_dataset_name
            )
    
    if not args.comparison_only:
        # Print summary
        print_summary(stats, model_a_short, model_b_short, baseline_name=baseline_short)
        
        # Print distribution analysis
        print_distribution_analysis(results, model_a_short, model_b_short)
        
        # Find and print interesting cases
        cases = find_interesting_cases(results, n_examples=args.n_examples)
        print_interesting_cases(cases, model_a_short, model_b_short)
        
        # Generate LaTeX
        print("\n" + "="*100)
        print(" LATEX OUTPUT")
        print("="*100)
        
        latex_summary = generate_latex_table(stats, model_a_short, model_b_short, baseline_name=baseline_short)
        print("\n% Summary Table")
        print(latex_summary)
        
        latex_examples = generate_examples_latex(cases, model_a_short, model_b_short)
        print("\n% Examples Table")
        print(latex_examples)
        
        if args.output_latex:
            with open(args.output_latex, 'w') as f:
                f.write("% Summary Table\n")
                f.write(latex_summary)
                f.write("\n\n% Examples Table\n")
                f.write(latex_examples)
            print(f"\nLaTeX saved to: {args.output_latex}")
    
    # Print final summary if comparison dataset was used
    if args.comparison_path or args.comparison_dataset:
        print("\n" + "="*100)
        print(f" FINAL SUMMARY ({comparison_dataset_name})")
        print("="*100)
        
        unified_stats = compute_simple_stats(results, has_baseline)
        comparison_pos_stats = compute_simple_stats(comparison_pos_results, has_baseline)
        comparison_neg_stats = compute_simple_stats(comparison_neg_results, has_baseline)
        
        print(f"\nKey Findings:")
        print(f"  1. {comparison_dataset_name} Positives (should be HIGH): {model_a_short}={comparison_pos_stats['sim_a_mean']:.4f}, {model_b_short}={comparison_pos_stats['sim_b_mean']:.4f}")
        print(f"  2. {comparison_dataset_name} Negatives (should be LOW): {model_a_short}={comparison_neg_stats['sim_a_mean']:.4f}, {model_b_short}={comparison_neg_stats['sim_b_mean']:.4f}")
        print(f"  3. Unified Negatives (should be LOW): {model_a_short}={unified_stats['sim_a_mean']:.4f}, {model_b_short}={unified_stats['sim_b_mean']:.4f}")
        
        # Separation gaps
        gap_a = comparison_pos_stats['sim_a_mean'] - comparison_neg_stats['sim_a_mean']
        gap_b = comparison_pos_stats['sim_b_mean'] - comparison_neg_stats['sim_b_mean']
        
        print(f"\n  {comparison_dataset_name} Positive-Negative Separation:")
        print(f"    {model_a_short}: {gap_a:.4f}")
        print(f"    {model_b_short}: {gap_b:.4f}")
        print(f"    Winner: {model_b_short if gap_b > gap_a else model_a_short} (by {abs(gap_b - gap_a):.4f})")
        
        # Unified negatives hardness
        unified_harder_a = unified_stats['sim_a_mean'] - comparison_neg_stats['sim_a_mean']
        unified_harder_b = unified_stats['sim_b_mean'] - comparison_neg_stats['sim_b_mean']
        
        print(f"\n  Unified Negatives are {'harder' if unified_harder_a > 0 else 'easier'} than {comparison_dataset_name} for {model_a_short} (diff: {unified_harder_a:+.4f})")
        print(f"  Unified Negatives are {'harder' if unified_harder_b > 0 else 'easier'} than {comparison_dataset_name} for {model_b_short} (diff: {unified_harder_b:+.4f})")
    
    # Save statistics to JSON if requested (for multi-seed aggregation)
    if args.save_stats:
        save_path = Path(args.save_stats)
        metadata = {
            "model_a_name": model_a_short,
            "model_b_name": model_b_short,
            "baseline_name": baseline_short,
            "data_path": args.data_path,
        }
        save_stats_to_json(stats, save_path, metadata)


if __name__ == "__main__":
    main()
