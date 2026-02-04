"""
Verifier Experiment Results Viewer.

Loads all_results.json from verifier experiments and generates LaTeX tables
for NanoBEIR performance and synthetic test set performance.

Supports multi-seed results:
- Detects seed_* subdirectories in the results path
- Aggregates metrics across seeds (computes mean +/- std)
- Displays error bars in tables and plots

The JSON structure has two NanoBEIR metric sets:
- nanobeir_encoder_only: Encoder-only cosine similarity retrieval
- nanobeir_with_verifier: Two-stage retrieval with verifier reranking
  - before_rerank: Metrics before verifier reranking
  - after_rerank: Metrics after verifier reranking
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from aggregate_utils import (
    find_seed_dirs,
    aggregate_nested_dict,
    load_json_safe,
    get_mean,
    get_std,
    is_aggregated_value,
    format_with_std_latex,
    format_diff_cell_latex,
    compute_diff_with_error,
)


# Verifier names and display names
# Exclude "None" as we use hardcoded baselines
VERIFIERS = ["F0", "F1", "F2", "F3", "F4"]
VERIFIER_NAMES = {
    "None": r"$\emptyset$ (None)",
    "F0": r"$F_0$",
    "F1": r"$F_1$",
    "F2": r"$F_2$",
    "F3": r"$F_3$",
    "F4": r"$F_4$",
}

# Training modes
TRAINING_MODES = ["frozen", "end_to_end"]
MODE_DISPLAY = {
    "frozen": "Frozen",
    "end_to_end": "End-to-End",
}

# NanoBEIR metrics to display
NANOBEIR_METRICS = [
    ("ndcg@10", "NDCG@10"),
    ("mrr@10", "MRR@10"),
    ("accuracy@1", "Acc@1"),
    ("accuracy@10", "Acc@10"),
]

# Synthetic test set categories
SYNTHETIC_CATEGORIES = ["negation", "binding", "spatial", "overall"]
SYNTHETIC_DISPLAY = {
    "negation": "Negation",
    "binding": "Binding",
    "spatial": "Spatial",
    "overall": "Overall",
}

SYNTHETIC_TITLES = {
    "negation": "Negation",
    "binding": "Binding",
    "spatial": "Spatial",
    "overall": "Overall (Mean)",
}

# =============================================================================
# HARDCODED BASELINES
# =============================================================================

BASELINE_METRICS = {
    "nanobeir": {
        "model_a": { # Baseline (QQP only)
            "ndcg@10": 0.348,
            "mrr@10": 0.395,
            "accuracy@1": 0.293,
            "accuracy@10": 0.623
        },
        "model_b": { # Structured (QQP + Synthetic)
            "ndcg@10": 0.253,
            "mrr@10": 0.274,
            "accuracy@1": 0.189,
            "accuracy@10": 0.490
        }
    },
    "synthetic": {
        "model_a": { # Baseline
            "negation": 0.948,
            "binding": 0.949,
            "spatial": 0.992,
            "overall": 0.964
        },
        "model_b": { # Structured (Fine-tuned)
            "negation": 0.638,
            "binding": 0.981,
            "spatial": -0.073,
            "overall": 0.512
        }
    }
}


def load_all_results(results_path: Path) -> Optional[Dict]:
    """
    Load all_results.json from the given path.
    
    Supports multi-seed results: detects seed_* subdirectories and
    aggregates results across seeds with mean +/- std.
    
    Args:
        results_path: Path to directory containing all_results.json or seed_* subdirs,
                      or direct path to all_results.json
    
    Returns:
        Parsed JSON data (possibly aggregated with mean/std) or None if not found
    """
    # Check for seed subdirectories first
    if results_path.is_dir():
        seed_dirs = find_seed_dirs(results_path)
        
        if seed_dirs:
            print(f"Found {len(seed_dirs)} seeds: {[d.name for d in seed_dirs]}")
            all_data = []
            for seed_dir in seed_dirs:
                json_path = seed_dir / "all_results.json"
                data = load_json_safe(json_path)
                if data is not None:
                    all_data.append(data)
            
            if all_data:
                print(f"Aggregating results from {len(all_data)} seeds...")
                return aggregate_nested_dict(all_data)
            else:
                print("Warning: No valid all_results.json files found in seed directories")
                return None
    
    # Fallback to single file behavior
    if results_path.is_file():
        json_path = results_path
    else:
        json_path = results_path / "all_results.json"
    
    if not json_path.exists():
        print(f"Error: {json_path} not found")
        return None
    
    try:
        with open(json_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return None


def format_value(value: Union[float, Dict], precision: int = 3, show_std: bool = True) -> str:
    """Format a metric value for display, supporting aggregated values with std."""
    mean = get_mean(value)
    std = get_std(value)
    
    if show_std and std > 0:
        return f"{mean:.{precision}f}$\\pm${std:.{precision}f}"
    return f"{mean:.{precision}f}"


def format_params(params: int) -> str:
    """Format parameter count with commas."""
    if params == 0:
        return "0"
    return f"{params:,}"


def generate_nanobeir_unified_table(data: Dict) -> str:
    """
    Generate unified LaTeX table for NanoBEIR performance.
    """
    lines = []
    
    # Table header
    lines.append("% NanoBEIR Unified Performance Table")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Verifier performance on NanoBEIR. ``Encoder'' shows cosine similarity retrieval; ``+Verifier'' shows retrieval with verifier reranking.}")
    lines.append(r"\label{tab:verifier-nanobeir}")
    
    # Column specification
    metric_cols = " ".join(["r"] * len(NANOBEIR_METRICS))
    lines.append(r"\begin{tabular}{lll" + metric_cols + r"r}")
    lines.append(r"\toprule")
    
    # Header row
    header_parts = ["Verifier", "Mode", "Retrieval"]
    header_parts.extend([name for _, name in NANOBEIR_METRICS])
    header_parts.append("Params")
    lines.append(" & ".join(header_parts) + r" \\")
    lines.append(r"\midrule")
    
    # Data rows - group by verifier
    for i, verifier in enumerate(VERIFIERS):
        # Check which modes have data for this verifier
        modes_with_data = []
        for mode in TRAINING_MODES:
            if mode in data and verifier in data[mode]:
                modes_with_data.append(mode)
        
        if not modes_with_data:
            continue
        
        # Each mode has 2 rows (encoder-only and +verifier)
        total_rows = len(modes_with_data) * 2
        row_idx = 0
        
        for j, mode in enumerate(modes_with_data):
            verifier_data = data[mode][verifier]
            nanobeir_enc = verifier_data.get("nanobeir_encoder_only", {})
            nanobeir_v = verifier_data.get("nanobeir_with_verifier", {})
            nanobeir_rerank = nanobeir_v.get("after_rerank", {})
            params = verifier_data.get("num_parameters", 0)
            
            # Row 1: Encoder-only
            row_parts = []
            if row_idx == 0:
                row_parts.append(r"\multirow{" + str(total_rows) + r"}{*}{" + VERIFIER_NAMES[verifier] + r"}")
            else:
                row_parts.append("")
            
            if row_idx % 2 == 0:
                row_parts.append(r"\multirow{2}{*}{" + MODE_DISPLAY[mode] + r"}")
            else:
                row_parts.append("")
            
            row_parts.append("Encoder")
            for metric_key, _ in NANOBEIR_METRICS:
                value = nanobeir_enc.get(metric_key, 0)
                row_parts.append(format_value(value))
            row_parts.append(format_params(params))
            lines.append(" & ".join(row_parts) + r" \\")
            row_idx += 1
            
            # Row 2: +Verifier
            row_parts = ["", "", "+Verifier"]
            for metric_key, _ in NANOBEIR_METRICS:
                value = nanobeir_rerank.get(metric_key, 0)
                row_parts.append(format_value(value))
            row_parts.append("")
            lines.append(" & ".join(row_parts) + r" \\")
            row_idx += 1
            
            # Add cline after each mode (except the last one for this verifier)
            if j < len(modes_with_data) - 1:
                lines.append(r"\cline{2-" + str(3 + len(NANOBEIR_METRICS) + 1) + r"}")
        
        # Add midrule between verifiers (not after last one)
        if i < len(VERIFIERS) - 1:
            has_next = any(
                mode in data and next_v in data[mode]
                for next_v in VERIFIERS[i+1:]
                for mode in TRAINING_MODES
            )
            if has_next:
                lines.append(r"\midrule")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def generate_synthetic_table(data: Dict) -> str:
    """
    Generate LaTeX table for synthetic test set performance.
    """
    lines = []
    
    # Table header
    lines.append("% Synthetic Test Set Performance Table")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Performance on synthetic test set. ``Encoder'' shows cosine similarity; ``+Verifier'' shows verifier scores. Lower = better.}")
    lines.append(r"\label{tab:verifier-synthetic}")
    
    # Column specification
    cat_cols = " ".join(["r"] * len(SYNTHETIC_CATEGORIES))
    lines.append(r"\begin{tabular}{lll" + cat_cols + r"}")
    lines.append(r"\toprule")
    
    # Header row
    header_parts = ["Verifier", "Mode", "Eval"]
    header_parts.extend([SYNTHETIC_DISPLAY[cat] for cat in SYNTHETIC_CATEGORIES])
    lines.append(" & ".join(header_parts) + r" \\")
    lines.append(r"\midrule")
    
    # Data rows - group by verifier
    for i, verifier in enumerate(VERIFIERS):
        # Check which modes have data for this verifier
        modes_with_data = []
        for mode in TRAINING_MODES:
            if mode in data and verifier in data[mode]:
                modes_with_data.append(mode)
        
        if not modes_with_data:
            continue
        
        # Each mode has 2 rows (encoder-only and +verifier)
        total_rows = len(modes_with_data) * 2
        row_idx = 0
        
        for j, mode in enumerate(modes_with_data):
            verifier_data = data[mode][verifier]
            # Support both old format (mrpc_metrics) and new format (mrpc_encoder_only, mrpc_with_verifier)
            mrpc_enc = verifier_data.get("mrpc_encoder_only", verifier_data.get("mrpc_metrics", {}))
            mrpc_v = verifier_data.get("mrpc_with_verifier", verifier_data.get("mrpc_metrics", {}))
            
            # Row 1: Encoder-only
            row_parts = []
            if row_idx == 0:
                row_parts.append(r"\multirow{" + str(total_rows) + r"}{*}{" + VERIFIER_NAMES[verifier] + r"}")
            else:
                row_parts.append("")
            
            if row_idx % 2 == 0:
                row_parts.append(r"\multirow{2}{*}{" + MODE_DISPLAY[mode] + r"}")
            else:
                row_parts.append("")
            
            row_parts.append("Encoder")
            for cat in SYNTHETIC_CATEGORIES:
                cat_data = mrpc_enc.get(cat, {})
                mean_val = cat_data.get("mean", 0)
                row_parts.append(format_value(mean_val))
            lines.append(" & ".join(row_parts) + r" \\")
            row_idx += 1
            
            # Row 2: +Verifier
            row_parts = ["", "", "+Verifier"]
            for cat in SYNTHETIC_CATEGORIES:
                cat_data = mrpc_v.get(cat, {})
                mean_val = cat_data.get("mean", 0)
                row_parts.append(format_value(mean_val))
            lines.append(" & ".join(row_parts) + r" \\")
            row_idx += 1
            
            # Add cline after each mode (except the last one for this verifier)
            if j < len(modes_with_data) - 1:
                lines.append(r"\cline{2-" + str(3 + len(SYNTHETIC_CATEGORIES)) + r"}")
        
        # Add midrule between verifiers (not after last one)
        if i < len(VERIFIERS) - 1:
            has_next = any(
                mode in data and next_v in data[mode]
                for next_v in VERIFIERS[i+1:]
                for mode in TRAINING_MODES
            )
            if has_next:
                lines.append(r"\midrule")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def print_summary(data: Dict):
    """Print a text summary of verifier results."""
    # Unified NanoBEIR Performance
    print("=" * 115)
    print("NanoBEIR Performance (Encoder = cosine similarity, +Verifier = with reranking)")
    print("=" * 115)
    
    # Header
    header = f"{'Verifier':<10} {'Mode':<12} {'Retrieval':<12}"
    for _, name in NANOBEIR_METRICS:
        header += f" {name:>10}"
    header += f" {'Params':>12}"
    print(header)
    print("-" * 115)
    
    for verifier in VERIFIERS:
        for mode in TRAINING_MODES:
            if mode not in data or verifier not in data[mode]:
                continue
            
            verifier_data = data[mode][verifier]
            nanobeir_enc = verifier_data.get("nanobeir_encoder_only", {})
            nanobeir_v = verifier_data.get("nanobeir_with_verifier", {})
            nanobeir_rerank = nanobeir_v.get("after_rerank", {})
            params = verifier_data.get("num_parameters", 0)
            
            # Encoder-only row
            row = f"{verifier:<10} {MODE_DISPLAY[mode]:<12} {'Encoder':<12}"
            for metric_key, _ in NANOBEIR_METRICS:
                value = nanobeir_enc.get(metric_key, 0)
                row += f" {value:>10.4f}"
            row += f" {params:>12,}"
            print(row)
            
            # +Verifier row
            row = f"{'':<10} {'':<12} {'+Verifier':<12}"
            for metric_key, _ in NANOBEIR_METRICS:
                value = nanobeir_rerank.get(metric_key, 0)
                row += f" {value:>10.4f}"
            row += f" {'':<12}"
            print(row)
        
        # Separator between verifiers
        print("-" * 115)
    
    print()
    
    # Synthetic Test Set Performance
    print("=" * 120)
    print("Synthetic Test Set Performance (MRPC negatives, lower = better)")
    print("=" * 120)
    
    # Header
    header = f"{'Verifier':<10} {'Mode':<12} {'Eval':<12}"
    for cat in SYNTHETIC_CATEGORIES:
        header += f" {SYNTHETIC_DISPLAY[cat]:>10}"
    print(header)
    print("-" * 120)
    
    for verifier in VERIFIERS:
        for mode in TRAINING_MODES:
            if mode not in data or verifier not in data[mode]:
                continue
            
            verifier_data = data[mode][verifier]
            # Support both old format (mrpc_metrics) and new format (mrpc_encoder_only, mrpc_with_verifier)
            mrpc_enc = verifier_data.get("mrpc_encoder_only", verifier_data.get("mrpc_metrics", {}))
            mrpc_v = verifier_data.get("mrpc_with_verifier", verifier_data.get("mrpc_metrics", {}))
            
            # Encoder-only row
            row = f"{verifier:<10} {MODE_DISPLAY[mode]:<12} {'Encoder':<12}"
            for cat in SYNTHETIC_CATEGORIES:
                cat_data = mrpc_enc.get(cat, {})
                mean_val = cat_data.get("mean", 0)
                row += f" {mean_val:>10.4f}"
            print(row)
            
            # +Verifier row
            row = f"{'':<10} {'':<12} {'+Verifier':<12}"
            for cat in SYNTHETIC_CATEGORIES:
                cat_data = mrpc_v.get(cat, {})
                mean_val = cat_data.get("mean", 0)
                row += f" {mean_val:>10.4f}"
            print(row)
        
        print("-" * 120)
    
    print()


def display_latex_tables(data: Dict):
    """Display LaTeX tables for verifier results."""
    print(r"% LaTeX Preamble (add to your document):")
    print(r"% \usepackage{booktabs}")
    print(r"% \usepackage{multirow}")
    print()
    
    print("% " + "=" * 60)
    print("% NanoBEIR Unified Performance Table")
    print("% " + "=" * 60)
    print()
    print(generate_nanobeir_unified_table(data))
    print()
    
    print("% " + "=" * 60)
    print("% Synthetic Test Set Performance Table")
    print("% " + "=" * 60)
    print()
    print(generate_synthetic_table(data))
    print()


def extract_nanobeir_data(data: Dict, metric_key: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Extract NanoBEIR data for a specific metric.
    """
    result = {}
    for verifier in VERIFIERS:
        result[verifier] = {}
        for mode in TRAINING_MODES:
            result[verifier][mode] = {"verifier": None, "encoder": None}
            if mode in data and verifier in data[mode]:
                verifier_data = data[mode][verifier]
                # Get verifier metrics (after reranking)
                nanobeir_ver = verifier_data.get("nanobeir_with_verifier", {})
                nanobeir_rerank = nanobeir_ver.get("after_rerank", {})
                result[verifier][mode]["verifier"] = nanobeir_rerank.get(metric_key, None)
                
                # Get encoder metrics
                enc_data = verifier_data.get("nanobeir_encoder_only", {})
                result[verifier][mode]["encoder"] = enc_data.get(metric_key, None)
    return result


def extract_synthetic_data(data: Dict, category: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Extract synthetic test set data for a specific category.
    """
    result = {}
    for verifier in VERIFIERS:
        result[verifier] = {}
        for mode in TRAINING_MODES:
            result[verifier][mode] = {"verifier": None, "encoder": None}
            if mode in data and verifier in data[mode]:
                verifier_data = data[mode][verifier]
                # Get verifier metrics
                mrpc_ver = verifier_data.get("mrpc_with_verifier", verifier_data.get("mrpc_metrics", {}))
                cat_data_ver = mrpc_ver.get(category, {})
                result[verifier][mode]["verifier"] = cat_data_ver.get("mean", None)
                
                # Get encoder metrics
                mrpc_enc = verifier_data.get("mrpc_encoder_only", verifier_data.get("mrpc_metrics", {}))
                cat_data_enc = mrpc_enc.get(category, {})
                result[verifier][mode]["encoder"] = cat_data_enc.get("mean", None)
    return result


def plot_retrieval(ax: plt.Axes, data: Dict, metric_key: str, title: str):
    """
    Plot 1: NanoBEIR Retrieval - Verifiers vs Baselines.
    
    Supports multi-seed results with error bars.
    """
    metric_data = extract_nanobeir_data(data, metric_key)
    x = np.arange(len(VERIFIERS))
    
    # 1. Baselines
    baseline_a = BASELINE_METRICS["nanobeir"]["model_a"].get(metric_key)
    baseline_b = BASELINE_METRICS["nanobeir"]["model_b"].get(metric_key)
    
    if baseline_a:
        ax.axhline(y=baseline_a, color="black", linestyle="-", alpha=0.6, linewidth=1.5, label="Model A (Standard Baseline)")
    if baseline_b:
        ax.axhline(y=baseline_b, color="gray", linestyle=":", alpha=0.8, linewidth=1.5, label="Model B (Structured Baseline)")
        
    # 2. Verifiers (Frozen & E2E) - Extract means and stds for aggregated values
    y_frozen_raw = [metric_data[v]["frozen"]["verifier"] for v in VERIFIERS]
    y_e2e_raw = [metric_data[v]["end_to_end"]["verifier"] for v in VERIFIERS]
    
    # Extract means and stds, filter Nones
    x_frozen = []
    y_frozen = []
    err_frozen = []
    for i, val in enumerate(y_frozen_raw):
        if val is not None:
            x_frozen.append(i)
            y_frozen.append(get_mean(val))
            err_frozen.append(get_std(val))
    
    x_e2e = []
    y_e2e = []
    err_e2e = []
    for i, val in enumerate(y_e2e_raw):
        if val is not None:
            x_e2e.append(i)
            y_e2e.append(get_mean(val))
            err_e2e.append(get_std(val))
    
    # Plot with error bars if we have std data
    if y_frozen:
        if any(e > 0 for e in err_frozen):
            ax.errorbar(x_frozen, y_frozen, yerr=err_frozen, color="#1f77b4", marker="o", 
                       linestyle="--", capsize=3, capthick=1, label="Frozen Encoder + Verifier")
        else:
            ax.plot(x_frozen, y_frozen, color="#1f77b4", marker="o", linestyle="--", 
                   label="Frozen Encoder + Verifier")
    if y_e2e:
        if any(e > 0 for e in err_e2e):
            ax.errorbar(x_e2e, y_e2e, yerr=err_e2e, color="#ff7f0e", marker="s", 
                       linestyle="--", capsize=3, capthick=1, label="E2E Encoder + Verifier")
        else:
            ax.plot(x_e2e, y_e2e, color="#ff7f0e", marker="s", linestyle="--", 
                   label="E2E Encoder + Verifier")
        
    ax.set_xticks(x)
    ax.set_xticklabels([VERIFIER_NAMES[v] for v in VERIFIERS], fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')


def plot_synthetic(ax: plt.Axes, data: Dict, category: str, title: str):
    """
    Plot 2: Synthetic Test - Verifiers vs Structured Baseline.
    
    Supports multi-seed results with error bars.
    """
    cat_data = extract_synthetic_data(data, category)
    x = np.arange(len(VERIFIERS))
    
    # 1. Baseline (Model B - Structured)
    baseline_b = BASELINE_METRICS["synthetic"]["model_b"].get(category)
    
    if baseline_b:
        ax.axhline(y=baseline_b, color="gray", linestyle=":", alpha=0.8, linewidth=1.5, label="Model B (Structured Baseline)")
        
    # 2. Verifiers - Extract means and stds for aggregated values
    y_frozen_raw = [cat_data[v]["frozen"]["verifier"] for v in VERIFIERS]
    y_e2e_raw = [cat_data[v]["end_to_end"]["verifier"] for v in VERIFIERS]
    
    # Extract means and stds, filter Nones
    x_frozen = []
    y_frozen = []
    err_frozen = []
    for i, val in enumerate(y_frozen_raw):
        if val is not None:
            x_frozen.append(i)
            y_frozen.append(get_mean(val))
            err_frozen.append(get_std(val))
    
    x_e2e = []
    y_e2e = []
    err_e2e = []
    for i, val in enumerate(y_e2e_raw):
        if val is not None:
            x_e2e.append(i)
            y_e2e.append(get_mean(val))
            err_e2e.append(get_std(val))

    # Plot with error bars if we have std data
    if y_frozen:
        if any(e > 0 for e in err_frozen):
            ax.errorbar(x_frozen, y_frozen, yerr=err_frozen, color="#1f77b4", marker="o",
                       linestyle="--", capsize=3, capthick=1, label="Frozen Encoder + Verifier")
        else:
            ax.plot(x_frozen, y_frozen, color="#1f77b4", marker="o", linestyle="--",
                   label="Frozen Encoder + Verifier")
    if y_e2e:
        if any(e > 0 for e in err_e2e):
            ax.errorbar(x_e2e, y_e2e, yerr=err_e2e, color="#ff7f0e", marker="s",
                       linestyle="--", capsize=3, capthick=1, label="E2E Encoder + Verifier")
        else:
            ax.plot(x_e2e, y_e2e, color="#ff7f0e", marker="s", linestyle="--",
                   label="E2E Encoder + Verifier")
        
    ax.set_xticks(x)
    ax.set_xticklabels([VERIFIER_NAMES[v] for v in VERIFIERS], fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Similarity (Lower is Better)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def plot_encoder_comparison(ax: plt.Axes, data: Dict, metric_key: str, title: str, metric_type="nanobeir"):
    """
    Plot 3: Encoder Quality Comparison - Does E2E training hurt the encoder?
    
    Supports multi-seed results with error bars.
    """
    x = np.arange(len(VERIFIERS))
    
    # 1. Baselines
    if metric_type == "nanobeir":
        baseline_a = BASELINE_METRICS["nanobeir"]["model_a"].get(metric_key)
        baseline_b = BASELINE_METRICS["nanobeir"]["model_b"].get(metric_key)
        ylabel = metric_key.upper()
    else:
        baseline_a = BASELINE_METRICS["synthetic"]["model_a"].get(metric_key)
        baseline_b = BASELINE_METRICS["synthetic"]["model_b"].get(metric_key)
        ylabel = "Mean Similarity"

    if baseline_a:
        ax.axhline(y=baseline_a, color="black", linestyle="-", alpha=0.6, label="Model A (Standard)")
    if baseline_b:
        ax.axhline(y=baseline_b, color="gray", linestyle=":", alpha=0.8, label="Model B (Structured)")
        
    # 2. E2E Encoders (The encoder part ONLY from the E2E trained systems)
    if metric_type == "nanobeir":
        metric_data = extract_nanobeir_data(data, metric_key)
        y_enc_raw = [metric_data[v]["end_to_end"]["encoder"] for v in VERIFIERS]
    else:
        metric_data = extract_synthetic_data(data, metric_key)
        y_enc_raw = [metric_data[v]["end_to_end"]["encoder"] for v in VERIFIERS]

    # Extract means and stds, filter Nones
    x_enc = []
    y_enc = []
    err_enc = []
    for i, val in enumerate(y_enc_raw):
        if val is not None:
            x_enc.append(i)
            y_enc.append(get_mean(val))
            err_enc.append(get_std(val))
    
    if y_enc:
        if any(e > 0 for e in err_enc):
            ax.errorbar(x_enc, y_enc, yerr=err_enc, color="#d62728", marker="D",
                       linestyle="-", capsize=3, capthick=1, label="Encoder (from E2E training)")
        else:
            ax.plot(x_enc, y_enc, color="#d62728", marker="D", linestyle="-",
                   label="Encoder (from E2E training)")

    ax.set_xticks(x)
    ax.set_xticklabels([VERIFIER_NAMES[v] for v in VERIFIERS], fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def generate_plots(
    data: Dict,
    output_dir: Optional[Path] = None,
    show: bool = True,
) -> None:
    """Generate all requested plots.
    
    Plot 1: NanoBEIR Retrieval - 4 metrics (2x2), baselines + verifiers only
    Plot 2: Synthetic Test - 4 categories (2x2), structured baseline + verifiers only
    Plot 3: Encoder Comparison - Compare E2E encoders (no verifier) vs baselines
    Plot 4: Short Retrieval - nDCG@10 and Acc@1 only (1x2)
    Plot 5: Short Encoder Comparison - nDCG@10 and Acc@1 only (1x2)
    """
    
    # 1. Retrieval Plot (4 metrics in 2x2 grid)
    retrieval_metrics = [
        ("mrr@10", "MRR@10"),
        ("ndcg@10", "NDCG@10"),
        ("accuracy@1", "Acc@1"),
        ("accuracy@10", "Acc@10"),
    ]
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (metric_key, metric_name) in zip(axes1.flat, retrieval_metrics):
        plot_retrieval(ax, data, metric_key, metric_name)
    fig1.suptitle("NanoBEIR Performance\n(Encoder + Verifier Reranking vs Baselines)", fontweight="bold")
    plt.tight_layout()
    
    # 2. Synthetic Plot (4 categories in 2x2 grid)
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    for ax, cat in zip(axes2.flat, SYNTHETIC_CATEGORIES):
        plot_synthetic(ax, data, cat, SYNTHETIC_TITLES[cat])
    fig2.suptitle("Synthetic Test Set Performance\n(Encoder + Verifier vs Structured Baseline)", fontweight="bold")
    plt.tight_layout()
    
    # 3. Encoder Comparison Plot (4 metrics in 2x2 grid)
    fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (metric_key, metric_name) in zip(axes3.flat, retrieval_metrics):
        plot_encoder_comparison(ax, data, metric_key, metric_name, "nanobeir")
    fig3.suptitle("Encoder Quality After E2E Training\n(Underlying Encoder Only, No Verifier)", fontweight="bold")
    plt.tight_layout()
    
    # 4. Short Retrieval Plot (nDCG@10 and Acc@1 only, 1x2 grid)
    short_metrics = [
        ("ndcg@10", "nDCG@10"),
        ("accuracy@1", "Acc@1"),
    ]
    fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (metric_key, metric_name) in zip(axes4.flat, short_metrics):
        plot_retrieval(ax, data, metric_key, metric_name)
    fig4.suptitle("NanoBEIR Performance (nDCG@10 & Acc@1)\n(Encoder + Verifier Reranking vs Baselines)", fontweight="bold")
    plt.tight_layout()
    
    # 5. Short Encoder Comparison Plot (nDCG@10 and Acc@1 only, 1x2 grid)
    fig5, axes5 = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (metric_key, metric_name) in zip(axes5.flat, short_metrics):
        plot_encoder_comparison(ax, data, metric_key, metric_name, "nanobeir")
    fig5.suptitle("Encoder Quality After E2E Training (nDCG@10 & Acc@1)\n(Underlying Encoder Only, No Verifier)", fontweight="bold")
    plt.tight_layout()
    
    if output_dir:
        fig1.savefig(output_dir / "plot_retrieval.png", dpi=300, bbox_inches="tight")
        fig1.savefig(output_dir / "plot_retrieval.pdf", bbox_inches="tight")
        print(f"Saved: {output_dir / 'plot_retrieval.png'}")
        
        fig2.savefig(output_dir / "plot_synthetic.png", dpi=300, bbox_inches="tight")
        fig2.savefig(output_dir / "plot_synthetic.pdf", bbox_inches="tight")
        print(f"Saved: {output_dir / 'plot_synthetic.png'}")
        
        fig3.savefig(output_dir / "plot_encoder_comparison.png", dpi=300, bbox_inches="tight")
        fig3.savefig(output_dir / "plot_encoder_comparison.pdf", bbox_inches="tight")
        print(f"Saved: {output_dir / 'plot_encoder_comparison.png'}")
        
        fig4.savefig(output_dir / "plot_retrieval_short.png", dpi=300, bbox_inches="tight")
        fig4.savefig(output_dir / "plot_retrieval_short.pdf", bbox_inches="tight")
        print(f"Saved: {output_dir / 'plot_retrieval_short.png'}")
        
        fig5.savefig(output_dir / "plot_encoder_comparison_short.png", dpi=300, bbox_inches="tight")
        fig5.savefig(output_dir / "plot_encoder_comparison_short.pdf", bbox_inches="tight")
        print(f"Saved: {output_dir / 'plot_encoder_comparison_short.png'}")

    if show:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
        plt.close(fig4)
        plt.close(fig5)


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables and plots from verifier experiment results"
    )
    parser.add_argument(
        "--results-dir", "-r",
        type=str,
        required=True,
        help="Path to directory containing all_results.json"
    )
    parser.add_argument(
        "--summary", "-s",
        action="store_true",
        help="Print text summary instead of LaTeX tables"
    )
    parser.add_argument(
        "--plot", "-p",
        action="store_true",
        help="Generate plots for synthetic test set results"
    )
    parser.add_argument(
        "--plot-output", "-po",
        type=str,
        default=None,
        help="Directory to save plots (uses results-dir if not specified)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't show plots interactively (only save to files)"
    )
    args = parser.parse_args()
    
    # Load results
    results_path = Path(args.results_dir)
    data = load_all_results(results_path)
    
    if data is None:
        return 1
    
    # Generate plots if requested
    if args.plot:
        plot_output_dir = Path(args.plot_output) if args.plot_output else results_path
        if plot_output_dir.is_file():
            plot_output_dir = plot_output_dir.parent
        generate_plots(
            data,
            output_dir=plot_output_dir,
            show=not args.no_show,
        )
    
    # Display results
    if args.summary:
        print_summary(data)
    elif not args.plot:
        # Only show LaTeX if not in plot mode (unless explicitly requested)
        display_latex_tables(data)
    
    return 0


if __name__ == "__main__":
    exit(main())
