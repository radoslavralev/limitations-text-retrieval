"""
Results Viewer Script.

Walks a results folder and displays metrics for model-a and model-b
variants, organized by backbone. Outputs LaTeX tables.

Supports multi-seed results:
- Detects seed_* subdirectories in each model folder
- Aggregates metrics across seeds (computes mean +/- std)
- Displays error bars in tables

Usage:
    python show_results.py [--results-dir <path>] [--format combined|per-backbone|subcolumns]
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, Union, Optional

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


# Dataset names (short names for table headers)
DATASETS = [
    ("NanoClimateFEVER", "NanoClimateFEVER"),
    ("NanoDBPedia", "NanoDBPedia"),
    ("NanoFEVER", "NanoFEVER"),
    ("NanoFiQA2018", "NanoFiQA2018"),
    ("NanoHotpotQA", "NanoHotpotQA"),
    ("NanoMSMARCO", "NanoMSMARCO"),
    ("NanoNFCorpus", "NanoNFCorpus"),
    ("NanoNQ", "NanoNQ"),
    ("NanoQuoraRetrieval", "NanoQuoraRetrieval"),
    ("NanoSCIDOCS", "NanoSCIDOCS"),
    ("NanoArguAna", "NanoArguAna"),
    ("NanoSciFact", "NanoSciFact"),
    ("NanoTouche2020", "NanoTouche2020"),
]

# Metrics to generate tables for
METRICS = [
    ("ndcg@10", "NDCG@10"),
    ("mrr@10", "MRR@10"),
    ("accuracy@1", "Acc@1"),
    ("accuracy@10", "Acc@10"),
]

# Backbone display names
BACKBONE_NAMES = {
    "all-MiniLM-L6-v2": "MiniLM-L6",
    "all-MiniLM-L12-v2": "MiniLM-L12",
    "gte-small": "GTE-Small",
    "gte-modernbert-base": "GTE-ModernBERT",
}


def load_metrics(metrics_path: Path) -> dict:
    """Load metrics from a final_metrics.json file."""
    with open(metrics_path, "r") as f:
        return json.load(f)


def load_metrics_with_seeds(model_dir: Path) -> Optional[dict]:
    """
    Load metrics from a model directory, supporting multi-seed structure.
    
    Checks for seed_* subdirectories first. If found, loads and aggregates
    metrics from all seeds. Otherwise, falls back to loading directly.
    
    Args:
        model_dir: Directory containing final_metrics.json or seed_* subdirs
        
    Returns:
        Metrics dict (possibly aggregated with mean/std), or None if not found
    """
    seed_dirs = find_seed_dirs(model_dir)
    
    if seed_dirs:
        # Multi-seed: load and aggregate
        print(f"  Found {len(seed_dirs)} seeds in {model_dir.name}")
        all_metrics = []
        for seed_dir in seed_dirs:
            metrics_path = seed_dir / "final_metrics.json"
            if metrics_path.exists():
                metrics = load_json_safe(metrics_path)
                if metrics is not None:
                    all_metrics.append(metrics)
        
        if all_metrics:
            return aggregate_nested_dict(all_metrics)
        return None
    else:
        # Single-seed: load directly
        metrics_path = model_dir / "final_metrics.json"
        if metrics_path.exists():
            return load_metrics(metrics_path)
        return None


def extract_all_metrics(metrics: dict) -> dict:
    """
    Extract all per-dataset metrics and mean metrics.
    
    Handles both single-value metrics and aggregated metrics with mean/std.
    """
    extracted = {}
    
    if "nano_beir" not in metrics:
        return extracted
    
    nb = metrics["nano_beir"]
    
    for metric_key, metric_name in METRICS:
        extracted[metric_name] = {}
        
        # Per-dataset metrics
        for dataset_full, dataset_short in DATASETS:
            key = f"{dataset_full}_cosine_{metric_key}"
            value = nb.get(key, 0)
            # Keep the aggregated structure if present
            extracted[metric_name][dataset_short] = value
        
        # Mean metric
        mean_key = f"NanoBEIR_mean_cosine_{metric_key}"
        value = nb.get(mean_key, 0)
        extracted[metric_name]["Mean"] = value
    
    return extracted


def walk_results(results_dir: Path) -> dict:
    """
    Walk the results directory and collect all metrics.
    
    Supports multi-seed results: detects seed_* subdirectories and
    aggregates metrics across seeds with mean +/- std.
    
    Returns:
        dict: {backbone_name: {model_name: metrics_dict}}
    """
    results = defaultdict(dict)
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return results
    
    print(f"Scanning results directory: {results_dir}")
    
    # Walk through results/<backbone>/<model>/
    for backbone_dir in sorted(results_dir.iterdir()):
        if not backbone_dir.is_dir():
            continue
        
        backbone_name = backbone_dir.name
        print(f"\nBackbone: {backbone_name}")
        
        for model_dir in sorted(backbone_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            
            model_name = model_dir.name
            
            try:
                # Use multi-seed aware loader
                metrics = load_metrics_with_seeds(model_dir)
                if metrics is not None:
                    results[backbone_name][model_name] = extract_all_metrics(metrics)
                    print(f"  Loaded: {model_name}")
            except Exception as e:
                print(f"  Error loading {model_dir}: {e}")
    
    return results


def format_value(value: Union[float, Dict], is_diff: bool = False, show_std: bool = True) -> str:
    """
    Format a metric value for LaTeX.
    
    Handles both simple floats and aggregated dicts with mean/std.
    """
    mean = get_mean(value)
    std = get_std(value)
    
    if is_diff:
        if show_std and std > 0:
            sign = "+" if mean >= 0 else ""
            return f"{sign}{mean:.3f}$\\pm${std:.3f}"
        else:
            if mean >= 0:
                return f"+{mean:.3f}"
            else:
                return f"{mean:.3f}"
    else:
        if show_std and std > 0:
            return f"{mean:.3f}$\\pm${std:.3f}"
        return f"{mean:.3f}"


def format_diff_cell(val_a: Union[float, Dict], val_b: Union[float, Dict], show_std: bool = True) -> str:
    """
    Format a diff cell (B - A) as a signed value with color.
    
    Handles aggregated values with error propagation.
    """
    diff_result = compute_diff_with_error(val_a, val_b)
    diff_mean = diff_result["mean"]
    diff_std = diff_result["std"]
    
    return format_diff_cell_latex(diff_mean, diff_std, precision=3, show_std=show_std)


def generate_latex_table(results: dict, metric_name: str) -> str:
    """
    Generate a LaTeX table for a specific metric.
    
    Structure:
    - Rows: backbone (with sub-rows for Model A, Model B, Diff)
    - Columns: datasets + Mean
    """
    # Get column headers (datasets + Mean)
    columns = [ds[1] for ds in DATASETS] + ["Mean"]
    num_cols = len(columns)
    
    # Build LaTeX table
    lines = []
    
    # Table header
    lines.append(f"% Table for {metric_name}")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + metric_name + r" comparison across datasets}")
    lines.append(r"\label{tab:" + metric_name.lower().replace("@", "") + r"}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    
    # Column specification: first col for backbone/model, then one for each dataset + mean
    col_spec = "l" + "c" * num_cols
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\toprule")
    
    # Header row
    header = "Model & " + " & ".join(columns) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    
    # Data rows - iterate over backbones
    backbone_order = ["all-MiniLM-L6-v2", "all-MiniLM-L12-v2", "gte-small", "gte-modernbert-base"]
    
    for backbone in backbone_order:
        if backbone not in results:
            continue
        
        models = results[backbone]
        backbone_display = BACKBONE_NAMES.get(backbone, backbone)
        
        # Find model-a and model-b
        model_a_metrics = None
        model_b_metrics = None
        
        for model_name, metrics in models.items():
            if "model-a" in model_name.lower():
                model_a_metrics = metrics.get(metric_name, {})
            elif "model-b" in model_name.lower():
                model_b_metrics = metrics.get(metric_name, {})
        
        if not model_a_metrics or not model_b_metrics:
            continue
        
        # Model A row
        values_a = [format_value(model_a_metrics.get(col, 0)) for col in columns]
        line_a = f"{backbone_display} (A) & " + " & ".join(values_a) + r" \\"
        lines.append(line_a)
        
        # Model B row
        values_b = [format_value(model_b_metrics.get(col, 0)) for col in columns]
        line_b = f"{backbone_display} (B) & " + " & ".join(values_b) + r" \\"
        lines.append(line_b)
        
        # Diff row
        diffs = []
        for col in columns:
            val_a = model_a_metrics.get(col, 0)
            val_b = model_b_metrics.get(col, 0)
            diffs.append(format_diff_cell(val_a, val_b))
        line_diff = f"{backbone_display} ($\\Delta$) & " + " & ".join(diffs) + r" \\"
        lines.append(line_diff)
        
        lines.append(r"\midrule")
    
    # Remove last \midrule and replace with \bottomrule
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def generate_per_backbone_tables(results: dict, metric_name: str) -> str:
    """
    Generate separate compact LaTeX tables for each backbone.
    
    Structure per table:
    - Rows: Backbone, A, B, Δ(A-Base), Δ(B-Base), Δ(B-A) (subrows)
    - Columns: datasets + Mean
    """
    # Get column headers (datasets + Mean) - use short names
    columns = [ds[1].replace("Nano", "") for ds in DATASETS] + ["Mean"]
    num_cols = len(columns)
    
    all_tables = []
    backbone_order = ["all-MiniLM-L6-v2", "all-MiniLM-L12-v2", "gte-small", "gte-modernbert-base"]
    
    for backbone in backbone_order:
        if backbone not in results:
            continue
        
        models = results[backbone]
        backbone_display = BACKBONE_NAMES.get(backbone, backbone)
        
        # Find backbone, model-a and model-b
        backbone_metrics = None
        model_a_metrics = None
        model_b_metrics = None
        
        for model_name, metrics in models.items():
            if model_name.lower() == "backbone":
                backbone_metrics = metrics.get(metric_name, {})
            elif "model-a" in model_name.lower():
                model_a_metrics = metrics.get(metric_name, {})
            elif "model-b" in model_name.lower():
                model_b_metrics = metrics.get(metric_name, {})
        
        if not model_a_metrics or not model_b_metrics:
            continue
        
        lines = []
        
        # Table header
        lines.append(f"% {backbone_display} - {metric_name}")
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{" + backbone_display + r": " + metric_name + r" on NanoBEIR}")
        safe_label = (backbone_display + "-" + metric_name).lower().replace("@", "").replace(" ", "-")
        lines.append(r"\label{tab:" + safe_label + r"}")
        lines.append(r"\resizebox{\textwidth}{!}{%")
        
        # Column specification
        col_spec = "l" + "c" * num_cols
        lines.append(r"\begin{tabular}{" + col_spec + r"}")
        lines.append(r"\toprule")
        
        # Header row
        header = " & " + " & ".join(columns) + r" \\"
        lines.append(header)
        lines.append(r"\midrule")
        
        # Get full column names for lookup
        full_columns = [ds[1] for ds in DATASETS] + ["Mean"]
        
        # Backbone row (if available)
        if backbone_metrics:
            values_base = [format_value(backbone_metrics.get(col, 0)) for col in full_columns]
            line_base = r"Backbone & " + " & ".join(values_base) + r" \\"
            lines.append(line_base)
        
        # Model A row
        values_a = [format_value(model_a_metrics.get(col, 0)) for col in full_columns]
        line_a = r"Model A (MSMARCO) & " + " & ".join(values_a) + r" \\"
        lines.append(line_a)
        
        # Model B row
        values_b = [format_value(model_b_metrics.get(col, 0)) for col in full_columns]
        line_b = r"Model B (+ Negatives) & " + " & ".join(values_b) + r" \\"
        lines.append(line_b)
        
        lines.append(r"\midrule")
        
        # Diff B-A row
        diffs_ba = []
        for col in full_columns:
            val_a = model_a_metrics.get(col, 0)
            val_b = model_b_metrics.get(col, 0)
            diffs_ba.append(format_diff_cell(val_a, val_b))
        line_diff_ba = r"$\Delta$ (B$-$A) & " + " & ".join(diffs_ba) + r" \\"
        lines.append(line_diff_ba)
        
        # Diff A-Backbone and B-Backbone (if backbone available)
        if backbone_metrics:
            diffs_a_base = []
            diffs_b_base = []
            for col in full_columns:
                val_base = backbone_metrics.get(col, 0)
                val_a = model_a_metrics.get(col, 0)
                val_b = model_b_metrics.get(col, 0)
                diffs_a_base.append(format_diff_cell(val_base, val_a))
                diffs_b_base.append(format_diff_cell(val_base, val_b))
            line_diff_a_base = r"$\Delta$ (A$-$Base) & " + " & ".join(diffs_a_base) + r" \\"
            line_diff_b_base = r"$\Delta$ (B$-$Base) & " + " & ".join(diffs_b_base) + r" \\"
            lines.append(line_diff_a_base)
            lines.append(line_diff_b_base)
        
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"}")
        lines.append(r"\end{table}")
        
        all_tables.append("\n".join(lines))
    
    return "\n\n".join(all_tables)


def generate_combined_backbone_table(results: dict, metric_name: str) -> str:
    """
    Generate a single combined table with all backbones.
    Each backbone has Base/A/B/Δ subrows, separated by midrules.
    
    Structure:
    - Rows: grouped by backbone (Base, A, B, Δ subrows)
    - Columns: datasets + Mean
    """
    # Use short names for columns
    columns = [ds[1].replace("Nano", "") for ds in DATASETS] + ["Mean"]
    full_columns = [ds[1] for ds in DATASETS] + ["Mean"]
    num_cols = len(columns)
    
    lines = []
    
    # Table header
    lines.append(f"% Combined table for {metric_name}")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + metric_name + r" comparison across backbones on NanoBEIR}")
    lines.append(r"\label{tab:" + metric_name.lower().replace("@", "") + r"-combined}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    
    # Column specification
    col_spec = "ll" + "c" * num_cols
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\toprule")
    
    # Header row
    header = "Backbone & Model & " + " & ".join(columns) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    
    backbone_order = ["all-MiniLM-L6-v2", "all-MiniLM-L12-v2", "gte-small", "gte-modernbert-base"]
    
    for i, backbone in enumerate(backbone_order):
        if backbone not in results:
            continue
        
        models = results[backbone]
        backbone_display = BACKBONE_NAMES.get(backbone, backbone)
        
        # Find backbone, model-a and model-b
        backbone_metrics = None
        model_a_metrics = None
        model_b_metrics = None
        
        for model_name, metrics in models.items():
            if model_name.lower() == "backbone":
                backbone_metrics = metrics.get(metric_name, {})
            elif "model-a" in model_name.lower():
                model_a_metrics = metrics.get(metric_name, {})
            elif "model-b" in model_name.lower():
                model_b_metrics = metrics.get(metric_name, {})
        
        if not model_a_metrics or not model_b_metrics:
            continue
        
        # Determine number of rows for this backbone
        num_rows = 4 if backbone_metrics else 3  # Base + A + B + Δ or just A + B + Δ
        
        # Backbone row (if available)
        if backbone_metrics:
            values_base = [format_value(backbone_metrics.get(col, 0)) for col in full_columns]
            line_base = r"\multirow{" + str(num_rows) + r"}{*}{" + backbone_display + r"} & Base & " + " & ".join(values_base) + r" \\"
            lines.append(line_base)
            
            # Model A row
            values_a = [format_value(model_a_metrics.get(col, 0)) for col in full_columns]
            line_a = r" & A & " + " & ".join(values_a) + r" \\"
            lines.append(line_a)
        else:
            # Model A row (with backbone name using multirow)
            values_a = [format_value(model_a_metrics.get(col, 0)) for col in full_columns]
            line_a = r"\multirow{" + str(num_rows) + r"}{*}{" + backbone_display + r"} & A & " + " & ".join(values_a) + r" \\"
            lines.append(line_a)
        
        # Model B row
        values_b = [format_value(model_b_metrics.get(col, 0)) for col in full_columns]
        line_b = r" & B & " + " & ".join(values_b) + r" \\"
        lines.append(line_b)
        
        # Diff row (B - A)
        diffs = []
        for col in full_columns:
            val_a = model_a_metrics.get(col, 0)
            val_b = model_b_metrics.get(col, 0)
            diffs.append(format_diff_cell(val_a, val_b))
        line_diff = r" & $\Delta$ & " + " & ".join(diffs) + r" \\"
        lines.append(line_diff)
        
        # Add midrule between backbones (not after last one)
        if i < len(backbone_order) - 1:
            lines.append(r"\midrule")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def generate_compact_latex_table(results: dict, metric_name: str) -> str:
    """
    Generate a more compact LaTeX table with backbone as main row,
    and A/B/Diff as sub-columns for each dataset.
    """
    # Get column headers (datasets + Mean)
    columns = [ds[1] for ds in DATASETS] + ["Mean"]
    
    lines = []
    
    # Table header
    lines.append(f"% Compact table for {metric_name}")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + metric_name + r" comparison across datasets}")
    lines.append(r"\label{tab:" + metric_name.lower().replace("@", "") + r"}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    
    # Column specification
    # Backbone | for each dataset: A | B | Δ
    num_data_cols = len(columns) * 3
    col_spec = "l" + "c" * num_data_cols
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\toprule")
    
    # Two-level header
    # First level: dataset names spanning 3 columns each
    header1_parts = [""]
    for col in columns:
        header1_parts.append(r"\multicolumn{3}{c}{" + col + r"}")
    header1 = " & ".join(header1_parts) + r" \\"
    lines.append(header1)
    
    # Cmidrules for each dataset group
    cmidrules = []
    for i, col in enumerate(columns):
        start = 2 + i * 3
        end = start + 2
        cmidrules.append(f"\\cmidrule(lr){{{start}-{end}}}")
    lines.append(" ".join(cmidrules))
    
    # Second level: A | B | Δ for each dataset
    header2_parts = ["Backbone"]
    for _ in columns:
        header2_parts.extend(["A", "B", r"$\Delta$"])
    header2 = " & ".join(header2_parts) + r" \\"
    lines.append(header2)
    lines.append(r"\midrule")
    
    # Data rows
    backbone_order = ["all-MiniLM-L6-v2", "all-MiniLM-L12-v2", "gte-small", "gte-modernbert-base"]
    
    for backbone in backbone_order:
        if backbone not in results:
            continue
        
        models = results[backbone]
        backbone_display = BACKBONE_NAMES.get(backbone, backbone)
        
        # Find model-a and model-b
        model_a_metrics = None
        model_b_metrics = None
        
        for model_name, metrics in models.items():
            if "model-a" in model_name.lower():
                model_a_metrics = metrics.get(metric_name, {})
            elif "model-b" in model_name.lower():
                model_b_metrics = metrics.get(metric_name, {})
        
        if not model_a_metrics or not model_b_metrics:
            continue
        
        row_parts = [backbone_display]
        for col in columns:
            val_a = model_a_metrics.get(col, 0)
            val_b = model_b_metrics.get(col, 0)
            
            row_parts.append(format_value(val_a))
            row_parts.append(format_value(val_b))
            row_parts.append(format_diff_cell(val_a, val_b))
        
        lines.append(" & ".join(row_parts) + r" \\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def generate_summary_table(results: dict) -> str:
    """
    Generate a summary LaTeX table showing mean NanoBEIR metrics.
    
    Format: Base → A → B with colored diffs for each transition.
    Shows all four metrics (nDCG@10, MRR@10, Acc@1, Acc@10) in columns.
    Supports multi-seed results with error bars.
    """
    lines = []
    
    # Table header
    lines.append(r"% Summary table for mean NanoBEIR metrics")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\Large")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Backbone} & \textbf{nDCG@10} & \textbf{MRR@10} & \textbf{Acc@1} & \textbf{Acc@10} \\")
    lines.append(r"\midrule")
    
    backbone_order = ["all-MiniLM-L6-v2", "all-MiniLM-L12-v2", "gte-small", "gte-modernbert-base"]
    metric_names = ["NDCG@10", "MRR@10", "Acc@1", "Acc@10"]
    
    # Color formatting for diffs with optional std
    def format_colored_diff_with_std(val_a, val_b) -> str:
        diff_result = compute_diff_with_error(val_a, val_b)
        diff_mean = diff_result["mean"]
        diff_std = diff_result["std"]
        
        sign = "+" if diff_mean >= 0 else ""
        if diff_std > 0:
            formatted = f"{sign}{diff_mean:.3f}$\\pm${diff_std:.3f}"
        else:
            formatted = f"{sign}{diff_mean:.3f}"
        
        if diff_mean >= 0:
            return r"\textcolor{ForestGreen}{" + formatted + r"}"
        else:
            return r"\textcolor{BrickRed}{" + formatted + r"}"
    
    for backbone in backbone_order:
        if backbone not in results:
            continue
        
        models = results[backbone]
        backbone_display = BACKBONE_NAMES.get(backbone, backbone)
        
        # Find backbone, model-a and model-b metrics
        backbone_metrics = None
        model_a_metrics = None
        model_b_metrics = None
        
        for model_name, metrics in models.items():
            if model_name.lower() == "backbone":
                backbone_metrics = metrics
            elif "model-a" in model_name.lower():
                model_a_metrics = metrics
            elif "model-b" in model_name.lower():
                model_b_metrics = metrics
        
        if not model_a_metrics or not model_b_metrics:
            continue
        
        # Build row with all four metrics
        cells = [backbone_display]
        
        for metric_name in metric_names:
            base_val = backbone_metrics.get(metric_name, {}).get("Mean", 0) if backbone_metrics else None
            a_val = model_a_metrics.get(metric_name, {}).get("Mean", 0)
            b_val = model_b_metrics.get(metric_name, {}).get("Mean", 0)
            
            # Extract mean values for display (handles both simple and aggregated)
            base_mean = get_mean(base_val) if base_val is not None else None
            a_mean = get_mean(a_val)
            b_mean = get_mean(b_val)
            a_std = get_std(a_val)
            b_std = get_std(b_val)
            
            # Format values with std if available
            a_str = f"{a_mean:.3f}" + (f"$\\pm${a_std:.3f}" if a_std > 0 else "")
            b_str = f"{b_mean:.3f}" + (f"$\\pm${b_std:.3f}" if b_std > 0 else "")
            
            if base_mean is not None:
                base_std = get_std(base_val) if base_val is not None else 0
                base_str = f"{base_mean:.3f}" + (f"$\\pm${base_std:.3f}" if base_std > 0 else "")
                
                # Format: Base → A → B with colored diffs on single line
                cell = (
                    f"${base_str} \\rightarrow {a_str}\\;({format_colored_diff_with_std(base_val, a_val)})$ "
                    f"$\\rightarrow {b_str}\\;({format_colored_diff_with_std(a_val, b_val)})$"
                )
            else:
                # No backbone, just show A → B
                cell = f"${a_str} \\rightarrow {b_str}\\;({format_colored_diff_with_std(a_val, b_val)})$"
            
            cells.append(cell)
        
        lines.append(" & ".join(cells) + r" \\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\caption{Mean NanoBEIR retrieval metrics: Backbone $\rightarrow$ Model A (MSMARCO fine-tuning) $\rightarrow$ Model B (+ MRPC structured negatives). Colored values show the change from the previous stage. Values shown as mean$\pm$std across seeds where available.}")
    lines.append(r"\label{tab:nanobeir-mean-summary}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def generate_short_summary_table(results: dict) -> str:
    """
    Generate a SHORT summary LaTeX table showing only nDCG@10 and Acc@1.
    
    Shows Model A, Model B, and Delta for each backbone in a compact format.
    Supports multi-seed results with error bars.
    """
    lines = []
    
    # Table header
    lines.append(r"% Short summary table (nDCG@10 and Acc@1 only)")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l ccc ccc}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{3}{c}{\textbf{nDCG@10}} & \multicolumn{3}{c}{\textbf{Acc@1}} \\")
    lines.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    lines.append(r"\textbf{Backbone} & Model A & Model B & $\Delta$ & Model A & Model B & $\Delta$ \\")
    lines.append(r"\midrule")
    
    backbone_order = ["all-MiniLM-L6-v2", "all-MiniLM-L12-v2", "gte-small", "gte-modernbert-base"]
    metric_names = ["NDCG@10", "Acc@1"]
    
    for backbone in backbone_order:
        if backbone not in results:
            continue
        
        models = results[backbone]
        backbone_display = BACKBONE_NAMES.get(backbone, backbone)
        
        # Find model-a and model-b metrics
        model_a_metrics = None
        model_b_metrics = None
        
        for model_name, metrics in models.items():
            if "model-a" in model_name.lower():
                model_a_metrics = metrics
            elif "model-b" in model_name.lower():
                model_b_metrics = metrics
        
        if not model_a_metrics or not model_b_metrics:
            continue
        
        # Build row
        cells = [backbone_display]
        
        for metric_name in metric_names:
            a_val = model_a_metrics.get(metric_name, {}).get("Mean", 0)
            b_val = model_b_metrics.get(metric_name, {}).get("Mean", 0)
            
            # Extract mean and std
            a_mean = get_mean(a_val)
            a_std = get_std(a_val)
            b_mean = get_mean(b_val)
            b_std = get_std(b_val)
            
            # Compute diff with error propagation
            diff_result = compute_diff_with_error(a_val, b_val)
            diff_mean = diff_result["mean"]
            diff_std = diff_result["std"]
            
            # Format values
            if a_std > 0:
                a_str = f"{a_mean:.3f}$\\pm${a_std:.3f}"
            else:
                a_str = f"{a_mean:.3f}"
            
            if b_std > 0:
                b_str = f"{b_mean:.3f}$\\pm${b_std:.3f}"
            else:
                b_str = f"{b_mean:.3f}"
            
            # Format diff with color
            sign = "+" if diff_mean >= 0 else ""
            if diff_std > 0:
                diff_formatted = f"{sign}{diff_mean:.3f}$\\pm${diff_std:.3f}"
            else:
                diff_formatted = f"{sign}{diff_mean:.3f}"
            
            if diff_mean > 0:
                diff_str = r"\textcolor{ForestGreen}{" + diff_formatted + r"}"
            elif diff_mean < 0:
                diff_str = r"\textcolor{BrickRed}{" + diff_formatted + r"}"
            else:
                diff_str = diff_formatted
            
            cells.extend([a_str, b_str, diff_str])
        
        lines.append(" & ".join(cells) + r" \\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Mean NanoBEIR retrieval performance (nDCG@10 and Acc@1). Model A: standard fine-tuning. Model B: + structured negatives. Values shown as mean$\pm$std across seeds.}")
    lines.append(r"\label{tab:nanobeir-short-summary}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def display_results(results: dict, table_format: str = "combined"):
    """
    Display results as LaTeX tables.
    
    Args:
        results: The results dictionary
        table_format: One of "combined", "per-backbone", "subcolumns"
            - combined: Single table with A/B/Δ subrows, backbones grouped
            - per-backbone: Separate table per backbone with A/B/Δ subrows
            - subcolumns: A/B/Δ as subcolumns for each dataset (wide)
    """
    if not results:
        print("No results found.")
        return
    
    # Preamble for LaTeX
    print(r"""% LaTeX Preamble (add to your document):
% \usepackage{booktabs}
% \usepackage{graphicx}
% \usepackage{multirow}  % needed for combined format
% \usepackage[dvipsnames]{xcolor}  % for colored deltas
""")
    
    # Generate a table for each metric
    for metric_key, metric_name in METRICS:
        print(f"\n% {'='*60}")
        print(f"% {metric_name} Table ({table_format})")
        print(f"% {'='*60}\n")
        
        if table_format == "per-backbone":
            latex_table = generate_per_backbone_tables(results, metric_name)
        elif table_format == "combined":
            latex_table = generate_combined_backbone_table(results, metric_name)
        else:  # subcolumns (original wide format)
            latex_table = generate_compact_latex_table(results, metric_name)
        
        print(latex_table)
        print()
    
    # Generate summary table at the end
    print(f"\n% {'='*60}")
    print(f"% Summary Table (Mean NanoBEIR metrics)")
    print(f"% {'='*60}\n")
    summary_table = generate_summary_table(results)
    print(summary_table)
    print()
    
    # Generate short summary table (nDCG@10 and MRR@10 only)
    print(f"\n% {'='*60}")
    print(f"% Short Summary Table (nDCG@10 and MRR@10 only)")
    print(f"% {'='*60}\n")
    short_summary_table = generate_short_summary_table(results)
    print(short_summary_table)
    print()


def display_results_compact(results: dict):
    """Display results as compact LaTeX tables (legacy, uses subcolumns)."""
    display_results(results, table_format="subcolumns")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from biencoder experiment results")
    parser.add_argument("--results-dir", "-r", type=str, default=None,
                        help="Path to results directory (default: ./results relative to script)")
    parser.add_argument("--format", "-f", type=str, default="combined",
                        choices=["combined", "per-backbone", "subcolumns"],
                        help="Table format: 'combined' (A/B/Δ subrows, all backbones in one table), "
                             "'per-backbone' (separate table per backbone), "
                             "'subcolumns' (A/B/Δ as subcolumns per dataset, wide)")
    parser.add_argument("--row-format", action="store_true",
                        help="[Deprecated] Use 'subcolumns' format")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file path (prints to stdout if not specified)")
    args = parser.parse_args()
    
    # Handle deprecated --row-format
    table_format = args.format
    if args.row_format:
        table_format = "subcolumns"
    
    # Determine results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        # Default: relative to this script
        script_dir = Path(__file__).parent
        results_dir = script_dir / "results"
    
    print(f"% Results from: {results_dir}")
    print(f"% Generated LaTeX tables for NanoBEIR benchmark comparison")
    print(f"% Format: {table_format}")
    print()
    
    results = walk_results(results_dir)
    display_results(results, table_format=table_format)


if __name__ == "__main__":
    main()
