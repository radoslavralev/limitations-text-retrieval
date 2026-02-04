"""
Aggregation Utilities for Multi-Seed Experiment Results.

Provides functions to:
- Detect seed subdirectories in result paths
- Aggregate numeric metrics across seeds (compute mean +/- std)
- Format values with error bars for display and LaTeX output

Usage:
    from aggregate_utils import find_seed_dirs, aggregate_nested_dict, format_with_std
    
    seed_dirs = find_seed_dirs(Path("results/model-a-baseline"))
    if seed_dirs:
        all_data = [load_json(d / "final_metrics.json") for d in seed_dirs]
        aggregated = aggregate_nested_dict(all_data)
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from collections import defaultdict

import numpy as np


def find_seed_dirs(parent_path: Path) -> List[Path]:
    """
    Find seed_* subdirectories in the given path.
    
    Args:
        parent_path: Directory to search for seed subdirectories
        
    Returns:
        Sorted list of paths to seed directories, or empty list if none found
    """
    if not parent_path.exists() or not parent_path.is_dir():
        return []
    
    seed_dirs = []
    seed_pattern = re.compile(r'^seed_(\d+)$')
    
    for child in parent_path.iterdir():
        if child.is_dir() and seed_pattern.match(child.name):
            seed_dirs.append(child)
    
    # Sort by seed number
    seed_dirs.sort(key=lambda p: int(seed_pattern.match(p.name).group(1)))
    return seed_dirs


def get_seed_number(seed_dir: Path) -> int:
    """Extract the seed number from a seed directory name."""
    match = re.match(r'^seed_(\d+)$', seed_dir.name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Invalid seed directory name: {seed_dir.name}")


def aggregate_metrics(values: List[float]) -> Dict[str, Any]:
    """
    Compute aggregate statistics for a list of numeric values.
    
    Args:
        values: List of numeric values from different seeds
        
    Returns:
        Dict with keys: mean, std, min, max, count, values
    """
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0, "values": []}
    
    values_array = np.array(values)
    return {
        "mean": float(np.mean(values_array)),
        "std": float(np.std(values_array)),
        "min": float(np.min(values_array)),
        "max": float(np.max(values_array)),
        "count": len(values),
        "values": values,
    }


def aggregate_nested_dict(dicts: List[Dict]) -> Dict:
    """
    Recursively aggregate numeric values from multiple dictionaries.
    
    For numeric values at leaf nodes, computes mean +/- std across all dicts.
    For nested dicts, recursively aggregates.
    For other types (strings, etc.), keeps the first value.
    
    Args:
        dicts: List of dictionaries with the same structure
        
    Returns:
        Aggregated dictionary with {"mean": X, "std": Y, ...} at leaf nodes
    """
    if not dicts:
        return {}
    
    # If all items are numbers, aggregate them
    if all(isinstance(d, (int, float)) for d in dicts):
        return aggregate_metrics([float(d) for d in dicts])
    
    # If all items are dicts, recursively aggregate
    if all(isinstance(d, dict) for d in dicts):
        result = {}
        # Get union of all keys
        all_keys = set()
        for d in dicts:
            all_keys.update(d.keys())
        
        for key in all_keys:
            # Collect values for this key from all dicts that have it
            child_values = [d[key] for d in dicts if key in d]
            if child_values:
                result[key] = aggregate_nested_dict(child_values)
        
        return result
    
    # If all items are lists of the same length, aggregate element-wise
    if all(isinstance(d, list) for d in dicts):
        if all(len(d) == len(dicts[0]) for d in dicts):
            # Aggregate each index
            result = []
            for i in range(len(dicts[0])):
                values = [d[i] for d in dicts]
                result.append(aggregate_nested_dict(values))
            return result
    
    # For mixed types or non-aggregatable types, return the first value
    return dicts[0]


def is_aggregated_value(value: Any) -> bool:
    """Check if a value is an aggregated dict with mean/std."""
    return (
        isinstance(value, dict) 
        and "mean" in value 
        and "std" in value
    )


def get_mean(value: Union[float, Dict]) -> float:
    """Extract mean from an aggregated value or return the value itself."""
    if is_aggregated_value(value):
        return value["mean"]
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def get_std(value: Union[float, Dict]) -> float:
    """Extract std from an aggregated value or return 0."""
    if is_aggregated_value(value):
        return value["std"]
    return 0.0


def format_with_std(
    mean: float, 
    std: float, 
    precision: int = 3,
    show_std: bool = True,
) -> str:
    """
    Format a value with optional standard deviation for display.
    
    Args:
        mean: Mean value
        std: Standard deviation
        precision: Number of decimal places
        show_std: Whether to show the +/- std
        
    Returns:
        Formatted string like "0.549 +/- 0.012" or just "0.549"
    """
    if show_std and std > 0:
        return f"{mean:.{precision}f} +/- {std:.{precision}f}"
    return f"{mean:.{precision}f}"


def format_with_std_latex(
    mean: float, 
    std: float, 
    precision: int = 3,
    show_std: bool = True,
) -> str:
    """
    Format a value with standard deviation for LaTeX.
    
    Args:
        mean: Mean value
        std: Standard deviation
        precision: Number of decimal places
        show_std: Whether to show the +/- std
        
    Returns:
        Formatted string like "0.549$\\pm$0.012" or just "0.549"
    """
    if show_std and std > 0:
        # Use smaller precision for std if it's very small
        std_precision = precision
        return f"{mean:.{precision}f}$\\pm${std:.{std_precision}f}"
    return f"{mean:.{precision}f}"


def format_value_auto(
    value: Union[float, Dict],
    precision: int = 3,
    latex: bool = False,
    show_std: bool = True,
) -> str:
    """
    Auto-format a value that may be aggregated or a simple number.
    
    Args:
        value: Either a number or an aggregated dict with mean/std
        precision: Number of decimal places
        latex: Whether to format for LaTeX
        show_std: Whether to show the +/- std
        
    Returns:
        Formatted string
    """
    mean = get_mean(value)
    std = get_std(value)
    
    if latex:
        return format_with_std_latex(mean, std, precision, show_std)
    return format_with_std(mean, std, precision, show_std)


def format_diff_with_std(
    diff_mean: float,
    diff_std: float,
    precision: int = 3,
    latex: bool = False,
) -> str:
    """
    Format a difference value with sign and optional std.
    
    Args:
        diff_mean: Mean difference
        diff_std: Standard deviation of difference
        precision: Number of decimal places
        latex: Whether to format for LaTeX
        
    Returns:
        Formatted string like "+0.012 +/- 0.003" or "-0.015$\\pm$0.002"
    """
    sign = "+" if diff_mean >= 0 else ""
    
    if latex:
        if diff_std > 0:
            return f"{sign}{diff_mean:.{precision}f}$\\pm${diff_std:.{precision}f}"
        return f"{sign}{diff_mean:.{precision}f}"
    else:
        if diff_std > 0:
            return f"{sign}{diff_mean:.{precision}f} +/- {diff_std:.{precision}f}"
        return f"{sign}{diff_mean:.{precision}f}"


def format_diff_cell_latex(
    diff_mean: float,
    diff_std: float = 0.0,
    precision: int = 3,
    show_std: bool = True,
) -> str:
    """
    Format a diff cell (B - A) with color for LaTeX.
    
    Args:
        diff_mean: Mean difference
        diff_std: Standard deviation of difference
        precision: Number of decimal places
        show_std: Whether to show std
        
    Returns:
        LaTeX formatted string with color based on sign
    """
    sign = "+" if diff_mean >= 0 else ""
    
    if show_std and diff_std > 0:
        formatted = f"{sign}{diff_mean:.{precision}f}$\\pm${diff_std:.{precision}f}"
    else:
        formatted = f"{sign}{diff_mean:.{precision}f}"
    
    if diff_mean > 0:
        return r"\textcolor{ForestGreen}{" + formatted + r"}"
    elif diff_mean < 0:
        return r"\textcolor{BrickRed}{" + formatted + r"}"
    else:
        return formatted


def load_json_safe(path: Path) -> Optional[Dict]:
    """Safely load a JSON file, returning None if it doesn't exist or fails."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load {path}: {e}")
        return None


def load_and_aggregate_seeds(
    parent_path: Path,
    json_filename: str = "all_results.json",
) -> Optional[Dict]:
    """
    Load JSON files from all seed subdirectories and aggregate them.
    
    If no seed directories are found, attempts to load the JSON file directly
    from parent_path (backward compatibility with single-seed results).
    
    Args:
        parent_path: Directory containing seed_* subdirs or the JSON file directly
        json_filename: Name of the JSON file to load from each seed dir
        
    Returns:
        Aggregated dictionary with mean/std at leaf nodes, or None if no data found
    """
    seed_dirs = find_seed_dirs(parent_path)
    
    if seed_dirs:
        print(f"Found {len(seed_dirs)} seeds: {[d.name for d in seed_dirs]}")
        all_data = []
        for seed_dir in seed_dirs:
            json_path = seed_dir / json_filename
            data = load_json_safe(json_path)
            if data is not None:
                all_data.append(data)
        
        if all_data:
            return aggregate_nested_dict(all_data)
        return None
    else:
        # Fallback: try loading directly from parent_path
        json_path = parent_path / json_filename
        if json_path.exists():
            return load_json_safe(json_path)
        
        # Also try if parent_path is the JSON file itself
        if parent_path.is_file() and parent_path.suffix == ".json":
            return load_json_safe(parent_path)
        
        return None


# Convenience function for computing differences with error propagation
def compute_diff_with_error(
    value_a: Union[float, Dict],
    value_b: Union[float, Dict],
) -> Dict[str, float]:
    """
    Compute the difference B - A with error propagation.
    
    For independent measurements: std_diff = sqrt(std_a^2 + std_b^2)
    
    Args:
        value_a: First value (may be aggregated)
        value_b: Second value (may be aggregated)
        
    Returns:
        Dict with mean and std of the difference
    """
    mean_a = get_mean(value_a)
    mean_b = get_mean(value_b)
    std_a = get_std(value_a)
    std_b = get_std(value_b)
    
    diff_mean = mean_b - mean_a
    # Error propagation for difference of independent variables
    diff_std = np.sqrt(std_a**2 + std_b**2)
    
    return {"mean": diff_mean, "std": diff_std}
