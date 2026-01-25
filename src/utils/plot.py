"""Plotting utilities."""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def plot_metric_curves(
    data: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: Optional[str] = None,
    ylabel: Optional[str] = None
):
    """Plot metric curves over severity for each corruption and model.
    
    Args:
        data: DataFrame with columns: model, corruption, severity, and metric
        metric: Column name for the metric to plot
        output_path: Path to save the plot
        title: Plot title (default: metric name)
        ylabel: Y-axis label (default: metric name)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    corruptions = data['corruption'].unique()
    
    for idx, corruption in enumerate(corruptions):
        ax = axes[idx]
        corruption_data = data[data['corruption'] == corruption]
        
        for model in corruption_data['model'].unique():
            model_data = corruption_data[corruption_data['model'] == model]
            model_data = model_data.sort_values('severity')
            ax.plot(
                model_data['severity'],
                model_data[metric],
                marker='o',
                label=model,
                linewidth=2,
                markersize=6
            )
        
        ax.set_xlabel('Severity', fontsize=12)
        ax.set_ylabel(ylabel or metric, fontsize=12)
        ax.set_title(f'{corruption.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks([0, 1, 2, 3, 4])
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_tiny_object_curves(
    data: pd.DataFrame,
    metrics: List[str],
    output_path: Path,
    corruption: str
):
    """Plot tiny object analysis curves.
    
    Args:
        data: DataFrame with severity and metric columns
        metrics: List of metric column names to plot
        output_path: Path to save the plot
        corruption: Corruption type name
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for model in data['model'].unique():
            model_data = data[data['model'] == model].sort_values('severity')
            ax.plot(
                model_data['severity'],
                model_data[metric],
                marker='o',
                label=model,
                linewidth=2,
                markersize=6
            )
        
        ax.set_xlabel('Severity', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{corruption.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks([0, 1, 2, 3, 4])
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
