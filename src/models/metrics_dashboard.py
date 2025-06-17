#!/usr/bin/env python3
"""
Comprehensive Metrics Dashboard for ES Futures Trading Strategy
Creates detailed performance analysis and visualization dashboards.

Week 2 deliverable: Comprehensive metrics tracking for walk-forward validation results.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Visualization Libraries
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsDashboard:
    """Comprehensive metrics dashboard for trading strategy analysis."""
    
    def __init__(self, results_dir: str = "reports/metrics_dashboard"):
        """Initialize metrics dashboard."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        logger.info(f"Initialized Metrics Dashboard")
        logger.info(f"Output directory: {self.results_dir}")
    
    def load_walk_forward_results(self, results_file: str) -> Dict[str, Any]:
        """Load walk-forward validation results."""
        logger.info(f"Loading walk-forward results from {results_file}")
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Convert individual splits to DataFrame
        splits_df = pd.DataFrame(results['individual_splits'])
        splits_df['test_start'] = pd.to_datetime(splits_df['test_start'])
        splits_df['test_end'] = pd.to_datetime(splits_df['test_end'])
        splits_df['train_start'] = pd.to_datetime(splits_df['train_start'])
        splits_df['train_end'] = pd.to_datetime(splits_df['train_end'])
        
        results['splits_df'] = splits_df
        
        logger.info(f"Loaded {len(splits_df)} splits covering {splits_df['test_start'].min()} to {splits_df['test_end'].max()}")
        
        return results
    
    def create_comprehensive_dashboard(self, results_file: str, save_individual: bool = True) -> str:
        """Create comprehensive performance dashboard."""
        logger.info("Creating comprehensive dashboard...")
        
        # Load results
        results = self.load_walk_forward_results(results_file)
        splits_df = results['splits_df']
        agg_metrics = results['aggregated_metrics']
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Main title
        fig.suptitle(
            f'Comprehensive Trading Strategy Performance Dashboard\n'
            f'Model: {results["model_type"].title()} | Validation: {results["method"].replace("_", " ").title()} | '
            f'{len(splits_df)} Splits | Period: {splits_df["test_start"].min().strftime("%Y-%m")} to {splits_df["test_end"].max().strftime("%Y-%m")}',
            fontsize=16, fontweight='bold'
        )
        
        # 1. Performance Over Time (top row, spanning 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_performance_over_time(ax1, splits_df)
        
        # 2. Trading Metrics Over Time (top row, spanning 2 columns)
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_trading_metrics_over_time(ax2, splits_df)
        
        # 3. Per-Class Performance (second row, left)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_per_class_performance(ax3, splits_df)
        
        # 4. Cumulative Returns (second row, center-left)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_cumulative_returns(ax4, splits_df)
        
        # 5. Risk-Return Scatter (second row, center-right)
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_risk_return_scatter(ax5, splits_df)
        
        # 6. Performance Distribution (second row, right)
        ax6 = fig.add_subplot(gs[1, 3])
        self._plot_performance_distribution(ax6, splits_df)
        
        # 7. Detailed Metrics Table (third row, full width)
        ax7 = fig.add_subplot(gs[2, :])
        self._plot_metrics_table(ax7, agg_metrics, results)
        
        # 8. Latency and Performance Analysis (fourth row, left half)
        ax8 = fig.add_subplot(gs[3, :2])
        self._plot_latency_analysis(ax8, splits_df)
        
        # 9. Training Performance (fourth row, right half)
        ax9 = fig.add_subplot(gs[3, 2:])
        self._plot_training_performance(ax9, splits_df)
        
        # Save comprehensive dashboard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dashboard_file = self.results_dir / f"comprehensive_dashboard_{timestamp}.png"
        plt.savefig(dashboard_file, dpi=300, bbox_inches='tight', facecolor='white')
        
        logger.info(f"Comprehensive dashboard saved to: {dashboard_file}")
        
        # Save individual plots if requested
        if save_individual:
            self._save_individual_plots(splits_df, agg_metrics, results, timestamp)
        
        # Create summary report
        summary_file = self._create_summary_report(results, timestamp)
        
        plt.show()
        
        return str(dashboard_file)
    
    def _plot_performance_over_time(self, ax, splits_df):
        """Plot key performance metrics over time."""
        ax2 = ax.twinx()
        
        # F1 Score (left axis)
        line1 = ax.plot(splits_df['test_start'], splits_df['f1_macro'], 'o-', 
                       color='blue', linewidth=2, markersize=6, label='F1 Macro', alpha=0.8)
        
        # Accuracy (left axis)
        line2 = ax.plot(splits_df['test_start'], splits_df['accuracy'], 's-', 
                       color='green', linewidth=2, markersize=6, label='Accuracy', alpha=0.8)
        
        # Sharpe Ratio (right axis)
        line3 = ax2.plot(splits_df['test_start'], splits_df['sharpe_ratio'], '^-', 
                        color='red', linewidth=2, markersize=6, label='Sharpe Ratio', alpha=0.8)
        
        ax.set_xlabel('Test Period Start')
        ax.set_ylabel('Score', color='blue')
        ax2.set_ylabel('Sharpe Ratio', color='red')
        ax.set_title('Performance Metrics Over Time', fontweight='bold')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', rotation=45)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Combined legend
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        
        # Add performance target lines
        ax.axhline(y=0.44, color='blue', linestyle='--', alpha=0.5, label='F1 Target')
        ax2.axhline(y=1.2, color='red', linestyle='--', alpha=0.5, label='Sharpe Target')
    
    def _plot_trading_metrics_over_time(self, ax, splits_df):
        """Plot trading-specific metrics over time."""
        ax2 = ax.twinx()
        
        # Win Rate (left axis)
        line1 = ax.plot(splits_df['test_start'], splits_df['win_rate'], 'o-', 
                       color='green', linewidth=2, markersize=6, label='Win Rate', alpha=0.8)
        
        # Max Drawdown (right axis, inverted)
        line2 = ax2.plot(splits_df['test_start'], -splits_df['max_drawdown'], 's-', 
                        color='red', linewidth=2, markersize=6, label='Max Drawdown', alpha=0.8)
        
        ax.set_xlabel('Test Period Start')
        ax.set_ylabel('Win Rate', color='green')
        ax2.set_ylabel('Max Drawdown (positive = bad)', color='red')
        ax.set_title('Trading Performance Metrics', fontweight='bold')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', rotation=45)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
    
    def _plot_per_class_performance(self, ax, splits_df):
        """Plot per-class F1 scores."""
        x = splits_df['test_start']
        
        ax.plot(x, splits_df['f1_flat'], 'o-', label='FLAT', color='gray', alpha=0.8)
        ax.plot(x, splits_df['f1_long'], 's-', label='LONG', color='green', alpha=0.8)
        ax.plot(x, splits_df['f1_short'], '^-', label='SHORT', color='red', alpha=0.8)
        
        ax.set_xlabel('Test Period')
        ax.set_ylabel('F1 Score')
        ax.set_title('Per-Class F1 Scores', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_cumulative_returns(self, ax, splits_df):
        """Plot cumulative returns."""
        splits_df['cumulative_return'] = splits_df['total_return'].cumsum()
        
        ax.plot(splits_df['test_start'], splits_df['cumulative_return'], 
               'o-', color='purple', linewidth=3, markersize=6, alpha=0.8)
        
        ax.set_xlabel('Test Period')
        ax.set_ylabel('Cumulative Return')
        ax.set_title('Cumulative Returns', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
        ax.tick_params(axis='x', rotation=45)
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Color regions
        positive_mask = splits_df['cumulative_return'] >= 0
        if positive_mask.any():
            ax.fill_between(splits_df['test_start'], splits_df['cumulative_return'], 0, 
                          where=positive_mask, alpha=0.3, color='green', interpolate=True)
        
        negative_mask = splits_df['cumulative_return'] < 0
        if negative_mask.any():
            ax.fill_between(splits_df['test_start'], splits_df['cumulative_return'], 0, 
                          where=negative_mask, alpha=0.3, color='red', interpolate=True)
    
    def _plot_risk_return_scatter(self, ax, splits_df):
        """Plot risk-return scatter."""
        scatter = ax.scatter(splits_df['volatility'], splits_df['avg_return_per_trade'], 
                           c=splits_df['sharpe_ratio'], s=splits_df['total_trades'], 
                           alpha=0.7, cmap='viridis')
        
        ax.set_xlabel('Volatility')
        ax.set_ylabel('Avg Return per Trade')
        ax.set_title('Risk-Return Analysis\n(Size=Total Trades, Color=Sharpe)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Sharpe Ratio')
        
        # Add quadrant lines
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    def _plot_performance_distribution(self, ax, splits_df):
        """Plot performance metric distributions."""
        metrics = ['f1_macro', 'accuracy', 'sharpe_ratio', 'win_rate']
        data = [splits_df[metric] for metric in metrics]
        labels = ['F1 Macro', 'Accuracy', 'Sharpe', 'Win Rate']
        
        box_plot = ax.boxplot(data, labels=labels, patch_artist=True)
        
        # Color boxes
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_title('Performance Distribution', fontweight='bold')
        ax.set_ylabel('Score')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_metrics_table(self, ax, agg_metrics, results):
        """Create detailed metrics table."""
        ax.axis('off')
        
        # Prepare table data
        table_data = [
            ['Metric', 'Mean', 'Std', 'Min', 'Max', 'Target', 'Status'],
            ['F1 Macro', f'{agg_metrics["f1_macro_mean"]:.4f}', f'{agg_metrics["f1_macro_std"]:.4f}', 
             f'{agg_metrics["f1_macro_min"]:.4f}', f'{agg_metrics["f1_macro_max"]:.4f}', '≥0.44', 
             '✓' if agg_metrics["f1_macro_mean"] >= 0.44 else '✗'],
            ['Accuracy', f'{agg_metrics["accuracy_mean"]:.4f}', f'{agg_metrics["accuracy_std"]:.4f}', 
             f'{agg_metrics["accuracy_min"]:.4f}', f'{agg_metrics["accuracy_max"]:.4f}', '≥0.70', 
             '✓' if agg_metrics["accuracy_mean"] >= 0.70 else '✗'],
            ['Sharpe Ratio', f'{agg_metrics["sharpe_ratio_mean"]:.4f}', f'{agg_metrics["sharpe_ratio_std"]:.4f}', 
             f'{agg_metrics["sharpe_ratio_mean"] - agg_metrics["sharpe_ratio_std"]:.4f}', 
             f'{agg_metrics["sharpe_ratio_mean"] + agg_metrics["sharpe_ratio_std"]:.4f}', '≥1.20', 
             '✓' if agg_metrics["sharpe_ratio_mean"] >= 1.20 else '✗'],
            ['Win Rate', f'{agg_metrics["win_rate_mean"]:.4f}', f'{agg_metrics["win_rate_std"]:.4f}', 
             f'{agg_metrics["win_rate_mean"] - agg_metrics["win_rate_std"]:.4f}', 
             f'{agg_metrics["win_rate_mean"] + agg_metrics["win_rate_std"]:.4f}', '≥0.50', 
             '✓' if agg_metrics["win_rate_mean"] >= 0.50 else '✗'],
            ['Avg Latency (ms)', f'{agg_metrics["avg_latency_ms_mean"]:.3f}', '-', '-', '-', '≤250', 
             '✓' if agg_metrics["avg_latency_ms_mean"] <= 250 else '✗'],
        ]
        
        # Create table
        table = ax.table(cellText=table_data[1:], colLabels=table_data[0], 
                        cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style status column
        for i in range(1, len(table_data)):
            status = table_data[i][-1]
            color = '#4CAF50' if status == '✓' else '#F44336'
            table[(i, 6)].set_facecolor(color)
            table[(i, 6)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Performance Metrics Summary', fontweight='bold', pad=20)
    
    def _plot_latency_analysis(self, ax, splits_df):
        """Plot latency analysis."""
        # Histogram
        ax.hist(splits_df['avg_latency_ms'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add target line
        ax.axvline(x=250, color='red', linestyle='--', linewidth=2, label='Target (≤250ms)')
        
        # Statistics
        mean_latency = splits_df['avg_latency_ms'].mean()
        max_latency = splits_df['avg_latency_ms'].max()
        
        ax.axvline(x=mean_latency, color='orange', linestyle='-', linewidth=2, label=f'Mean ({mean_latency:.1f}ms)')
        
        ax.set_xlabel('Average Latency (ms)')
        ax.set_ylabel('Frequency')
        ax.set_title('Prediction Latency Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add text box with stats
        stats_text = f'Mean: {mean_latency:.2f}ms\nMax: {max_latency:.2f}ms\nTarget: ≤250ms'
        ax.text(0.7, 0.8, stats_text, transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    def _plot_training_performance(self, ax, splits_df):
        """Plot training performance metrics."""
        # Training time vs Performance
        scatter = ax.scatter(splits_df['training_time'], splits_df['f1_macro'], 
                           c=splits_df['train_samples'], s=60, alpha=0.7, cmap='plasma')
        
        ax.set_xlabel('Training Time (seconds)')
        ax.set_ylabel('F1 Macro Score')
        ax.set_title('Training Efficiency\n(Color=Train Samples)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Training Samples')
        
        # Add trend line
        z = np.polyfit(splits_df['training_time'], splits_df['f1_macro'], 1)
        p = np.poly1d(z)
        ax.plot(splits_df['training_time'], p(splits_df['training_time']), "r--", alpha=0.8)
    
    def _save_individual_plots(self, splits_df, agg_metrics, results, timestamp):
        """Save individual plots for detailed analysis."""
        logger.info("Saving individual plots...")
        
        # Create subdirectory
        plots_dir = self.results_dir / f"individual_plots_{timestamp}"
        plots_dir.mkdir(exist_ok=True)
        
        # 1. Time series performance
        fig, ax = plt.subplots(figsize=(12, 8))
        self._plot_performance_over_time(ax, splits_df)
        plt.tight_layout()
        plt.savefig(plots_dir / "performance_over_time.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Trading metrics
        fig, ax = plt.subplots(figsize=(12, 8))
        self._plot_trading_metrics_over_time(ax, splits_df)
        plt.tight_layout()
        plt.savefig(plots_dir / "trading_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation_cols = ['f1_macro', 'accuracy', 'sharpe_ratio', 'win_rate', 
                          'volatility', 'max_drawdown', 'total_trades', 'avg_latency_ms']
        corr_matrix = splits_df[correlation_cols].corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=ax, fmt='.3f')
        ax.set_title('Performance Metrics Correlation Matrix', fontweight='bold')
        plt.tight_layout()
        plt.savefig(plots_dir / "correlation_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Individual plots saved to: {plots_dir}")
    
    def _create_summary_report(self, results, timestamp) -> str:
        """Create detailed summary report."""
        logger.info("Creating summary report...")
        
        splits_df = results['splits_df']
        agg_metrics = results['aggregated_metrics']
        
        report_file = self.results_dir / f"performance_summary_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# Trading Strategy Performance Summary\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Model:** {results['model_type'].title()}\n")
            f.write(f"**Validation Method:** {results['method'].replace('_', ' ').title()}\n")
            f.write(f"**Number of Splits:** {len(splits_df)}\n")
            f.write(f"**Test Period:** {splits_df['test_start'].min().strftime('%Y-%m-%d')} to {splits_df['test_end'].max().strftime('%Y-%m-%d')}\n\n")
            
            if results.get('hyperparameter_optimization', False):
                f.write(f"**Hyperparameter Optimization:** {results['n_trials']} trials\n")
                f.write(f"**Best Parameters:** {results['best_params']}\n\n")
            
            f.write("## Performance Summary\n\n")
            f.write("| Metric | Mean | Std | Target | Status |\n")
            f.write("|--------|------|-----|---------|--------|\n")
            
            metrics_info = [
                ('F1 Macro', 'f1_macro_mean', 'f1_macro_std', 0.44),
                ('Accuracy', 'accuracy_mean', 'accuracy_std', 0.70),
                ('Sharpe Ratio', 'sharpe_ratio_mean', 'sharpe_ratio_std', 1.20),
                ('Win Rate', 'win_rate_mean', 'win_rate_std', 0.50),
                ('Avg Latency (ms)', 'avg_latency_ms_mean', None, 250)
            ]
            
            for name, mean_key, std_key, target in metrics_info:
                mean_val = agg_metrics[mean_key]
                std_val = agg_metrics[std_key] if std_key else 0
                status = "✅" if mean_val >= target else "❌"
                
                if std_key:
                    f.write(f"| {name} | {mean_val:.4f} | {std_val:.4f} | ≥{target} | {status} |\n")
                else:
                    f.write(f"| {name} | {mean_val:.3f} | - | ≤{target} | {status} |\n")
            
            f.write("\n## Key Insights\n\n")
            
            # Performance insights
            best_split = splits_df.loc[splits_df['f1_macro'].idxmax()]
            worst_split = splits_df.loc[splits_df['f1_macro'].idxmin()]
            
            f.write(f"- **Best Performance:** {best_split['f1_macro']:.4f} F1 score in {best_split['test_start'].strftime('%Y-%m')}\n")
            f.write(f"- **Worst Performance:** {worst_split['f1_macro']:.4f} F1 score in {worst_split['test_start'].strftime('%Y-%m')}\n")
            f.write(f"- **Performance Stability:** {agg_metrics['f1_macro_std']:.4f} standard deviation\n")
            f.write(f"- **Total Training Time:** {agg_metrics['training_time_total']:.1f} seconds\n")
            f.write(f"- **Average Latency:** {agg_metrics['avg_latency_ms_mean']:.3f} ms (Target: ≤250ms)\n\n")
            
            # Trading insights
            total_return = splits_df['total_return'].sum()
            max_dd = splits_df['max_drawdown'].min()
            
            f.write(f"## Trading Performance\n\n")
            f.write(f"- **Total Cumulative Return:** {total_return:.6f}\n")
            f.write(f"- **Worst Drawdown:** {max_dd:.6f}\n")
            f.write(f"- **Average Win Rate:** {agg_metrics['win_rate_mean']:.1%}\n")
            f.write(f"- **Total Trades:** {splits_df['total_trades'].sum():,}\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            if agg_metrics['f1_macro_mean'] >= 0.44:
                f.write("✅ **F1 Score Target Met** - Model performance is satisfactory\n")
            else:
                f.write("❌ **F1 Score Below Target** - Consider additional feature engineering or model tuning\n")
            
            if agg_metrics['sharpe_ratio_mean'] >= 1.20:
                f.write("✅ **Sharpe Ratio Target Met** - Risk-adjusted returns are excellent\n")
            else:
                f.write("❌ **Sharpe Ratio Below Target** - Consider improving risk management or signal quality\n")
            
            if agg_metrics['avg_latency_ms_mean'] <= 250:
                f.write("✅ **Latency Target Met** - System is suitable for real-time trading\n")
            else:
                f.write("❌ **Latency Too High** - Optimize model inference or infrastructure\n")
        
        logger.info(f"Summary report saved to: {report_file}")
        return str(report_file)
    
    def analyze_model_comparison_results(self, comparison_file: str) -> str:
        """Create dashboard for model comparison results."""
        logger.info("Creating model comparison dashboard...")
        
        with open(comparison_file, 'r') as f:
            results = json.load(f)
        
        models_data = []
        for model_name, metrics in results['models'].items():
            models_data.append({
                'model': model_name,
                'f1_macro': metrics['test_f1_macro'],
                'accuracy': metrics['test_accuracy'],
                'latency_ms': metrics['avg_latency_ms'],
                'training_time': metrics['training_time']
            })
        
        models_df = pd.DataFrame(models_data)
        
        # Create comparison dashboard
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Comparison Dashboard', fontsize=16, fontweight='bold')
        
        # 1. F1 Score comparison
        models_df.plot(x='model', y='f1_macro', kind='bar', ax=axes[0,0], color='skyblue')
        axes[0,0].set_title('F1 Macro Score by Model')
        axes[0,0].set_ylabel('F1 Score')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Accuracy comparison
        models_df.plot(x='model', y='accuracy', kind='bar', ax=axes[0,1], color='lightgreen')
        axes[0,1].set_title('Accuracy by Model')
        axes[0,1].set_ylabel('Accuracy')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Latency comparison
        models_df.plot(x='model', y='latency_ms', kind='bar', ax=axes[1,0], color='lightcoral')
        axes[1,0].set_title('Inference Latency by Model')
        axes[1,0].set_ylabel('Latency (ms)')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Training time comparison
        models_df.plot(x='model', y='training_time', kind='bar', ax=axes[1,1], color='lightyellow')
        axes[1,1].set_title('Training Time by Model')
        axes[1,1].set_ylabel('Training Time (s)')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_dashboard = self.results_dir / f"model_comparison_dashboard_{timestamp}.png"
        plt.savefig(comparison_dashboard, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        logger.info(f"Model comparison dashboard saved to: {comparison_dashboard}")
        return str(comparison_dashboard)

def main():
    """Main execution function."""
    
    # Initialize dashboard
    dashboard = MetricsDashboard()
    
    # Example usage - analyze model comparison results
    try:
        comparison_file = "reports/comprehensive_comparison/comprehensive_comparison_results.json"
        dashboard.analyze_model_comparison_results(comparison_file)
        
        logger.info("Dashboard creation completed successfully")
        
    except FileNotFoundError as e:
        logger.error(f"Required results file not found: {e}")
        logger.info("Please ensure walk-forward validation has been completed first")
    except Exception as e:
        logger.error(f"Dashboard creation failed: {e}")
        raise

if __name__ == "__main__":
    main()