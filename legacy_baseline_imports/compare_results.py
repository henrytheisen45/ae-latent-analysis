"""
Script to compare Optuna optimization results across different datasets
"""

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir='optuna_results'):
    """Load all optimization results from the results directory"""
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        print(f"Results directory {results_dir} does not exist!")
        return None
    
    results = {}
    for result_file in results_dir.glob('*_results.json'):
        with open(result_file, 'r') as f:
            data = json.load(f)
            dataset_name = data['dataset']
            results[dataset_name] = data
    
    return results


def create_comparison_table(results):
    """Create a comparison table of results"""
    data = []
    
    for dataset, result in results.items():
        params = result['best_params']
        row = {
            'Dataset': dataset,
            'Best Val Loss': result['best_value'],
            'N Trials': result['n_trials'],
            'Base Channels': params['base_channels'],
            'Num Levels': params['num_levels'],
            'Kernel Size': params['kernel_size'],
            'Learning Rate': params['learning_rate'],
            'Batch Size': params['batch_size'],
            'Pooling Size': params.get('pooling_size', 2)
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    df = df.sort_values('Best Val Loss')
    
    return df


def plot_comparison(results, save_path='comparison_plot.png'):
    """Create visualization comparing results across datasets"""
    datasets = list(results.keys())
    val_losses = [results[d]['best_value'] for d in datasets]
    n_trials = [results[d]['n_trials'] for d in datasets]
    base_channels = [results[d]['best_params']['base_channels'] for d in datasets]
    num_levels = [results[d]['best_params']['num_levels'] for d in datasets]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Validation loss comparison
    axes[0, 0].bar(datasets, val_losses, color='steelblue')
    axes[0, 0].set_ylabel('Best Validation Loss')
    axes[0, 0].set_title('Best Validation Loss by Dataset')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Number of trials
    axes[0, 1].bar(datasets, n_trials, color='coral')
    axes[0, 1].set_ylabel('Number of Trials')
    axes[0, 1].set_title('Optimization Trials by Dataset')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Base channels
    axes[1, 0].bar(datasets, base_channels, color='lightgreen')
    axes[1, 0].set_ylabel('Base Channels')
    axes[1, 0].set_title('Best Base Channels by Dataset')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Number of levels
    axes[1, 1].bar(datasets, num_levels, color='plum')
    axes[1, 1].set_ylabel('Number of Levels')
    axes[1, 1].set_title('Best Network Depth by Dataset')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to {save_path}")


def print_detailed_comparison(results):
    """Print detailed comparison of all results"""
    print("\n" + "="*80)
    print("DETAILED RESULTS COMPARISON")
    print("="*80)
    
    for dataset, result in results.items():
        print(f"\n{dataset.upper()}")
        print("-" * 40)
        print(f"Best Validation Loss: {result['best_value']:.6f}")
        print(f"Number of Trials: {result['n_trials']}")
        print(f"\nBest Hyperparameters:")
        for param, value in result['best_params'].items():
            print(f"  {param}: {value}")
        print()


def analyze_patterns(results):
    """Analyze patterns in optimal hyperparameters across datasets"""
    print("\n" + "="*80)
    print("PATTERN ANALYSIS")
    print("="*80)
    
    # Collect parameters
    base_channels = [r['best_params']['base_channels'] for r in results.values()]
    num_levels = [r['best_params']['num_levels'] for r in results.values()]
    kernel_sizes = [r['best_params']['kernel_size'] for r in results.values()]
    learning_rates = [r['best_params']['learning_rate'] for r in results.values()]
    batch_sizes = [r['best_params']['batch_size'] for r in results.values()]
    
    print(f"\nBase Channels:")
    print(f"  Range: {min(base_channels)} - {max(base_channels)}")
    print(f"  Mean: {np.mean(base_channels):.1f}")
    print(f"  Most common: {max(set(base_channels), key=base_channels.count)}")
    
    print(f"\nNetwork Depth (Num Levels):")
    print(f"  Range: {min(num_levels)} - {max(num_levels)}")
    print(f"  Mean: {np.mean(num_levels):.1f}")
    print(f"  Most common: {max(set(num_levels), key=num_levels.count)}")
    
    print(f"\nKernel Size:")
    print(f"  Most common: {max(set(kernel_sizes), key=kernel_sizes.count)}")
    
    print(f"\nLearning Rate:")
    print(f"  Range: {min(learning_rates):.6f} - {max(learning_rates):.6f}")
    print(f"  Mean: {np.mean(learning_rates):.6f}")
    
    print(f"\nBatch Size:")
    print(f"  Range: {min(batch_sizes)} - {max(batch_sizes)}")
    print(f"  Most common: {max(set(batch_sizes), key=batch_sizes.count)}")
    print()


def main():
    # Load results
    results = load_results()
    
    if results is None or len(results) == 0:
        print("No results found! Run optuna_train.py first.")
        return
    
    print(f"Found results for {len(results)} dataset(s): {', '.join(results.keys())}")
    
    # Create comparison table
    df = create_comparison_table(results)
    print("\n" + "="*80)
    print("RESULTS SUMMARY TABLE")
    print("="*80)
    print(df.to_string(index=False))
    
    # Save table to CSV
    df.to_csv('optuna_results/comparison_table.csv', index=False)
    print("\nTable saved to optuna_results/comparison_table.csv")
    
    # Create comparison plots
    if len(results) > 1:
        plot_comparison(results)
    
    # Print detailed comparison
    print_detailed_comparison(results)
    
    # Analyze patterns
    if len(results) > 1:
        analyze_patterns(results)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()
