# Quick Start Guide

## What Has Changed

Your original scripts have been refactored into a modular system:

### Original → New
- `unet_train.py` → Split into multiple specialized files
- `unet_utils.py` → Split into `unet_model.py`, `dataset_utils.py`, `train_utils.py`

### New Files

1. **unet_model.py** - Flexible UNet architecture
   - Takes architecture parameters as constructor arguments
   - Adapts to any image size and channel count
   - Configurable depth, kernel sizes, channels, etc.

2. **dataset_utils.py** - Dataset loading
   - Supports MNIST, Fashion-MNIST, CIFAR-10/100, CelebA, SVHN
   - Automatic train/val splitting
   - Easy to add new datasets

3. **train_utils.py** - Training utilities
   - Early stopping support
   - Validation tracking
   - Optuna integration for pruning

4. **optuna_train.py** - Main optimization script
   - Runs Optuna hyperparameter search
   - Trains best model after optimization
   - Saves results and visualizations

5. **simple_example.py** - Manual training example
   - Shows how to use the model without Optuna
   - Good for testing specific configurations

6. **compare_results.py** - Results analysis
   - Compare optimization results across datasets
   - Generate summary tables and plots

## Getting Started

### 1. Quick Test (5-10 minutes)
```bash
# Fast test on small MNIST subset
python optuna_train.py --dataset mnist --n_trials 10 --num_epochs 5 --subset_size 1000
```

### 2. Full MNIST Optimization (1-2 hours)
```bash
# Complete optimization on MNIST
python optuna_train.py --dataset mnist --n_trials 50 --num_epochs 30
```

### 3. Train Best Model (30-60 minutes)
```bash
# After optimization, train final model
python optuna_train.py --dataset mnist --train_best
```

### 4. Optimize Multiple Datasets
```bash
# Run overnight for comprehensive comparison
python optuna_train.py --dataset mnist --n_trials 50 --num_epochs 25
python optuna_train.py --dataset cifar10 --n_trials 50 --num_epochs 25
python optuna_train.py --dataset fashion_mnist --n_trials 50 --num_epochs 25

# Then compare results
python compare_results.py
```

## Key Features

### No More Hardcoded Architecture
```python
# OLD (hardcoded):
model = UNetAutoencoder(in_channels=1, base_channels=64)

# NEW (flexible):
model = FlexibleUNetAutoencoder(
    in_channels=1,
    img_size=28,
    base_channels=64,    # Can be optimized
    num_levels=3,        # Can be optimized
    kernel_size=3,       # Can be optimized
    pooling_size=2       # Can be optimized
)
```

### Automatic Architecture Search
Optuna searches over:
- Base channels: [32, 64, 128]
- Network depth: [2, 3, 4, 5] levels
- Kernel size: [3, 5]
- Learning rate: [1e-4, 1e-2]
- Batch size: [32, 64, 128, 256]

### Smart Pruning
- Stops bad trials early (saves time)
- Early stopping prevents overfitting
- Handles OOM errors gracefully

## Workflow

```
┌─────────────────────────────────────────────────────────┐
│  1. Run Optuna Optimization (optuna_train.py)          │
│     - Tries many architectures                          │
│     - Finds best hyperparameters                        │
│     - Saves results to optuna_results/                  │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  2. Train Best Model (optuna_train.py --train_best)    │
│     - Uses best hyperparameters                         │
│     - Trains for more epochs                            │
│     - Saves final model                                 │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  3. Compare Results (compare_results.py)                │
│     - Analyze across datasets                           │
│     - Generate comparison plots                         │
│     - Identify patterns                                 │
└─────────────────────────────────────────────────────────┘
```

## Configuration Tips

### For Fast Experimentation
```bash
python optuna_train.py \
    --dataset mnist \
    --n_trials 5 \
    --num_epochs 3 \
    --subset_size 500
```

### For Production
```bash
python optuna_train.py \
    --dataset cifar10 \
    --n_trials 100 \
    --num_epochs 50 \
    --storage sqlite:///studies.db  # Can resume if interrupted
```

### For Different Datasets

**Simple (MNIST, Fashion-MNIST):**
- n_trials: 20-50
- num_epochs: 20-30
- Usually finds good solution quickly

**Medium (CIFAR-10):**
- n_trials: 50-100
- num_epochs: 30-50
- Needs more exploration

**Complex (CIFAR-100, CelebA):**
- n_trials: 100+
- num_epochs: 40+
- Start with subset_size for initial tests

## Troubleshooting

**"Out of memory" errors:**
- Reduce batch_size search space in optuna_train.py
- Reduce base_channels maximum
- Use subset_size

**Training too slow:**
- Use subset_size during optimization
- Reduce num_epochs for trials
- Train best model separately with full data

**Results not good enough:**
- Increase n_trials
- Increase num_epochs
- Expand search space in optuna_train.py

## Next Steps

1. Start with MNIST to verify everything works
2. Move to your target dataset
3. Adjust search space if needed
4. Run overnight for thorough optimization
5. Train best model for production use

## Files You'll Want to Keep

After optimization:
- `optuna_results/{dataset}_results.json` - Best hyperparameters
- `{dataset}_best_unet.pth` - Trained model
- `{dataset}_best_reconstructions.png` - Quality check

You can delete intermediate trial checkpoints and logs.
