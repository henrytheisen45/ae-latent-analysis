# Flexible UNet Autoencoder with Optuna Optimization

This repository contains a flexible UNet autoencoder implementation that uses Optuna to automatically find optimal architecture parameters for different image datasets.

## Features

- **Flexible Architecture**: UNet that adapts to different image sizes, channels, and complexity levels
- **Automatic Hyperparameter Optimization**: Uses Optuna to find the best:
  - Base number of channels
  - Network depth (number of encoder/decoder levels)
  - Kernel sizes
  - Learning rate
  - Batch size
  - Pooling sizes
- **Multiple Dataset Support**: MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, SVHN, CelebA
- **Early Stopping**: Prevents overfitting with validation-based early stopping
- **Pruning**: Optuna's MedianPruner stops unpromising trials early
- **Persistence**: Can save and resume optimization studies

## File Structure

```
.
├── unet_model.py          # Flexible UNet architecture
├── dataset_utils.py       # Dataset loading utilities
├── train_utils.py         # Training and validation functions
├── optuna_train.py        # Main script for Optuna optimization
└── README.md              # This file
```

## Installation

```bash
pip install torch torchvision optuna scikit-learn matplotlib numpy
```

For visualization support (optional):
```bash
pip install plotly kaleido
```

## Quick Start

### 1. Run Optimization on MNIST (Fast Test)

Use a subset of data for quick experimentation:

```bash
python optuna_train.py --dataset mnist --n_trials 20 --num_epochs 10 --subset_size 5000
```

### 2. Run Full Optimization on MNIST

```bash
python optuna_train.py --dataset mnist --n_trials 100 --num_epochs 30
```

### 3. Run Optimization on CIFAR-10

```bash
python optuna_train.py --dataset cifar10 --n_trials 100 --num_epochs 30
```

### 4. Train Best Model After Optimization

After optimization completes, train a final model with the best parameters:

```bash
python optuna_train.py --dataset mnist --train_best
```

## Command Line Arguments

```
--dataset          Dataset to use (mnist, fashion_mnist, cifar10, cifar100, celeba, svhn)
--n_trials         Number of Optuna trials (default: 100)
--num_epochs       Number of epochs per trial (default: 30)
--device           Device to use: cuda or cpu (default: cuda)
--data_root        Root directory for datasets (default: ./data)
--subset_size      Use subset for faster experimentation (default: None)
--storage          Database URL for persistence (e.g., sqlite:///optuna.db)
--train_best       Train final model with best parameters
--best_params_file JSON file with best parameters (for --train_best)
```

## Usage Examples

### Example 1: Quick Test on Small Dataset

```bash
# Fast optimization with subset
python optuna_train.py \
    --dataset mnist \
    --n_trials 10 \
    --num_epochs 5 \
    --subset_size 1000
```

### Example 2: Full Optimization with Persistent Storage

```bash
# Run optimization with database storage (can resume if interrupted)
python optuna_train.py \
    --dataset cifar10 \
    --n_trials 100 \
    --num_epochs 30 \
    --storage sqlite:///cifar10_study.db
```

### Example 3: Multi-Dataset Optimization

```bash
# Optimize for different datasets
for dataset in mnist fashion_mnist cifar10; do
    python optuna_train.py \
        --dataset $dataset \
        --n_trials 50 \
        --num_epochs 25
done
```

### Example 4: Train Final Model

```bash
# After optimization, train the best model for 100 epochs
python optuna_train.py --dataset mnist --train_best
```

## Hyperparameter Search Space

The optimization searches over dataset-appropriate ranges:

### MNIST / Fashion-MNIST
- **base_channels**: [16, 32, 64] - Smaller for simpler data
- **num_levels**: [2, 3, 4] - Shallower networks
- **Max parameters**: 1M (typical best: 200K-500K)
- **kernel_size**: [3, 5]
- **learning_rate**: [1e-4, 1e-2] - Log-uniform distribution
- **batch_size**: [64, 128, 256]

### CIFAR-10 / SVHN
- **base_channels**: [32, 64, 128]
- **num_levels**: [3, 4, 5]
- **Max parameters**: 5M (typical best: 1M-3M)
- **kernel_size**: [3, 5]
- **learning_rate**: [1e-4, 1e-2]
- **batch_size**: [32, 64, 128, 256]

### CIFAR-100 / CelebA
- **base_channels**: [64, 128, 256] - Larger for complex data
- **num_levels**: [3, 4, 5]
- **Max parameters**: 20M (typical best: 5M-15M)
- **kernel_size**: [3, 5]
- **learning_rate**: [1e-4, 1e-2]
- **batch_size**: [16, 32, 64, 128]

The search space automatically adapts based on the dataset to prevent oversized models.

## Output Files

After optimization, the following files are created:

```
optuna_results/
├── {dataset}_unet_optimization_results.json   # Best parameters and metrics
├── {dataset}_unet_optimization_history.png    # Optimization history plot
└── {dataset}_unet_optimization_importance.png # Parameter importance plot

{dataset}_best_unet.pth                         # Trained model (if --train_best used)
{dataset}_best_reconstructions.png              # Sample reconstructions
```

## Architecture Details

The `FlexibleUNetAutoencoder` class automatically:
- Adjusts to different input image sizes and channels
- Creates encoder/decoder pairs based on `num_levels`
- Uses skip connections between encoder and decoder
- Handles size mismatches with bilinear interpolation
- Scales channel counts exponentially through the network

## Tips for Different Datasets

### MNIST / Fashion-MNIST (28x28, 1 channel)
- Usually needs fewer parameters (target: 100K-500K)
- Recommended: 20-50 trials, 20-30 epochs per trial
- **If you see >1M parameters, the model is too large!**

### CIFAR-10/100 (32x32, 3 channels)
- More complex than MNIST
- CIFAR-10: target 500K-3M parameters
- CIFAR-100: target 1M-10M parameters
- Recommended: 50-100 trials, 30-50 epochs per trial

### CelebA (64x64, 3 channels)
- High complexity, large images
- Target: 2M-15M parameters
- Recommended: 100+ trials, 40+ epochs per trial
- Consider using subset_size initially

### General Tips
1. Start with `subset_size` for quick experimentation
2. Use `--storage` for long optimizations (allows resuming)
3. Increase `num_epochs` if validation loss is still decreasing
4. After finding good architecture, train longer with `--train_best`
5. **Check parameter count!** See PARAMETER_GUIDE.md for reference tables

## Customization

### Adding New Datasets

Edit `dataset_utils.py` and add to `get_dataset_info()` and `get_dataloaders()`:

```python
dataset_configs = {
    'your_dataset': {'channels': 3, 'img_size': 64, 'num_classes': 10},
    # ...
}
```

### Modifying Search Space

Edit the `objective()` function in `optuna_train.py`:

```python
base_channels = trial.suggest_categorical('base_channels', [16, 32, 64, 128, 256])
num_levels = trial.suggest_int('num_levels', 2, 6)
# Add more suggestions as needed
```

### Custom Loss Functions

Edit `train_utils.py` to change the loss function in `train_autoencoder()`:

```python
criterion = nn.MSELoss()  # Change to your preferred loss
```

## Troubleshooting

**Out of Memory Errors:**
- Reduce `batch_size` in search space
- Reduce `base_channels` maximum value
- Reduce `num_levels` maximum value
- Use `subset_size` for initial experiments

**Slow Training:**
- Use `subset_size` for faster iteration
- Reduce `num_epochs` during optimization
- Reduce `n_trials`
- Use better hardware or CPU if GPU is full

**Poor Results:**
- Increase `n_trials` for more thorough search
- Increase `num_epochs` if underfitting
- Check if early stopping is too aggressive
- Try different datasets

## Performance Notes

- Optuna's pruner will stop unpromising trials early
- Early stopping prevents overfitting during each trial
- GPU is strongly recommended for image datasets
- Expect 1-6 hours for full optimization depending on dataset

## License

MIT
