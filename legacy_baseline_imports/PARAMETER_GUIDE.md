# UNet Parameter Count Reference

This guide helps you understand how architecture choices affect model size.

## Parameter Count Formula

For a UNet with `num_levels` and `base_channels`:
- Channels at each level: base × 2^level
- Each DoubleConv has: ~2 × (in × out × kernel² + out)
- Bottleneck has highest channel count: base × 2^num_levels

## Example Configurations

### MNIST (28×28, 1 channel)
Recommended: **100K - 500K parameters**

| base_channels | num_levels | Parameters | Notes |
|--------------|------------|------------|-------|
| 16           | 2          | ~70K       | Minimal, may underfit |
| 16           | 3          | ~180K      | ✓ Good baseline |
| 32           | 2          | ~150K      | ✓ Good baseline |
| 32           | 3          | ~450K      | ✓ Sweet spot |
| 64           | 2          | ~400K      | ✓ Upper range |
| 64           | 3          | ~1.2M      | May overfit |
| 64           | 4          | ~4.5M      | ⚠️ Too large |
| 128          | 3          | ~4.8M      | ⚠️ Too large |
| 128          | 4          | ~18M       | ❌ Massive overkill |
| 128          | 5          | ~72M       | ❌ Extreme overkill |

### Fashion-MNIST (28×28, 1 channel)
Recommended: **200K - 800K parameters** (slightly more complex than MNIST)

| base_channels | num_levels | Parameters | Notes |
|--------------|------------|------------|-------|
| 32           | 3          | ~450K      | ✓ Good baseline |
| 64           | 2          | ~400K      | ✓ Alternative |
| 64           | 3          | ~1.2M      | Acceptable, watch for overfitting |

### CIFAR-10 (32×32, 3 channels)
Recommended: **500K - 3M parameters**

| base_channels | num_levels | Parameters | Notes |
|--------------|------------|------------|-------|
| 32           | 3          | ~700K      | ✓ Efficient |
| 32           | 4          | ~2.3M      | ✓ Good performance |
| 64           | 3          | ~1.8M      | ✓ Sweet spot |
| 64           | 4          | ~6.8M      | Upper limit |
| 128          | 3          | ~7M        | May overfit |
| 128          | 4          | ~27M       | ⚠️ Too large |

### CIFAR-100 (32×32, 3 channels, 100 classes)
Recommended: **1M - 10M parameters** (more complex than CIFAR-10)

| base_channels | num_levels | Parameters | Notes |
|--------------|------------|------------|-------|
| 64           | 3          | ~1.8M      | ✓ Baseline |
| 64           | 4          | ~6.8M      | ✓ Better capacity |
| 128          | 3          | ~7M        | ✓ Good for complexity |
| 128          | 4          | ~27M       | Upper reasonable limit |

### CelebA (64×64, 3 channels)
Recommended: **2M - 15M parameters** (high resolution, complex)

| base_channels | num_levels | Parameters | Notes |
|--------------|------------|------------|-------|
| 64           | 4          | ~6.8M      | ✓ Baseline |
| 64           | 5          | ~26M       | Large but reasonable |
| 128          | 4          | ~27M       | ✓ Good capacity |
| 128          | 5          | ~107M      | ⚠️ Very large |

## Quick Calculation Script

Use this to check parameter count before training:

```python
from unet_model import FlexibleUNetAutoencoder, count_parameters

# Example for MNIST
model = FlexibleUNetAutoencoder(
    in_channels=1,
    img_size=28,
    base_channels=32,
    num_levels=3,
    kernel_size=3,
    padding=1,
    pooling_size=2
)

print(f"Parameters: {count_parameters(model):,}")
```

## How Channel Counts Grow

For `base_channels=64`, `num_levels=4`:

```
Level 0 (Encoder): 64 channels
Level 1 (Encoder): 128 channels
Level 2 (Encoder): 256 channels
Level 3 (Encoder): 512 channels
Bottleneck:        1024 channels  ← Most parameters here!
Level 3 (Decoder): 512 channels
Level 2 (Decoder): 256 channels
Level 1 (Decoder): 128 channels
Level 0 (Decoder): 64 channels
```

The bottleneck layer has the most channels, and this is where parameters explode!

## Updated Search Spaces

The new Optuna script now uses dataset-aware search spaces:

### MNIST / Fashion-MNIST
- base_channels: [16, 32, 64]
- num_levels: [2, 3, 4]
- Max parameters: 1M
- Expected best: ~200K-500K

### CIFAR-10 / SVHN
- base_channels: [32, 64, 128]
- num_levels: [3, 4, 5]
- Max parameters: 5M
- Expected best: ~1M-3M

### CIFAR-100 / CelebA
- base_channels: [64, 128, 256]
- num_levels: [3, 4, 5]
- Max parameters: 20M
- Expected best: ~5M-15M

## Tips

1. **Start small**: Begin with lower base_channels and fewer levels
2. **Monitor overfitting**: If training loss << validation loss, reduce model size
3. **Dataset complexity matters**: MNIST needs way less capacity than CelebA
4. **Memory constraints**: Larger models need more GPU memory
5. **Training time**: Parameters roughly correlate with training time

## Rule of Thumb

- Simple grayscale (MNIST): base_channels ≤ 32, num_levels ≤ 3
- Medium RGB (CIFAR-10): base_channels ≤ 64, num_levels ≤ 4
- Complex RGB (CIFAR-100, CelebA): base_channels ≤ 128, num_levels ≤ 4

**If you see >10M parameters for MNIST, something is wrong!**
