"""
Simple example script demonstrating how to use the flexible UNet autoencoder
without Optuna optimization.
"""

import torch
from unet_model import FlexibleUNetAutoencoder, count_parameters
from dataset_utils import get_dataloaders, visualize_reconstructions
from train_utils import train_autoencoder


def main():
    # Configuration
    dataset_name = 'mnist'  # Change to: fashion_mnist, cifar10, cifar100, etc.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Manual architecture configuration
    # For MNIST (28x28, 1 channel) - ~100K-500K parameters
    config = {
        'in_channels': 1,
        'img_size': 28,
        'base_channels': 32,  # Reduced from 64
        'num_levels': 3,
        'kernel_size': 3,
        'padding': 1,
        'pooling_size': 2
    }
    
    # For CIFAR-10 (32x32, 3 channels) - ~1M-3M parameters, uncomment:
    # config = {
    #     'in_channels': 3,
    #     'img_size': 32,
    #     'base_channels': 64,
    #     'num_levels': 4,
    #     'kernel_size': 3,
    #     'padding': 1,
    #     'pooling_size': 2
    # }
    
    print(f"Using device: {device}")
    print(f"Dataset: {dataset_name}")
    print(f"\nModel Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create model
    model = FlexibleUNetAutoencoder(**config)
    print(f"\nModel has {count_parameters(model):,} trainable parameters")
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, dataset_info = get_dataloaders(
        dataset_name=dataset_name,
        batch_size=128,
        data_root='./data'
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Train model
    print("\nTraining model...")
    best_val_loss = train_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        learning_rate=1e-3,
        device=device,
        patience=10,
        verbose=True
    )
    
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.6f}")
    
    # Save model
    save_path = f'{dataset_name}_manual_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'val_loss': best_val_loss
    }, save_path)
    print(f"Model saved to {save_path}")
    
    # Visualize reconstructions
    print("\nGenerating reconstructions...")
    visualize_reconstructions(
        model, 
        val_loader, 
        device=device,
        save_path=f'{dataset_name}_manual_reconstructions.png'
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
