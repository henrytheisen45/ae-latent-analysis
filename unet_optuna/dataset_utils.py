import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np


def get_dataset_info(dataset_name):
    """
    Get basic information about a dataset
    
    Returns:
        dict with keys: 'channels', 'img_size', 'num_classes'
    """
    dataset_configs = {
        'mnist': {'channels': 1, 'img_size': 28, 'num_classes': 10},
        'fashion_mnist': {'channels': 1, 'img_size': 28, 'num_classes': 10},
        'cifar10': {'channels': 3, 'img_size': 32, 'num_classes': 10},
        'cifar100': {'channels': 3, 'img_size': 32, 'num_classes': 100},
        'celeba': {'channels': 3, 'img_size': 64, 'num_classes': None},  # Will be resized
        'svhn': {'channels': 3, 'img_size': 32, 'num_classes': 10},
    }
    
    if dataset_name.lower() not in dataset_configs:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_configs.keys())}")
    
    return dataset_configs[dataset_name.lower()]


def get_dataloaders(dataset_name, batch_size=128, num_workers=2, data_root='./data', 
                   val_split=0.1, subset_size=None):
    """
    Get train and validation dataloaders for a specified dataset
    
    Args:
        dataset_name: Name of the dataset ('mnist', 'cifar10', etc.)
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        data_root: Root directory for dataset storage
        val_split: Fraction of training data to use for validation
        subset_size: If not None, use only this many samples for faster experimentation
    
    Returns:
        train_loader, val_loader, dataset_info
    """
    dataset_info = get_dataset_info(dataset_name)
    img_size = dataset_info['img_size']
    
    # Define transforms
    if dataset_name.lower() in ['mnist', 'fashion_mnist']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    elif dataset_name.lower() in ['cifar10', 'cifar100', 'svhn']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    elif dataset_name.lower() == 'celeba':
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    
    # Load dataset
    if dataset_name.lower() == 'mnist':
        full_dataset = datasets.MNIST(root=data_root, train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root=data_root, train=False, transform=transform, download=True)

    elif dataset_name.lower() == 'fashion_mnist':
        full_dataset = datasets.FashionMNIST(root=data_root, train=True, transform=transform, download=True)
        test_dataset = datasets.FashionMNIST(root=data_root, train=False, transform=transform, download=True)

    elif dataset_name.lower() == 'cifar10':
        full_dataset = datasets.CIFAR10(root=data_root, train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(root=data_root, train=False, transform=transform, download=True)

    elif dataset_name.lower() == 'cifar100':
        full_dataset = datasets.CIFAR100(root=data_root, train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR100(root=data_root, train=False, transform=transform, download=True)

    elif dataset_name.lower() == 'svhn':
        full_dataset = datasets.SVHN(root=data_root, split='train', transform=transform, download=True)
        test_dataset = datasets.SVHN(root=data_root, split='test', transform=transform, download=True)

    elif dataset_name.lower() == 'celeba':
        full_dataset = datasets.CelebA(root=data_root, split='train', transform=transform, download=True)
        test_dataset = datasets.CelebA(root=data_root, split='test', transform=transform, download=True)
        
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented")
    
    # Create subset if requested (for faster experimentation)
    if subset_size is not None and subset_size < len(full_dataset):
        #TODO make seed an arg
        rng = np.random.default_rng(42)
        indices = rng.choice(len(full_dataset), subset_size, replace=False)
        full_dataset = Subset(full_dataset, indices)
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, dataset_info


def visualize_reconstructions(model, data_loader, device='cuda', num_images=10, save_path='reconstructions.png'):
    """Visualize original and reconstructed images"""
    import matplotlib.pyplot as plt
    
    model.eval()
    with torch.no_grad():
        # Get a batch of images
        data, _ = next(iter(data_loader))
        data = data[:num_images].to(device)
        
        # Reconstruct
        reconstructions = model(data)
        
        # Move to CPU for visualization
        data = data.cpu()
        reconstructions = reconstructions.cpu()
        
        # Determine if grayscale or RGB
        is_grayscale = data.shape[1] == 1
        
        # Plot
        fig, axes = plt.subplots(2, num_images, figsize=(15, 3))
        for i in range(num_images):
            # Original images
            if is_grayscale:
                axes[0, i].imshow(data[i].squeeze(), cmap='gray')
            else:
                axes[0, i].imshow(data[i].permute(1, 2, 0))
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original', fontsize=10)
            
            # Reconstructed images
            if is_grayscale:
                axes[1, i].imshow(reconstructions[i].squeeze(), cmap='gray')
            else:
                axes[1, i].imshow(reconstructions[i].permute(1, 2, 0))
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved reconstructions to {save_path}")
