import torch
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import argparse
import json
from pathlib import Path

from unet_model import FlexibleUNetAutoencoder, count_parameters
from dataset_utils import get_dataloaders, visualize_reconstructions, get_dataset_info
from train_utils import train_autoencoder


def objective(trial, dataset_name, num_epochs, device, data_root, subset_size):
    """
    Optuna objective function to optimize UNet architecture
    
    Args:
        trial: Optuna trial object
        dataset_name: Name of the dataset to use
        num_epochs: Number of training epochs
        device: Device to train on
        data_root: Root directory for datasets
        subset_size: Size of subset for faster experimentation (None for full dataset)
    
    Returns:
        best_val_loss: Best validation loss achieved
    """
    # Get dataset info
    dataset_info = get_dataset_info(dataset_name)
    in_channels = dataset_info['channels']
    img_size = dataset_info['img_size']
    
    # Define dataset-specific search spaces based on complexity
    # Simple datasets (MNIST, Fashion-MNIST): smaller architectures
    # Complex datasets (CIFAR-100, CelebA): larger architectures
    if dataset_name.lower() in ['mnist', 'fashion_mnist']:
        base_channels_choices = [16, 32, 64]
        num_levels_range = (2, 4)  # 2-4 levels
        batch_size_choices = [64, 128, 256]
    elif dataset_name.lower() in ['cifar10', 'svhn']:
        base_channels_choices = [32, 64, 128]
        num_levels_range = (3, 5)  # 3-5 levels
        batch_size_choices = [32, 64, 128, 256]
    elif dataset_name.lower() in ['cifar100', 'celeba']:
        base_channels_choices = [64, 64]
        num_levels_range = (3, 3)  # 3-5 levels
        batch_size_choices = [64, 64]
    else:
        # Default for unknown datasets
        base_channels_choices = [32, 64, 128]
        num_levels_range = (2, 5)
        batch_size_choices = [32, 64, 128, 256]
    
    # Suggest hyperparameters
    base_channels = trial.suggest_categorical('base_channels', base_channels_choices)
    num_levels = trial.suggest_int('num_levels', num_levels_range[0], num_levels_range[1])
    kernel_size = trial.suggest_categorical('kernel_size', [3, 3])
    base_lr = trial.suggest_float('learning_rate', 2.2e-4, 2.2e-4, log=True)
    batch_size = trial.suggest_categorical('batch_size', batch_size_choices)
    learning_rate = base_lr * (batch_size / base_channels_choices[0])
    
    # Padding is typically (kernel_size - 1) // 2 for 'same' padding
    padding = (kernel_size - 1) // 2
    
    # Pooling size (typically 2)
    pooling_size = trial.suggest_categorical('pooling_size', [2])
    
    # Calculate if architecture is feasible (image must be divisible by pooling at each level)
    min_size = pooling_size ** num_levels
    if img_size < min_size:
        raise optuna.TrialPruned(f"Image size {img_size} too small for {num_levels} levels with pooling {pooling_size}")
    
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}")
    print(f"{'='*60}")
    print(f"Architecture parameters:")
    print(f"  Base channels: {base_channels}")
    print(f"  Num levels: {num_levels}")
    print(f"  Kernel size: {kernel_size}")
    print(f"  Pooling size: {pooling_size}")
    print(f"Training parameters:")
    print(f"  Learning rate: {learning_rate:.6f}")
    print(f"  Batch size: {batch_size}")
    print(f"{'='*60}\n")
    
    # Create model
    model = FlexibleUNetAutoencoder(
        in_channels=in_channels,
        img_size=img_size,
        base_channels=base_channels,
        num_levels=num_levels,
        kernel_size=kernel_size,
        padding=padding,
        pooling_size=pooling_size
    )
    
    num_params_trials = []
    num_params = count_parameters(model)
    num_params_trials.append(num_params)
    print(f"Model has {num_params:,} parameters")
    
    # Define reasonable parameter limits based on dataset
    if dataset_name.lower() in ['mnist', 'fashion_mnist']:
        max_params = 1_000_000  # 1M max for simple datasets
    elif dataset_name.lower() in ['cifar10', 'svhn']:
        max_params = 5_000_000  # 5M max for medium datasets
    elif dataset_name.lower() in ['cifar100', 'celeba']:
        max_params = 20_000_000  # 20M max for complex datasets
    else:
        max_params = 10_000_000  # Default 10M
    
    # Prune if model is too large
    if num_params > max_params:
        print(f"Model too large ({num_params:,} > {max_params:,} params). Pruning trial.")
        raise optuna.TrialPruned(f"Parameter count {num_params:,} exceeds limit {max_params:,}")
    
    # Load data
    train_loader, val_loader, _ = get_dataloaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
        data_root=data_root,
        subset_size=subset_size
    )
    
    # Train model
    try:
        best_val_loss = train_autoencoder(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
            patience=10,
            verbose=True,
            trial=trial
        )
    except RuntimeError as e:
        # Handle out of memory errors
        if "out of memory" in str(e):
            print(f"Out of memory error! Pruning trial.")
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()
        else:
            raise e
    
    return best_val_loss


def run_optimization(dataset_name, n_trials=100, num_epochs=50, device='cuda', 
                    data_root='./data', subset_size=None, study_name=None,
                    storage=None):
    """
    Run Optuna optimization study
    
    Args:
        dataset_name: Name of the dataset
        n_trials: Number of trials to run
        num_epochs: Number of epochs per trial
        device: Device to use
        data_root: Root directory for datasets
        subset_size: Size of subset for faster experimentation
        study_name: Name for the study (for persistence)
        storage: Database URL for study persistence
    
    Returns:
        study: Optuna study object
    """
    # Create study
    if study_name is None:
        study_name = f"{dataset_name}_unet_optimization"
    
    sampler = TPESampler()
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, dataset_name, num_epochs, device, data_root, subset_size),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Print results
    print("\n" + "="*60)
    print("Optimization Complete!")
    print("="*60)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_value:.6f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results_dir = Path('optuna_results')
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / f"{study_name}_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'dataset': dataset_name
        }, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Generate optimization history plot
    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances
        import matplotlib.pyplot as plt
        
        fig = plot_optimization_history(study)
        fig.write_image(str(results_dir / f"{study_name}_history.png"))
        
        fig = plot_param_importances(study)
        fig.write_image(str(results_dir / f"{study_name}_importance.png"))
        
        print(f"Visualizations saved to {results_dir}")
    except Exception as e:
        print(f"Could not generate visualizations: {e}")
    
    return study


def train_best_model(dataset_name, best_params, num_epochs=100, device='cuda', 
                    data_root='./data', save_path='best_model.pth'):
    """
    Train a model with the best hyperparameters found by Optuna
    
    Args:
        dataset_name: Name of the dataset
        best_params: Best hyperparameters from Optuna
        num_epochs: Number of epochs to train
        device: Device to use
        data_root: Root directory for datasets
        save_path: Path to save the trained model
    
    Returns:
        model: Trained model
    """
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    


    # Get dataset info
    dataset_info = get_dataset_info(dataset_name)
    in_channels = dataset_info['channels']
    img_size = dataset_info['img_size']
    
    # Extract parameters
    base_channels = best_params['base_channels']
    num_levels = best_params['num_levels']
    kernel_size = best_params['kernel_size']
    batch_size = best_params['batch_size']
    base_lr = best_params['learning_rate']
    pooling_size = best_params['pooling_size']
    padding = (kernel_size - 1) // 2

    #TODO make flexible
    learning_rate = base_lr * (batch_size / 64)
    
    print(f"\n{'='*60}")
    print(f"Training Best Model")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Architecture: base_channels={base_channels}, num_levels={num_levels}")
    print(f"Kernel size: {kernel_size}, Pooling: {pooling_size}")
    print(f"Training: lr={learning_rate:.6f}, batch_size={batch_size}, epochs={num_epochs}")
    print(f"{'='*60}\n")
    
    # Create model
    model = FlexibleUNetAutoencoder(
        in_channels=in_channels,
        img_size=img_size,
        base_channels=base_channels,
        num_levels=num_levels,
        kernel_size=kernel_size,
        padding=padding,
        pooling_size=pooling_size
    )
    
    print(f"Model has {count_parameters(model):,} parameters\n")
    
    # Load data (use full dataset for final training)
    train_loader, val_loader, _ = get_dataloaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
        data_root=data_root,
        subset_size=None  # Use full dataset
    )
    
    # Train
    best_val_loss = train_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        patience=15,
        verbose=True
    )
    
    print(f"\nFinal validation loss: {best_val_loss:.6f}")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'hyperparameters': best_params,
        'dataset_info': dataset_info,
        'val_loss': best_val_loss
    }, save_path)
    print(f"Model saved to {save_path}")

    torch.save(model.state_dict(), save_path.replace(".pth", "_weights.pth"))

    
    # Visualize reconstructions
    try:
        visualize_reconstructions(model, val_loader, device=device, 
                            save_path=f'{dataset_name}_best_reconstructions.png')
    except Exception as e:
        print("Visualization failed:", e)

    
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimize UNet Autoencoder with Optuna')
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'celeba', 'svhn'],
                       help='Dataset to use')
    parser.add_argument('--n_trials', type=int, default=100,
                       help='Number of Optuna trials')
    parser.add_argument('--num_epochs', type=int, default=30,
                       help='Number of epochs per trial')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--data_root', type=str, default='./data',
                       help='Root directory for datasets')
    parser.add_argument('--subset_size', type=int, default=None,
                       help='Use subset of data for faster experimentation')
    parser.add_argument('--storage', type=str, default=None,
                       help='Database URL for study persistence (e.g., sqlite:///optuna.db)')
    parser.add_argument('--train_best', action='store_true',
                       help='Train the best model after optimization')
    parser.add_argument('--best_params_file', type=str, default=None,
                       help='JSON file with best parameters (for --train_best)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    if args.train_best:
        # Load best parameters and train final model
        if args.best_params_file is None:
            results_file = Path('optuna_results') / f"{args.dataset}_unet_optimization_results.json"
        else:
            results_file = Path(args.best_params_file)
        
        if not results_file.exists():
            raise FileNotFoundError(f"Best parameters file not found: {results_file}")
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        best_params = results['best_params']
        
        train_best_model(
            dataset_name=args.dataset,
            best_params=best_params,
            num_epochs=100,
            device=device,
            data_root=args.data_root,
            save_path=f'{args.dataset}_best_unet.pth'
        )
    else:
        # Run optimization
        study = run_optimization(
            dataset_name=args.dataset,
            n_trials=args.n_trials,
            num_epochs=args.num_epochs,
            device=device,
            data_root=args.data_root,
            subset_size=args.subset_size,
            storage=args.storage
        )
