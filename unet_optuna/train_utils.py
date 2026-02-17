import copy

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler



class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    def __init__(self, patience=7, min_delta=0, verbose=False):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as an improvement
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        
        return self.early_stop


 def train_epoch(model, train_loader, optimizer, criterion, device, use_amp=False, scaler=None):
    model.train()
    total_loss = 0.0
    n = 0

    for data, _ in train_loader:
        data = data.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            output = model(data)
            loss = criterion(output, data)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = data.size(0)
        total_loss += loss.item() * bs
        n += bs

    return total_loss / n




def validate(model, val_loader, criterion, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for data, _ in val_loader:
            data = data.to(device)
            with autocast(enabled=use_amp):
                output = model(data)
                loss = criterion(output, data)
            bs = data.size(0)
            total_loss += loss.item() * bs
            n += bs
    return total_loss / n


def train_autoencoder(model, train_loader, val_loader, num_epochs=100, 
                     learning_rate=1e-3, device='cuda', patience=10, 
                     verbose=True, trial=None):
    """
    Train the autoencoder with validation and early stopping
    
    Args:
        model: The autoencoder model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Maximum number of epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on
        patience: Patience for early stopping
        verbose: Whether to print training progress
        trial: Optuna trial object for pruning
    
    Returns:
        best_val_loss: Best validation loss achieved
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=patience, verbose=verbose)
    use_amp = (str(device).startswith("cuda") and torch.cuda.is_available())
    scaler = GradScaler(enabled=use_amp)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, use_amp=use_amp, scaler=scaler)

        
        # Validate
        val_loss = validate(model, val_loader, criterion, device, use_amp=use_amp)

        
        if verbose:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Track best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        # Report to Optuna for pruning (if using Optuna)
        if trial is not None:
            trial.report(val_loss, epoch)
            
            # Handle pruning based on the intermediate value
            if trial.should_prune():
                if verbose:
                    print(f"Trial pruned at epoch {epoch+1}")
                raise optuna.TrialPruned()
        
        # Early stopping check
        if early_stopping(val_loss, model):
            if verbose:
                print(f'Early stopping triggered at epoch {epoch+1}')
            # Restore best model
            model.load_state_dict(early_stopping.best_model_state)
            break
    
    # If not early stopped, restore best model anyway
    if not early_stopping.early_stop and early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)
    
    return best_val_loss


# Import for pruning
try:
    import optuna
except ImportError:
    pass
