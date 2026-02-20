

def visualize_reconstructions(
    model,
    data_loader,
    device=None,
    num_images=10,
    save_path=None,
    denormalize=None,
    seed=42
):
    """
    Visualize original and reconstructed images (originals + recon in a 2-row grid).

    Assumes [-1,1] std=0.5 normalization by default:
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    For display, we revert to [0, 1].

    Args:
        model: Trained autoencoder (callable on a batch: [B,C,H,W] -> [B,C,H,W]).
        data_loader: PyTorch DataLoader yielding (images, labels) or images only.
        device: torch.device or str. If None, inferred from model parameters.
        num_images: Number of images to visualize from a single batch.
        save_path: If provided, saves the figure to this path.
        denormalize: Optional callable that maps a batch tensor back to display range.
                     If None, uses default inverse for mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5).

    Returns:
        fig: matplotlib.figure.Figure
        axes: numpy.ndarray of Axes with shape (2, N)
        batch: dict with CPU tensors:
            {
              "original": Tensor[N,C,H,W] in display range [0,1],
              "reconstruction": Tensor[N,C,H,W] in display range [0,1],
            }
    """
    import torch
    import matplotlib.pyplot as plt

    # Infer device from model if not provided
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device) if not isinstance(device, torch.device) else device

    # Default denormalize for Normalize(mean=0.5, std=0.5) per channel
    if denormalize is None:
        def denormalize(x: torch.Tensor) -> torch.Tensor:
            # x: [B,C,H,W]
            C = x.shape[1]
            mean = torch.full((1, C, 1, 1), 0.5, device=x.device, dtype=x.dtype)
            std  = torch.full((1, C, 1, 1), 0.5, device=x.device, dtype=x.dtype)
            return (x * std + mean).clamp(0, 1)

    model.eval()
    with torch.no_grad():
        batch = next(iter(data_loader))
        data = batch[0] if isinstance(batch, (list, tuple)) else batch

        # Guard against too-large num_images
        n = min(num_images, data.shape[0])
        
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(data), generator=g)[:n]

        data = data[idx].to(device)
        recon = model(data)

        # Move to CPU and denormalize for display
        data_disp = denormalize(data).cpu()
        recon_disp = denormalize(recon).cpu()

    is_grayscale = data_disp.shape[1] == 1

    fig, axes = plt.subplots(2, n, figsize=(max(1.5 * n, 6), 3.2))
    # If n == 1, axes may not be 2D; normalize it
    if n == 1:
        axes = axes.reshape(2, 1)
        
    for i in range(n):
        if is_grayscale:
            axes[0, i].imshow(data_disp[i].squeeze(0), cmap="gray")
        else:
            axes[0, i].imshow(data_disp[i].permute(1, 2, 0))
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original", fontsize=10)

        if is_grayscale:
            axes[1, i].imshow(recon_disp[i].squeeze(0), cmap="gray")
        else:
            axes[1, i].imshow(recon_disp[i].permute(1, 2, 0))
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Reconstructed", fontsize=10)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print("hii")
    return fig, axes, {"original": data_disp, "reconstruction": recon_disp}



def visualize_raw_images(
    dataset,
    num_images=10,
    seed=42,
    save_path=None,
):
    """
    Visualize raw (pre-transform) images directly from the dataset.

    Args:
        dataset: torchvision dataset (e.g. CelebA, CIFAR, MNIST)
        num_images: number of images to show
        seed: RNG seed
        save_path: optional path to save figure
    """
    import torch
    import matplotlib.pyplot as plt
    import numpy as np

    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=rng)[:num_images]

    fig, axes = plt.subplots(1, num_images, figsize=(1.5 * num_images, 3))

    if num_images == 1:
        axes = [axes]

    for i, j in enumerate(idx):
        sample = dataset[j]

        # torchvision datasets usually return (PIL, label) before transform
        if isinstance(sample, (tuple, list)):
            img = sample[0]
        else:
            img = sample

        # Convert PIL → numpy if needed
        if hasattr(img, "convert"):
            img = np.array(img)

        axes[i].imshow(img)
        axes[i].axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, axes