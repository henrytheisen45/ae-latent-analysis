# AE Latent Analysis

A reproducible research pipeline for studying the geometry of latent spaces in convolutional autoencoders.

This project investigates how latent dimension, architecture, and training choices affect the intrinsic geometry of learned representations — with a focus on intrinsic dimension (ID), neighborhood structure, and manifold smoothness.

---

## Research Goal

This project studies how autoencoders organize high-dimensional image data in latent space. The goal is to understand:

- Intrinsic dimensionality of learned representations
- Examine neighborhood and class structure
- Stability of embeddings across models and datasets
- Test intuitions regarding geometry of latent representions on simple models, before looking at more complex ones

---

## Project Structure
ae-latent-analysis/
│
├── src/
│ └── ae_latent/
│ ├── models/ # Autoencoder architectures + builders
│ ├── training/ # Training loops and optimization logic
│ ├── analysis/ # Latent extraction + intrinsic dimension analysis
│ ├── data/ # Deterministic dataloaders + dataset utilities
│ ├── cli/ # Command-line entrypoints
│ ├── tuning/ # Hyperparameter search utilities
│ └── utils/ # Config loading + reproducibility helpers
│
├── configs/ # Experiment configuration files
├── notebooks/ # Exploratory analysis (kept separate from core logic)
├── experiments/ # Research notes + documented experiment results

---

##  Model Philosophy

The default model is a vector-bottleneck convolutional autoencoder:

- No skip connections
- Latent is a true vector `z ∈ ℝ^d`
- Controlled downsampling (`num_levels`)
- Minimal architectural adjustments

We intentionally isolate the bottleneck as the only information path to avoid contaminating latent geometry with bypass connections.
This is a "start simple" model. More to come later.