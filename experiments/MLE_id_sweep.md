# Experiment: Intrinsic Dimension Estimation in Latent Space (kNN-MLE)

## Goal

Estimate the intrinsic dimension (ID) of autoencoder latent representations and analyze how the estimated ID changes as bottleneck size `z_dim` varies.

---

## Models Evaluated

### Architecture

All models are vector-bottleneck convolutional autoencoders with a true information bottleneck (no skip connections).

Encoder:
- Convolutional downsampling stack
- Fixed depth and number of levels
- Fixed base channel width
- Final feature map flattened and projected to latent vector `z`

Bottleneck:
- Fully-connected projection to `z ∈ ℝ^{z_dim}`

Decoder:
- Fully-connected expansion from latent vector
- Mirrored convolutional upsampling stack
- No skip connections

Output:
- `tanh` activation
- L1 reconstruction loss

### Controlled Variables

Held constant across all models:

- Dataset: CelebA (same split)
- Preprocessing and normalization
- Image resolution
- Architecture depth and width
- Loss function
- Optimizer and learning rate schedule
- Batch size
- Early stopping patience
- Random seed

### Experimental Variable

The only parameter varied is: z_dim ∈ {8, 16, 32, 64, 128, 256, 512}


This isolates the effect of latent capacity on representation geometry.
The projection from the final encoder layer necissarilly varies accoring to Z dim. 

---

## Data

For each trained model, latents are exported as `.npz` files containing:

- `Z`: latent vectors, shape `(N, z_dim)`
- `y`: labels (unused)
- `base_idx`: dataset index
- `dataset`, `split`: metadata

The experiment operates only on `Z`.

---

## Method: Levina–Bickel kNN MLE

We estimate intrinsic dimension using the local kNN-based MLE estimator (Levina & Bickel, 2005).

For each query point `i`, let `r_{i,j}` denote the distance to its `j`th nearest neighbor (excluding itself). With `k` neighbors:

m_i = (k−1) / Σ_{j=1}^{k−1} log(r_{i,k} / r_{i,j})

The global ID estimate is the mean of `m_i` across valid query points.

---

## k-Sweep for Stability

Because the estimator depends on `k`, we evaluate:k ∈ {5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80}

- Small k → high variance
- Large k → curvature and density bias

Scalar ID values are derived from the plateau region (typically k ≈ 15–40). 
Other Ks are included to confirm the behavior above. 

---

## Sampling Strategy

To control computation:

- Reference set: all N latent vectors
- Query set: 10,000 randomly sampled points
- Fixed random seed so all k values use identical query points

This provides stable estimates without requiring full N×N distance computation.

---

## FAISS Configuration

Approximate kNN is performed using FAISS IVF.

Parameters:

- `nlist = 1024`
- `nprobe = 32`

These settings balance recall and computational efficiency for N ≈ 150k.

---

## Biases and Limitations

The MLE estimator measures effective local dimensionality of the learned representation and is influenced by:

- Finite-sample effects
- Curvature of the manifold
- Density variation
- Representation noise
- Latent collapse
- Approximate neighbor search

Interpretation therefore relies on stability across k and consistency across models.

---

## Outputs

For each `z_dim`:

- CSV file: (k, ID(k), n_used)
- Plot: ID(k) vs k

Additionally:

- Combined table (model × k)
- Overlay plot comparing all models

---

## Extensions

- Multi-seed stability analysis
- Checkpoint-wise ID evolution
- Comparison against reconstruction error
