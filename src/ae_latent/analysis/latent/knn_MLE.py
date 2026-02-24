from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Union

import numpy as np


@dataclass(frozen=True)
class MLEIDResult:
    id_estimate: float
    per_point_ids: np.ndarray  # length N, NaN for non-queried points
    k: int
    n_used: int
    backend: str


def _as_Z(record_or_Z: Union[Dict[str, Any], np.ndarray]) -> np.ndarray:
    if isinstance(record_or_Z, dict):
        if "Z" not in record_or_Z:
            raise KeyError("Expected key 'Z' in record dict.")
        Z = record_or_Z["Z"]
    else:
        Z = record_or_Z

    Z = np.asarray(Z)
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2D (N,d). Got shape {Z.shape}.")
    if Z.shape[0] < 3:
        raise ValueError("Need at least 3 points to estimate ID.")
    return np.ascontiguousarray(Z, dtype=np.float32)  # float32 for FAISS + speed


def _lb_mle_from_knn_dists(dists: np.ndarray, k: int, eps: float = 1e-12) -> np.ndarray:
    """
    Compute per-point Levina–Bickel MLE ID from kNN distances.
    dists: shape (M, k+1) if it includes self at col0, or (M, k) if no self.
    We'll assume caller passes (M, k) distances to the k nearest neighbors EXCLUDING self.
    Returns m_i shape (M,) with NaNs for invalid.
    """
    if dists.ndim != 2 or dists.shape[1] != k:
        raise ValueError(f"Expected dists shape (M,{k}). Got {dists.shape}.")

    # Sort just in case backend doesn’t guarantee it (FAISS generally does)
    dists = np.sort(dists, axis=1)

    r_k = np.maximum(dists[:, k - 1], eps)          # (M,)
    r_j = np.maximum(dists[:, : k - 1], eps)        # (M, k-1)

    logs = np.log(r_k[:, None] / r_j)
    denom = np.sum(logs, axis=1)

    m_i = np.full((dists.shape[0],), np.nan, dtype=np.float64)
    valid = denom > eps
    m_i[valid] = (k - 1) / denom[valid]
    return m_i


def mle_intrinsic_dim_knn_large(
    record_or_Z: Union[Dict[str, Any], np.ndarray],
    k: int = 20,
    *,
    max_points: int = 10_000,
    seed: int = 0,
    backend: str = "faiss",   # "faiss" or "sklearn"
    # FAISS knobs
    approx: bool = True,
    faiss_nlist: int = 4096,   # IVF lists (only used if approx=True)
    faiss_nprobe: int = 16,    # how many IVF lists to probe (bigger => better recall)
    # sklearn knobs
    sklearn_n_jobs: int = -1,
    eps: float = 1e-12,
) -> MLEIDResult:
    """
    Scalable Levina–Bickel kNN MLE ID estimator.

    Strategy:
    - Subsample M query points (max_points) from N to keep runtime sane.
    - Build an index over ALL N reference points (unless you change this yourself).
    - Query k nearest neighbors (excluding self) for those M points.
    - Compute local MLEs and average.

    For N=150k, d=256:
    - FAISS (approx) is the right tool.
    - sklearn brute force over all refs will be expensive; only use for smaller sanity checks.

    Returns per_point_ids length N with NaN for non-queried points.
    """
    Z = _as_Z(record_or_Z)
    N, d = Z.shape
    if not (2 <= k < N):
        raise ValueError(f"k must satisfy 2 <= k < N. Got k={k}, N={N}.")

    rng = np.random.default_rng(seed)
    M = min(int(max_points), N)
    q_idx = rng.choice(N, size=M, replace=False)
    Q = Z[q_idx]

    if backend.lower() == "faiss":
        try:
            import faiss  # type: ignore
        except Exception as e:
            raise ImportError(
                "FAISS not available. Install faiss-cpu or faiss-gpu, or use backend='sklearn'."
            ) from e

        # Build index
        if not approx:
            # Exact search (still can be heavy but feasible-ish with FAISS; memory OK)
            index = faiss.IndexFlatL2(d)
            index.add(Z)
            # Search k+1 then drop self if Q is subset of Z (self will be at distance ~0)
            D2, I = index.search(Q, k + 1)
        else:
            # IVF index: faster, approximate
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, int(faiss_nlist), faiss.METRIC_L2)
            if not index.is_trained:
                # Training on a subset is fine; use up to 100k points
                train_n = min(N, 100_000)
                train_idx = rng.choice(N, size=train_n, replace=False)
                index.train(Z[train_idx])
            index.add(Z)
            index.nprobe = int(faiss_nprobe)
            D2, I = index.search(Q, k + 1)

        # Convert squared L2 distances to L2 distances
        D = np.sqrt(np.maximum(D2, 0.0)).astype(np.float64)

        # Drop self neighbor if present:
        # For queries that come from the reference set, FAISS typically returns the point itself first.
        # Safer: drop the smallest-distance neighbor always (should be self ~0); keep next k.
        D_k = D[:, 1 : k + 1]  # (M,k)

        m_q = _lb_mle_from_knn_dists(D_k, k=k, eps=eps)
        backend_used = "faiss"

    elif backend.lower() == "sklearn":
        from sklearn.neighbors import NearestNeighbors  # type: ignore

        # NOTE: In high-d, trees degrade; brute is usually what sklearn uses effectively.
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", n_jobs=sklearn_n_jobs, metric="euclidean")
        nn.fit(Z)
        D, I = nn.kneighbors(Q, return_distance=True)
        D = D.astype(np.float64)

        # Drop self (same idea: remove first neighbor)
        D_k = D[:, 1 : k + 1]
        m_q = _lb_mle_from_knn_dists(D_k, k=k, eps=eps)
        backend_used = "sklearn"

    else:
        raise ValueError("backend must be 'faiss' or 'sklearn'.")

    id_est = float(np.nanmean(m_q))
    if not np.isfinite(id_est):
        raise RuntimeError(
            "ID estimate became non-finite. Likely causes: too many duplicates/collapsed latents, "
            "k too small, or neighbor distances ~0."
        )

    per_point = np.full((N,), np.nan, dtype=np.float64)
    per_point[q_idx] = m_q
    n_used = int(np.sum(np.isfinite(m_q)))

    return MLEIDResult(
        id_estimate=id_est,
        per_point_ids=per_point,
        k=k,
        n_used=n_used,
        backend=backend_used,
    )


def _as_Z(record_or_Z: Union[Dict[str, Any], np.ndarray]) -> np.ndarray:
    if isinstance(record_or_Z, dict):
        if "Z" not in record_or_Z:
            raise KeyError("Expected key 'Z' in record dict.")
        Z = record_or_Z["Z"]
    else:
        Z = record_or_Z

    Z = np.asarray(Z)
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2D (N,d). Got shape {Z.shape}.")
    if Z.shape[0] < 3:
        raise ValueError("Need at least 3 points to estimate ID.")
    return np.ascontiguousarray(Z, dtype=np.float32)


def _lb_mle_from_knn_dists(dists: np.ndarray, k: int, eps: float = 1e-12) -> np.ndarray:
    """
    dists: (M, k) distances to k nearest neighbors EXCLUDING self
    """
    if dists.ndim != 2 or dists.shape[1] != k:
        raise ValueError(f"Expected dists shape (M,{k}). Got {dists.shape}.")

    dists = np.sort(dists, axis=1)
    r_k = np.maximum(dists[:, k - 1], eps)
    r_j = np.maximum(dists[:, : k - 1], eps)

    denom = np.sum(np.log(r_k[:, None] / r_j), axis=1)
    m = np.full((dists.shape[0],), np.nan, dtype=np.float64)
    valid = denom > eps
    m[valid] = (k - 1) / denom[valid]
    return m


def _knn_dists_fixed_queries(
    Z: np.ndarray,
    q_idx: np.ndarray,
    k: int,
    *,
    backend: str,
    # FAISS
    approx: bool,
    faiss_nlist: int,
    faiss_nprobe: int,
    faiss_train_n: int,
    # sklearn
    sklearn_n_jobs: int,
) -> np.ndarray:
    """
    Returns distances shape (M, k) to k nearest neighbors excluding self.
    """
    N, d = Z.shape
    Q = Z[q_idx]

    if backend.lower() == "faiss":
        import faiss  # type: ignore

        if not approx:
            index = faiss.IndexFlatL2(d)
            index.add(Z)
            D2, _ = index.search(Q, k + 1)
        else:
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, int(faiss_nlist), faiss.METRIC_L2)

            if not index.is_trained:
                rng = np.random.default_rng(0)
                train_n = min(int(faiss_train_n), N)
                train_idx = rng.choice(N, size=train_n, replace=False)
                index.train(Z[train_idx])

            index.add(Z)
            index.nprobe = int(faiss_nprobe)
            D2, _ = index.search(Q, k + 1)

        D = np.sqrt(np.maximum(D2, 0.0)).astype(np.float64)
        # drop nearest (self ~0)
        return D[:, 1 : k + 1]

    elif backend.lower() == "sklearn":
        from sklearn.neighbors import NearestNeighbors  # type: ignore

        nn = NearestNeighbors(
            n_neighbors=k + 1, algorithm="brute", metric="euclidean", n_jobs=sklearn_n_jobs
        )
        nn.fit(Z)
        D, _ = nn.kneighbors(Q, return_distance=True)
        D = D.astype(np.float64)
        return D[:, 1 : k + 1]

    else:
        raise ValueError("backend must be 'faiss' or 'sklearn'.")


def mle_id_stability_sweep(
    record_or_Z: Union[Dict[str, Any], np.ndarray],
    ks: Sequence[int] = (5, 10, 15, 20, 30, 40),
    *,
    max_points: int = 10_000,
    seed: int = 0,
    backend: str = "faiss",   # "faiss" or "sklearn"
    eps: float = 1e-12,
    # FAISS knobs
    approx: bool = True,
    faiss_nlist: int = 4096,
    faiss_nprobe: int = 16,
    faiss_train_n: int = 100_000,
    # sklearn knobs
    sklearn_n_jobs: int = -1,
    # plotting
    plot: bool = True,
):
    """
    Sweep ID estimate across ks using the same query subsample.
    Returns:
      {
        'k': (len(ks),),
        'id_estimate': (len(ks),),
        'n_used': (len(ks),),
        'q_idx': (M,),
        'backend': str,
        'M': int,
        'N': int,
        'd': int
      }
    """
    Z = _as_Z(record_or_Z)
    N, d = Z.shape

    M = min(int(max_points), N)
    rng = np.random.default_rng(seed)
    q_idx = rng.choice(N, size=M, replace=False)

    ks = [int(k) for k in ks]
    ids = np.zeros((len(ks),), dtype=np.float64)
    n_used = np.zeros((len(ks),), dtype=np.int64)

    for i, k in enumerate(ks):
        if not (2 <= k < N):
            raise ValueError(f"k must satisfy 2 <= k < N. Got k={k}, N={N}.")
        D_k = _knn_dists_fixed_queries(
            Z, q_idx, k,
            backend=backend,
            approx=approx,
            faiss_nlist=faiss_nlist,
            faiss_nprobe=faiss_nprobe,
            faiss_train_n=faiss_train_n,
            sklearn_n_jobs=sklearn_n_jobs,
        )
        m_q = _lb_mle_from_knn_dists(D_k, k=k, eps=eps)
        ids[i] = float(np.nanmean(m_q))
        n_used[i] = int(np.sum(np.isfinite(m_q)))

    out = {
        "k": np.array(ks, dtype=int),
        "id_estimate": ids,
        "n_used": n_used,
        "q_idx": q_idx,
        "backend": backend,
        "M": M,
        "N": N,
        "d": d,
    }

    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(out["k"], out["id_estimate"], marker="o")
        plt.xlabel("k (neighbors)")
        plt.ylabel("Estimated intrinsic dimension (MLE)")
        plt.title(f"ID stability sweep (backend={backend}, M={M}, N={N}, d={d})")
        plt.show()

    return out