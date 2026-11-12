"""Data generators and a converged Lloyd baseline for the reproduction (NumPy only)."""
import numpy as np


def sq_dists(x, c):
    return np.maximum((x * x).sum(1)[:, None] - 2.0 * x @ c.T + (c * c).sum(1)[None, :], 0.0)


def assign(x, c, chunk=16384):
    lab, dist = [], []
    for i in range(0, len(x), chunk):
        d = sq_dists(x[i:i + chunk], c)
        j = d.argmin(1)
        lab.append(j)
        dist.append(d[np.arange(len(j)), j])
    return np.concatenate(lab), np.concatenate(dist)


def mse(x, c):
    return float(assign(x, c)[1].mean())


def lloyd(x, init, max_iter=300, info=None):
    """Lloyd's algorithm to convergence (stops when no label changes). If `info` is a dict it
    receives `converged` (False when max_iter was reached first) and `iterations`."""
    c = np.array(init, dtype=np.float64, copy=True)
    k = len(c)
    labels, dist = assign(x, c)
    for _ in range(max_iter):
        counts = np.bincount(labels, minlength=k)
        sums = np.zeros_like(c)
        np.add.at(sums, labels, x)
        c[counts > 0] = sums[counts > 0] / counts[counts > 0, None]
        for j in np.flatnonzero(counts == 0):       # empty cluster: farthest point
            far = dist.argmax()
            c[j], dist[far] = x[far], 0.0
        new, dist = assign(x, c)
        if not (new != labels).any():
            if info is not None:
                info.update(converged=True, iterations=_ + 1)
            break
        labels = new
    else:
        if info is not None:
            info.update(converged=False, iterations=max_iter)
    return c


def random_init(x, k, seed):
    return x[np.random.RandomState(seed).permutation(len(x))[:k]].copy()


def kmeanspp_init(x, k, seed):
    rng = np.random.RandomState(seed)
    c = [x[rng.randint(len(x))]]
    d = sq_dists(x, np.array(c))[:, 0]
    for _ in range(1, k):
        c.append(x[rng.choice(len(x), p=d / d.sum())])
        d = np.minimum(d, sq_dists(x, c[-1][None])[:, 0])
    return np.array(c)


def zipf_sizes(n, k, a):
    w = 1 / np.arange(1, k + 1) ** a
    return np.maximum(1, np.round(n * w / w.sum())).astype(int)


def mixture(seed, n, k, d=8, zipf=None, spread="equal", shape="iso", tails=None, box=25.0, shift=0.0,
            outliers=0.0):
    """Gaussian mixture: centres uniform in [-box, box]^d; sizes equal or Zipf (1/rank^zipf);
    spread 'equal' (std 1) or 'unequal' (std log-uniform in [0.5, 2]); shape 'iso' or 'aniso'
    (random rotation, axis stds log-uniform in [0.5, 3]); tails None or 't3' (Student-t, 3 d.o.f.,
    unit variance); outliers = fraction of rows replaced by uniform points in [-3 box, 3 box]^d."""
    rng = np.random.RandomState(seed)
    centers = rng.uniform(-box, box, (k, d))
    m = int(outliers * n)
    sizes = zipf_sizes(n - m, k, zipf) if zipf else np.full(k, (n - m) // k)
    stds = np.exp(rng.uniform(np.log(0.5), np.log(2.0), k)) if spread == "unequal" else np.ones(k)
    parts = []
    for j in range(k):
        z = rng.standard_t(3, size=(sizes[j], d)) / np.sqrt(3.0) if tails == "t3" else rng.randn(sizes[j], d)
        if shape == "aniso":
            q, _ = np.linalg.qr(rng.randn(d, d))
            z = (z * np.exp(rng.uniform(np.log(0.5), np.log(3.0), d))) @ q.T
        parts.append(centers[j] + stds[j] * z)
    if m:
        parts.append(rng.uniform(-3 * box, 3 * box, (m, d)))
    return np.concatenate(parts) + shift, centers + shift


def far_rare(seed, n=20000, share=0.006, d=8):
    """Seven equal clusters with centres in [-10, 10]^d plus one rare cluster 40 from the origin."""
    rng = np.random.RandomState(seed)
    centers = rng.uniform(-10, 10, (7, d))
    v = rng.randn(d)
    centers = np.vstack([centers, 40 * v / np.linalg.norm(v)])
    m = int(round(share * n))
    sizes = [(n - m) // 7] * 7 + [m]
    return np.concatenate([c + rng.randn(sz, d) for c, sz in zip(centers, sizes)]), centers


def unstructured(seed, n, d):
    return np.random.RandomState(seed).randn(n, d), None
