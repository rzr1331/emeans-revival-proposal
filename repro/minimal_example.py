"""Smallest reproduction: 7 equal clusters plus one small distant cluster, started from the true
centres. Needs only numpy and faiss.contrib.e_means (or the vendored copy in third_party/)."""
import os, sys
import numpy as np
try:
    from faiss.contrib.e_means import NumpyEmeans
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party"))
    from e_means_ref.e_means import NumpyEmeans

rng = np.random.RandomState(0)
d, n, small = 8, 20000, 120                       # the small cluster holds 0.6% of the rows
centers = rng.uniform(-10, 10, (7, d))
v = rng.randn(d)
centers = np.vstack([centers, 40 * v / np.linalg.norm(v)]).astype("float32")
sizes = [(n - small) // 7] * 7 + [small]
x = np.concatenate([c + rng.randn(m, d) for c, m in zip(centers, sizes)]).astype("float32")


def mse(c):
    dist = (x * x).sum(1)[:, None] - 2.0 * x @ c.T + (c * c).sum(1)[None, :]
    return float(dist.min(1).mean())


print("MSE at the true centres:            %.3f" % mse(centers))
for name, kw in (("NumpyEmeans, defaults", {}), ("NumpyEmeans, unused_threshold=0", dict(unused_threshold=0.0))):
    m = NumpyEmeans(d, 8, seed=0, **kw)
    c, _ = m.train(x, init_centroids=centers)
    gap = np.linalg.norm(c - centers[7], axis=1).min()
    print("%-34s  MSE %.3f, splits %d, nearest centroid to the small cluster: %.1f away" % (name, mse(c), m.n_splits, gap))
