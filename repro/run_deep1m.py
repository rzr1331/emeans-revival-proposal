"""DEEP 1M in the paper's Table 3 configuration (k=16384, test = the next 10,000 rows, seeds 10+),
run on CPU with the NumPy reference class from the local faiss branch: revival="mass" (the
paper's rule) against revival="separation". Differences from the paper's run: float32 NumpyEmeans
with faiss.knn instead of the GPU BF16 kernels, and NumPy random-row starts instead of the CUDA
generator, so absolute numbers can differ slightly from Table 3."""
import os, sys, time, json
import numpy as np, faiss
from faiss.contrib.e_means import NumpyEmeans
faiss.omp_set_num_threads(8)
N, K, D = 10**6, 16384, 96
raw = np.memmap("base.1B.fbin.part", dtype="float32", mode="r", offset=8, shape=(N + 10000, D))
train = np.ascontiguousarray(raw[:N]); test = np.ascontiguousarray(raw[N:])

def error(x, c):
    _, lab = faiss.knn(x, c, 1)
    diff = x.astype("float64") - c[lab[:, 0]].astype("float64")
    return float((diff * diff).sum() / len(x)), np.bincount(lab[:, 0], minlength=len(c))

out = []
for seed in (10, 11, 12):
    init = train[np.random.RandomState(seed).permutation(N)[:K]].copy()
    for epochs in (3, 5):
        for rule in ("mass", "separation"):
            t0 = time.time()
            m = NumpyEmeans(D, K, nepochs=epochs, seed=seed, revival=rule)
            c, _ = m.train(train, init_centroids=init)
            secs = time.time() - t0
            te, _ = error(test, c); tr, sizes = error(train, c)
            rec = dict(seed=seed, epochs=epochs, rule=rule, test_mse=te, train_mse=tr, splits=m.n_splits,
                       tiny=int((sizes < 10).sum()), imbalance=float(K * (sizes.astype(float) ** 2).sum() / N ** 2),
                       seconds=secs, batch=m.recipe["batch_size"])
            out.append(rec); print(json.dumps(rec), flush=True)
            json.dump(out, open("deep1m_results.json", "w"), indent=1)
