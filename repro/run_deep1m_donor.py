"""DEEP 1M, Table 3 configuration: Meta's mass rule with only the donor changed to largest SSE."""
import sys, time, json
import numpy as np, faiss
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from emeans_plus import NumpyEmeansPlus
faiss.omp_set_num_threads(8)
N, K, D = 10**6, 16384, 96
raw = np.memmap("base.1B.fbin.part", dtype="float32", mode="r", offset=8, shape=(N + 10000, D))
train = np.ascontiguousarray(raw[:N]); test = np.ascontiguousarray(raw[N:])
def error(x, c):
    _, lab = faiss.knn(x, c, 1)
    diff = x.astype("float64") - c[lab[:, 0]].astype("float64")
    return float((diff * diff).sum() / len(x)), np.bincount(lab[:, 0], minlength=len(c))
out = []
for epochs in (3, 5):
    for seed in (10, 11, 12):
        init = train[np.random.RandomState(seed).permutation(N)[:K]].copy()
        t0 = time.time()
        m = NumpyEmeansPlus(D, K, nepochs=epochs, seed=seed, separation=False, shadow=False,
                            warmup_rows_per_centroid=0, scaled_split=False, donor_by_sse=True)
        c, _ = m.train(train, init_centroids=init)
        secs = time.time() - t0
        te, _ = error(test, c); tr, sizes = error(train, c)
        rec = dict(seed=seed, epochs=epochs, rule="mass + donor SSE", test_mse=te, train_mse=tr, splits=m.n_splits,
                   tiny=int((sizes < 10).sum()), imbalance=float(K * (sizes.astype(float) ** 2).sum() / N ** 2), seconds=secs)
        out.append(rec); print(json.dumps(rec), flush=True)
        json.dump(out, open("deep1m_donor_results.json", "w"), indent=1)
