"""DEEP 1M: the combined switch (separation test only when the mass threshold >= 10 rows; here it
is 6.1 rows, so seen centroids follow the mass rule), with the shadow test and donors by SSE."""
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
        m = NumpyEmeansPlus(D, K, nepochs=epochs, seed=seed, min_rows=10, warmup_rows_per_centroid=0, scaled_split=False)
        c, _ = m.train(train, init_centroids=init)
        secs = time.time() - t0
        te, _ = error(test, c); tr, sizes = error(train, c)
        rec = dict(seed=seed, epochs=epochs, rule="auto", active=bool(m._separation_active), test_mse=te, train_mse=tr, splits=m.n_splits,
                   tiny=int((sizes < 10).sum()), imbalance=float(K * (sizes.astype(float) ** 2).sum() / N ** 2), seconds=secs)
        out.append(rec); print(json.dumps(rec), flush=True)
        json.dump(out, open("deep1m_auto_results.json", "w"), indent=1)
