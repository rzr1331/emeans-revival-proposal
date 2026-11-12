"""Reproduce the revival findings on Meta's NumpyEmeans and evaluate the proposed patch.

Variants, all running Meta's code path (float32, faiss.knn assignment):
  reference                 NumpyEmeans, defaults
  revival off               NumpyEmeans(unused_threshold=0)
  separation (3 changes)    NumpyEmeansPlus with the three proposed changes (the PR's rule)
  ... + warm-up + scaled    the same plus two optional changes
  absolute threshold        NumpyEmeans(unused_threshold=10 k / n), with and without donor by SSE

Every variant gets the same rows, the same initial centroids and the same seed. The baseline is
Lloyd's algorithm run to convergence from the same initial centroids. All MSEs are training MSE on the
full data, after reassigning every row to the final centroids. E-Means runs use the default T=5 epochs.

    python repro/run.py [--workers 4] [--csv path/to/extra_data.csv]
"""
import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "repro"))

from multiprocessing import Pool

import faiss
import numpy as np

from common import assign, far_rare, kmeanspp_init, lloyd, mixture, mse, random_init, unstructured
from emeans_plus import NumpyEmeans, NumpyEmeansPlus

faiss.omp_set_num_threads(1)
S = 7000
Z = dict(zipf=1.5)
# (group, name, maker, k, start, seeds)
SCEN = [
    ("good", "far rare cluster, 0.3% of rows", lambda s: far_rare(S + s, share=0.003), 8, "true centres", 10),
    ("good", "far rare cluster, 0.6%", lambda s: far_rare(S + s, share=0.006), 8, "true centres", 10),
    ("good", "far rare cluster, 1.25%", lambda s: far_rare(S + s, share=0.0125), 8, "true centres", 10),
    ("good", "far rare cluster, 2.5%", lambda s: far_rare(S + s, share=0.025), 8, "true centres", 10),
    ("good", "far rare cluster, 0.6%, n=200k", lambda s: far_rare(S + s, n=200000), 8, "true centres", 5),
    ("good", "Zipf sizes a=1.2, k=30, n=20k", lambda s: mixture(S + 100 + s, 20000, 30, zipf=1.2), 30, "true centres", 10),
    ("good", "Zipf sizes a=1.2, k=30, n=20k", lambda s: mixture(S + 100 + s, 20000, 30, zipf=1.2), 30, "k-means++", 10),
    ("good", "Zipf sizes a=1.5, k=40, n=40k", lambda s: mixture(S + 200 + s, 40000, 40, **Z), 40, "true centres", 10),
    ("good", "Zipf a=1.5, unequal spreads", lambda s: mixture(S + 300 + s, 40000, 40, spread="unequal", **Z), 40, "k-means++", 10),
    ("good", "Zipf a=1.2, elongated clusters", lambda s: mixture(S + 400 + s, 30000, 30, zipf=1.2, shape="aniso"), 30, "k-means++", 10),
    ("good", "Zipf a=1.5, heavy tails (t3)", lambda s: mixture(S + 500 + s, 40000, 40, tails="t3", **Z), 40, "k-means++", 10),
    ("good", "batch < k: n=2000, k=50", lambda s: mixture(S + 600 + s, 2000, 50, box=20.0), 50, "true centres", 10),
    ("good", "d=32, k=256, n=100k, Zipf a=1.2", lambda s: mixture(S + 700 + s, 100000, 256, d=32, zipf=1.2, box=10.0), 256, "k-means++", 3),
    ("good", "1% scattered outliers, k=30", lambda s: mixture(S + 800 + s, 20000, 30, outliers=0.01), 30, "k-means++", 10),
    ("good", "overlapping clusters (box 4), Zipf", lambda s: mixture(S + 900 + s, 30000, 30, zipf=1.2, box=4.0), 30, "k-means++", 10),
    ("random", "balanced, k=30, n=20k", lambda s: mixture(S + 1000 + s, 20000, 30), 30, "random rows", 15),
    ("random", "balanced, k=60, n=30k, unequal spreads", lambda s: mixture(S + 1100 + s, 30000, 60, spread="unequal"), 60, "random rows", 15),
    ("random", "balanced, k=40, heavy tails (t3)", lambda s: mixture(S + 1200 + s, 30000, 40, tails="t3"), 40, "random rows", 15),
    ("random", "Zipf a=1.5, k=40, n=40k", lambda s: mixture(S + 200 + s, 40000, 40, **Z), 40, "random rows", 10),
    ("random", "batch < k: n=2000, k=50", lambda s: mixture(S + 600 + s, 2000, 50, box=20.0), 50, "random rows", 10),
    ("random", "1% scattered outliers, k=30", lambda s: mixture(S + 800 + s, 20000, 30, outliers=0.01), 30, "random rows", 10),
    ("random", "overlapping clusters (box 4), Zipf", lambda s: mixture(S + 900 + s, 30000, 30, zipf=1.2, box=4.0), 30, "random rows", 10),
    ("random", "d=32, k=256, n=100k, Zipf a=1.2", lambda s: mixture(S + 700 + s, 100000, 256, d=32, zipf=1.2, box=10.0), 256, "random rows", 3),
    ("flat", "unstructured Gaussian, d=16, k=512, n=100k", lambda s: unstructured(S + 1300 + s, 100000, 16), 512, "random rows", 5),
    ("flat", "unstructured Gaussian, d=32, k=1024, n=100k", lambda s: unstructured(S + 1400 + s, 100000, 32), 1024, "random rows", 5),
]
CORE = dict(warmup_rows_per_centroid=0, scaled_split=False)
VARIANTS = {
    "reference": lambda d, k, n, s: NumpyEmeans(d, k, seed=s),
    "revival off": lambda d, k, n, s: NumpyEmeans(d, k, seed=s, unused_threshold=0.0),
    "separation (3 changes)": lambda d, k, n, s: NumpyEmeansPlus(d, k, seed=s, **CORE),
    "separation + warm-up + scaled split": lambda d, k, n, s: NumpyEmeansPlus(d, k, seed=s),
    # switch: separation test only when the mass threshold is at least 10 rows (n/k >= 100)
    "auto (separation if threshold >= 10 rows)": lambda d, k, n, s: NumpyEmeansPlus(d, k, seed=s, min_rows=10, **CORE),
    # simpler alternatives: "dead" = fewer than 10 rows, i.e. unused_threshold = 10 k / n
    "absolute threshold (10 rows)": lambda d, k, n, s: NumpyEmeans(d, k, seed=s, unused_threshold=10.0 * k / n),
    "absolute threshold + donor by SSE": lambda d, k, n, s: NumpyEmeansPlus(
        d, k, seed=s, unused_threshold=10.0 * k / n, separation=False, shadow=False, **CORE),
}
MAIN = "separation (3 changes)"
ABLATION = {
    "all five": {},
    "no warm-up": dict(warmup_rows_per_centroid=0),
    "no separation rule": dict(separation=False),
    "no shadow test": dict(shadow=False),
    "donor by mass": dict(donor_by_sse=False),
    "multiplicative split": dict(scaled_split=False),
}
ABL_SCEN = ["far rare cluster, 0.3% of rows|true centres", "Zipf sizes a=1.5, k=40, n=40k|true centres",
            "batch < k: n=2000, k=50|true centres", "balanced, k=30, n=20k|random rows"]
CSV = None


def load_csv(path, log1p):
    import pandas as pd
    df = pd.read_csv(path).select_dtypes("number")
    df = df.fillna(df.median())
    if log1p:                       # log1p every non-negative column with skewness > 2
        cols = [c for c in df if df[c].min() >= 0 and df[c].skew() > 2]
        df[cols] = np.log1p(df[cols])
    X = df.to_numpy(dtype=np.float64)
    return (X - X.mean(0)) / X.std(0)


def job(a):
    sid, s = a
    group, name, make, k, start, _ = SCEN[sid]
    X, centers = make(s)
    X32 = np.ascontiguousarray(X, dtype=np.float32)
    X = X32.astype(np.float64)
    init = {"true centres": lambda: centers.copy(), "k-means++": lambda: kmeanspp_init(X, k, s),
            "random rows": lambda: random_init(X, k, s)}[start]().astype(np.float32)
    linfo = {}
    base = None if group == "flat" else mse(X, lloyd(X, init.astype(np.float64), info=linfo))
    makers = dict(VARIANTS)
    if f"{name}|{start}" in ABL_SCEN:
        for an, kw in ABLATION.items():
            makers["abl:" + an] = (lambda d, k, n, s, kw=kw: NumpyEmeansPlus(d, k, seed=s, **kw))
    out = {"_lloyd": linfo}
    for v, mk in makers.items():
        m = mk(X.shape[1], k, len(X), s)
        c, _ = m.train(X32, init_centroids=init)
        lab, dist = assign(X, c.astype(np.float64))
        out[v] = dict(m=float(dist.mean()), r=None if base is None else float(dist.mean() / base),
                      splits=m.n_splits, tiny=int((np.bincount(lab, minlength=k) < 10).sum()))
    return sid, s, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--csv", help="optional numeric CSV to add as real data (median-imputed, z-scored; "
                                  "run twice: as is, and with log1p on skewed columns)")
    ap.add_argument("--quick", action="store_true", help="3 seeds per setting, skip n >= 100k")
    args = ap.parse_args()
    if args.csv:
        for tag, lg in (("z-scored", False), ("log1p + z-scored", True)):
            X = load_csv(args.csv, lg)
            for k in (8, 50):
                for start in ("k-means++", "random rows"):
                    SCEN.append(("real", f"{os.path.basename(args.csv)} ({tag}), k={k}", lambda s, X=X: (X, None),
                                 k, start, 10))
    if args.quick:
        SCEN[:] = [sc[:5] + (min(sc[5], 3),) for sc in SCEN if "100k" not in sc[1] and "200k" not in sc[1]]
    jobs = [(sid, s) for sid, sc in enumerate(SCEN) for s in range(sc[5])]
    with Pool(args.workers) as p:
        res = p.map(job, jobs)
    by = {}
    for sid, s, out in res:
        by.setdefault(sid, []).append(out)
    names = list(VARIANTS)
    import json
    lloyd_runs = [o["_lloyd"] for sid, s, o in res if o["_lloyd"]]
    with open(os.path.join(ROOT, "results", "runs_quick.json" if args.quick else "runs.json"), "w") as f:
        json.dump([dict(data=SCEN[sid][1], start=SCEN[sid][4], group=SCEN[sid][0], seed=s, **o)
                   for sid, s, o in res], f)
    L = ["# Results\n", "Meta's `NumpyEmeans` (float32, unmodified) vs the proposed patch. Same rows, initial centroids "
         "and seed for every variant. Baseline: Lloyd to convergence from the same initial centroids.\n",
         f"Lloyd baselines: {len(lloyd_runs)}, of which {sum(l['converged'] for l in lloyd_runs)} converged "
         f"within 300 iterations (max {max(l['iterations'] for l in lloyd_runs)}). Per-seed records: "
         "`results/runs.json`.\n"]

    def fails(sid, v):
        r = np.array([o[v]["r"] for o in by[sid]])
        return f"{(r >= 1.2).sum()}/{len(r)}, {np.median(r):.2f}x, {np.mean([o[v]['splits'] for o in by[sid]]):.1f}"

    def paired(sid, v, ref="reference"):
        d = np.array([o[v]["m"] / o[ref]["m"] - 1 for o in by[sid]])
        return f"{100 * d.mean():+.1f}% +/- {100 * d.std(ddof=1) / np.sqrt(len(d)):.1f}"

    L.append("## 1. Good starts (true centres or k-means++)\n")
    L.append("Cell: runs with MSE >= 1.2x converged Lloyd / seeds, median MSE / Lloyd, mean splits per run.\n")
    L.append("| data | start | " + " | ".join(names) + " |\n|---|---|" + "---|" * len(names))
    pooled = {st: {v: [0, 0] for v in names} for st in ("true centres", "k-means++")}
    same = [0, 0]      # E-Means+ failures, and how many of them also fail with revival off (same seed)
    for sid, sc in enumerate(SCEN):
        if sc[0] in ("good", "real") and sc[4] != "random rows":
            for v in names:
                r = np.array([o[v]["r"] for o in by[sid]])
                pooled[sc[4]][v][0] += int((r >= 1.2).sum()); pooled[sc[4]][v][1] += len(r)
            for o in by[sid]:
                if o[MAIN]["r"] >= 1.2:
                    same[0] += 1
                    same[1] += o["revival off"]["r"] >= 1.2
            L.append(f"| {sc[1]} | {sc[4]} | " + " | ".join(fails(sid, v) for v in names) + " |")
    for st in pooled:
        L.append(f"| **all {st} runs** | | " + " | ".join(f"**{a}/{b} = {100 * a / b:.0f}%**" for a, b in pooled[st].values()) + " |")
    L.append(f"\n'{MAIN}' runs at >= 1.2x Lloyd: {same[0]}; of these, also >= 1.2x with revival off on the same seed: {same[1]}.")
    L.append("\n## 2. Random-row starts (the paper's default)\n")
    L.append("Cell: mean paired MSE change vs the reference +/- standard error; median MSE / Lloyd in brackets.\n")
    L.append("| data | seeds | reference | " + " | ".join(names[1:]) + " |\n|---|---|---|" + "---|" * (len(names) - 1))
    for sid, sc in enumerate(SCEN):
        if sc[4] == "random rows" and sc[0] != "flat":
            med = lambda v: f"[{np.median([o[v]['r'] for o in by[sid]]):.2f}x]"
            L.append(f"| {sc[1]} | {len(by[sid])} | — {med('reference')} | "
                     + " | ".join(f"{paired(sid, v)} {med(v)}" for v in names[1:]) + " |")
    L.append("\n## 3. Unstructured data (no Lloyd baseline)\n")
    L.append("Cell: mean paired MSE change vs the reference +/- standard error; mean number of final clusters "
             "with < 10 rows in brackets.\n")
    L.append("| data | seeds | reference | " + " | ".join(names[1:]) + " |\n|---|---|---|" + "---|" * (len(names) - 1))
    for sid, sc in enumerate(SCEN):
        if sc[0] == "flat":
            t = lambda v: f"[{np.mean([o[v]['tiny'] for o in by[sid]]):.1f}]"
            L.append(f"| {sc[1]} | {len(by[sid])} | — {t('reference')} | "
                     + " | ".join(f"{paired(sid, v)} {t(v)}" for v in names[1:]) + " |")
    L.append("\n## 4. Ablation: all five changes with one switched off\n")
    L.append("Good-start rows: failures / seeds, median MSE / Lloyd. Random-start row: mean paired MSE change vs "
             "the reference.\n")
    L.append("| data | start | " + " | ".join(ABLATION) + " |\n|---|---|" + "---|" * len(ABLATION))
    for sid, sc in enumerate(SCEN):
        if f"{sc[1]}|{sc[4]}" in ABL_SCEN:
            if sc[4] == "random rows":
                cells = [paired(sid, "abl:" + a) for a in ABLATION]
            else:
                cells = []
                for a in ABLATION:
                    r = np.array([o["abl:" + a]["r"] for o in by[sid]])
                    cells.append(f"{(r >= 1.2).sum()}/{len(r)}, {np.median(r):.2f}x")
            L.append(f"| {sc[1]} | {sc[4]} | " + " | ".join(cells) + " |")
    text = "\n".join(L) + "\n"
    with open(os.path.join(ROOT, "results", "results_quick.md" if args.quick else "results.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
