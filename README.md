# E-Means dead-cluster revival overwrites real small clusters when k is small

A reproducible report on the dead-cluster revival step in E-Means (Bessa et al., "e-means:
outperforming k-means (in one-tenth the epochs)", arXiv:2611.09501; `faiss/contrib/e_means`),
with a candidate opt-in rule, evaluated on the unmodified reference code.

Everything runs Meta's `NumpyEmeans` (float32, `faiss.knn` assignment). The reference files are
vendored unmodified in `third_party/` (faiss commit `2b648f06`, MIT licence).

## The problem in one example

`python repro/minimal_example.py`: seven equal clusters plus one distant cluster holding 0.6% of
20,000 rows, k=8, started from the true centres.

| | training MSE | splits | nearest centroid to the small cluster |
|---|---|---|---|
| true centres | 7.960 | | |
| `NumpyEmeans`, defaults | 15.801 | 1 | 35.6 away |
| `NumpyEmeans(unused_threshold=0)` | 7.957 | 0 | 0.2 away |

Revival treats a centroid with mass below 0.1x the mean as dead. With k=8 that is 250 rows, so the
120-row cluster is below it for the whole run and its centroid is moved.

## Scope

- The paper targets n = 1M-1B with k between 4*sqrt(n) and 32*sqrt(n) on embedding data, and its
  Table 14 shows revival helping there. This report is about a different regime: few, real
  clusters (k = 8-256) of unequal size. The paper's Appendix D already covers high n/k and Zipf
  mixtures on MSE; what is new here is the destructive effect on small clusters at low k.
- faiss PR #5854 (9 Nov 2026) switched revival off for d <= 2. The cases here are at d = 8-32 and
  depend on cluster structure, not dimension. As of 11 Nov 2026 we found no faiss issue on this.
- The settings were built to expose the mechanism (well-separated components, a rare component
  near the threshold, k equal to the number of components). They are regression cases; the pooled
  rates below do not estimate how often this happens on data in general.
- Measure: training MSE after reassigning every row to the final centroids, relative to Lloyd's
  algorithm from the same initial centroids (a local baseline, not ground truth: 305 of 306
  baselines converged within 300 iterations, and several methods beat it). E-Means at T=5.
  Held-out error and recovery of the minority component were not measured here.

## Findings on the reference code

| start | runs | ended at >= 1.2x Lloyd: defaults | with `unused_threshold=0` |
|---|---|---|---|
| k-means++, synthetic (7 settings) | 63 | 37 | 1 |
| true centres (8 settings; Lloyd stays near the supplied solution) | 75 | 66 | 0 |
| k-means++, real dataset (4 settings) | 40 | 0 | 0 |

Mechanisms, each with its rows in `results/results.md`:

1. **The threshold is relative to the mean mass.** A cluster with less than about 0.1/k of the
   rows is below it permanently (far cluster with <= 1.25% of rows, k=8: 10/10 runs).
2. **The mass is an EMA over a few batches, tested every 5 batches.** A sparse cluster above the
   nominal threshold still dips below it by chance (Zipf sizes, smallest cluster 0.55% against a
   0.33% threshold: 10/10 from true centres, 8/10 from k-means++).
3. **The first check comes after 5 batches,** before a small cluster has necessarily received a
   row; with batch < k many centroids are overwritten at once (n=2000, k=50: 10/10).
4. **Donors are drawn by mass.** On clustered data the centroid worth splitting covers two
   clusters and has about twice the mean mass, so it is rarely drawn (see the donor ablation).

Revival still matters from random-row starts: switching it off costs 15-91% on five clustered
settings and is within noise on three others (+2.0% +/- 4.2, +1.4% +/- 1.7, -0.1% +/- 0.3).

Real data (Kaggle "Credit Card Dataset for Clustering", 8,950 rows, not included): z-scored, the
defaults end 4% (k=8) and 9% (k=50) above Lloyd and `unused_threshold=0` at 0.99-1.00x; with
log1p on skewed columns first, all variants agree. Keeping extreme rows as their own clusters
lowers training MSE but is not evidence that they are meaningful groups.

## Candidate rule

`emeans_plus.py` subclasses `NumpyEmeans` and changes only the revival path. The same three
changes are implemented in `NumpyEmeans` behind `revival="separation"` on a branch (not yet
public); `results/results.md` column "separation (3 changes)".

1. **Separation test.** A low-mass centroid is revived only if the mean over its rows of
   `d2(second nearest) - d2(nearest)` is below the global mean within-cluster squared distance.
   Both are EMAs with the decay of the masses, so the ratio does not move with batch counts.
2. **Shadow test.** A centroid with no row yet is revived at once only if another centroid is
   that close; otherwise it waits (dead after a full pass, or half the run below two passes).
3. **Donor by SSE.** Split the clusters with the largest within-cluster squared error.

Costs: the two nearest centroids of every training row; at each revival check a nearest-centroid
search for every centroid that has no row yet (up to k searches early in a run, so O(k^2 d) in
the worst case); five float64 arrays and one boolean array of size k. Statistics must come from
`accumulate`. Rows are assumed shuffled.

## Results

Good starts, runs ending at >= 1.2x Lloyd (`results/results.md`, section 1):

| start | defaults | revival off | separation | absolute threshold (10 rows) | absolute + donor by SSE |
|---|---|---|---|---|---|
| k-means++, synthetic, 63 runs | 37 | 1 | 1 | 4 | 1 |
| true centres, 75 runs | 66 | 0 | 0 | 25 | 11 |

The one failing separation run is the same seed that fails with revival off.

Random-row starts, mean paired training-MSE change vs the defaults (+/- standard error):

| data | revival off | separation | absolute + donor by SSE |
|---|---|---|---|
| balanced, k=30 | +23.5% +/- 6.1 | -43.1% +/- 6.2 | -8.6% +/- 6.5 |
| balanced, k=60, unequal spreads | +15.4% +/- 3.0 | -45.0% +/- 3.4 | -19.5% +/- 3.8 |
| balanced, k=40, heavy tails | +48.4% +/- 15.5 | -65.5% +/- 3.8 | -42.7% +/- 6.4 |
| Zipf a=1.5, k=40 | +2.0% +/- 4.2 | -9.7% +/- 3.6 | -4.2% +/- 3.8 |
| batch < k: n=2000, k=50 | +91.1% +/- 18.8 | -68.3% +/- 4.0 | -82.9% +/- 1.9 |
| d=32, k=256, n=100k (3 seeds) | +1.4% +/- 1.7 | -17.6% +/- 0.4 | -17.4% +/- 0.9 |
| 1% scattered outliers, k=30 | +89.9% +/- 6.4 | -3.2% +/- 6.2 | -12.5% +/- 3.8 |
| overlapping clusters | -0.1% +/- 0.3 | -0.6% +/- 0.4 | -0.4% +/- 0.4 |
| unstructured Gaussian, d=32, k=1024, n=100k | +1.1% +/- 0.1 | +0.1% +/- 0.0 | +0.0% +/- 0.0 |

The absolute threshold (`unused_threshold = 10 * k / n`, "dead" = fewer than 10 rows) needs no
new code but still loses clusters to chance dips. With donor by SSE it is the simplest candidate
and is better with outliers and with batch < k; the separation rule protects small clusters best
and gains more on the three balanced settings.

## On DEEP 1M the candidate rule is slightly worse

n=1M, k=16,384, test error on the next 10,000 rows (the faiss benchmark's `deep1b-1M` data and k;
the fixed-budget comparison of the paper's Table 5), seeds 10-12, CPU `NumpyEmeans` (float32,
NumPy random-row starts; the paper uses BF16 GPU kernels and a CUDA generator, and its held-out
split is not verified to be the same rows). `repro/run_deep1m.py` needs a faiss whose `contrib.e_means` has the `revival` option (the branch),
and all three DEEP scripts expect the 388 MB `base.1B.fbin.part` prefix in the working directory.

| passes | test MSE, defaults | test MSE, separation | difference (range over seeds) | clusters < 10 rows | seconds |
|---|---|---|---|---|---|
| 3 | 0.38716 | 0.38776 | +0.15% (+0.07% to +0.21%) | 28 vs 466 | 251 vs 247 |
| 5 | 0.38635 | 0.38691 | +0.14% (+0.11% to +0.19%) | 10 vs 119 | 374 vs 371 |

Changing only the donor (mass rule kept, donors by SSE; `repro/run_deep1m_donor.py`,
`results/deep1m_donor_results.json`), against the defaults on the same seeds:

| passes | test MSE | training MSE | clusters < 10 rows | size imbalance |
|---|---|---|---|---|
| 3 | -0.04% (-0.08% to +0.01%) | -0.18% (-0.19% to -0.16%) | 28 vs 28 | 1.212 vs 1.188 |
| 5 | -0.01% (-0.04% to +0.01%) | -0.16% (-0.17% to -0.15%) | 13 vs 10 | 1.211 vs 1.178 |

Training MSE is lower in all six pairs; test MSE is level (lower in four, higher in two, all
within 0.08%); cluster sizes are slightly less even. Three seeds, so the test difference is not
established.

The separation rule is worse in all six pairs, with similar measured runtime (defaults always ran first). At about 61
rows per cluster the mass threshold is about 6 rows, and the rule shows no benefit there. It is
therefore proposed as an opt-in, with the default unchanged.

## A switch that is level on DEEP 1M and keeps the gains at low k

`min_rows=10` in `emeans_plus.py` applies the separation test only when the mass threshold is at
least 10 rows (0.1 n/k >= 10, i.e. 100 or more rows per cluster); below that, every low-mass
centroid holds fewer than 10 rows (the paper's definition of a dead cluster) and the mass rule
applies. The shadow test and donors by SSE stay on. The cutoff is that definition, not a tuned
value, and it has only been tried on the cases below. `results/results.md`, column "auto";
`repro/run_deep1m_auto.py`, `results/deep1m_auto_results.json`.

| | defaults | separation | switch |
|---|---|---|---|
| good starts ending >= 1.2x Lloyd: k-means++ (63 runs) / true centres (75) | 37 / 66 | 1 / 0 | 1 / 0 |
| random-row starts, the nine settings above | — | as tabled | same to 0.1 point |
| unstructured d=32, k=1024: clusters < 10 rows | 0.0 | 2.4 | 0.0 |
| DEEP 1M test MSE, 3 passes (3 seeds) | 0.38716 | +0.15% | +0.00% (-0.09% to +0.07%) |
| DEEP 1M test MSE, 5 passes | 0.38635 | +0.14% | +0.02% (-0.06% to +0.08%) |
| DEEP 1M training MSE | — | +0.22% / +0.03% | -0.15% / -0.14% |
| DEEP 1M clusters < 10 rows, 3 / 5 passes | 28 / 10 | 466 / 119 | 27 / 11 |

On DEEP 1M the switch is level with the defaults on test MSE (three wins, three losses), slightly
lower on training MSE, with slightly less even cluster sizes (imbalance 1.21 vs 1.18). It is not
an improvement on the paper's benchmark. It is in the branch as `revival="auto"` (bit-identical to this implementation on 21 runs).

## Other limits

- Not tried with the GPU, quantised, distributed or IVF paths, or beyond n=1M.
- Scattered outliers: the separation rule keeps centroids on isolated points (k-means++ start, 1%
  outliers: defaults 0.69x Lloyd, separation 0.77x).
- Unstructured d=32, k=1024: separation leaves 2.4 final clusters with < 10 rows (defaults 0.0).
- The rule was developed on other seeds and settings of the same generator family.
- In earlier tests with a NumPy port, a greedy k-means++ start reached the optimum on
  well-separated clusters for Lloyd and E-Means alike; the random-start gains matter where such a
  start is too costly.

## Reproduce

```
pip install -r requirements.txt       # numpy 2.0.2, faiss-cpu 1.13.0, pandas 2.3.3 (Python 3.9)
python repro/minimal_example.py
python repro/run.py --workers 8       # results/results.md and per-seed results/runs.json
python repro/run.py --quick           # 3 seeds, no n >= 100k
```

`--csv file.csv` adds a numeric CSV twice: median-imputed and z-scored, and with log1p on skewed
non-negative columns first.

| file | content |
|---|---|
| `repro/minimal_example.py` | the example above |
| `emeans_plus.py` | the candidate rule as a subclass of `NumpyEmeans` |
| `repro/run.py`, `repro/common.py` | settings, Lloyd baseline, tables |
| `results/` | `results.md`, per-seed `runs.json`, `deep1m_results.json` |
| `third_party/e_means_ref/` | unmodified `base.py` and `e_means.py` from faiss `2b648f06` |

Licence: MIT (`LICENSE`); the vendored faiss files keep their own MIT licence (`third_party/LICENSE.faiss`).
