# Results

Meta's `NumpyEmeans` (float32, unmodified) vs the proposed patch. Same rows, initial centroids and seed for every variant. Baseline: Lloyd to convergence from the same initial centroids.

Lloyd baselines: 306, of which 305 converged within 300 iterations (max 300). Per-seed records: `results/runs.json`.

## 1. Good starts (true centres or k-means++)

Cell: runs with MSE >= 1.2x converged Lloyd / seeds, median MSE / Lloyd, mean splits per run.

| data | start | reference | revival off | separation (3 changes) | separation + warm-up + scaled split | auto (separation if threshold >= 10 rows) | absolute threshold (10 rows) | absolute threshold + donor by SSE |
|---|---|---|---|---|---|---|---|---|
| far rare cluster, 0.3% of rows | true centres | 10/10, 1.46x, 1.3 | 0/10, 1.00x, 0.0 | 0/10, 1.00x, 0.0 | 0/10, 1.00x, 0.0 | 0/10, 1.00x, 0.0 | 9/10, 1.41x, 1.0 | 7/10, 1.32x, 1.2 |
| far rare cluster, 0.6% | true centres | 10/10, 1.91x, 1.3 | 0/10, 1.00x, 0.0 | 0/10, 1.00x, 0.0 | 0/10, 1.00x, 0.0 | 0/10, 1.00x, 0.0 | 0/10, 1.00x, 0.0 | 0/10, 1.00x, 0.0 |
| far rare cluster, 1.25% | true centres | 10/10, 2.83x, 1.4 | 0/10, 1.00x, 0.0 | 0/10, 1.00x, 0.0 | 0/10, 1.00x, 0.0 | 0/10, 1.00x, 0.0 | 0/10, 1.00x, 0.0 | 0/10, 1.00x, 0.0 |
| far rare cluster, 2.5% | true centres | 1/10, 1.00x, 0.1 | 0/10, 1.00x, 0.0 | 0/10, 1.00x, 0.0 | 0/10, 1.00x, 0.0 | 0/10, 1.00x, 0.0 | 0/10, 1.00x, 0.0 | 0/10, 1.00x, 0.0 |
| far rare cluster, 0.6%, n=200k | true centres | 5/5, 1.96x, 1.4 | 0/5, 1.00x, 0.0 | 0/5, 1.00x, 0.0 | 0/5, 1.00x, 0.0 | 0/5, 1.00x, 0.0 | 0/5, 1.00x, 0.0 | 0/5, 1.00x, 0.0 |
| Zipf sizes a=1.2, k=30, n=20k | true centres | 10/10, 3.87x, 7.2 | 0/10, 1.00x, 0.0 | 0/10, 1.00x, 0.0 | 0/10, 1.00x, 0.0 | 0/10, 1.00x, 0.0 | 1/10, 1.00x, 0.2 | 1/10, 1.00x, 0.1 |
| Zipf sizes a=1.2, k=30, n=20k | k-means++ | 8/10, 1.89x, 5.5 | 1/10, 1.00x, 0.0 | 1/10, 0.97x, 0.3 | 1/10, 0.97x, 0.3 | 1/10, 0.97x, 0.3 | 1/10, 1.00x, 0.3 | 1/10, 1.00x, 0.3 |
| Zipf sizes a=1.5, k=40, n=40k | true centres | 10/10, 4.46x, 35.9 | 0/10, 1.00x, 0.0 | 0/10, 1.00x, 0.0 | 0/10, 1.00x, 0.0 | 0/10, 1.00x, 0.0 | 5/10, 1.23x, 2.0 | 3/10, 1.17x, 1.9 |
| Zipf a=1.5, unequal spreads | k-means++ | 10/10, 2.78x, 30.8 | 0/10, 1.00x, 0.0 | 0/10, 0.81x, 1.5 | 0/10, 0.81x, 1.5 | 0/10, 0.81x, 1.5 | 2/10, 1.02x, 1.7 | 0/10, 0.82x, 2.0 |
| Zipf a=1.2, elongated clusters | k-means++ | 6/10, 1.32x, 4.3 | 0/10, 1.00x, 0.0 | 0/10, 0.98x, 0.5 | 0/10, 0.98x, 0.5 | 0/10, 0.98x, 0.5 | 0/10, 1.00x, 0.1 | 0/10, 1.00x, 0.1 |
| Zipf a=1.5, heavy tails (t3) | k-means++ | 10/10, 2.80x, 29.6 | 0/10, 1.00x, 0.0 | 0/10, 0.83x, 2.9 | 0/10, 0.83x, 2.9 | 0/10, 0.83x, 2.9 | 1/10, 0.98x, 3.3 | 0/10, 0.82x, 3.3 |
| batch < k: n=2000, k=50 | true centres | 10/10, 5.22x, 15.8 | 0/10, 1.00x, 0.0 | 0/10, 1.00x, 0.0 | 0/10, 1.00x, 0.0 | 0/10, 1.00x, 0.0 | 10/10, 3.33x, 35.2 | 0/10, 1.00x, 19.1 |
| d=32, k=256, n=100k, Zipf a=1.2 | k-means++ | 3/3, 1.34x, 86.3 | 0/3, 1.00x, 0.0 | 0/3, 0.90x, 12.3 | 0/3, 0.90x, 12.3 | 0/3, 0.90x, 12.3 | 0/3, 1.00x, 22.0 | 0/3, 0.89x, 18.7 |
| 1% scattered outliers, k=30 | k-means++ | 0/10, 0.69x, 19.8 | 0/10, 1.01x, 0.0 | 0/10, 0.77x, 1.9 | 0/10, 0.79x, 1.8 | 0/10, 0.77x, 1.9 | 0/10, 0.68x, 17.0 | 0/10, 0.65x, 9.9 |
| overlapping clusters (box 4), Zipf | k-means++ | 0/10, 0.99x, 1.7 | 0/10, 0.98x, 0.0 | 0/10, 0.99x, 0.9 | 0/10, 0.99x, 0.9 | 0/10, 0.99x, 0.9 | 0/10, 0.98x, 0.2 | 0/10, 0.98x, 0.2 |
| CC GENERAL.csv (z-scored), k=8 | k-means++ | 0/10, 1.04x, 4.8 | 0/10, 1.00x, 0.0 | 0/10, 1.00x, 0.2 | 0/10, 1.00x, 0.2 | 0/10, 1.00x, 0.2 | 0/10, 1.00x, 2.1 | 0/10, 1.01x, 2.6 |
| CC GENERAL.csv (z-scored), k=50 | k-means++ | 0/10, 1.09x, 23.3 | 0/10, 1.00x, 0.0 | 0/10, 0.99x, 4.7 | 0/10, 0.99x, 4.6 | 0/10, 0.99x, 4.7 | 0/10, 1.06x, 17.6 | 0/10, 1.03x, 37.6 |
| CC GENERAL.csv (log1p + z-scored), k=8 | k-means++ | 0/10, 1.00x, 0.0 | 0/10, 1.00x, 0.0 | 0/10, 1.00x, 0.0 | 0/10, 1.00x, 0.0 | 0/10, 1.00x, 0.0 | 0/10, 1.00x, 0.0 | 0/10, 1.00x, 0.0 |
| CC GENERAL.csv (log1p + z-scored), k=50 | k-means++ | 0/10, 0.99x, 3.7 | 0/10, 0.99x, 0.0 | 0/10, 0.99x, 2.0 | 0/10, 0.99x, 2.0 | 0/10, 0.99x, 2.0 | 0/10, 0.99x, 1.7 | 0/10, 0.99x, 1.8 |
| **all true centres runs** | | **66/75 = 88%** | **0/75 = 0%** | **0/75 = 0%** | **0/75 = 0%** | **0/75 = 0%** | **25/75 = 33%** | **11/75 = 15%** |
| **all k-means++ runs** | | **37/103 = 36%** | **1/103 = 1%** | **1/103 = 1%** | **1/103 = 1%** | **1/103 = 1%** | **4/103 = 4%** | **1/103 = 1%** |

'separation (3 changes)' runs at >= 1.2x Lloyd: 1; of these, also >= 1.2x with revival off on the same seed: 1.

## 2. Random-row starts (the paper's default)

Cell: mean paired MSE change vs the reference +/- standard error; median MSE / Lloyd in brackets.

| data | seeds | reference | revival off | separation (3 changes) | separation + warm-up + scaled split | auto (separation if threshold >= 10 rows) | absolute threshold (10 rows) | absolute threshold + donor by SSE |
|---|---|---|---|---|---|---|---|---|
| balanced, k=30, n=20k | 15 | — [0.86x] | +23.5% +/- 6.1 [1.00x] | -43.1% +/- 6.2 [0.48x] | -44.3% +/- 6.3 [0.48x] | -43.1% +/- 6.2 [0.48x] | +11.5% +/- 3.7 [1.00x] | -8.6% +/- 6.5 [0.74x] |
| balanced, k=60, n=30k, unequal spreads | 15 | — [0.93x] | +15.4% +/- 3.0 [1.07x] | -45.0% +/- 3.4 [0.48x] | -44.6% +/- 3.4 [0.48x] | -45.0% +/- 3.4 [0.48x] | +7.0% +/- 2.2 [0.97x] | -19.5% +/- 3.8 [0.68x] |
| balanced, k=40, heavy tails (t3) | 15 | — [0.69x] | +48.4% +/- 15.5 [0.94x] | -65.5% +/- 3.8 [0.22x] | -65.6% +/- 3.8 [0.22x] | -65.5% +/- 3.8 [0.22x] | +26.9% +/- 11.7 [0.83x] | -42.7% +/- 6.4 [0.41x] |
| Zipf a=1.5, k=40, n=40k | 10 | — [0.96x] | +2.0% +/- 4.2 [0.95x] | -9.7% +/- 3.6 [0.81x] | -9.2% +/- 3.4 [0.81x] | -9.7% +/- 3.6 [0.81x] | -1.2% +/- 3.9 [0.92x] | -4.2% +/- 3.8 [0.91x] |
| batch < k: n=2000, k=50 | 10 | — [0.47x] | +91.1% +/- 18.8 [0.99x] | -68.3% +/- 4.0 [0.14x] | -66.5% +/- 3.9 [0.15x] | -68.4% +/- 4.1 [0.13x] | -37.3% +/- 4.6 [0.31x] | -82.9% +/- 1.9 [0.09x] |
| 1% scattered outliers, k=30 | 10 | — [0.53x] | +89.9% +/- 6.4 [1.00x] | -3.2% +/- 6.2 [0.49x] | -3.8% +/- 5.3 [0.50x] | -3.2% +/- 6.2 [0.49x] | +6.0% +/- 5.1 [0.56x] | -12.5% +/- 3.8 [0.45x] |
| overlapping clusters (box 4), Zipf | 10 | — [0.99x] | -0.1% +/- 0.3 [1.00x] | -0.6% +/- 0.4 [0.99x] | -0.5% +/- 0.4 [0.99x] | -0.6% +/- 0.4 [0.99x] | -0.3% +/- 0.2 [0.99x] | -0.4% +/- 0.4 [0.99x] |
| d=32, k=256, n=100k, Zipf a=1.2 | 3 | — [0.97x] | +1.4% +/- 1.7 [0.97x] | -17.6% +/- 0.4 [0.80x] | -16.6% +/- 0.4 [0.81x] | -17.6% +/- 0.4 [0.80x] | -3.3% +/- 0.7 [0.93x] | -17.4% +/- 0.9 [0.80x] |
| CC GENERAL.csv (z-scored), k=8 | 10 | — [1.03x] | -3.5% +/- 0.7 [1.00x] | -3.5% +/- 0.7 [1.00x] | -3.5% +/- 0.7 [1.00x] | -3.5% +/- 0.7 [1.00x] | -3.1% +/- 0.8 [1.00x] | -3.5% +/- 0.6 [0.99x] |
| CC GENERAL.csv (z-scored), k=50 | 10 | — [1.10x] | -10.2% +/- 0.7 [0.99x] | -10.4% +/- 0.8 [0.99x] | -10.4% +/- 0.8 [0.98x] | -10.4% +/- 0.8 [0.99x] | -4.7% +/- 0.9 [1.04x] | -6.8% +/- 1.1 [1.02x] |
| CC GENERAL.csv (log1p + z-scored), k=8 | 10 | — [1.00x] | +0.0% +/- 0.0 [1.00x] | +0.0% +/- 0.0 [1.00x] | +0.0% +/- 0.0 [1.00x] | +0.0% +/- 0.0 [1.00x] | +0.0% +/- 0.0 [1.00x] | +0.0% +/- 0.0 [1.00x] |
| CC GENERAL.csv (log1p + z-scored), k=50 | 10 | — [0.99x] | -0.2% +/- 0.2 [0.99x] | -0.3% +/- 0.2 [0.99x] | -0.2% +/- 0.2 [0.99x] | -0.3% +/- 0.2 [0.99x] | -0.1% +/- 0.1 [0.99x] | -0.3% +/- 0.2 [0.99x] |

## 3. Unstructured data (no Lloyd baseline)

Cell: mean paired MSE change vs the reference +/- standard error; mean number of final clusters with < 10 rows in brackets.

| data | seeds | reference | revival off | separation (3 changes) | separation + warm-up + scaled split | auto (separation if threshold >= 10 rows) | absolute threshold (10 rows) | absolute threshold + donor by SSE |
|---|---|---|---|---|---|---|---|---|
| unstructured Gaussian, d=16, k=512, n=100k | 5 | — [0.0] | +0.0% +/- 0.0 [1.2] | -0.0% +/- 0.0 [0.0] | -0.0% +/- 0.0 [0.0] | -0.0% +/- 0.0 [0.0] | -0.0% +/- 0.0 [0.0] | -0.0% +/- 0.0 [0.0] |
| unstructured Gaussian, d=32, k=1024, n=100k | 5 | — [0.0] | +1.1% +/- 0.1 [124.4] | +0.1% +/- 0.0 [2.4] | +0.1% +/- 0.0 [3.0] | +0.0% +/- 0.0 [0.0] | -0.0% +/- 0.0 [0.0] | +0.0% +/- 0.0 [0.0] |

## 4. Ablation: all five changes with one switched off

Good-start rows: failures / seeds, median MSE / Lloyd. Random-start row: mean paired MSE change vs the reference.

| data | start | all five | no warm-up | no separation rule | no shadow test | donor by mass | multiplicative split |
|---|---|---|---|---|---|---|---|
| far rare cluster, 0.3% of rows | true centres | 0/10, 1.00x | 0/10, 1.00x | 10/10, 1.46x | 1/10, 1.00x | 0/10, 1.00x | 0/10, 1.00x |
| Zipf sizes a=1.5, k=40, n=40k | true centres | 0/10, 1.00x | 0/10, 1.00x | 10/10, 2.59x | 3/10, 1.13x | 0/10, 1.00x | 0/10, 1.00x |
| batch < k: n=2000, k=50 | true centres | 0/10, 1.00x | 0/10, 1.00x | 0/10, 1.00x | 0/10, 1.00x | 0/10, 1.00x | 0/10, 1.00x |
| balanced, k=30, n=20k | random rows | -44.3% +/- 6.3 | -44.3% +/- 6.3 | -44.3% +/- 6.3 | -44.3% +/- 6.3 | +0.0% +/- 0.0 | -43.1% +/- 6.2 |
