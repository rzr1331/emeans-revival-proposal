"""E-Means+ : proposed dead-cluster revival changes, as a subclass of Meta's NumpyEmeans.

Only the revival path changes. Assignment, the EMA update, the cosine schedule, the scaling
laws and the closing means are inherited unchanged.

Changes (each can be switched off to recover the reference behaviour):
  1. warm-up        no revival until `warmup_rows_per_centroid * k` rows have been seen;
  2. separation     a low-mass centroid is revived only if its points are not well separated:
                    mean over its points of (d2_second - d2_nearest) < kappa * global mean
                    within-cluster squared distance per point (both are EMAs with the same
                    decay as the masses, so the ratio does not swing with batch counts);
  3. shadow         a centroid with no rows since it was created is revived at once only if
                    another centroid lies within kappa * spread (squared distance); an
                    isolated one waits, and is dead after a full epoch without a hit;
  4. donor by SSE   donors are the clusters with the largest within-cluster SSE (EMA),
                    not a mass-proportional random draw;
  5. scaled split   donor/child offset = split_scale * donor's RMS spread per coordinate,
                    added, instead of the multiplicative c * (1 +/- 1e-4).

Extra cost: the second-nearest distance per row (top-2 instead of top-1) and two EMAs per
centroid.
"""
import numpy as np
import faiss

import os
import sys

# The vendored, unmodified reference (faiss commit 2b648f06) is imported first so the code under
# test cannot change with the installed faiss version.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party"))
from e_means_ref.e_means import NumpyEmeans, _separate_splits

class NumpyEmeansPlus(NumpyEmeans):
    def __init__(self, d, k, *args, warmup_rows_per_centroid=10, kappa=1.0, split_scale=1e-3,
                 separation=True, shadow=True, donor_by_sse=True, scaled_split=True, min_rows=0, **kwargs):
        super().__init__(d, k, *args, **kwargs)
        self.warmup_rows_per_centroid = warmup_rows_per_centroid
        self.kappa = float(kappa)
        self.split_scale = float(split_scale)
        self.separation = separation
        self.shadow = shadow
        self.donor_by_sse = donor_by_sse
        self.scaled_split = scaled_split
        # The separation test is applied only when the mass threshold, in rows, is at least
        # min_rows (0 = always). Below that every low-mass centroid holds fewer than min_rows rows.
        self.min_rows = min_rows
        self._separation_active = True
        if self.spherical:      # removal cost is not tracked for inner-product assignment:
            self.warmup_rows_per_centroid = 0          # keep the reference revival rule
            self.separation = self.shadow = self.donor_by_sse = self.scaled_split = False

    # ---- state -----------------------------------------------------------------
    def begin(self, global_n, init_centroids):
        super().begin(global_n, init_centroids)
        k = self.k
        self._R = np.zeros(k)                 # EMA removal cost: sum of (d2_second - d2_nearest)
        self._E = np.zeros(k)                 # EMA within-cluster SSE
        self._rb = np.zeros(k)                # per-batch accumulators
        self._eb = np.zeros(k)
        self._rows_seen = 0
        self._separation_active = self._unused_threshold * self._n / k >= self.min_rows
        # dead after this many rows without a hit: one pass, or half the run below two passes
        self._stale_rows = min(self._n, sum(self.recipe["epoch_rows"]) / 2.0)
        self._last_hit = np.zeros(k)          # rows seen at the last hit (or at birth)
        self._hit_since_birth = np.zeros(k, dtype=bool)

    def _assign_rows(self, x, centroids):
        if self.spherical or self.k < 2 or self._usage is None:   # not training: reference path
            return super()._assign_rows(x, centroids)
        dist, lab = faiss.knn(x, centroids, 2)
        labels = lab[:, 0]
        d1 = dist[:, 0].astype("float64")
        self._rb += np.bincount(labels, weights=dist[:, 1].astype("float64") - d1, minlength=self.k)
        self._eb += np.bincount(labels, weights=d1, minlength=self.k)
        return dist[:, 0], labels

    def _apply_statistics(self, ids, sums, counts, objective, alpha, decay, replace_usage=False):
        touched = counts > 0
        self._rows_seen += int(round(float(counts.sum())))
        super()._apply_statistics(ids, sums, counts, objective, alpha, decay, replace_usage)
        if replace_usage:                     # same EMA as the masses
            self._R[touched], self._E[touched] = self._rb[touched], self._eb[touched]
        else:
            self._R *= decay
            self._E *= decay
            self._R += alpha * self._rb
            self._E += alpha * self._eb
        self._last_hit[touched] = self._rows_seen
        self._hit_since_birth |= touched
        self._rb.fill(0)
        self._eb.fill(0)

    # ---- revival ---------------------------------------------------------------
    def _revive_unused(self, rng):
        if self._rows_seen < self.warmup_rows_per_centroid * self.k:
            return 0
        return super()._revive_unused(rng)

    def _choose_splits(self, rng):
        if not (self.separation or self.donor_by_sse):
            return super()._choose_splits(rng)
        k = self.k
        N = self._usage.astype("float64")
        unused = N < N.sum() * self._unused_threshold / k
        if self.separation:
            spread = self._E.sum() / max(N.sum(), 1e-300)
            rbar = np.where(N > 1e-12, self._R / np.maximum(N, 1e-300), 0.0)
            not_separated = rbar < self.kappa * spread
            if not self._separation_active:      # mass rule for centroids that have rows
                not_separated = np.ones(k, dtype=bool)
            if self.shadow:
                seen = self._hit_since_birth
                idle = self._rows_seen - self._last_hit      # seen: a full pass; never seen: _stale_rows
                stale = np.where(seen, idle >= self._n, idle >= self._stale_rows)
                dead_unseen = np.zeros(k, dtype=bool)
                idx = np.flatnonzero(unused & ~seen)
                if len(idx) and k > 1:   # nearest other centroid = second neighbour (first is itself)
                    dd, _ = faiss.knn(self._centroids[idx], self._centroids, 2)
                    dead_unseen[idx] = dd[:, 1] < self.kappa * spread
                keep = stale | (seen & not_separated) | dead_unseen
            else:
                keep = not_separated
            donors = (~unused) & (N > 0)      # low-mass centroids that are kept are not donors either
            unused = unused & keep
        else:
            donors = (~unused) & (N > 0)
        nsplit = min(int(unused.sum()), int(donors.sum()))
        if nsplit == 0:
            return None, None
        keys = np.full(k, -np.inf)
        if self.donor_by_sse:
            keys[donors] = self._E[donors]
            source = np.argsort(keys, kind="stable")[-nsplit:][::-1]
        else:
            uniform = np.maximum(rng.random_sample(k), np.finfo("float64").tiny)
            keys[donors] = np.log(N[donors]) - np.log(-np.log(uniform[donors]))
            source = np.argsort(keys)[-nsplit:][::-1]
        return source, np.flatnonzero(unused)[:nsplit]

    def _apply_splits(self, source, target):
        if self.scaled_split:
            centroids = self._centroids[source].astype("float64")
            sign = np.ones(self.d)
            sign[1::2] = -1.0
            spread = np.sqrt(self._E[source] / np.maximum(self._usage[source].astype("float64"), 1e-300) / self.d)
            offset = self.split_scale * spread[:, None] * sign[None, :]
            donor = (centroids - offset).astype("float32")
            child = (centroids + offset).astype("float32")
            _separate_splits(donor, child)
            half = self._usage[source] / 2.0
            self._usage[source] = half
            self._centroids[source] = donor
            self._usage[target] = half
            self._centroids[target] = child
            self.n_splits += len(source)
            self._invalidate_readout()
            n = len(source)
        else:
            n = super()._apply_splits(source, target)
        for a in (self._R, self._E):          # children share the donor's statistics
            a[source] /= 2
            a[target] = a[source]
        self._last_hit[target] = self._rows_seen     # a revived slot is newborn
        self._hit_since_birth[target] = False
        return n
