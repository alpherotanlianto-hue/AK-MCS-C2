"""Apply AK-MCS-C2 to the floor-system summary CSVs (v13, v13c, v13d).

Failure event: a configuration is "hard to design", i.e. at most
``--min-converged`` of the five structural models (BEAM, TRUSS, WALL, PLATE,
STRUSS) converged.  g(x) = n_converged(x) - (min_converged + 0.5), so g <= 0
means failure.

The CSVs hold only ~230 evaluated configurations, so the limit state is a
data-driven oracle: inverse-distance k-NN regression of n_converged on the
standardised inputs.  The Monte Carlo population is a smoothed bootstrap of
the observed inputs.  Pf is therefore an estimate for the *empirical input
distribution*, not a property of any physical population.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from akmcsc2 import AKMCS, ConformalC2, LOOGPRegressor, ReliabilityProblem

INPUTS = ["Lx", "Ly", "max_span", "storeys", "headroom", "h_transfer",
          "h_podium", "bearing_mm", "is_office"]


def load(paths):
    frames = []
    for rank, p in enumerate(sorted(paths)):
        df = pd.read_csv(p)
        df["src"] = os.path.basename(p).split("_")[1]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    # v13c re-runs part of v13: keep the latest version of each cid.
    order = {"v13": 0, "v13c": 1, "v13d": 2}
    df["_o"] = df["src"].map(order)
    df = df.sort_values("_o").drop_duplicates("cid", keep="last").drop(columns="_o")
    df["is_office"] = (df["use"] == "office").astype(float)
    return df.reset_index(drop=True)


def build_problem(df, min_converged, k, jitter):
    X = df[INPUTS].to_numpy(float)
    y = df["n_converged"].to_numpy(float) - (min_converged + 0.5)
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = (X - mu) / sd
    nn = NearestNeighbors(n_neighbors=k).fit(Z)

    def g(Xq):
        d, i = nn.kneighbors((np.atleast_2d(Xq) - mu) / sd)
        w = 1.0 / (d + 1e-6)
        return (w * y[i]).sum(1) / w.sum(1)

    def sampler(n, rng):
        base = X[rng.integers(0, len(X), n)]
        return base + jitter * sd * rng.standard_normal(base.shape)

    pf_emp = float(np.mean(y <= 0))
    return ReliabilityProblem(
        name="floors", dim=len(INPUTS), g=g, sampler=sampler,
        pf_ref=pf_emp, description="n_converged <= %d" % min_converged,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_dir")
    ap.add_argument("--min-converged", type=int, default=2)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--jitter", type=float, default=0.15)
    ap.add_argument("--n-mc", type=int, default=20_000)
    ap.add_argument("--n-init", type=int, default=30)
    ap.add_argument("--max-calls", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()

    df = load(glob.glob(os.path.join(a.csv_dir, "*floors_v13*_summary_configs.csv")))
    print(f"{len(df)} unique configurations; empirical share with "
          f"n_converged <= {a.min_converged}: "
          f"{np.mean(df.n_converged <= a.min_converged):.3f}")
    prob = build_problem(df, a.min_converged, a.k, a.jitter)

    rows = []
    for seed in range(a.seeds):
        algo = AKMCS(
            problem=prob,
            learning=ConformalC2(pf_gap_tol=5e-3, batch_q=1),
            conformal="j-mm-gp",
            gp_factory=lambda: LOOGPRegressor(kernel="matern52", noise=1e-4),
            n_mc=a.n_mc, n_init=a.n_init, max_calls=a.max_calls,
            track_coverage=True,
        )
        r = algo.run(seed=seed)
        rows.append(dict(seed=seed, pf=r.pf, lower=r.pf_lower, upper=r.pf_upper,
                         pf_crude_mc=r.pf_mc_ref, calls=r.n_calls,
                         converged=r.converged))
        print(rows[-1])
    out = pd.DataFrame(rows)
    out.to_csv("floors_akmcsc2_results.csv", index=False)
    print(out.describe().T[["mean", "std"]])


if __name__ == "__main__":
    main()
