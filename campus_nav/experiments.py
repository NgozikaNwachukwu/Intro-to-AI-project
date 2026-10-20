#!/usr/bin/env python3
"""
Preliminary controlled experiments: Greedy vs A* vs UCS.

Each experiment varies ONE environment factor and keeps the rest fixed.
For every trial we generate a fresh random campus (seeded -> reproducible),
pick a start/goal, and run all three algorithms on the SAME instance.

Metrics (per algorithm, per instance):
  found, cost, suboptimality = cost / UCS cost - 1, nodes_expanded,
  nodes_generated, time_ms (best of 3 repeats to reduce timer noise)

Usage:
  python experiments.py                 # all experiments, default trials
  python experiments.py --trials 30     # quicker
Outputs go to results/ (CSV per experiment, summary.json, PNG charts).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics as st
import zlib
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from campus_nav.graph import PROFILES
from campus_nav.heuristics import euclidean, manhattan, scaled, zero
from campus_nav.maps import random_campus, random_endpoints
from campus_nav.search import a_star, greedy_best_first, uniform_cost

OUT = "results"
ALGS = [("Greedy", greedy_best_first), ("A*", a_star), ("UCS", uniform_cost)]
COLORS = {"Greedy": "#2a78d6", "A*": "#eb6834", "UCS": "#1baf7a"}  # fixed categorical order


def timed(fn, *a, repeats=3, **kw):
    best = None
    for _ in range(repeats):
        r = fn(*a, **kw)
        if best is None or r.time_ms < best.time_ms:
            best = r
    return best


def opposite_sides(g, n_cols, n_rows, rng):
    """Start west of the barrier, goal east of it, both in the middle rows."""
    bc = n_cols // 2 - 1
    rows = range(n_rows // 4, n_rows - n_rows // 4)
    s = f"n{rng.choice(rows)}_{rng.randint(0, max(0, bc - 1))}"
    t = f"n{rng.choice(rows)}_{rng.randint(min(n_cols - 1, bc + 2), n_cols - 1)}"
    return s, t


def run_instance(g, s, t, profile, heuristic=None):
    rows = []
    ref = None
    for name, fn in ALGS:
        r = timed(fn, g, s, t, profile, heuristic)
        if name == "UCS":
            ref = r
        rows.append(r)
    out = []
    for r in rows:
        sub = (r.cost / ref.cost - 1.0) if (r.found and ref.found and ref.cost > 0) else None
        out.append(dict(algorithm=r.algorithm, found=r.found,
                        cost=round(r.cost, 2) if r.found else None,
                        suboptimality=sub, optimal=(sub is not None and sub < 1e-9),
                        expanded=r.nodes_expanded, generated=r.nodes_generated,
                        time_ms=r.time_ms, path_len=len(r.path),
                        straight_line=round(g.straight_line(s, t), 1),
                        n_nodes=len(g), n_edges=g.num_edges))
    return out


def experiment(name, param, values, make_instance, trials, profile="no_stairs"):
    """Run `trials` instances for every value of `param`."""
    records = []
    for v in values:
        for i in range(trials):
            # Stable seed per (experiment, trial): every parameter value is tested on
            # the same sequence of seeds (common random numbers) -> fair comparison,
            # and results are identical on every machine/run.
            rng = random.Random(zlib.crc32(f"{name}|{i}".encode()))
            g, s, t, h = make_instance(v, i, rng)
            for row in run_instance(g, s, t, PROFILES[profile], h):
                row.update(experiment=name, param=param, value=v, trial=i)
                records.append(row)
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    return records


def summarize(records):
    """{value: {alg: stats}}"""
    groups = defaultdict(list)
    for r in records:
        groups[(r["value"], r["algorithm"])].append(r)
    out = defaultdict(dict)
    for (v, a), rs in groups.items():
        solved = [r for r in rs if r["found"]]
        subs = [r["suboptimality"] for r in rs if r["suboptimality"] is not None]
        out[v][a] = dict(
            n=len(rs),
            success_rate=len(solved) / len(rs),
            mean_expanded=st.mean(r["expanded"] for r in rs),
            # effort on instances where a route exists (all algs agree on solvability)
            mean_expanded_solved=st.mean(r["expanded"] for r in solved) if solved else 0.0,
            median_time_solved_ms=st.median(r["time_ms"] for r in solved) if solved else 0.0,
            median_time_ms=st.median(r["time_ms"] for r in rs),
            mean_subopt_pct=100 * st.mean(subs) if subs else None,
            max_subopt_pct=100 * max(subs) if subs else None,
            pct_optimal=100 * st.mean(1.0 if s < 1e-9 else 0.0 for s in subs) if subs else None,
        )
    return dict(out)  # keeps the order the values were run in


def chart(summary, title, xlabel, fname, xlabels=None):
    """Two small multiples (separate y-axes, never dual-axis):
    left = nodes expanded per algorithm, right = Greedy extra cost vs optimal."""
    xs = list(summary.keys())
    xpos = list(range(len(xs)))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for alg in ("Greedy", "A*", "UCS"):
        ys = [summary[x][alg]["mean_expanded_solved"] for x in xs]
        a1.plot(xpos, ys, marker="o", lw=2, ms=6, color=COLORS[alg], label=alg)
        a1.annotate(alg, (xpos[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color="#333")
    a1.set_title("Nodes expanded (mean, solvable instances)", fontsize=10, loc="left")
    g = [summary[x]["Greedy"]["mean_subopt_pct"] or 0 for x in xs]
    a2.bar(xpos, g, color=COLORS["Greedy"], width=0.6)
    for x, y in zip(xpos, g):
        a2.text(x, y, f"{y:.1f}%", ha="center", va="bottom", fontsize=8, color="#333")
    a2.set_title("Greedy: extra route cost vs optimal (mean %)", fontsize=10, loc="left")
    for ax in (a1, a2):
        ax.set_xticks(xpos)
        ax.set_xticklabels(xlabels or [str(x) for x in xs], fontsize=8)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.grid(axis="y", color="#e5e5e5", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color("#bbb"); ax.spines["bottom"].set_color("#bbb")
    a1.legend(frameon=False, fontsize=8, loc="best")
    a1.set_xlim(-0.3, len(xs) - 0.4)
    fig.suptitle(title, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    args = ap.parse_args()
    T = args.trials
    summary = {}

    # E1 - size of the search space
    def e1(size, i, rng):
        g = random_campus(size, size, seed=rng.randrange(10**9))
        s, t = random_endpoints(g, rng, min_dist=0.5 * size * 10 * 10)
        return g, s, t, None
    sizes = [5, 10, 15, 20, 30, 40]
    r = experiment("E1_size", "grid side (nodes = side^2)", sizes, e1, T)
    summary["E1_size"] = summarize(r)
    chart(summary["E1_size"], "E1 - Search-space size (no_stairs profile)",
          "campus size (locations)", "E1_size.png", [str(s * s) for s in sizes])

    # E2 - path density
    def e2(d, i, rng):
        g = random_campus(15, 15, density=d, seed=rng.randrange(10**9))
        s, t = random_endpoints(g, rng, min_dist=750)
        return g, s, t, None
    dens = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    r = experiment("E2_density", "path density", dens, e2, T)
    summary["E2_density"] = summarize(r)
    chart(summary["E2_density"], "E2 - Path density, 225 locations (no_stairs)",
          "extra-path probability (0 = tree, 1 = full grid + diagonals)", "E2_density.png")

    # E3 - accessible-network sparsity (stairs fraction)
    def e3(sf, i, rng):
        g = random_campus(15, 15, stairs_frac=sf, seed=rng.randrange(10**9))
        s, t = random_endpoints(g, rng, min_dist=750)
        return g, s, t, None
    sfs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    r = experiment("E3_stairs", "stairs fraction", sfs, e3, T)
    summary["E3_stairs"] = summarize(r)
    chart(summary["E3_stairs"], "E3 - Fraction of stairs-only paths, 225 locations (no_stairs)",
          "fraction of connections that are stairs", "E3_stairs.png",
          [f"{int(x * 100)}%" for x in sfs])

    # E4 - barrier forcing a detour (the 'wrong way first' condition)
    conds = ["none", "barrier_2_gaps", "barrier_1_gap"]

    def e4(c, i, rng):
        bar = c != "none"
        gaps = 1 if c.endswith("1_gap") else 2
        g = random_campus(16, 16, barrier=bar, barrier_gaps=gaps, seed=rng.randrange(10**9))
        s, t = opposite_sides(g, 16, 16, rng)
        return g, s, t, None
    r = experiment("E4_barrier", "barrier", conds, e4, T)
    summary["E4_barrier"] = summarize(r)
    chart(summary["E4_barrier"], "E4 - Barrier with stairs-only direct crossings, 256 locations",
          "condition (start and goal on opposite sides)", "E4_barrier.png",
          ["no barrier", "2 accessible\ncrossings", "1 accessible\ncrossing"])

    # E5 - heuristic quality for A* (and Greedy)
    hs = {"zero": lambda g: zero(g), "0.5x euclid": lambda g: scaled(g, 0.5),
          "euclid": lambda g: euclidean(g), "manhattan": lambda g: manhattan(g),
          "1.5x euclid": lambda g: scaled(g, 1.5), "3x euclid": lambda g: scaled(g, 3.0)}

    def e5(hname, i, rng):
        g = random_campus(20, 20, seed=rng.randrange(10**9))
        s, t = random_endpoints(g, rng, min_dist=1000)
        return g, s, t, hs[hname](g)
    r = experiment("E5_heuristic", "heuristic", list(hs), e5, T)
    s5 = summarize(r)
    summary["E5_heuristic"] = s5
    xs = list(hs)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    exp = [s5[x]["A*"]["mean_expanded_solved"] for x in xs]
    sub = [s5[x]["A*"]["mean_subopt_pct"] for x in xs]
    a1.bar(range(len(xs)), exp, color=COLORS["A*"], width=0.6)
    a1.axhline(s5["euclid"]["UCS"]["mean_expanded_solved"], color="#888", ls="--", lw=1)
    a1.text(len(xs) - 0.5, s5["euclid"]["UCS"]["mean_expanded_solved"], " UCS", va="bottom",
            ha="right", fontsize=8, color="#555")
    a1.set_title("A*: nodes expanded (mean)", fontsize=10, loc="left")
    a2.bar(range(len(xs)), sub, color=COLORS["A*"], width=0.6)
    for k, y in enumerate(sub):
        a2.text(k, y, f"{y:.1f}%", ha="center", va="bottom", fontsize=8)
    a2.set_title("A*: extra route cost vs optimal (mean %)", fontsize=10, loc="left")
    for ax in (a1, a2):
        ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, fontsize=8, rotation=15)
        ax.grid(axis="y", color="#e5e5e5"); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("E5 - Heuristic quality, 400 locations (no_stairs)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(f"{OUT}/E5_heuristic.png", dpi=160); plt.close(fig)

    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=str)

    # Console report
    for ename, s in summary.items():
        print(f"\n### {ename}")
        print(f"{'value':>16} | {'alg':6} | succ | exp(solved) | time ms(solved) | subopt% | optimal%")
        for v, algs in s.items():
            for a, m in algs.items():
                so = "-" if m["mean_subopt_pct"] is None else f"{m['mean_subopt_pct']:.2f}"
                po = "-" if m["pct_optimal"] is None else f"{m['pct_optimal']:.0f}"
                print(f"{str(v):>16} | {a:6} | {m['success_rate']:.2f} | "
                      f"{m['mean_expanded_solved']:11.1f} | {m['median_time_solved_ms']:15.3f} | {so:>7} | {po:>7}")


if __name__ == "__main__":
    main()
