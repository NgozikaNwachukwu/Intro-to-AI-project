#!/usr/bin/env python3
"""
Accessibility-Aware Campus Routing - command-line app.

Examples
--------
  # Proposal map, Greedy (Quick Preview), wheelchair profile, with a picture
  python main.py --map proposal --start S --goal G --profile no_stairs --algorithm greedy --plot route.png

  # Compare all three modes on the 'wrong way first' map
  python main.py --map detour --start S --goal G --profile no_stairs --algorithm all --plot compare.png

  # Random 15x15 campus with a barrier, random start/goal
  python main.py --map random --size 15 --barrier --seed 7 --algorithm all --plot random.png

  # Menu-driven mode for the demo
  python main.py --interactive
"""
from __future__ import annotations

import argparse
import random
import sys

from campus_nav.graph import CampusGraph, PROFILES
from campus_nav.heuristics import HEURISTICS
from campus_nav.maps import MAPS, random_campus, random_endpoints
from campus_nav.search import ALGORITHMS, MODE_NAMES
from campus_nav.visualize import plot_results, text_route


def build_map(args) -> CampusGraph:
    if args.map in MAPS:
        return MAPS[args.map]()
    if args.map == "random":
        return random_campus(args.size, args.size, density=args.density,
                             stairs_frac=args.stairs, detour=args.detour,
                             barrier=args.barrier, seed=args.seed)
    return CampusGraph.load(args.map)  # path to a JSON map file


def run(g: CampusGraph, start: str, goal: str, profile_name: str,
        algorithm: str, heuristic: str = "euclidean", plot: str | None = None):
    profile = PROFILES[profile_name]
    h = HEURISTICS[heuristic](g)
    algs = list(ALGORITHMS) if algorithm == "all" else [algorithm]
    results = []
    print(f"\nMap: {g.name} ({len(g)} locations, {g.num_edges} connections)")
    print(f"Profile: {profile.name} - {profile.description}")
    print(f"Heuristic: {heuristic}\n")
    for a in algs:
        res = ALGORITHMS[a](g, start, goal, profile, h)
        results.append(res)
        print(f"== {MODE_NAMES[a]} ==")
        print(text_route(g, res, profile) if len(g) <= 60 else res.summary())
        print()
    if len(results) > 1 and all(r.found for r in results):
        best = min(r.cost for r in results)
        print("Comparison (cost relative to best found):")
        for r in results:
            print(f"  {r.algorithm:7s} cost {r.cost:8.1f}  (+{100 * (r.cost / best - 1):5.1f}%)"
                  f"  expanded {r.nodes_expanded:5d}  time {r.time_ms:8.3f} ms")
    if plot:
        plot_results(g, results, profile, plot, label_nodes=len(g) <= 40)
        print(f"\nSaved visualization to {plot}")
    return results


def interactive():
    print("=== Accessibility-Aware Campus Routing ===")
    maps = list(MAPS) + ["random"]
    m = _choose("Map", maps, "proposal")
    if m == "random":
        size = int(input("  Grid size [10]: ") or 10)
        barrier = (input("  Add barrier? (y/N): ").lower() == "y")
        seed = int(input("  Seed [random]: ") or random.randrange(10**6))
        g = random_campus(size, size, barrier=barrier, seed=seed)
    else:
        g = MAPS[m]()
    if len(g) <= 30:
        print("  Locations: " + ", ".join(f"{n.id}={n.name}" for n in g.nodes.values()))
    start = input("  Start location id [random]: ").strip()
    goal = input("  Goal location id [random]: ").strip()
    if not start or not goal:
        s, t = random_endpoints(g, random.Random())
        start, goal = start or s, goal or t
    prof = _choose("Accessibility profile", list(PROFILES), "no_stairs")
    alg = _choose("Algorithm", list(ALGORITHMS) + ["all"], "all")
    run(g, start, goal, prof, alg, plot="route.png")


def _choose(label, options, default):
    print(f"{label}:")
    for i, o in enumerate(options, 1):
        print(f"  {i}. {o}{' (default)' if o == default else ''}")
    ans = input("  > ").strip()
    if ans.isdigit() and 1 <= int(ans) <= len(options):
        return options[int(ans) - 1]
    return ans if ans in options else default


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--map", default="proposal",
                   help="proposal | detour | random | path/to/map.json")
    p.add_argument("--size", type=int, default=10, help="random map grid size (NxN)")
    p.add_argument("--density", type=float, default=0.5)
    p.add_argument("--stairs", type=float, default=0.25, help="fraction of stairs edges")
    p.add_argument("--detour", type=float, default=0.3)
    p.add_argument("--barrier", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--start", help="start node id (random if omitted)")
    p.add_argument("--goal", help="goal node id (random if omitted)")
    p.add_argument("--profile", default="standard", choices=list(PROFILES))
    p.add_argument("--algorithm", default="greedy", choices=list(ALGORITHMS) + ["all"])
    p.add_argument("--heuristic", default="euclidean", choices=list(HEURISTICS))
    p.add_argument("--plot", help="save a PNG visualization to this path")
    p.add_argument("--save-map", help="save the generated map as JSON")
    p.add_argument("--list-nodes", action="store_true")
    args = p.parse_args(argv)

    if args.interactive:
        return interactive()
    g = build_map(args)
    if args.save_map:
        g.save(args.save_map)
    if args.list_nodes:
        for n in g.nodes.values():
            print(f"{n.id:10s} {n.name:20s} ({n.x:.1f}, {n.y:.1f})")
        return
    rng = random.Random(args.seed)
    if not args.start or not args.goal:
        s, t = random_endpoints(g, rng, min_dist=0.0)
        args.start, args.goal = args.start or s, args.goal or t
    for nid in (args.start, args.goal):
        if nid not in g.nodes:
            sys.exit(f"Unknown location {nid!r}. Use --list-nodes to see ids.")
    run(g, args.start, args.goal, args.profile, args.algorithm, args.heuristic, args.plot)


if __name__ == "__main__":
    main()
