# Accessibility-Aware Campus Routing — SOFE 3720 Group 1

Route-planning agent (Project Option 1) that finds the lowest-cost route between two
campus locations that fits a user's accessibility needs, and compares
**Greedy Best-First Search** (Quick Preview), **A\*** (Get Directions) and
**Uniform Cost Search** (Verified Accessible).

All search algorithms are implemented from scratch (`heapq` only). matplotlib is
used for visualization only.

## Quick start

```bash
pip install -r requirements.txt          # matplotlib
python main.py --map proposal --start S --goal G --profile no_stairs --algorithm greedy --plot route.png
python main.py --map detour --start S --goal G --profile no_stairs --algorithm all --plot compare.png
python main.py --map random --size 15 --barrier --seed 7 --algorithm all --plot random.png
python main.py --interactive             # menu-driven demo
python -m unittest discover -s tests -v  # 12 correctness tests
python experiments.py --trials 100       # preliminary experiments -> results/
```

## Problem formulation

| Element | Definition |
|---|---|
| States | Campus locations (nodes) with (x, y) coordinates |
| Initial / goal state | Chosen or random start / destination location |
| Goal test | current node == goal node |
| Actions | Move along a connection: walkway (flat), ramp, stairs, elevator, outdoor path |
| Step cost | Edge length (m) × profile multiplier (≥ 1); blocked edge types are unusable |
| Constraints | Every cost > 0 and ≥ the edge's straight-line length (enforced in `add_edge`) |
| Objective | Lowest total-cost route that satisfies the accessibility profile |

**Accessibility profiles** (`campus_nav/graph.py`): `standard`, `no_stairs`
(wheelchair), `poor_weather` (no outdoor paths), `accessibility_challenge`
(no stairs, outdoor ×1.5, ramps ×1.2).

**Heuristic** h(n) = Euclidean distance from n to the goal, in metres. It is
admissible and consistent because every edge costs at least its straight-line
length (the triangle inequality does the rest).

## Project layout

```
campus_nav/graph.py        CampusGraph, edge types, accessibility profiles, JSON save/load
campus_nav/maps.py         proposal map, 'wrong-way-first' detour map, random campus generator
campus_nav/heuristics.py   euclidean, zero, manhattan, scaled(w)
campus_nav/search/greedy.py      Greedy Best-First Search (Quick Preview)
campus_nav/search/best_first.py  UCS and A* (comparison baselines)
campus_nav/visualize.py    matplotlib route/exploration plots + turn-by-turn text
main.py                    CLI (start/goal/map/profile/algorithm/heuristic selection)
experiments.py             E1–E5 controlled experiments, CSVs + charts in results/
tests/test_search.py       unit + randomized property tests
```

## Greedy Best-First Search — design notes

* Priority = h(n) only; g(n) is tracked solely to report the route's real cost.
* Graph search with a `reached` set: each location enters the frontier once.
  Since h(n) never changes, re-opening a node could not change its priority.
* Ties broken FIFO (insertion counter) → deterministic, reproducible runs.
* Goal tested when popped (same as UCS/A*, so expansion counts are comparable);
  `early_goal_test=True` stops when the goal is generated (Quick Preview speed-up).
* Only edges usable under the profile are followed, so every route it returns
  is accessible. Complete on finite graphs; **not optimal**.

## Random campus generator parameters

`size` (grid side), `density` (extra paths beyond a spanning tree), `stairs_frac`
(sparsity of the accessible network), `outdoor_frac`, `detour` (how much real
paths deviate from straight lines), `barrier` + `barrier_gaps` (a central
barrier whose direct crossings are stairs-only — forces "wrong-way-first" routes),
`seed` (reproducible).
