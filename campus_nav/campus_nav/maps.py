"""
Campus environments.

* proposal_map()   - the exact 7-location campus from our proposal slides
* detour_map()     - small hand-built case where the direct way has stairs and
                     the only accessible route starts by moving AWAY from the goal
* random_campus()  - randomized campus generator with tunable size, path density,
                     stairs fraction, cost noise, and an optional "barrier"
                     (river / building block) whose direct crossings are stairs-only
"""
from __future__ import annotations

import math
import random
from typing import Optional

from .graph import (CampusGraph, ELEVATOR, FLAT, OUTDOOR, RAMP, STAIRS)


def proposal_map() -> CampusGraph:
    """Campus graph from the proposal slide (1 unit = 15 m)."""
    g = CampusGraph("proposal_campus", metres_per_unit=15.0)
    g.add_node("S", 0, 4, "Main Entrance")
    g.add_node("L", 2, 8, "Library")
    g.add_node("B", 5, 4, "Business")
    g.add_node("E", 7, 8, "Engineering")
    g.add_node("R", 2, 0, "Residence")
    g.add_node("SC", 7, 0, "Science")
    g.add_node("G", 10, 4, "Gym")

    g.add_edge("S", "L", 100, STAIRS)
    g.add_edge("S", "B", 150, ELEVATOR)
    g.add_edge("S", "R", 200, RAMP)
    g.add_edge("L", "E", 120, STAIRS)
    g.add_edge("L", "B", 80, RAMP)        # "accessible (ramp)"
    g.add_edge("E", "B", 70, ELEVATOR)
    g.add_edge("E", "G", 100, RAMP)
    g.add_edge("R", "B", 100, STAIRS)
    g.add_edge("R", "SC", 80, OUTDOOR)
    g.add_edge("B", "SC", 90, FLAT)
    g.add_edge("SC", "G", 150, ELEVATOR)
    return g


def detour_map() -> CampusGraph:
    """Hand-built 'wrong way first' case (1 unit = 10 m).

    S -> X -> G is short but X->G is stairs.  The cheapest accessible route
    goes WEST first (away from the goal) through W, north around the building,
    and back east (343 m).  Under the no_stairs profile Greedy is pulled toward
    X and then down a 'decoy' branch D1-D2 that looks close to G but ends in a
    long switchback ramp, so it returns a much costlier route (420 m) while
    UCS and A* find the 343 m one.
    """
    g = CampusGraph("detour_campus", metres_per_unit=10.0)
    pts = {
        "S": (0, 0), "X": (6, 0), "G": (12, 0),
        "W": (-3, 0), "N1": (-3, 8), "N2": (6, 9), "N3": (12, 8),
        "D1": (8, -2), "D2": (10, -4),
    }
    for k, (x, y) in pts.items():
        g.add_node(k, x, y)
    g.add_edge("S", "X", 60, FLAT)
    g.add_edge("X", "G", 62, STAIRS)
    g.add_edge("X", "D1", 30, FLAT)          # decoy: looks close to G...
    g.add_edge("D1", "D2", 30, FLAT)
    g.add_edge("D2", "G", 300, RAMP)         # ...but ends in a long switchback ramp
    g.add_edge("S", "W", 30, FLAT)
    g.add_edge("W", "N1", 80, RAMP)
    g.add_edge("N1", "N2", 92, ELEVATOR)
    g.add_edge("N2", "N3", 61, FLAT)
    g.add_edge("N3", "G", 80, RAMP)
    return g


def random_campus(n_cols: int = 10, n_rows: int = 10, *,
                  density: float = 0.5,
                  stairs_frac: float = 0.25,
                  outdoor_frac: float = 0.3,
                  detour: float = 0.3,
                  barrier: bool = False,
                  barrier_gaps: int = 1,
                  spacing: float = 10.0,
                  metres_per_unit: float = 10.0,
                  seed: Optional[int] = None) -> CampusGraph:
    """Generate a random campus on a jittered grid.

    density      - probability of adding each extra (non-tree) grid/diagonal edge.
                   0 -> a tree (exactly one route between any pair);
                   1 -> every grid + diagonal connection exists.
    stairs_frac  - fraction of edges that are stairs-only.
    outdoor_frac - fraction of non-stairs edges that are outdoor paths.
    detour       - edge length = straight line * (1 + U(0, detour)). Controls how
                   far real paths deviate from straight lines (heuristic quality).
    barrier      - add a vertical barrier through the middle; direct crossings are
                   stairs-only and `barrier_gaps` accessible crossings are placed at
                   the far top/bottom edges of the campus (forces detours).
    """
    rng = random.Random(seed)
    g = CampusGraph(f"random_{n_cols}x{n_rows}_s{seed}", metres_per_unit)

    def nid(c, r):
        return f"n{r}_{c}"

    jitter = spacing * 0.25
    for r in range(n_rows):
        for c in range(n_cols):
            g.add_node(nid(c, r),
                       c * spacing + rng.uniform(-jitter, jitter),
                       r * spacing + rng.uniform(-jitter, jitter))

    barrier_col = n_cols // 2 - 1  # barrier between barrier_col and barrier_col+1

    def crosses(c1, c2):
        return barrier and min(c1, c2) == barrier_col and max(c1, c2) == barrier_col + 1

    # Accessible gap rows at the extremes of the campus
    gap_rows = set()
    if barrier:
        candidates = [0, n_rows - 1, 1, n_rows - 2]
        gap_rows = set(candidates[:max(0, barrier_gaps)])

    # Candidate edges: 4-neighbour grid + diagonals
    cand = []
    for r in range(n_rows):
        for c in range(n_cols):
            for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
                c2, r2 = c + dc, r + dr
                if 0 <= c2 < n_cols and 0 <= r2 < n_rows:
                    cand.append(((c, r), (c2, r2), dc != 0 and dr != 0))
    rng.shuffle(cand)

    # Union-find to build a random spanning tree on non-diagonal edges first
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    chosen = []
    rest = []
    for a, b, diag in cand:
        if not diag and find(a) != find(b):
            parent[find(a)] = find(b)
            chosen.append((a, b, diag))
        else:
            rest.append((a, b, diag))
    for a, b, diag in rest:
        p = density * (0.5 if diag else 1.0)
        if rng.random() < p:
            chosen.append((a, b, diag))

    # With a barrier, guarantee the accessible gaps exist
    if barrier:
        have = {(min(a, b), max(a, b)) for a, b, _ in chosen}
        for r in gap_rows:
            key = ((barrier_col, r), (barrier_col + 1, r))
            if key not in have:
                chosen.append((key[0], key[1], False))

    for (c1, r1), (c2, r2), diag in chosen:
        u, v = nid(c1, r1), nid(c2, r2)
        straight = g.straight_line(u, v)
        length = straight * (1.0 + rng.uniform(0.0, detour))
        if crosses(c1, c2):
            kind = RAMP if (r1 == r2 and r1 in gap_rows) else STAIRS
        elif rng.random() < stairs_frac:
            kind = STAIRS
        else:
            kind = OUTDOOR if rng.random() < outdoor_frac else rng.choice((FLAT, RAMP, ELEVATOR))
        g.add_edge(u, v, math.ceil(length * 10) / 10, kind)  # round UP: keep >= straight line
    return g


def random_endpoints(g: CampusGraph, rng: random.Random, min_dist: float = 0.0):
    """Pick a random (start, goal) pair at least `min_dist` metres apart."""
    ids = list(g.nodes)
    for _ in range(1000):
        s, t = rng.sample(ids, 2)
        if g.straight_line(s, t) >= min_dist:
            return s, t
    return rng.sample(ids, 2)


MAPS = {
    "proposal": proposal_map,
    "detour": detour_map,
}
