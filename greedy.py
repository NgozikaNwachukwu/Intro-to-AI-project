"""
Greedy Best-First Search (GBFS) - the app's "Quick Preview" mode.

Idea
----
Always expand the frontier node that *looks* closest to the goal, i.e. the
node with the smallest heuristic value h(n). The cost already travelled, g(n),
is ignored when choosing what to expand.

    priority(n) = h(n)            (UCS uses g(n), A* uses g(n) + h(n))

Why it is fast: it heads straight for the goal and usually expands very few
nodes. Why it can be wrong: when the cheapest accessible route must first go
*away* from the goal (around a building, to the only ramp), the straight-line
estimate is misleading, so GBFS commits to whatever branch looks closest and
may return a costlier route. GBFS is complete on finite graphs (graph-search
version with a reached set) but NOT optimal.

Implementation notes
--------------------
* Graph search: each node enters the frontier at most once (`reached`).
  Because the priority depends only on h(n) (fixed per node), finding a second,
  cheaper path to a node would not change its priority, so we never re-open it.
  The parent recorded is the first one that discovered the node; the returned
  cost is the real accumulated cost along that parent chain.
* Ties in h are broken by insertion order (FIFO) via a counter, which makes the
  search deterministic and reproducible for experiments.
* `early_goal_test=True` stops as soon as the goal is *generated* instead of
  when it is popped. This is the classic "quick preview" speed-up; it never
  changes the route GBFS returns here (the goal has h=0, so it would be popped
  next anyway) but saves one expansion. Default False so all three algorithms
  count expansions the same way.
* Only the profile's usable edges are followed (e.g. stairs removed for the
  no_stairs profile), so any route returned satisfies the accessibility needs.
"""
from __future__ import annotations

import heapq
import itertools
import time
from typing import Dict, Optional

from ..graph import AccessibilityProfile, CampusGraph, PROFILES
from ..heuristics import Heuristic, euclidean
from .common import SearchResult, reconstruct


def greedy_best_first(graph: CampusGraph, start: str, goal: str,
                      profile: Optional[AccessibilityProfile] = None,
                      heuristic: Optional[Heuristic] = None,
                      early_goal_test: bool = False) -> SearchResult:
    if start not in graph.nodes or goal not in graph.nodes:
        raise KeyError("start/goal not in graph")
    profile = profile or PROFILES["standard"]
    h = heuristic or euclidean(graph)

    t0 = time.perf_counter()
    counter = itertools.count()                      # FIFO tie-breaker
    frontier = [(h(start, goal), next(counter), start)]
    parent: Dict[str, Optional[str]] = {start: None}
    g_cost: Dict[str, float] = {start: 0.0}          # cost along the parent chain
    expanded = set()
    res = SearchResult("Greedy", start, goal, found=False, nodes_generated=1,
                       max_frontier=1)

    def finish(node: str) -> SearchResult:
        res.found = True
        res.path = reconstruct(parent, node)
        res.cost = g_cost[node]
        res.time_ms = (time.perf_counter() - t0) * 1000
        return res

    if early_goal_test and start == goal:
        return finish(start)

    while frontier:
        _, _, node = heapq.heappop(frontier)
        if node in expanded:            # cannot happen with `reached`, kept for safety
            continue
        if node == goal:                # goal test on expansion
            return finish(node)
        expanded.add(node)
        res.nodes_expanded += 1
        res.expansion_order.append(node)

        for nb, step, _edge in graph.neighbors(node, profile):
            if nb in parent:            # already reached: never re-open in GBFS
                continue
            parent[nb] = node
            g_cost[nb] = g_cost[node] + step
            if early_goal_test and nb == goal:
                return finish(nb)
            heapq.heappush(frontier, (h(nb, goal), next(counter), nb))
            res.nodes_generated += 1
        res.max_frontier = max(res.max_frontier, len(frontier))

    res.time_ms = (time.perf_counter() - t0) * 1000
    return res                          # frontier exhausted: no accessible route
