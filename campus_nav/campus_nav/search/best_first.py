"""
Uniform Cost Search and A* (reference implementations used as comparison
baselines for Greedy). Both are best-first graph searches that differ only in
the priority:

    UCS : f(n) = g(n)
    A*  : f(n) = g(n) + h(n)

Both keep the best known g(n) per node and use lazy deletion: an outdated
frontier entry is skipped when popped. A node is re-opened if a cheaper path is
found later, so A* stays correct even with an inconsistent heuristic (and is
optimal whenever h is admissible).
"""
from __future__ import annotations

import heapq
import itertools
import time
from typing import Callable, Dict, Optional

from ..graph import AccessibilityProfile, CampusGraph, PROFILES
from ..heuristics import Heuristic, euclidean
from .common import SearchResult, reconstruct


def _best_first(name: str, graph: CampusGraph, start: str, goal: str,
                profile: Optional[AccessibilityProfile],
                h: Callable[[str, str], float]) -> SearchResult:
    if start not in graph.nodes or goal not in graph.nodes:
        raise KeyError("start/goal not in graph")
    profile = profile or PROFILES["standard"]
    t0 = time.perf_counter()
    counter = itertools.count()
    best_g: Dict[str, float] = {start: 0.0}
    parent: Dict[str, Optional[str]] = {start: None}
    frontier = [(h(start, goal), next(counter), 0.0, start)]
    closed: Dict[str, float] = {}
    res = SearchResult(name, start, goal, found=False, nodes_generated=1, max_frontier=1)

    while frontier:
        _, _, g, node = heapq.heappop(frontier)
        if g > best_g.get(node, float("inf")):      # stale entry
            continue
        if node == goal:
            res.found = True
            res.path = reconstruct(parent, node)
            res.cost = g
            res.time_ms = (time.perf_counter() - t0) * 1000
            return res
        if node in closed and closed[node] <= g:
            continue
        closed[node] = g
        res.nodes_expanded += 1
        res.expansion_order.append(node)
        for nb, step, _ in graph.neighbors(node, profile):
            ng = g + step
            if ng < best_g.get(nb, float("inf")):
                best_g[nb] = ng
                parent[nb] = node
                heapq.heappush(frontier, (ng + h(nb, goal), next(counter), ng, nb))
                res.nodes_generated += 1
        res.max_frontier = max(res.max_frontier, len(frontier))

    res.time_ms = (time.perf_counter() - t0) * 1000
    return res


def uniform_cost(graph: CampusGraph, start: str, goal: str,
                 profile: Optional[AccessibilityProfile] = None,
                 heuristic: Optional[Heuristic] = None) -> SearchResult:
    """UCS - 'Verified Accessible' mode. Ignores any heuristic."""
    return _best_first("UCS", graph, start, goal, profile, lambda n, t: 0.0)


def a_star(graph: CampusGraph, start: str, goal: str,
           profile: Optional[AccessibilityProfile] = None,
           heuristic: Optional[Heuristic] = None) -> SearchResult:
    """A* - 'Get Directions' mode."""
    h = heuristic or euclidean(graph)
    return _best_first("A*", graph, start, goal, profile, h)
