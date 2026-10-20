"""
Heuristic functions h(n): estimated cost from node n to the goal.

euclidean  - straight-line distance in metres (the project heuristic).
             Admissible because every edge cost >= its straight-line length,
             so no real path can be shorter than the straight line.
             Also consistent (triangle inequality), so A* graph search is optimal.
zero       - h(n) = 0. Turns A* into UCS; turns Greedy into an arbitrary-order
             search. Useful as a 'no information' baseline.
manhattan  - |dx| + |dy|. NOT admissible here (paths may cut diagonally),
             included to study how an overestimating heuristic affects A*.
scaled(w)  - w * euclidean. w < 1 is a weaker (still admissible) heuristic;
             w > 1 is inadmissible (Weighted A*). Used for heuristic-quality
             experiments.
"""
from __future__ import annotations

from typing import Callable

from .graph import CampusGraph

Heuristic = Callable[[str, str], float]


def euclidean(g: CampusGraph) -> Heuristic:
    def h(n: str, goal: str) -> float:
        return g.straight_line(n, goal)
    h.__name__ = "euclidean"
    return h


def zero(g: CampusGraph) -> Heuristic:
    def h(n: str, goal: str) -> float:
        return 0.0
    h.__name__ = "zero"
    return h


def manhattan(g: CampusGraph) -> Heuristic:
    def h(n: str, goal: str) -> float:
        a, b = g.nodes[n], g.nodes[goal]
        return (abs(a.x - b.x) + abs(a.y - b.y)) * g.metres_per_unit
    h.__name__ = "manhattan"
    return h


def scaled(g: CampusGraph, w: float) -> Heuristic:
    def h(n: str, goal: str) -> float:
        return w * g.straight_line(n, goal)
    h.__name__ = f"euclidean_x{w:g}"
    return h


HEURISTICS = {"euclidean": euclidean, "zero": zero, "manhattan": manhattan}
