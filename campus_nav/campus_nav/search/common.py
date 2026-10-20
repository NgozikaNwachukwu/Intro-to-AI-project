"""Shared result type and helpers for all search algorithms."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SearchResult:
    algorithm: str
    start: str
    goal: str
    found: bool
    path: List[str] = field(default_factory=list)
    cost: float = float("inf")
    nodes_expanded: int = 0        # nodes popped from the frontier and expanded
    nodes_generated: int = 0       # nodes pushed onto the frontier
    max_frontier: int = 0          # peak frontier size (memory proxy)
    time_ms: float = 0.0
    expansion_order: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.found:
            return (f"{self.algorithm}: NO ROUTE from {self.start} to {self.goal} "
                    f"(expanded {self.nodes_expanded} nodes, {self.time_ms:.3f} ms)")
        return (f"{self.algorithm}: cost {self.cost:.1f} m | "
                f"{len(self.path) - 1} steps | expanded {self.nodes_expanded} | "
                f"generated {self.nodes_generated} | {self.time_ms:.3f} ms\n"
                f"  path: {' -> '.join(self.path)}")


def reconstruct(parent: Dict[str, Optional[str]], goal: str) -> List[str]:
    path = [goal]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    path.reverse()
    return path
