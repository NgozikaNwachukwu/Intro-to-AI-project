from .common import SearchResult
from .greedy import greedy_best_first
from .best_first import a_star, uniform_cost

ALGORITHMS = {
    "greedy": greedy_best_first,   # Quick Preview
    "astar": a_star,               # Get Directions
    "ucs": uniform_cost,           # Verified Accessible
}

MODE_NAMES = {
    "greedy": "Quick Preview (Greedy Best-First)",
    "astar": "Get Directions (A*)",
    "ucs": "Verified Accessible (UCS)",
}

__all__ = ["SearchResult", "greedy_best_first", "a_star", "uniform_cost",
           "ALGORITHMS", "MODE_NAMES"]
