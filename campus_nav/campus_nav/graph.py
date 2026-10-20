"""
Campus graph model for the Accessibility-Aware Campus Routing project.

States  = campus locations (nodes) with (x, y) map coordinates.
Actions = moving along a connection (edge): walkway, ramp, stairs, elevator, outdoor path.

Cost model (from the proposal):
    * every edge cost > 0
    * every edge cost >= the edge's physical (straight-line) length
These two rules are what keep the Euclidean heuristic admissible for A*.

An AccessibilityProfile describes what the user can/cannot use:
    * blocked edge types are removed from the search (e.g. stairs for a wheelchair user)
    * penalty multipliers (>= 1.0) make some edge types more expensive
      (e.g. outdoor paths in bad weather). Multipliers are clamped to >= 1 so
      cost never drops below the physical length.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Edge types
# ---------------------------------------------------------------------------
STAIRS = "stairs"
RAMP = "ramp"
ELEVATOR = "elevator"
FLAT = "flat"
OUTDOOR = "outdoor"

EDGE_TYPES = (STAIRS, RAMP, ELEVATOR, FLAT, OUTDOOR)

# Which edge types are wheelchair-accessible
ACCESSIBLE_TYPES = {RAMP, ELEVATOR, FLAT, OUTDOOR}


@dataclass(frozen=True)
class Node:
    id: str
    name: str
    x: float
    y: float


@dataclass(frozen=True)
class Edge:
    u: str
    v: str
    length: float          # metres, the base cost
    kind: str = FLAT       # one of EDGE_TYPES
    outdoor: bool = False  # exposed to weather?


@dataclass(frozen=True)
class AccessibilityProfile:
    """What a user can use, and how much they dislike each edge type."""
    name: str
    blocked: frozenset = frozenset()
    multipliers: Tuple[Tuple[str, float], ...] = ()  # (edge kind or "outdoor", multiplier)
    description: str = ""

    def edge_cost(self, edge: Edge) -> Optional[float]:
        """Return the cost of using `edge`, or None if the edge is unusable."""
        if edge.kind in self.blocked:
            return None
        if edge.outdoor and "outdoor" in self.blocked:
            return None
        mult = 1.0
        for key, m in self.multipliers:
            if key == edge.kind or (key == "outdoor" and edge.outdoor):
                mult *= max(1.0, m)  # never below physical length
        return edge.length * mult


PROFILES: Dict[str, AccessibilityProfile] = {
    "standard": AccessibilityProfile(
        "standard", description="All paths available; cost = distance."),
    "no_stairs": AccessibilityProfile(
        "no_stairs", blocked=frozenset({STAIRS}),
        description="Wheelchair / mobility aid: stairs unavailable, must use ramps or elevators."),
    "poor_weather": AccessibilityProfile(
        "poor_weather", blocked=frozenset({"outdoor"}),
        description="Poor weather: outdoor paths unavailable, indoor routes only."),
    "accessibility_challenge": AccessibilityProfile(
        "accessibility_challenge", blocked=frozenset({STAIRS}),
        multipliers=(("outdoor", 1.5), (RAMP, 1.2)),
        description="No stairs; outdoor paths cost 1.5x and ramps 1.2x (effort-weighted)."),
}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
class CampusGraph:
    """Undirected weighted graph of campus locations.

    `metres_per_unit` converts map coordinates to metres so the heuristic and
    edge lengths are in the same unit (proposal: 1 unit = 15 m).
    """

    def __init__(self, name: str = "campus", metres_per_unit: float = 15.0):
        self.name = name
        self.metres_per_unit = metres_per_unit
        self.nodes: Dict[str, Node] = {}
        self._adj: Dict[str, List[Edge]] = {}

    # ---- construction -----------------------------------------------------
    def add_node(self, node_id: str, x: float, y: float, name: Optional[str] = None) -> None:
        if node_id in self.nodes:
            raise ValueError(f"duplicate node {node_id!r}")
        self.nodes[node_id] = Node(node_id, name or node_id, float(x), float(y))
        self._adj[node_id] = []

    def add_edge(self, u: str, v: str, length: Optional[float] = None,
                 kind: str = FLAT, outdoor: Optional[bool] = None) -> Edge:
        if u not in self.nodes or v not in self.nodes:
            raise KeyError(f"unknown node in edge {u}-{v}")
        if kind not in EDGE_TYPES:
            raise ValueError(f"unknown edge type {kind!r}")
        physical = self.straight_line(u, v)
        if length is None:
            length = physical
        if length <= 0:
            raise ValueError(f"edge {u}-{v} must have positive cost")
        if length + 1e-9 < physical:
            raise ValueError(
                f"edge {u}-{v} length {length:.1f} m is shorter than the straight-line "
                f"distance {physical:.1f} m (breaks heuristic admissibility)")
        if outdoor is None:
            outdoor = (kind == OUTDOOR)
        e = Edge(u, v, float(length), kind, bool(outdoor))
        self._adj[u].append(e)
        self._adj[v].append(Edge(v, u, e.length, e.kind, e.outdoor))
        return e

    # ---- queries ----------------------------------------------------------
    def straight_line(self, a: str, b: str) -> float:
        """Euclidean distance between two nodes, in metres."""
        na, nb = self.nodes[a], self.nodes[b]
        return math.hypot(na.x - nb.x, na.y - nb.y) * self.metres_per_unit

    def neighbors(self, node_id: str,
                  profile: Optional[AccessibilityProfile] = None
                  ) -> Iterator[Tuple[str, float, Edge]]:
        """Yield (neighbor, step_cost, edge) for every usable edge."""
        profile = profile or PROFILES["standard"]
        for e in self._adj[node_id]:
            c = profile.edge_cost(e)
            if c is not None:
                yield e.v, c, e

    def edges(self) -> Iterator[Edge]:
        """Each undirected edge once."""
        seen = set()
        for u, lst in self._adj.items():
            for e in lst:
                key = tuple(sorted((e.u, e.v))) + (e.kind,)
                if key not in seen:
                    seen.add(key)
                    yield e

    def edge_between(self, u: str, v: str,
                     profile: Optional[AccessibilityProfile] = None) -> Optional[Tuple[float, Edge]]:
        """Cheapest usable edge between u and v under the profile."""
        best = None
        for nb, c, e in self.neighbors(u, profile):
            if nb == v and (best is None or c < best[0]):
                best = (c, e)
        return best

    def path_cost(self, path: List[str],
                  profile: Optional[AccessibilityProfile] = None) -> float:
        total = 0.0
        for a, b in zip(path, path[1:]):
            step = self.edge_between(a, b, profile)
            if step is None:
                raise ValueError(f"no usable edge {a}-{b} under this profile")
            total += step[0]
        return total

    def __len__(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return sum(1 for _ in self.edges())

    # ---- persistence ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "metres_per_unit": self.metres_per_unit,
            "nodes": [{"id": n.id, "name": n.name, "x": n.x, "y": n.y}
                      for n in self.nodes.values()],
            "edges": [{"u": e.u, "v": e.v, "length": e.length, "kind": e.kind,
                       "outdoor": e.outdoor} for e in self.edges()],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CampusGraph":
        g = cls(d.get("name", "campus"), d.get("metres_per_unit", 15.0))
        for n in d["nodes"]:
            g.add_node(n["id"], n["x"], n["y"], n.get("name"))
        for e in d["edges"]:
            g.add_edge(e["u"], e["v"], e["length"], e.get("kind", FLAT), e.get("outdoor"))
        return g

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CampusGraph":
        with open(path) as f:
            return cls.from_dict(json.load(f))
