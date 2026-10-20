"""
Route visualization (matplotlib).

* Edges are styled by type (stairs = red dashed, ramp = blue, elevator = purple,
  flat = grey, outdoor = green dotted). Edges the profile cannot use are faded.
* Nodes the algorithm expanded are shaded orange (darker = expanded earlier).
* The returned route is drawn as a thick line with its total cost in the title.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .graph import AccessibilityProfile, CampusGraph, PROFILES
from .search import SearchResult

EDGE_STYLE = {
    "stairs":   dict(color="#d62728", ls="--"),
    "ramp":     dict(color="#1f77b4", ls="-"),
    "elevator": dict(color="#9467bd", ls="-"),
    "flat":     dict(color="#7f7f7f", ls="-"),
    "outdoor":  dict(color="#2ca02c", ls=":"),
}
PATH_COLOR = "#111111"


def _draw_one(ax, g: CampusGraph, res: Optional[SearchResult],
              profile: AccessibilityProfile, label_nodes: bool, title: str):
    small = len(g) <= 40
    for e in g.edges():
        a, b = g.nodes[e.u], g.nodes[e.v]
        st = EDGE_STYLE[e.kind]
        usable = profile.edge_cost(e) is not None
        ax.plot([a.x, b.x], [a.y, b.y], color=st["color"], ls=st["ls"],
                lw=1.6 if small else 0.9, alpha=0.85 if usable else 0.15, zorder=1)
        if small and label_nodes:
            ax.text((a.x + b.x) / 2, (a.y + b.y) / 2, f"{e.length:.0f}",
                    fontsize=7, color=st["color"], ha="center", va="center",
                    bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5), zorder=2)

    expanded = res.expansion_order if res else []
    rank = {n: i for i, n in enumerate(expanded)}
    xs = [n.x for n in g.nodes.values()]
    ys = [n.y for n in g.nodes.values()]
    colors = []
    for nid in g.nodes:
        if nid in rank:
            t = rank[nid] / max(1, len(expanded) - 1)
            colors.append((1.0, 0.55 + 0.35 * t, 0.15 + 0.6 * t))
        else:
            colors.append("white")
    ax.scatter(xs, ys, s=260 if small else 28, c=colors, edgecolors="#333",
               linewidths=0.8, zorder=3)

    if res and res.found and len(res.path) > 1:
        px = [g.nodes[n].x for n in res.path]
        py = [g.nodes[n].y for n in res.path]
        ax.plot(px, py, color=PATH_COLOR, lw=4 if small else 2.5, alpha=0.75,
                zorder=4, solid_capstyle="round")
    if res:
        for nid, col, lab in ((res.start, "#2ca02c", "START"), (res.goal, "#d62728", "GOAL")):
            n = g.nodes[nid]
            ax.scatter([n.x], [n.y], s=420 if small else 120, c=col, marker="*",
                       edgecolors="black", zorder=5)
    if small and label_nodes:
        for n in g.nodes.values():
            ax.text(n.x, n.y, n.id, fontsize=8, ha="center", va="center",
                    weight="bold", zorder=6)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def legend_handles():
    hs = [Line2D([0], [0], color=v["color"], ls=v["ls"], lw=2, label=k)
          for k, v in EDGE_STYLE.items()]
    hs.append(Line2D([0], [0], color=PATH_COLOR, lw=4, alpha=0.75, label="route found"))
    hs.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=(1, 0.6, 0.2),
                     markeredgecolor="#333", markersize=10, label="expanded node"))
    return hs


def plot_results(g: CampusGraph, results: Sequence[SearchResult],
                 profile: Optional[AccessibilityProfile] = None,
                 outfile: str = "route.png", label_nodes: bool = True) -> str:
    """Plot one panel per result (side by side) and save to `outfile`."""
    profile = profile or PROFILES["standard"]
    n = max(1, len(results))
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.8), squeeze=False)
    for ax, res in zip(axes[0], results or [None]):
        if res is None:
            title = g.name
        elif res.found:
            title = (f"{res.algorithm}: cost {res.cost:.0f} m\n"
                     f"expanded {res.nodes_expanded} nodes")
        else:
            title = f"{res.algorithm}: NO ACCESSIBLE ROUTE\nexpanded {res.nodes_expanded}"
        _draw_one(ax, g, res, profile, label_nodes, title)
    fig.suptitle(f"{g.name}  |  profile: {profile.name}", fontsize=11)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=7, fontsize=8,
               frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile


def text_route(g: CampusGraph, res: SearchResult,
               profile: Optional[AccessibilityProfile] = None) -> str:
    """Turn-by-turn textual description of a route."""
    profile = profile or PROFILES["standard"]
    if not res.found:
        return f"[{res.algorithm}] No route satisfies profile '{profile.name}'."
    lines = [f"[{res.algorithm}] {g.nodes[res.start].name} -> {g.nodes[res.goal].name}"]
    total = 0.0
    for i, (a, b) in enumerate(zip(res.path, res.path[1:]), 1):
        c, e = g.edge_between(a, b, profile)
        total += c
        lines.append(f"  {i}. {g.nodes[a].name} -> {g.nodes[b].name} "
                     f"via {e.kind}{' (outdoor)' if e.outdoor and e.kind != 'outdoor' else ''}"
                     f"  {e.length:.0f} m  (cost {c:.0f}, running {total:.0f})")
    lines.append(f"  Total cost: {res.cost:.1f}  |  nodes expanded: {res.nodes_expanded}"
                 f"  |  time: {res.time_ms:.3f} ms")
    return "\n".join(lines)
