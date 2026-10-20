
import heapq
import itertools
import time
import tracemalloc

from algorithms.astar import heuristic as straight_line


def greedy_best_first(graph, start, goal, coordinates,
                      avoid_stairs=False, indoor_only=False,
                      heuristic=None):
    """
    Greedy Best-First Search for accessibility-aware campus routing.

    Always expands the frontier node that looks closest to the goal,
    using f(n) = h(n) only. Fast, but NOT optimal: when the cheapest
    accessible route first moves away from the goal, Greedy can return
    a costlier route. Complete on finite graphs (graph search).

    heuristic(node, goal, coordinates) defaults to the straight-line
    distance used by A*. Returns results in the same format as UCS/A*.
    """

    h = heuristic or straight_line

    tracemalloc.start()
    start_time = time.perf_counter()

    # Ties in h(n) are broken by insertion order (FIFO) so runs are
    # deterministic and reproducible.
    counter = itertools.count()

    # Priority queue: (h_cost, insertion_order, node)
    frontier = [(h(start, goal, coordinates), next(counter), start)]

    # Previous node for route reconstruction. A node is "reached" once
    # it has a parent; since h(n) never changes, a second path to it
    # could not change its priority, so it is never re-opened.
    parent = {start: None}

    # Actual distance travelled along the parent chain
    path_cost = {start: 0}

    visited = set()
    nodes_explored = 0
    max_frontier_size = 1

    while frontier:

        # Select node that looks closest to the goal
        _, _, current = heapq.heappop(frontier)

        if current in visited:
            continue

        visited.add(current)
        nodes_explored += 1

        # Stop when destination is reached
        if current == goal:
            break

        for edge in graph.get(current, []):
            neighbor = edge["to"]
            edge_cost = edge["cost"]
            edge_type = edge["type"]

            if edge_cost < 0:
                raise ValueError(
                    "Greedy search requires non-negative edge costs."
                )

            # Apply accessibility restrictions
            if avoid_stairs and edge_type == "stairs":
                continue

            if indoor_only and edge.get("environment") != "indoor":
                continue

            # Already reached: never re-open in Greedy
            if neighbor in parent:
                continue

            parent[neighbor] = current
            path_cost[neighbor] = path_cost[current] + edge_cost

            heapq.heappush(
                frontier,
                (h(neighbor, goal, coordinates), next(counter), neighbor)
            )

        max_frontier_size = max(max_frontier_size, len(frontier))

    solution_found = goal in visited

    path = []
    solution_cost = None

    if solution_found:
        node = goal

        while node is not None:
            path.append(node)
            node = parent[node]

        path.reverse()
        solution_cost = path_cost[goal]

    execution_time = time.perf_counter() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "path": path,
        "solution_cost": solution_cost,
        "solution_found": solution_found,
        "nodes_explored": nodes_explored,
        "max_frontier_size": max_frontier_size,
        "execution_time": execution_time,
        "max_memory": peak_memory
    }


if __name__ == "__main__":
    from graphs.sample_graph import graph, coordinates

    result = greedy_best_first(graph, "S", "G", coordinates)

    print("Greedy Best-First Search Results")
    print("Path:", result["path"])
    print("Total Distance:", result["solution_cost"], "metres")
    print("Nodes Explored:", result["nodes_explored"])
    print("Execution Time:", result["execution_time"])
