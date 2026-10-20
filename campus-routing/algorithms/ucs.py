
import heapq
import time
import tracemalloc


def ucs(graph, start, goal, avoid_stairs=False, indoor_only=False):
    """
    Uniform Cost Search (UCS) for accessibility-aware campus routing.

    Returns the lowest-cost valid route and performance metrics.
    All edge costs must be non-negative.
    """

    tracemalloc.start()
    start_time = time.perf_counter()

    frontier = [(0, start)]
    best_cost = {start: 0}
    parent = {start: None}
    visited = set()

    nodes_explored = 0
    max_frontier_size = 1

    while frontier:
        current_cost, current = heapq.heappop(frontier)

        if current in visited:
            continue

        visited.add(current)
        nodes_explored += 1

        if current == goal:
            break

        for edge in graph.get(current, []):
            neighbor = edge["to"]
            edge_cost = edge["cost"]
            edge_type = edge["type"]

            if edge_cost < 0:
                raise ValueError("UCS requires non-negative edge costs.")

            # Apply accessibility restrictions
            if avoid_stairs and edge_type == "stairs":
                continue

            # Edges without indoor/outdoor metadata cannot be
            # verified as indoor, so exclude them in indoor-only mode.
            if indoor_only and edge.get("environment") != "indoor":
                continue

            new_cost = current_cost + edge_cost

            if neighbor in visited:
                continue

            if neighbor not in best_cost or new_cost < best_cost[neighbor]:
                best_cost[neighbor] = new_cost
                parent[neighbor] = current
                heapq.heappush(frontier, (new_cost, neighbor))

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
        solution_cost = best_cost[goal]

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
