
import heapq
import math
import time
import tracemalloc


def heuristic(node, goal, coordinates):
    """
    Estimate straight-line distance to the destination.
    Each coordinate unit represents 15 metres.
    """
    x1, y1 = coordinates[node]
    x2, y2 = coordinates[goal]

    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    return distance * 15


def astar(graph, start, goal, coordinates,
          avoid_stairs=False, indoor_only=False):
    """
    A* Search for accessibility-aware campus routing.

    Finds the lowest-cost valid route using f(n) = g(n) + h(n).
    Assumes a consistent heuristic and non-negative edge costs.
    Returns the route and performance metrics.
    """

    tracemalloc.start()
    start_time = time.perf_counter()

    # Priority queue: (f_cost, g_cost, node)
    frontier = [(heuristic(start, goal, coordinates), 0, start)]

    # Best known cost to reach each node
    best_cost = {start: 0}

    # Store previous nodes for route reconstruction
    parent = {start: None}

    # Track explored nodes
    visited = set()
    nodes_explored = 0
    max_frontier_size = 1

    while frontier:

        # Select node with lowest estimated total cost
        f_cost, current_cost, current = heapq.heappop(frontier)

        # Skip already explored nodes
        if current in visited:
            continue

        # Skip outdated queue entries
        if current_cost > best_cost[current]:
            continue

        visited.add(current)
        nodes_explored += 1

        # Stop when destination is reached
        if current == goal:
            break

        # Explore neighboring campus locations
        for edge in graph.get(current, []):
            neighbor = edge["to"]
            edge_cost = edge["cost"]
            edge_type = edge["type"]

            # Reject negative edge costs
            if edge_cost < 0:
                raise ValueError(
                    "A* requires non-negative edge costs."
                )

            # Apply accessibility restrictions
            if avoid_stairs and edge_type == "stairs":
                continue

            if indoor_only and edge.get("environment") != "indoor":
                continue

            # Calculate actual distance travelled
            new_cost = current_cost + edge_cost

            if neighbor in visited:
                continue

            # Update if a cheaper path is discovered
            if neighbor not in best_cost or new_cost < best_cost[neighbor]:

                best_cost[neighbor] = new_cost
                parent[neighbor] = current

                # A* formula: f(n) = g(n) + h(n)
                estimated_total = new_cost + heuristic(
                    neighbor, goal, coordinates
                )

                # Add neighbor to priority queue
                heapq.heappush(
                    frontier,
                    (estimated_total, new_cost, neighbor)
                )

        # Track maximum frontier size
        max_frontier_size = max(
            max_frontier_size, len(frontier)
        )

    # Check whether destination was reached
    solution_found = goal in visited

    path = []
    solution_cost = None

    # Reconstruct route from goal back to start
    if solution_found:
        node = goal

        while node is not None:
            path.append(node)
            node = parent[node]

        # Reverse path to get start-to-goal directions
        path.reverse()

        solution_cost = best_cost[goal]

    # Calculate performance metrics
    execution_time = time.perf_counter() - start_time

    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Return results in same format as UCS
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

    result = astar(graph, "S", "G", coordinates)

    print("A* Search Results")
    print("Path:", result["path"])
    print("Total Distance:", result["solution_cost"], "metres")
    print("Nodes Explored:", result["nodes_explored"])
    print("Execution Time:", result["execution_time"])
