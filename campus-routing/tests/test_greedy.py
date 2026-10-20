
import unittest
from pathlib import Path
import sys

# Allow imports from campus-routing
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.greedy import greedy_best_first
from algorithms.ucs import ucs
from graphs.sample_graph import graph, coordinates


class TestGreedy(unittest.TestCase):

    def test_normal_route(self):
        # h(S)=150, h(L)=h(R)=134.2, h(B)=75 -> expand S, then B.
        # From B, E and SC tie at h=75; FIFO picks E, which reaches G.
        result = greedy_best_first(graph, "S", "G", coordinates)

        self.assertTrue(result["solution_found"])
        self.assertEqual(result["path"], ["S", "B", "E", "G"])
        self.assertEqual(result["solution_cost"], 320)
        self.assertEqual(result["nodes_explored"], 4)

    def test_avoid_stairs(self):
        result = greedy_best_first(
            graph, "S", "G", coordinates,
            avoid_stairs=True
        )

        self.assertTrue(result["solution_found"])
        self.assertEqual(result["path"], ["S", "B", "E", "G"])
        self.assertEqual(result["solution_cost"], 320)

    def test_start_equals_goal(self):
        result = greedy_best_first(graph, "S", "S", coordinates)

        self.assertTrue(result["solution_found"])
        self.assertEqual(result["path"], ["S"])
        self.assertEqual(result["solution_cost"], 0)

    def test_unreachable_destination(self):
        result = greedy_best_first(graph, "G", "S", coordinates)

        self.assertFalse(result["solution_found"])
        self.assertEqual(result["path"], [])
        self.assertIsNone(result["solution_cost"])

    def test_accessibility_blocks_route(self):
        test_graph = {
            "A": [
                {"to": "B", "cost": 5, "type": "stairs"}
            ],
            "B": []
        }

        test_coordinates = {
            "A": (0, 0),
            "B": (0, 0)
        }

        result = greedy_best_first(
            test_graph, "A", "B", test_coordinates,
            avoid_stairs=True
        )

        self.assertFalse(result["solution_found"])

    def test_indoor_only(self):
        test_graph = {
            "A": [
                {"to": "B", "cost": 5, "type": "flat",
                 "environment": "outdoor"},
                {"to": "C", "cost": 8, "type": "ramp",
                 "environment": "indoor"}
            ],
            "B": [
                {"to": "D", "cost": 5, "type": "flat",
                 "environment": "indoor"}
            ],
            "C": [
                {"to": "D", "cost": 8, "type": "elevator",
                 "environment": "indoor"}
            ],
            "D": []
        }

        test_coordinates = {
            "A": (0, 0),
            "B": (0, 0),
            "C": (0, 0),
            "D": (0, 0)
        }

        result = greedy_best_first(
            test_graph, "A", "D", test_coordinates,
            indoor_only=True
        )

        self.assertTrue(result["solution_found"])
        self.assertEqual(result["path"], ["A", "C", "D"])
        self.assertEqual(result["solution_cost"], 16)

    def test_not_optimal_when_route_looks_wrong_way(self):
        # B looks closer to D, but B->D is a long ramp. The cheapest
        # route goes through C, which first moves away from D.
        test_graph = {
            "A": [
                {"to": "B", "cost": 50, "type": "flat"},
                {"to": "C", "cost": 50, "type": "flat"}
            ],
            "B": [
                {"to": "D", "cost": 200, "type": "ramp"}
            ],
            "C": [
                {"to": "D", "cost": 80, "type": "flat"}
            ],
            "D": []
        }

        test_coordinates = {
            "A": (0, 0),
            "B": (3, 0),
            "C": (0, 3),
            "D": (4, 0)
        }

        greedy = greedy_best_first(
            test_graph, "A", "D", test_coordinates
        )
        optimal = ucs(test_graph, "A", "D")

        self.assertEqual(greedy["path"], ["A", "B", "D"])
        self.assertEqual(greedy["solution_cost"], 250)
        self.assertEqual(optimal["solution_cost"], 130)
        self.assertLessEqual(
            greedy["nodes_explored"], optimal["nodes_explored"]
        )

    def test_custom_heuristic(self):
        def zero(node, goal, coords):
            return 0

        result = greedy_best_first(
            graph, "S", "G", coordinates, heuristic=zero
        )

        self.assertTrue(result["solution_found"])
        self.assertEqual(result["path"][0], "S")
        self.assertEqual(result["path"][-1], "G")

    def test_negative_cost_rejected(self):
        test_graph = {
            "A": [
                {"to": "B", "cost": -1, "type": "flat"}
            ],
            "B": []
        }

        with self.assertRaises(ValueError):
            greedy_best_first(
                test_graph, "A", "B", {"A": (0, 0), "B": (0, 0)}
            )

    def test_metrics_returned(self):
        result = greedy_best_first(graph, "S", "G", coordinates)

        self.assertGreater(result["nodes_explored"], 0)
        self.assertGreaterEqual(result["max_frontier_size"], 1)
        self.assertGreaterEqual(result["execution_time"], 0)
        self.assertGreaterEqual(result["max_memory"], 0)


if __name__ == "__main__":
    unittest.main()
