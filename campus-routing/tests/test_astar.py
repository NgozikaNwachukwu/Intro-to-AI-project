
import unittest
from pathlib import Path
import sys

# Allow imports from campus-routing
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.astar import astar
from graphs.sample_graph import graph, coordinates


class TestAStar(unittest.TestCase):

    def test_normal_route(self):
        result = astar(graph, "S", "G", coordinates)

        self.assertTrue(result["solution_found"])
        self.assertEqual(result["solution_cost"], 320)
        self.assertIn(
            result["path"],
            [
                ["S", "L", "E", "G"],
                ["S", "B", "E", "G"]
            ]
        )

    def test_avoid_stairs(self):
        result = astar(
            graph, "S", "G", coordinates,
            avoid_stairs=True
        )

        self.assertTrue(result["solution_found"])
        self.assertEqual(result["path"], ["S", "B", "E", "G"])
        self.assertEqual(result["solution_cost"], 320)

    def test_start_equals_goal(self):
        result = astar(graph, "S", "S", coordinates)

        self.assertTrue(result["solution_found"])
        self.assertEqual(result["path"], ["S"])
        self.assertEqual(result["solution_cost"], 0)

    def test_unreachable_destination(self):
        result = astar(graph, "G", "S", coordinates)

        self.assertFalse(result["solution_found"])
        self.assertEqual(result["path"], [])
        self.assertIsNone(result["solution_cost"])

    def test_cheaper_alternative_path(self):
        test_graph = {
            "A": [
                {"to": "B", "cost": 10, "type": "flat"},
                {"to": "C", "cost": 2, "type": "flat"}
            ],
            "B": [
                {"to": "D", "cost": 1, "type": "flat"}
            ],
            "C": [
                {"to": "B", "cost": 3, "type": "flat"}
            ],
            "D": []
        }

        test_coordinates = {
            "A": (0, 0),
            "B": (0, 0),
            "C": (0, 0),
            "D": (0, 0)
        }

        result = astar(
            test_graph, "A", "D", test_coordinates
        )

        self.assertTrue(result["solution_found"])
        self.assertEqual(result["path"], ["A", "C", "B", "D"])
        self.assertEqual(result["solution_cost"], 6)

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

        result = astar(
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

        result = astar(
            test_graph, "A", "D", test_coordinates,
            indoor_only=True
        )

        self.assertTrue(result["solution_found"])
        self.assertEqual(result["path"], ["A", "C", "D"])
        self.assertEqual(result["solution_cost"], 16)

    def test_metrics_returned(self):
        result = astar(graph, "S", "G", coordinates)

        self.assertGreater(result["nodes_explored"], 0)
        self.assertGreaterEqual(result["max_frontier_size"], 1)
        self.assertGreaterEqual(result["execution_time"], 0)
        self.assertGreaterEqual(result["max_memory"], 0)


if __name__ == "__main__":
    unittest.main()
