"""Correctness tests. Run with:  python -m unittest discover -s tests -v"""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from campus_nav.graph import CampusGraph, PROFILES, STAIRS, FLAT
from campus_nav.heuristics import euclidean
from campus_nav.maps import detour_map, proposal_map, random_campus, random_endpoints
from campus_nav.search import a_star, greedy_best_first, uniform_cost


def path_is_valid(g, res, profile):
    """Every consecutive pair is joined by a usable edge and cost adds up."""
    if res.path[0] != res.start or res.path[-1] != res.goal:
        return False
    try:
        return abs(g.path_cost(res.path, profile) - res.cost) < 1e-6
    except ValueError:
        return False


def brute_force_best(g, s, t, profile):
    """Cheapest simple path by exhaustive DFS (small graphs only)."""
    best = [float("inf")]

    def dfs(n, seen, cost):
        if cost >= best[0]:
            return
        if n == t:
            best[0] = cost
            return
        for nb, c, _ in g.neighbors(n, profile):
            if nb not in seen:
                seen.add(nb)
                dfs(nb, seen, cost + c)
                seen.remove(nb)
    dfs(s, {s}, 0.0)
    return best[0]


class TestGraph(unittest.TestCase):
    def test_rejects_edge_shorter_than_straight_line(self):
        g = CampusGraph(metres_per_unit=10)
        g.add_node("a", 0, 0); g.add_node("b", 3, 4)   # 50 m apart
        with self.assertRaises(ValueError):
            g.add_edge("a", "b", 40)

    def test_rejects_non_positive_cost(self):
        g = CampusGraph()
        g.add_node("a", 0, 0); g.add_node("b", 0, 0)
        with self.assertRaises(ValueError):
            g.add_edge("a", "b", 0)

    def test_no_stairs_profile_blocks_stairs(self):
        g = proposal_map()
        nbrs = {n for n, _, _ in g.neighbors("S", PROFILES["no_stairs"])}
        self.assertNotIn("L", nbrs)              # S-L is stairs only
        self.assertEqual(nbrs, {"B", "R"})

    def test_json_round_trip(self):
        g = random_campus(5, 5, seed=3)
        g2 = CampusGraph.from_dict(g.to_dict())
        self.assertEqual(len(g), len(g2))
        self.assertEqual(g.num_edges, g2.num_edges)


class TestGreedy(unittest.TestCase):
    def test_proposal_map_hand_trace(self):
        # h(S)=150, h(L)=h(R)=134.2, h(B)=75 -> expand S, then B;
        # from B, E and SC tie at h=75 -> FIFO picks E (generated first); E reaches G.
        g = proposal_map()
        r = greedy_best_first(g, "S", "G", PROFILES["standard"])
        self.assertTrue(r.found)
        self.assertEqual(r.expansion_order, ["S", "B", "E"])
        self.assertEqual(r.path, ["S", "B", "E", "G"])
        self.assertAlmostEqual(r.cost, 320.0)

    def test_route_respects_no_stairs(self):
        g = proposal_map()
        prof = PROFILES["no_stairs"]
        for s in g.nodes:
            for t in g.nodes:
                if s == t:
                    continue
                r = greedy_best_first(g, s, t, prof)
                self.assertTrue(r.found)
                self.assertTrue(path_is_valid(g, r, prof))
                for a, b in zip(r.path, r.path[1:]):
                    _, e = g.edge_between(a, b, prof)
                    self.assertNotEqual(e.kind, STAIRS)

    def test_start_equals_goal(self):
        g = proposal_map()
        r = greedy_best_first(g, "B", "B")
        self.assertTrue(r.found)
        self.assertEqual(r.path, ["B"])
        self.assertEqual(r.cost, 0)

    def test_reports_no_route(self):
        g = CampusGraph(metres_per_unit=10)
        for i, (x, y) in enumerate([(0, 0), (1, 0), (2, 0)]):
            g.add_node(str(i), x, y)
        g.add_edge("0", "1", 10, FLAT)
        g.add_edge("1", "2", 10, STAIRS)
        r = greedy_best_first(g, "0", "2", PROFILES["no_stairs"])
        self.assertFalse(r.found)
        self.assertEqual(r.path, [])
        self.assertEqual(r.nodes_expanded, 2)  # explored everything reachable

    def test_detour_map_greedy_is_suboptimal(self):
        g = detour_map()
        prof = PROFILES["no_stairs"]
        gr = greedy_best_first(g, "S", "G", prof)
        uc = uniform_cost(g, "S", "G", prof)
        self.assertAlmostEqual(gr.cost, 420.0)
        self.assertAlmostEqual(uc.cost, 343.0)
        self.assertEqual(uc.path, ["S", "W", "N1", "N2", "N3", "G"])
        self.assertLess(gr.nodes_expanded, uc.nodes_expanded)

    def test_early_goal_test_same_route(self):
        rng = random.Random(11)
        for i in range(50):
            g = random_campus(8, 8, seed=i)
            s, t = random_endpoints(g, rng)
            a = greedy_best_first(g, s, t)
            b = greedy_best_first(g, s, t, early_goal_test=True)
            self.assertEqual(a.path, b.path)
            self.assertLessEqual(b.nodes_expanded, a.nodes_expanded)


class TestAgainstBaselines(unittest.TestCase):
    def test_random_instances(self):
        """On many random campuses: completeness, validity, and cost ordering."""
        rng = random.Random(2026)
        for i in range(300):
            g = random_campus(rng.randint(4, 12), rng.randint(4, 12),
                              density=rng.random(), stairs_frac=rng.random() * 0.6,
                              barrier=rng.random() < 0.3, seed=i)
            s, t = random_endpoints(g, rng)
            prof = PROFILES[rng.choice(list(PROFILES))]
            gr = greedy_best_first(g, s, t, prof)
            uc = uniform_cost(g, s, t, prof)
            ast = a_star(g, s, t, prof)
            self.assertEqual(gr.found, uc.found)          # GBFS graph search is complete
            self.assertEqual(ast.found, uc.found)
            if uc.found:
                self.assertTrue(path_is_valid(g, gr, prof))
                self.assertTrue(path_is_valid(g, uc, prof))
                self.assertGreaterEqual(gr.cost + 1e-6, uc.cost)   # UCS optimal
                self.assertAlmostEqual(ast.cost, uc.cost, places=6)  # A* optimal (admissible h)
                self.assertLessEqual(ast.nodes_expanded, uc.nodes_expanded)

    def test_ucs_matches_brute_force(self):
        rng = random.Random(5)
        for i in range(40):
            g = random_campus(4, 3, density=0.7, stairs_frac=0.3, seed=100 + i)
            s, t = random_endpoints(g, rng)
            for pname in ("standard", "no_stairs"):
                prof = PROFILES[pname]
                best = brute_force_best(g, s, t, prof)
                uc = uniform_cost(g, s, t, prof)
                if best == float("inf"):
                    self.assertFalse(uc.found)
                else:
                    self.assertAlmostEqual(uc.cost, best, places=6)


if __name__ == "__main__":
    unittest.main()
