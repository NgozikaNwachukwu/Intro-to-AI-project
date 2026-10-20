# Sample campus graph for testing UCS, A*, and Greedy Best-First Search.
# Edge costs represent distances in metres.
# All edges are directed, as in our proposal's example.

graph = {
    "S": [
        {"to": "L", "cost": 100, "type": "stairs"},
        {"to": "B", "cost": 150, "type": "elevator"},
        {"to": "R", "cost": 200, "type": "ramp"}
    ],
    "L": [
        {"to": "B", "cost": 80, "type": "ramp"},
        {"to": "E", "cost": 120, "type": "stairs"}
    ],
    "B": [
        {"to": "E", "cost": 70, "type": "elevator"},
        {"to": "SC", "cost": 90, "type": "flat"},
        {"to": "R", "cost": 100, "type": "stairs"}
    ],
    "R": [
        {"to": "SC", "cost": 80, "type": "flat"}
    ],
    "E": [
        {"to": "G", "cost": 100, "type": "ramp"}
    ],
    "SC": [
        {"to": "G", "cost": 150, "type": "elevator"}
    ],
    "G": []
}

# Coordinates used by A* and Greedy for Euclidean heuristics.
# Scale: 1 coordinate unit = 15 metres.
coordinates = {
    "S": (0, 4),
    "L": (2, 8),
    "B": (5, 4),
    "R": (2, 0),
    "E": (7, 8),
    "SC": (7, 0),
    "G": (10, 4)
}

location_names = {
    "S": "Main Entrance",
    "L": "Library",
    "B": "Business",
    "R": "Residence",
    "E": "Engineering",
    "SC": "Science",
    "G": "Gym"
}