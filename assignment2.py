"""
Treasure Hunt Pathfinding with Uniform‐Cost Search (UCS)

This script implements a solver for the “treasure hunt” assignment:
a single‐agent must collect all treasures on a grid map. Stepping on certain
cells (traps or rewards) dynamically modifies movement cost (gravity/speed),
or teleports the agent (trap3), or removes all remaining treasures (trap4).

The chosen search algorithm is Uniform‐Cost Search because:
1. Costs are nonnegative but vary dynamically when traps/rewards are collected.
2. UCS guarantees optimality without needing a heuristic.
3. It cleanly handles dynamic edge costs (gravity × (1/speed)).

---------------------------------------------------------------
Author: [Milky]
Assignment: CSC3206 Assignment 2 – Treasure Hunt
Date: [1/6/2025]
Python Version: 3.x
---------------------------------------------------------------
"""

import heapq
from collections import namedtuple

# ------------------------------------------------------------
# 1) HARD-CODED 6×10 HEX MAP (even-r flat-top), with the “proper” neighbors you provided
# ------------------------------------------------------------

# 1A) CONTENT AT EACH COORDINATE
#    Keys: (row, col)  where row ∈ [0..5], col ∈ [0..9]
#    Values: one of:
#       'start'     → entry point
#       'empty'     → plain hex
#       'trap1','trap2','trap3','trap4' → trap types 1–4
#       'reward1','reward2'   → reward types 1–2
#       'treasure'  → collectible “C”
#       'obstacle'  → impassable “X”
#
cell_type = {
    (0,0): 'empty',
    (0,1): 'empty',
    (0,2): 'empty',
    (0,3): 'empty',
    (0,4): 'reward1',
    (0,5): 'empty',
    (0,6): 'empty',
    (0,7): 'empty',
    (0,8): 'empty',
    (0,9): 'empty',

    (1,0): 'empty',
    (1,1): 'trap2',
    (1,2): 'empty',
    (1,3): 'trap1',
    (1,4): 'treasure',
    (1,5): 'empty',
    (1,6): 'trap3',
    (1,7): 'empty',
    (1,8): 'obstacle',
    (1,9): 'empty',

    (2,0): 'empty',
    (2,1): 'empty',
    (2,2): 'obstacle',
    (2,3): 'empty',
    (2,4): 'obstacle',
    (2,5): 'empty',
    (2,6): 'empty',
    (2,7): 'reward2',
    (2,8): 'trap1',
    (2,9): 'empty',

    (3,0): 'obstacle',
    (3,1): 'reward1',
    (3,2): 'empty',
    (3,3): 'obstacle',
    (3,4): 'empty',
    (3,5): 'trap3',
    (3,6): 'obstacle',
    (3,7): 'treasure',
    (3,8): 'empty',
    (3,9): 'treasure',

    (4,0): 'empty',
    (4,1): 'empty',
    (4,2): 'trap2',
    (4,3): 'treasure',
    (4,4): 'obstacle',
    (4,5): 'empty',
    (4,6): 'obstacle',
    (4,7): 'obstacle',
    (4,8): 'empty',
    (4,9): 'empty',

    (5,0): 'empty',
    (5,1): 'empty',
    (5,2): 'empty',
    (5,3): 'empty',
    (5,4): 'empty',
    (5,5): 'reward2',
    (5,6): 'empty',
    (5,7): 'empty',
    (5,8): 'empty',
    (5,9): 'empty',
}

# 1B) EXACT NEIGHBOR LIST YOU PROVIDED FOR EACH COORDINATE
#    Keys: (row, col). Values: list of all neighbors (no 'trap4' in this map).
#
neighbors = {
    (0,0): [(1,0), (0,1)],

    (0,1): [(0,0), (1,0), (1,1), (0,2)],

    (0,2): [(0,1), (1,1), (1,2), (0,3)],

    (0,3): [(0,2), (1,2), (1,3), (0,4)],

    (0,4): [(0,3), (1,3), (1,4), (0,5)],

    (0,5): [(0,4), (1,4), (1,5), (0,6)],

    (0,6): [(0,5), (1,5), (1,6), (0,7)],

    (0,7): [(0,6), (1,6), (1,7), (0,8)],

    (0,8): [(0,7), (1,7), (1,8), (0,9)],

    (0,9): [(0,8), (1,8), (1,9)],

    (1,0): [(0,1), (0,0), (2,0), (2,1), (1,1)],

    (1,1): [(0,2), (0,1), (1,0), (2,1), (2,2), (1,2)],

    (1,2): [(0,3), (0,2), (1,1), (2,2), (2,3), (1,3)],

    (1,3): [(0,4), (0,3), (1,2), (2,3), (2,4), (1,4)],

    (1,4): [(0,5), (0,4), (1,3), (2,4), (2,5), (1,5)],

    (1,5): [(0,6), (0,5), (1,4), (2,5), (2,6), (1,6)],

    (1,6): [(0,7), (0,6), (1,5), (2,6), (2,7), (1,7)],

    (1,7): [(0,8), (0,7), (1,6), (2,7), (2,8), (1,8)],

    (1,8): [(0,9), (0,8), (1,7), (2,8), (2,9)],

    (1,9): [(0,9), (0,8), (1,8), (2,9)],

    (2,0): [(1,1), (1,0), (3,0), (3,1)],

    (2,1): [(1,2), (1,1), (2,0), (3,1), (3,2), (2,2)],

    (2,2): [(1,3), (1,2), (2,1), (3,2), (3,3)],

    (2,3): [(1,4), (1,3), (2,2), (3,3), (3,4), (2,4)],

    (2,4): [(1,5), (1,4), (2,3), (3,4), (3,5)],

    (2,5): [(1,6), (1,5), (2,4), (3,5), (3,6), (2,6)],

    (2,6): [(1,7), (1,6), (2,5), (3,6), (3,7), (2,7)],

    (2,7): [(1,8), (1,7), (2,6), (3,7), (3,8), (2,8)],

    (2,8): [(1,9), (1,8), (2,7), (3,8), (3,9), (2,9)],

    (2,9): [(1,9), (1,8), (2,8), (3,9)],

    (3,0): [(2,1), (2,0), (4,0), (4,1)],

    (3,1): [(2,2), (2,1), (3,0), (4,1), (4,2), (3,2)],

    (3,2): [(2,3), (2,2), (3,1), (4,2), (4,3), (3,3)],

    (3,3): [(2,4), (2,3), (3,2), (4,3), (4,4)],

    (3,4): [(2,5), (2,4), (3,3), (4,4), (4,5), (3,5)],

    (3,5): [(2,6), (2,5), (3,4), (4,5), (4,6), (3,6)],

    (3,6): [(2,7), (2,6), (3,5), (4,6), (4,7)],

    (3,7): [(2,8), (2,7), (3,6), (4,7), (4,8), (3,8)],

    (3,8): [(2,9), (2,8), (3,7), (4,8), (4,9), (3,9)],

    (3,9): [(2,9), (2,8), (3,8), (4,9)],

    (4,0): [(3,1), (3,0), (5,0), (5,1)],

    (4,1): [(3,2), (3,1), (4,0), (5,1), (5,2), (4,2)],

    (4,2): [(3,3), (3,2), (4,1), (5,2), (5,3), (4,3)],

    (4,3): [(3,4), (3,3), (4,2), (5,3), (5,4), (4,4)],

    (4,4): [(3,5), (3,4), (4,3), (5,4), (5,5)],

    (4,5): [(3,6), (3,5), (4,4), (5,5), (5,6), (4,6)],

    (4,6): [(3,7), (3,6), (4,5), (5,6), (5,7), (4,7)],

    (4,7): [(3,8), (3,7), (4,6), (5,7), (5,8), (4,8)],

    (4,8): [(3,9), (3,8), (4,7), (5,8), (5,9), (4,9)],

    (4,9): [(3,9), (3,8), (4,8), (5,9)],

    (5,0): [(4,1), (4,0), (5,1)],

    (5,1): [(4,2), (4,1), (5,0), (5,2)],

    (5,2): [(4,3), (4,2), (5,1), (5,3)],

    (5,3): [(4,4), (4,3), (5,2), (5,4)],

    (5,4): [(4,5), (4,4), (5,3), (5,5)],

    (5,5): [(4,6), (4,5), (5,4), (5,6)],

    (5,6): [(4,7), (4,6), (5,5), (5,7)],

    (5,7): [(4,8), (4,7), (5,6), (5,8)],

    (5,8): [(4,9), (4,8), (5,7), (5,9)],

    (5,9): [(4,9), (5,8)],
}

# 1C) If you want to treat obstacles ('obstacle') as impassable or trap4 as
#     blocked, you can filter them out of each neighbor list. In this map,
#     there is no 'trap4', so only obstacles would be filtered. Example:
# for coord, nbr_list in neighbors.items():
#     neighbors[coord] = [n for n in nbr_list if cell_type[n] != 'obstacle']


# ------------------------------------------------------------
# 2) BUILD GRID & TRACK START/TREASURES
# ------------------------------------------------------------

Cell = namedtuple('Cell', ['type', 'value'])
# type ∈ {'empty','obstacle','start','treasure','trap1','trap2','trap3','trap4','reward1','reward2'}

def make_grid():
    """
    Converts cell_type → a 6×10 2D grid of Cell objects, and returns:
      - grid: list of lists (6×10) of Cell
      - start: (r,c) for start
      - all_treasures: frozenset of (r,c) for each treasure
    """
    grid = [[None]*10 for _ in range(6)]
    start = None
    treasures = set()

    for (r, c), kind in cell_type.items():
        if kind == 'empty':
            grid[r][c] = Cell('empty', None)
        elif kind == 'obstacle':
            grid[r][c] = Cell('obstacle', None)
        elif kind == 'start':
            grid[r][c] = Cell('start', None)
            start = (r, c)
        elif kind == 'treasure':
            grid[r][c] = Cell('treasure', None)
            treasures.add((r, c))
        elif kind == 'trap1':
            grid[r][c] = Cell('trap1', None)
        elif kind == 'trap2':
            grid[r][c] = Cell('trap2', None)
        elif kind == 'trap3':
            grid[r][c] = Cell('trap3', None)
        elif kind == 'trap4':
            grid[r][c] = Cell('trap4', None)
        elif kind == 'reward1':
            grid[r][c] = Cell('reward1', None)
        elif kind == 'reward2':
            grid[r][c] = Cell('reward2', None)
        else:
            raise ValueError(f"Unknown map code {kind} at {(r,c)}")

    # If there was no explicit 'start' coordinate in your list, you could default
    # one, but the map you gave has no 'start' label. If needed, you can pick
    # one (e.g. (0,0) or wherever). For now, we'll treat (0,0) as 'empty' unless
    # you specify 'start' explicitly in cell_type. If you want (0,0) to be start,
    # change cell_type[(0,0)] = 'start' above.

    return grid, start, frozenset(treasures)


def get_neighbors_hardcoded(pos):
    """
    Return the pre‐computed list of neighbors for pos = (r,c).
    """
    return neighbors.get(pos, [])


# ------------------------------------------------------------
# 3) CELL EFFECTS & UCS IMPLEMENTATION
# ------------------------------------------------------------

class State:
    __slots__ = ('pos', 'remaining', 'gravity', 'speed', 'used_effects', 'last_move')

    def __init__(self, pos, remaining, gravity=1.0, speed=1.0,
                 used_effects=frozenset(), last_move=(0, 0)):
        self.pos = pos
        self.remaining = remaining      # frozenset of treasure coords still to collect
        self.gravity = gravity          # 1.0 = normal; 2.0 = doubled gravity, etc.
        self.speed = speed              # 1.0 = normal; 0.5 = slow, etc.
        self.used_effects = used_effects # frozenset of (r,c) where trap/reward already applied
        self.last_move = last_move      # direction vector (dr,dc) used to step here

    def __eq__(self, other):
        return (
            isinstance(other, State)
            and self.pos == other.pos
            and self.remaining == other.remaining
            and abs(self.gravity - other.gravity) < 1e-9
            and abs(self.speed - other.speed) < 1e-9
            and self.used_effects == other.used_effects
        )

    def __hash__(self):
        return hash((self.pos, self.remaining,
                     round(self.gravity, 9), round(self.speed, 9),
                     self.used_effects))


class SearchNode:
    __slots__ = ('state', 'cost', 'parent', 'triggered')

    def __init__(self, state, cost, parent=None):
        self.state = state
        self.cost = cost
        self.parent = parent
        self.triggered = None   # will store (effect_type, coord) if a trap/reward fired here

    def __lt__(self, other):
        return self.cost < other.cost


def step_cost(state):
    """
    Energy to move one step from this state:
      cost = 1.0 × gravity × (1.0 / speed)
    """
    return 1.0 * state.gravity * (1.0 / state.speed)


def apply_cell_effect(state, move_dir, grid):
    """
    Once you step onto state.pos, apply exactly one trap/reward at that coordinate
    (only if not already in used_effects). Returns (new_state, triggered) or
    (None, triggered) if Trap4 invalidates the path.

    Trap3 now “pushes you two hex-steps backward” (opposite of move_dir).
    """
    r, c = state.pos
    remaining = set(state.remaining)
    gravity = state.gravity
    speed = state.speed
    used = set(state.used_effects)
    dr, dc = move_dir
    ctype = grid[r][c].type
    triggered = None

    # 1) If it's a treasure, collect it
    if ctype == 'treasure' and (r, c) in remaining:
        remaining.remove((r, c))
        triggered = ('treasure', (r, c))

    # 2) If it's a trap/reward and not yet used, apply exactly one:
    if ctype in ('trap1','trap2','trap3','trap4','reward1','reward2') and ((r, c) not in used):
        used.add((r, c))
        triggered = (ctype, (r, c))

        if ctype == 'trap1':
            # Double gravity
            gravity *= 2.0

        elif ctype == 'reward1':
            # Halve gravity
            gravity *= 0.5

        elif ctype == 'trap2':
            # Halve speed
            speed *= 0.5

        elif ctype == 'reward2':
            # Double speed
            speed *= 2.0

        elif ctype == 'trap4':
            # If any treasures remain, this path is invalid
            if remaining:
                return None, triggered
            remaining.clear()

        elif ctype == 'trap3':
            # ── “push you two hex-steps backward” (opposite of move_dir) ──
            back1_r, back1_c = r - dr, c - dc
            back2_r, back2_c = r - 2*dr, c - 2*dc
            R, C = len(grid), len(grid[0])

            if (0 <= back2_r < R and 0 <= back2_c < C
                    and grid[back2_r][back2_c].type != 'obstacle'):
                # Land on (back2_r, back2_c)
                r, c = back2_r, back2_c
                landing = grid[r][c].type

                # If landing on a treasure
                if landing == 'treasure' and (r, c) in remaining:
                    remaining.remove((r, c))
                    triggered = ('treasure', (r, c))

                # If landing on another unused trap/reward, apply once more
                if landing in ('trap1','trap2','trap4','reward1','reward2') and ((r, c) not in used):
                    used.add((r, c))
                    triggered = (landing, (r, c))
                    if landing == 'trap1':
                        gravity *= 2.0
                    elif landing == 'reward1':
                        gravity *= 0.5
                    elif landing == 'trap2':
                        speed *= 0.5
                    elif landing == 'reward2':
                        speed *= 2.0
                    elif landing == 'trap4':
                        if remaining:
                            return None, triggered
                        remaining.clear()
                # Do NOT chain-teleport if landing on another trap3
            else:
                # If two-back is invalid or obstacle, you simply stay on (r,c)
                pass

    # Construct and return the updated State
    new_state = State(
        pos=(r, c),
        remaining=frozenset(remaining),
        gravity=gravity,
        speed=speed,
        used_effects=frozenset(used),
        last_move=(dr, dc)
    )
    return new_state, triggered


def uniform_cost_search(start_state, grid):
    """
    UCS over states, using the hard-coded neighbors. Returns the goal SearchNode
    (where state.remaining is empty) or None if no solution.
    """
    frontier = []
    heapq.heappush(frontier, SearchNode(start_state, cost=0.0, parent=None))
    explored = {start_state: 0.0}

    while frontier:
        node = heapq.heappop(frontier)
        state = node.state
        cost = node.cost

        # Goal test: no remaining treasures
        if not state.remaining:
            return node

        r, c = state.pos
        for (nr, nc) in get_neighbors_hardcoded((r, c)):
            if grid[nr][nc].type == 'obstacle':
                continue

            dr, dc = nr - r, nc - c
            temp = State(
                pos=(nr, nc),
                remaining=state.remaining,
                gravity=state.gravity,
                speed=state.speed,
                used_effects=state.used_effects,
                last_move=(dr, dc)
            )
            new_state, triggered = apply_cell_effect(temp, (dr, dc), grid)
            if new_state is None:
                # Trap4 invalidated this path
                continue

            new_cost = cost + step_cost(state)
            if new_state not in explored or new_cost < explored[new_state]:
                explored[new_state] = new_cost
                child = SearchNode(new_state, cost=new_cost, parent=node)
                child.triggered = triggered
                heapq.heappush(frontier, child)

    return None


# ------------------------------------------------------------
# 4) PATH RECONSTRUCTION & PRINTING
# ------------------------------------------------------------

def reconstruct_path(goal_node):
    """
    Walk back through parent pointers, collecting:
      - path: list of (r,c) from start → goal
      - triggers: parallel list of (effect_type, coord) or None
    """
    path = []
    triggers = []
    n = goal_node
    while n is not None:
        path.append(n.state.pos)
        triggers.append(getattr(n, 'triggered', None))
        n = n.parent
    return list(reversed(path)), list(reversed(triggers))


def print_path_on_map(grid, path):
    """
    Prints an ASCII version of the 6×10 map, but overwrites any visited cell
    (including traps/rewards) with '*' to show the full path.
    """
    symbol_map = {
        'empty': '.', 'obstacle': '#',
        'start': 'S', 'treasure': 'T',
        'trap1': '1', 'trap2': '2', 'trap3': '3', 'trap4': '4',
        'reward1': 'A', 'reward2': 'B'
    }
    char_map = [[symbol_map[grid[r][c].type] for c in range(10)] for r in range(6)]

    for (r, c) in path:
        char_map[r][c] = '*'

    for row in char_map:
        print(''.join(row))


# ------------------------------------------------------------
# 5) MAIN
# ------------------------------------------------------------

if __name__ == "__main__":
    # Build the grid from the hard-coded cell_type
    grid, start_pos, all_treasures = make_grid()

    # If you want (0,0) to be the “start,” set that explicitly:
    # e.g. uncomment the next line if start_pos was None
    # grid[0][0] = Cell('start', None)
    # start_pos = (0, 0)

    if start_pos is None:
        # Default to (0,0) as start if not specified
        start_pos = (0, 0)
        grid[0][0] = Cell('start', None)

    # Create initial state
    start_state = State(
        pos=start_pos,
        remaining=all_treasures,
        gravity=1.0,
        speed=1.0,
        used_effects=frozenset(),
        last_move=(0, 0)
    )

    print("Starting Uniform-Cost Search...")
    goal_node = uniform_cost_search(start_state, grid)

    if goal_node is None:
        print("No solution found.")
    else:
        path, triggers = reconstruct_path(goal_node)
        print(f"Total cost: {goal_node.cost:.2f}\n")

        # Print each step with any triggered effect
        for i, ((r, c), trig) in enumerate(zip(path, triggers), 1):
            if i == 1:
                print(f" Step {i}: ({r},{c}) -- start")
            else:
                if trig is not None:
                    effect, coord = trig
                    print(f" Step {i}: ({r},{c}) -- triggered {effect} at {coord}")
                else:
                    print(f" Step {i}: ({r},{c}) -- moved")

        print("\nASCII map with '*' marking the full path (including traps/rewards):")
        print_path_on_map(grid, path)
