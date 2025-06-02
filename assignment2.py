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
# 1) HARD‐CODED MAP DATA (cell_type)
# ------------------------------------------------------------
# We index rows 0..5 (top→bottom) and cols 0..9 (left→right).
# Legend for values:
#   'empty'    = plain hex
#   'obstacle' = blocks movement
#   'treasure' = must collect
#   'trap1'    = doubles gravity
#   'trap2'    = halves speed
#   'trap3'    = pushes you two hex‐steps backward
#   'trap4'    = invalidates path if any treasure remains
#   'reward1'  = halves gravity
#   'reward2'  = doubles speed
#
cell_type = {
    (0,0): 'empty',    (0,1): 'empty',    (0,2): 'empty',    (0,3): 'empty',    (0,4): 'reward1',
    (0,5): 'empty',    (0,6): 'empty',    (0,7): 'empty',    (0,8): 'empty',    (0,9): 'empty',

    (1,0): 'empty',    (1,1): 'trap2',    (1,2): 'empty',    (1,3): 'trap1',    (1,4): 'treasure',
    (1,5): 'empty',    (1,6): 'trap3',    (1,7): 'empty',    (1,8): 'obstacle', (1,9): 'empty',

    (2,0): 'empty',    (2,1): 'empty',    (2,2): 'obstacle', (2,3): 'empty',    (2,4): 'obstacle',
    (2,5): 'empty',    (2,6): 'empty',    (2,7): 'reward2',  (2,8): 'trap1',    (2,9): 'empty',

    (3,0): 'obstacle', (3,1): 'reward1',  (3,2): 'empty',    (3,3): 'obstacle', (3,4): 'empty',
    (3,5): 'trap3',    (3,6): 'obstacle', (3,7): 'treasure', (3,8): 'empty',    (3,9): 'treasure',

    (4,0): 'empty',    (4,1): 'empty',    (4,2): 'trap2',    (4,3): 'treasure', (4,4): 'obstacle',
    (4,5): 'empty',    (4,6): 'obstacle', (4,7): 'obstacle', (4,8): 'empty',    (4,9): 'empty',

    (5,0): 'empty',    (5,1): 'empty',    (5,2): 'empty',    (5,3): 'empty',    (5,4): 'empty',
    (5,5): 'reward2',  (5,6): 'empty',    (5,7): 'empty',    (5,8): 'empty',    (5,9): 'empty',
}

# ------------------------------------------------------------
# 2) DYNAMIC NEIGHBOR CALCULATION (even‐r layout)
# ------------------------------------------------------------
EVEN_OFFSETS = [
    (-1,  0),  # N
    (-1, -1),  # NW
    ( 0, -1),  # W
    ( 1, -1),  # SW
    ( 1,  0),  # S
    ( 0,  1)   # E
]

ODD_OFFSETS = [
    (-1,  1),  # NE
    (-1,  0),  # N
    ( 0, -1),  # W
    ( 1,  0),  # S
    ( 1,  1),  # SE
    ( 0,  1)   # E
]

def get_neighbors(pos, grid):
    """
    Return a list of valid neighbor coordinates for hex pos=(r,c)
    using even‐r horizontal layout, skipping obstacles/trap4.
    """
    rows, cols = len(grid), len(grid[0])
    r, c = pos
    offsets = EVEN_OFFSETS if (r % 2 == 0) else ODD_OFFSETS
    for dr, dc in offsets:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            if grid[nr][nc].type not in ('obstacle', 'trap4'):
                yield (nr, nc)

# ------------------------------------------------------------
# 3) BUILD GRID & TRACK START/TREASURES
# ------------------------------------------------------------
Cell = namedtuple('Cell', ['type', 'value'])

def make_grid():
    """
    Reads cell_type and builds:
      - grid: 6×10 list of Cell
      - start: coordinate of 'start' (if none specified, defaults to (0,0))
      - treasures: frozenset of all treasure coords
    """
    rows, cols = 6, 10
    grid = [[None]*cols for _ in range(rows)]
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
            raise ValueError(f"Unknown cell code '{kind}' at {(r,c)}")

    # If no 'start' was specified, default to (0,0)
    if start is None:
        start = (0, 0)
        grid[0][0] = Cell('start', None)

    return grid, start, frozenset(treasures)

# ------------------------------------------------------------
# 4) STATE & SEARCH NODE CLASSES
# ------------------------------------------------------------
class State:
    __slots__ = ('pos', 'remaining', 'gravity', 'speed', 'used_effects', 'last_move')

    def __init__(self, pos, remaining, gravity=1.0, speed=1.0,
                 used_effects=frozenset(), last_move=(0, 0)):
        self.pos = pos
        self.remaining = remaining      # frozenset of treasure coords left
        self.gravity = gravity          # 1.0 normal, 2.0 doubled
        self.speed = speed              # 1.0 normal, 0.5 halved
        self.used_effects = used_effects # frozenset of (r,c) for traps/rewards used
        self.last_move = last_move      # (dr, dc) of the move taken

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
        self.triggered = None   # (effect_type, coord) when trap/reward fired

    def __lt__(self, other):
        return self.cost < other.cost

# ------------------------------------------------------------
# 5) STEP COST & CELL EFFECTS
# ------------------------------------------------------------
def step_cost(state):
    """
    Energy to move one step: cost = gravity × (1/speed)
    """
    return 1.0 * state.gravity * (1.0 / state.speed)

def apply_cell_effect(state, move_dir, grid):
    """
    After stepping onto state.pos, apply at most one trap/reward.
    Returns (new_state, triggered) or (None, triggered) if trap4 invalidates.
    """
    r, c = state.pos
    remaining = set(state.remaining)
    gravity = state.gravity
    speed = state.speed
    used = set(state.used_effects)
    dr, dc = move_dir
    ctype = grid[r][c].type
    triggered = None

    # 1) If this is a treasure and uncollected, collect it
    if ctype == 'treasure' and (r, c) in remaining:
        remaining.remove((r, c))
        triggered = ('treasure', (r, c))

    # 2) If unused trap/reward, apply effect once
    if ctype in ('trap1','trap2','trap3','trap4','reward1','reward2') and ((r, c) not in used):
        used.add((r, c))
        triggered = (ctype, (r, c))

        if ctype == 'trap1':
            gravity *= 2.0

        elif ctype == 'reward1':
            gravity *= 0.5

        elif ctype == 'trap2':
            speed *= 0.5

        elif ctype == 'reward2':
            speed *= 2.0

        elif ctype == 'trap4':
            # Invalidates path if any treasure remains
            if remaining:
                return None, triggered
            remaining.clear()

        elif ctype == 'trap3':
            # Push two hex‐steps backward (opposite of move_dir)
            back1 = (r - dr, c - dc)
            back2 = (r - 2*dr, c - 2*dc)
            R, C = len(grid), len(grid[0])

            if (0 <= back2[0] < R and 0 <= back2[1] < C and
                    grid[back2[0]][back2[1]].type != 'obstacle'):
                # Teleport to back2
                r, c = back2
                landing = grid[r][c].type

                # If landing on a treasure, collect it
                if landing == 'treasure' and (r, c) in remaining:
                    remaining.remove((r, c))
                    triggered = ('treasure', (r, c))

                # If landing on unused trap/reward (except trap3), apply it
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
            # else: stay on trap3 cell

    # Build the new state
    new_state = State(
        pos=(r, c),
        remaining=frozenset(remaining),
        gravity=gravity,
        speed=speed,
        used_effects=frozenset(used),
        last_move=(dr, dc)
    )
    return new_state, triggered

# ------------------------------------------------------------
# 6) UNIFORM‐COST SEARCH
# ------------------------------------------------------------
def uniform_cost_search(start_state, grid):
    """
    Standard UCS: frontier is a min‐heap of SearchNode. Explored maps State→best_cost.
    Returns the goal SearchNode (all treasures collected) or None.
    """
    frontier = []
    heapq.heappush(frontier, SearchNode(start_state, cost=0.0, parent=None))
    explored = {start_state: 0.0}

    while frontier:
        node = heapq.heappop(frontier)
        state = node.state
        cost = node.cost

        # Goal test
        if not state.remaining:
            return node

        r, c = state.pos
        for (nr, nc) in get_neighbors((r, c), grid):
            # Skip obstacle/trap4 neighbors
            if grid[nr][nc].type in ('obstacle', 'trap4'):
                continue

            # If stepping on trap3 that’s already used, skip
            if grid[nr][nc].type == 'trap3' and ((nr, nc) in state.used_effects):
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
                # Trap4 invalidated
                continue

            new_cost = cost + step_cost(state)
            if new_state not in explored or new_cost < explored[new_state]:
                explored[new_state] = new_cost
                child = SearchNode(new_state, cost=new_cost, parent=node)
                child.triggered = triggered
                heapq.heappush(frontier, child)

    return None

# ------------------------------------------------------------
# 7) PATH RECONSTRUCTION & PRINTING
# ------------------------------------------------------------
def reconstruct_path(goal_node):
    """
    Follows parent pointers from goal_node back to start,
    returning (path, triggers) both as lists from start→goal.
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
    Prints ASCII map (6×10). Overwrites every visited cell in path with '*'.
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
# 8) MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    grid, start_pos, all_treasures = make_grid()

    start_state = State(
        pos=start_pos,
        remaining=all_treasures,
        gravity=1.0,
        speed=1.0,
        used_effects=frozenset(),
        last_move=(0, 0)
    )

    print("Starting Uniform‐Cost Search...")
    goal_node = uniform_cost_search(start_state, grid)

    if goal_node is None:
        print("No solution found.")
    else:
        path, triggers = reconstruct_path(goal_node)
        print(f"Total cost: {goal_node.cost:.2f}\n")
        for i, ((r, c), trig) in enumerate(zip(path, triggers), 1):
            if i == 1:
                print(f" Step {i}: ({r},{c}) -- start")
            else:
                if trig is not None:
                    effect, coord = trig
                    print(f" Step {i}: ({r},{c}) -- triggered {effect} at {coord}")
                else:
                    print(f" Step {i}: ({r},{c}) -- moved")

        print("\nASCII map with '*' marking the full path:")
        print_path_on_map(grid, path)

