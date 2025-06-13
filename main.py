import heapq
from collections import namedtuple


# ------------------------------------------------------------
# 1) Defining the state space 
# ------------------------------------------------------------ 
# Hardcoded adjacent list instead of edge list because: 
#   -> The map is static
#   -> Expansion of nodes require less resources compared to edge list (extra step required)
#   -> Fyi the state space taught in tutorial is an edge list

state_space = {
    (0,0):  [(0,1),(1,0),(1,1)],
    (0,1):  [(0,0),(0,2),(1,1)],
    (0,2):  [(0,1),(0,3),(1,1),(1,2),(1,3)],
    (0,3):  [(0,2),(0,4),(1,3)],
    (0,4):  [(0,3),(0,5),(1,3),(1,4),(1,5)],
    (0,5):  [(0,4),(0,6),(1,5)],
    (0,6):  [(0,5),(0,7),(1,5),(1,6),(1,7)],
    (0,7):  [(0,6),(0,8),(1,7)],
    (0,8):  [(0,7),(0,9),(1,7),(1,8),(1,9)],
    (0,9):  [(0,8),(1,9)],

    (1,0):  [(0,0),(1,1),(2,0),(2,1)],
    (1,1):  [(0,0),(0,1),(0,2),(1,0),(1,2),(2,1)],
    (1,2):  [(0,2),(1,1),(1,3),(2,1),(2,2),(2,3)],
    (1,3):  [(0,2),(0,3),(0,4),(1,2),(1,4),(2,3)],
    (1,4):  [(0,4),(1,3),(1,5),(2,3),(2,4),(2,5)],
    (1,5):  [(0,4),(0,5),(0,6),(1,4),(2,5),(1,6)],
    (1,6):  [(0,6),(1,5),(1,7),(2,5),(2,6),(2,7)],
    (1,7):  [(0,6),(0,7),(0,8),(1,6),(1,8),(2,7)],
    (1,8):  [(0,8),(1,7),(1,9),(2,7),(2,8),(2,9)],
    (1,9):  [(0,8),(0,9),(1,8),(2,9)],

    (2,0):  [(1,0),(2,1),(3,0),(3,1)],
    (2,1):  [(1,0),(1,1),(1,2),(2,0),(2,2),(3,1)],
    (2,2):  [(1,2),(2,1),(2,3),(3,1),(3,2),(3,3)],
    (2,3):  [(1,2),(1,3),(1,4),(2,2),(2,4),(3,3)],
    (2,4):  [(1,4),(2,3),(2,5),(3,3),(3,4),(3,5)],
    (2,5):  [(1,4),(1,5),(1,6),(2,4),(2,6),(3,5)],
    (2,6):  [(1,6),(2,5),(2,7),(3,5),(3,6),(3,7)],
    (2,7):  [(1,6),(1,7),(1,8),(2,6),(2,8),(3,7)],
    (2,8):  [(1,8),(2,7),(2,9),(3,7),(3,8),(3,9)],
    (2,9):  [(1,8),(1,9),(2,8),(3,9)],

    (3,0):  [(2,0),(3,1),(4,0),(4,1)],
    (3,1):  [(2,0),(2,1),(2,2),(3,0),(3,2),(4,1)],
    (3,2):  [(2,2),(3,1),(3,3),(4,1),(4,2),(4,3)],
    (3,3):  [(2,2),(2,3),(2,4),(3,2),(3,4),(4,3)],
    (3,4):  [(2,4),(3,3),(3,5),(4,3),(4,4),(4,5)],
    (3,5):  [(2,4),(2,5),(2,6),(3,4),(3,6),(4,5)],
    (3,6):  [(2,6),(3,5),(3,7),(4,5),(4,6),(4,7)],
    (3,7):  [(2,6),(2,7),(2,8),(3,6),(3,8),(4,7)],
    (3,8):  [(3,7),(2,8),(3,9),(4,9),(4,8),(4,7)],
    (3,9):  [(2,8),(2,9),(3,8),(4,9)],

    (4,0):  [(3,0),(4,1),(5,0),(5,1)],
    (4,1):  [(3,0),(3,1),(3,2),(4,0),(4,2),(5,1)],
    (4,2):  [(3,2),(4,1),(4,3),(5,1),(5,2),(5,3)],
    (4,3):  [(3,2),(3,3),(3,4),(4,2),(4,4),(5,3)],
    (4,4):  [(3,4),(4,3),(4,5),(5,3),(5,4),(5,5)],
    (4,5):  [(3,4),(3,5),(3,6),(4,4),(4,6),(5,5)],
    (4,6):  [(3,6),(4,5),(4,7),(5,5),(5,6),(5,7)],
    (4,7):  [(3,6),(3,7),(3,8),(4,6),(4,8),(5,7)],
    (4,8):  [(3,8),(4,7),(4,9),(5,9),(5,8),(5,7)],
    (4,9):  [(3,8),(3,9),(4,8),(5,9)],

    (5,0):  [(4,0),(5,1)],
    (5,1):  [(4,0),(4,1),(4,2),(5,0),(5,2)],
    (5,2):  [(4,2),(5,1),(5,3)],
    (5,3):  [(4,2),(4,3),(4,4),(5,2),(5,4)],
    (5,4):  [(4,4),(5,3),(5,5)],
    (5,5):  [(4,4),(4,5),(4,6),(5,4),(5,6)],
    (5,6):  [(4,6),(5,5),(5,7)],
    (5,7):  [(4,6),(4,7),(4,8),(5,6),(5,8)],
    (5,8):  [(4,8),(5,7),(5,9)],
    (5,9):  [(4,9),(4,8),(5,8)],
}

# Offset for each cell based on column
ODD_COL_DIRS = [
    (-1, 0), # N
    (-1, 1), # NE
    (0, 1), # SE
    (1, 0), # S
    (0, -1), # SW
    (-1, -1) # NW
]

EVEN_COL_DIRS  = [
    (-1, 0), # N
    (0, 1), # NE
    (1, 1), # SE
    (1, 0), # S
    (1, -1), # SW
    (0, -1) # NW 
] 

# ------------------------------------------------------------
# 2) Defining the cell types
# ------------------------------------------------------------
# We index rows 0..5 (top→bottom) and columns 0..9 (left→right). Trap3 and rewards are one-time uses. 
# Legend for values:
#   'empty'    = plain hex
#   'obstacle' = blocks movement (these are kept for grid customization flexibility)
#   'start'    = start cell
#   'treasure' = must collect
#   'trap1'    = doubles gravity
#   'trap2'    = halves speed 
#   'trap3'    = pushes you two steps forward
#   'trap4'    = ends the game (so we kind of hardcoded the search algorithm to avoid this)
#   'reward1'  = halves gravity
#   'reward2'  = doubles speed

cell_type = {
    (0,0): 'start',    (0,1): 'empty',    (0,2): 'empty',    (0,3): 'empty',    (0,4): 'reward1',
    (0,5): 'empty',    (0,6): 'empty',    (0,7): 'empty',    (0,8): 'empty',    (0,9): 'empty',

    (1,0): 'empty',    (1,1): 'trap2',    (1,2): 'empty',    (1,3): 'trap4',    (1,4): 'treasure',
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

def get_neighbors(pos):
    return state_space.get(pos, [])


# ------------------------------------------------------------
# 3. Mapping out the 2D grid
# ------------------------------------------------------------
Cell = namedtuple('Cell', ['type', 'value']) # class Cell with 2 fields

def make_grid():

    rows, cols = 6, 10
    grid = [[None]*cols for _ in range(rows)]
    start = None
    treasures = set()

    # Loop over all the items in cell_type dictionary to create actual map
    # This setup lets user have flexibility in changing trap / reward / treasure / obstacle locations
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

    # If start node not explicitly defined, default to (0, 0)
    if start is None:
        start = (0, 0)
        grid[0][0] = Cell('start', None)

    return grid, start, frozenset(treasures)


# ------------------------------------------------------------
# 4) State class & Search node class
# ------------------------------------------------------------
class State:
    __slots__ = ('pos', 'remaining', 'gravity', 'speed', 'used_rewards', 'used_trap3')

    def __init__(self, pos, remaining, gravity=1.0, speed=1.0,
                 used_rewards=frozenset(), used_trap3=frozenset()):
        self.pos = pos
        self.remaining = remaining
        self.gravity = gravity
        self.speed = speed
        self.used_rewards = used_rewards
        self.used_trap3 = used_trap3

    # Equality check to detect new states (e.g. explored list)
    # This is to let python compare the values between different state instances
    def __eq__(self, other):
        return (
            isinstance(other, State)
            and self.pos == other.pos
            and self.remaining == other.remaining
            and abs(self.gravity - other.gravity) < 1e-9
            and abs(self.speed - other.speed) < 1e-9
            and self.used_rewards == other.used_rewards
            and self.used_trap3 == other.used_trap3
        )

    # Lets python use state object as an index in a hash table
    # Required for it to work properly in sets and dicts (e.g. explored list)
    def __hash__(self):
        return hash((
            self.pos, self.remaining,
            round(self.gravity, 9), round(self.speed, 9),
            self.used_rewards, self.used_trap3
        ))

    # Debug
    def __repr__(self):
        return (f"State(pos={self.pos}, remaining={len(self.remaining)}, "
                f"gravity={self.gravity:.2f}, speed={self.speed:.2f}, "
                f"used_rewards={list(self.used_rewards)}, "
                f"used_trap3={list(self.used_trap3)})")

# A wrapper around State class that tracks how you got to this state (parent) and total cost
# Helps with backtracking to full path later on
class SearchNode:
    __slots__ = ('state', 'cost', 'parent', 'triggered')

    def __init__(self, state, cost, parent=None):
        self.state = state
        self.cost = cost
        self.parent = parent
        self.triggered = None

    # This allows heapq to sort nodes by lowest cost
    def __lt__(self, other):
        return self.cost < other.cost
    
    # Debug
    def __repr__(self):
        return (f"SearchNode(state={self.state}, "
               f"cost={self.cost}, "
               f"parent={self.parent}, "
               f"triggered={self.triggered})")


# ------------------------------------------------------------
# 5) Functions for Trap/Reward/Movement logic
# ------------------------------------------------------------
def step_cost(state):
    # Moving one hex costs gravity × (1 / speed).
    return state.gravity * (1.0 / state.speed)


def apply_cell_effect(state, grid):

    # Get current position & properties
    r, c = state.pos
    remaining_treasure = set(state.remaining)
    gravity = state.gravity
    speed = state.speed
    used_rewards = set(state.used_rewards)
    used_trap3 = set(state.used_trap3)
    ctype = grid[r][c].type
    triggered = None # helps with printing the output later

    # Collect treasure if present
    if ctype == 'treasure' and (r, c) in remaining_treasure:
        remaining_treasure.remove((r, c))
        triggered = ('treasure', (r, c))

    # Apply traps & rewards
    # Note: all rewards & trap3 can only be used once
    elif ctype == 'trap1':
        gravity *= 2.0
        triggered = ('trap1', (r, c))

    elif ctype == 'trap2':
        speed *= 0.5
        triggered = ('trap2', (r, c))

    elif ctype == 'trap3' and ((r,c) not in used_trap3):
        used_trap3.add((r, c))
        triggered = ('trap3', (r, c))

    elif ctype == 'trap4':
        triggered = ('trap4', (r, c))
        if remaining_treasure:
            return None, triggered
        remaining_treasure.clear()

    elif ctype == 'reward1' and ((r, c) not in used_rewards):
        gravity *= 0.5
        used_rewards.add((r, c))
        triggered = ('reward1', (r, c))

    elif ctype == 'reward2' and ((r, c) not in used_rewards):
        speed *= 2.0
        used_rewards.add((r, c))
        triggered = ('reward2', (r, c))

    new_state = State(
        pos=(r, c),
        remaining=frozenset(remaining_treasure),
        gravity=gravity,
        speed=speed,
        used_rewards=frozenset(used_rewards),
        used_trap3=frozenset(used_trap3)
    )
    return new_state, triggered


# ------------------------------------------------------------
# 6) Behold behold the Search Algorithm
# ------------------------------------------------------------
def uniform_cost_search(start_state, grid):

    frontier = []
    heapq.heappush(frontier, SearchNode(start_state, cost=0.0)) # heapq sorts the frontier based on cost so we dont have to do it manually
    explored = {}

    while frontier:
        node = heapq.heappop(frontier)
        state = node.state
        cost = node.cost

        # Duplication check
        if state in explored and explored[state] <= cost:
            continue
        explored[state] = cost # Store state & cost in 'explored'

        # Goal check
        if not state.remaining:
            return node

        (r, c) = state.pos
        ctype = grid[r][c].type

        # Sidenote: This is ugly coding but it's the only way I can think of to include the trap3 node itself into the final search path without completely omitting it
        # If we put the trap3's logic at apply_cell_effect, I can't rlly return both the state of trap3's node and the location after the boost
        if ctype == 'trap3' and node.parent:

            r0, c0 = node.parent.state.pos        # gives us parent node position
            dr, dc = r - r0, c - c0               # offset from parent position 

            # First we get the direction based on how we reach current state from parent node
            parent_dirs = ODD_COL_DIRS if c0 % 2 else EVEN_COL_DIRS

            for i, (drr, dcc) in enumerate(parent_dirs):
                if (dr, dc) == (drr, dcc):

                    # Now we know which direction we moving towards(i)
                    # Apply direction from trap3 using the current cell's parity
                    current_direction = ODD_COL_DIRS if c % 2 else EVEN_COL_DIRS
                    step1_r = r + current_direction[i][0]
                    step1_c = c + current_direction[i][1]
                    step1 = (step1_r, step1_c)

                    # Apply same direction from step1 using its parity
                    next_direction = ODD_COL_DIRS if step1_c % 2 else EVEN_COL_DIRS
                    step2_r = step1_r + next_direction[i][0]
                    step2_c = step1_c + next_direction[i][1]
                    step2 = (step2_r, step2_c)
                    break

            else:
                step1 = step2 = None # Invalid direction

            rows, cols = len(grid), len(grid[0])
            valid_boost = all(
                pos and 0 <= pos[0] < rows and 0 <= pos[1] < cols and grid[pos[0]][pos[1]].type not in ('obstacle', 'trap4')
                for pos in [step1, step2]
            )
            
            if valid_boost:
                # If we're not out of bounds or on an obstacle/trap4, add the node that we boosted into into the frontier n continue processing the current node (trap3)
                boosted_state = State(
                    pos=step2,
                    remaining=state.remaining,
                    gravity=state.gravity,
                    speed=state.speed,
                    used_rewards=state.used_rewards,
                    used_trap3=state.used_trap3
                )
                boosted_node = SearchNode(boosted_state, cost=cost, parent=node)
                heapq.heappush(frontier, boosted_node)
                continue

        for (nr, nc) in get_neighbors((r, c)):
            neighbor_type = grid[nr][nc].type

            # If neighbor is an obstacle: skip / ignore
            if neighbor_type == 'obstacle':
                continue

            # If hex contains trap3 multiply the cost by 3
            if grid[nr][nc].type == 'trap3':
                move_cost = step_cost(state) * 3
            else:
                move_cost = step_cost(state)

            new_cost = cost + move_cost
            actual_pos = (nr, nc)

            # Copy current state...
            temp_state = State(
                pos=actual_pos,
                remaining=state.remaining,
                gravity=state.gravity,
                speed=state.speed,
                used_rewards=state.used_rewards,
                used_trap3=state.used_trap3
            )

            # ...then apply neighbor cell effects in apply_cell_effect
            new_state, triggered = apply_cell_effect(temp_state, grid)
            if new_state is None:
                continue

            # Put would-be state into queue list
            if new_state not in explored or new_cost < explored[new_state]:
                child = SearchNode(new_state, cost=new_cost, parent=node)
                child.triggered = triggered
                heapq.heappush(frontier, child)

    return None


# ------------------------------------------------------------
# 7) Retracing path + Visuals
# ------------------------------------------------------------
def retrace_path(goal_node):

    path, triggers, costs = [], [], []
    while goal_node is not None:
        path.append(goal_node.state.pos)
        triggers.append(goal_node.triggered)
        costs.append(goal_node.cost)
        goal_node = goal_node.parent

    # Reverse so that it becomes start -> end
    path.reverse()
    triggers.reverse()
    costs.reverse()

    return path, triggers, costs

def print_solution(goal_node, grid):
    path, triggers, costs = retrace_path(goal_node)
    print("\n=== SOLUTION FOUND ===")
    print(f"Total cost: {goal_node.cost:.3f}")
    print(f"Path length: {len(path)} steps")
    print(f"Final state: {goal_node.state}")

    print("\n=== DETAILED PATH WITH COST VERIFICATION ===")
    for i, pos in enumerate(path):
        r, c = pos
        ctype = grid[r][c].type
        cost = costs[i]
        if i == 0:
            print(f"Step {i+1:2d}: {pos} [{ctype:8s}] -- START (cumulative: {cost:.3f})")
        else:
            step_cost_val = costs[i] - costs[i-1] # calculate cost of THIS step
            trig = triggers[i]
            if trig:
                effect, coord = trig
                print(f"Step {i+1:2d}: {pos} [{ctype:8s}] -- {effect} at {coord} "
                      f"(step: {step_cost_val:.3f}, cumulative: {cost:.3f})")
            else:
                print(f"Step {i+1:2d}: {pos} [{ctype:8s}] -- moved "
                      f"(step: {step_cost_val:.3f}, cumulative: {cost:.3f})")


if __name__ == "__main__":
    grid, start_pos, all_treasures = make_grid()

    print("=== TREASURE HUNT SETUP ===")
    print(f"Grid size: {len(grid)} x {len(grid[0])}")
    print(f"Start position: {start_pos}")
    print(f"Treasures: {sorted(all_treasures)}")
    print(f"Number of treasures: {len(all_treasures)}")

    start_state = State(
        pos=start_pos,
        remaining=all_treasures,
        gravity=1.0,
        speed=1.0,
        used_rewards=frozenset(),
        used_trap3=frozenset()
    )

    print(f"\nStarting state: {start_state}")
    print("\nStarting Uniform‐Cost Search...\n")

    goal_node = uniform_cost_search(start_state, grid)

    if goal_node is None:
        print("No solution found.")
    else:
        print_solution(goal_node, grid)
