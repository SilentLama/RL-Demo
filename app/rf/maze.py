import numpy as np


class Maze:
    """The maze should only be created via classmethods
    """
    def __init__(self, paths, rewards, start):
        self.paths = paths
        self.rewards = rewards
        self.start = start
        self.rows, self.cols = self.paths.shape
        goals_x, goals_y = np.where(rewards > 0)
        self.goal_states = [(x, y) for x, y in zip(goals_x, goals_y)]
        self.action_map = ((-1, 0), (0, 1), (1, 0), (0, -1)) # ("up", "right", "down", "left")

    @property
    def shape(self):
        return self.paths.shape

    @classmethod
    def load_from_numpy_array(cls, file, start):
        paths, rewards = np.load(file)
        return cls(paths, rewards, start)

    def from_json(self):
        pass

    def __getitem__(self, *args):
        return self.paths.__getitem__(*args)

    def execute_action(self, state, action):
        """Execute an action in the maze and return the reward and next state as a tuple
        
        If an action leads to an invalid state, next_state will be the current state
        """
        # state: (x, y)
        #action: ("up", "right", "down", "left") = (0, 1, 2, 3)
        s_x, s_y = state
        a_x, a_y = self.action_map[action]
        next_state = max(0, min(s_x + a_x, self.rows - 1)), max(0, min(s_y + a_y, self.cols - 1)) # do_nothing if invalid next state
        if self[next_state] == 0: # check if it's a wall
            next_state = state
        
        return self.rewards[next_state], next_state

    def is_terminal_state(self, state):
        return state in self.goal_states

    def get_random_state(self):
        state = np.random.choice(self.rows), np.random.choice(self.cols)
        while not self.is_valid_state(state):
            state = np.random.choice(self.rows), np.random.choice(self.cols)
        return state

    def get_start_state(self):
        return self.start if self.start is not None else self.get_random_state()

    def get_random_action(self):
        return np.random.choice(len(self.action_map))

    def is_valid_state(self, state):
        """Return whether a given state is valid. In this case it check whether a state is a wall"""
        return self.paths[state] > 0


class MazeGenerator:

    @staticmethod
    def get_neighbour_cells(cell, x_max, y_max):
        neighbours = [(max(0, min(y_max, y)), max(0, min(x_max, x))) for (y, x) in MazeGenerator.get_surrounding_cells(*cell)]
        return [neighbour for neighbour in neighbours if neighbour != cell]

    @staticmethod
    def get_surrounding_cells(x, y):
        return (y, x + 1), (y + 1, x), (y, x - 1), (y - 1, x)

    @staticmethod
    def _generate_prims(rows, cols):
        maze = np.full((rows, cols), False)

        def get_surrounding_walls(x, y):
            return [cell for cell in MazeGenerator.get_neighbour_cells((x, y), cols - 1, rows - 1) if not maze[cell]]

        def get_number_of_path_neighbours(x, y):
            return sum(maze[wall] for wall in MazeGenerator.get_neighbour_cells((x, y), cols - 1, rows - 1))


        start_y, start_x = np.random.randint(1, rows - 1), np.random.randint(1, cols - 1)

        walls = []
        maze[start_y, start_x] = True
        for wall in ((start_y - 1, start_x), (start_y, start_x - 1), 
                    (start_y, start_x + 1), (start_y + 1, start_x)):
            walls.append(wall)

        for wall in walls:
            maze[wall] = True

        while walls:    
            wall_y, wall_x = walls.pop(np.random.randint(0, len(walls)))
            if ((wall_x not in (0, cols - 1) and not maze[wall_y, wall_x - 1] and maze[wall_y, wall_x + 1])
            or (wall_x not in (0, cols - 1) and not maze[wall_y, wall_x + 1] and maze[wall_y, wall_x - 1])
            or (wall_y not in (0, rows - 1) and not maze[wall_y - 1, wall_x] and maze[wall_y + 1, wall_x])
            or (wall_y not in (0, rows - 1) and not maze[wall_y + 1, wall_x] and maze[wall_y - 1, wall_x])):
                if get_number_of_path_neighbours(wall_x, wall_y) <= 1:
                    maze[wall_y, wall_x] = True
                    surrounding_walls = get_surrounding_walls(wall_x, wall_y)
                    for wall in surrounding_walls:
                        if wall not in walls: walls.append(wall)
        return maze

    @staticmethod
    def generate_rewards(maze_paths, goal_state, base_reward = 0, goal_reward = 1):
        rewards = np.full(maze_paths.shape, base_reward)
        rewards[goal_state] = goal_reward
        return rewards
        
    
    @staticmethod
    def generate_start_and_goal(maze_paths, selection_chance = 1):
        y_max, x_max = maze_paths.shape[0] - 1, maze_paths.shape[1] - 1

        def search_path(*cell_stack):
            stack = []
            for cell in cell_stack:
                if maze_paths[cell] and np.random.rand() < selection_chance:
                    return cell
                for neighbour in MazeGenerator.get_neighbour_cells(cell, x_max, y_max):
                    stack.append(neighbour)
            return search_path(*cell_stack, *stack)

        corners = ((0, 0), (y_max, 0), (0, x_max), (y_max, x_max))
        start_corner = corners[np.random.randint(0, len(corners))]
        end_corner = corners[np.random.randint(0, len(corners))]
        while end_corner == start_corner:
            end_corner = corners[np.random.randint(0, len(corners))]
        return search_path(start_corner), search_path(end_corner)

    @staticmethod
    def generate(rows, cols, algorithm = "prims", border_walls = False, base_reward = 0, goal_reward = 1):
        if not border_walls:
            rows, cols = rows + 1, cols + 1
        if algorithm.lower() == "prims":
            maze_paths = MazeGenerator._generate_prims(rows, cols)

        if not border_walls:
            maze_paths = maze_paths[1:-1,1:-1]
        start, goal = MazeGenerator.generate_start_and_goal(maze_paths)
        rewards = MazeGenerator.generate_rewards(maze_paths, goal, base_reward = base_reward, goal_reward = goal_reward)
        return Maze(maze_paths, rewards, start) 
