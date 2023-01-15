import numpy as np


class Obstacle:
    def __init__(self, x, y, widht, height) -> None:
        self.x = x
        self.y = y
        self.cols = widht
        self.rows = height
        


class ObstacleEnvironment:
    ROTATION = 10 # possible rotation in degrees

    def __init__(self, rows, cols, obstacles, start, goal_states) -> None:
        self.rows = rows # discrete steps in y dimension
        self.cols = cols # discrete steps in x dimension
        
        self.obstacles = obstacles # list of Obstacle objects
        self.environment = np.full((rows, cols), True)
        self.environment_mask = np.full(self.environment.shape, False)
        self.apply_obstacles()
        self.start = start
        self.goal_states = goal_states # state = (x, y, rotation)
        self.action_map = ((-1, 0, 0), (0, 1, 0), (1, 0, 0), (0, -1, 0),
                            (-1, 0, 1), (0, 1, 1), (1, 0, 1), (0, -1, 1),
                            (-1, 0, 1), (0, 1, 1), (1, 0, 1), (0, -1, -1),
                            (0, 0, -1), (0, 0, 1)) 
        self.rewards = np.zeros((cols, rows, 365))

    def apply_obstacles(self):
        # modify the environment to take the obstacles into account
        for obstacle in self.obstacles:
            # assume for now the obstacles are rectangles
            self.environment[obstacle.y: obstacle.y + obstacle.height, obstacle.x: obstacle.x + obstacle.width]

    def get_blank_value_function(self):
        rows, cols = self.environment.shape
        return np.zeros((rows, cols, 365, len(self.action_map) - 1))

    @property
    def shape(self):
        return self.environment.shape

    def execute_action(self, state, action, length):
        """Execute an action in the environment
        """
        s_x, s_y, s_rotation = state
        a_x, a_y, a_rotation = self.action_map[action]
        next_state = (max(0, min(s_x + a_x, self.rows - 1)), 
                    max(0, min(s_y + a_y, self.cols - 1)), 
                    (s_rotation + a_rotation * self.ROTATION) % 365)
        if not self.is_valid_state(next_state, length):
            next_state = state
        return self.rewards[next_state], next_state


    def is_terminal_state(self, state):
        return state in self.goal_states

    def get_random_action(self):
        return np.random.choice(len(self.action_map))

    def is_valid_state(self, state, length):
        """Return whether a given state is valid. 
        In this case it check whether the player collides with an obstacle"""
        y, x, rotation = state
        self.environment_mask[:] = False # mark where the rod is on the environment
        self.environment_mask[y, x] = True
        # calculate left edge
        rotation = self.degree_to_radian(rotation)
        length_samples = np.arange(1, length, 0.1)
        px = np.round(x + length_samples * np.cos(rotation))
        py = np.round(y + length_samples * np.sin(rotation))
        if np.any(px < 0) or np.any(px > self.environment_mask.shape[1] - 1) or np.any(py < 0) or np.any(py > self.environment_mask.shape[0] - 1):
            return False
        
        indices = np.stack((py, px), axis = 1).astype(int)
        for row, col in indices:
            self.environment_mask[row, col] = True

        px = np.round(x - length_samples * np.cos(rotation))
        py = np.round(y - length_samples * np.sin(rotation))
        if np.any(px < 0) or np.any(px > self.environment_mask.shape[1] - 1) or np.any(py < 0) or np.any(py > self.environment_mask.shape[0] - 1):
            return False
        
        indices = np.stack((py, px), axis = 1).astype(int)
        for row, col in indices:
            self.environment_mask[row, col] = True
        return not np.any(self.environment_mask & ~self.environment) # check for overlaps with colliders

    @staticmethod
    def degree_to_radian(degree):
        return degree * (np.pi / 180)
