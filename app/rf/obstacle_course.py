import numpy as np


class Obstacle:
    def __init__(self, x, y, widht, height) -> None:
        self.x = x
        self.y = y
        self.width = widht
        self.height = height
        


class ObstacleEnvironment:
    ROTATION = 15 # possible rotation in degrees

    def __init__(self, width, height, obstacles, start, goal_states) -> None:
        self.width = width # discrete steps in x dimension
        self.height = height # discrete steps in y dimension
        self.obstacles = obstacles # list of Obstacle objects
        self.environment = np.full((height, width), True)
        self.environment_mask = np.full(self.environment.shape, False)
        self.apply_obstacles()
        self.start = start
        goal_states = goal_states # state = (x, y, rotation)
        self.action_map = ((-1, 0, 0), (0, 1, 0), (1, 0, 0), (0, -1, 0),
                            (-1, 0, 1), (0, 1, 1), (1, 0, 1), (0, -1, 1),
                            (-1, 0, 1), (0, 1, 1), (1, 0, 1), (0, -1, -1),
                            (0, 0, -1), (0, 0, 1)) 
        self.rewards = np.zeros((width, height, len(self.action_map) - 1))

    def apply_obstacles(self):
        # modify the environment to take the obstacles into account
        for obstacle in self.obstacles:
            # assume for now the obstacles are rectangles
            self.environment[obstacle.y: obstacle.y + obstacle.height, obstacle.x: obstacle.x + obstacle.width]

    def get_blank_value_function(self):
        rows, cols = self.environment.shape
        return np.zeros((rows, cols, len(self.action_map)))

    @property
    def shape(self):
        return self.environment.shape

    def execute_action(self, state, action, length):
        """Execute an action in the environment
        """
        s_x, s_y, s_rotation = state
        a_x, a_y, a_rotation = self.action_map[action]
        next_state = (max(0, min(s_x + a_x, self.height - 1)), 
                    max(0, min(s_y + a_y, self.width - 1)), 
                    max(0, min(s_rotation + a_rotation * self.ROTATION, 365)))
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
        for r in np.arange(1, length / 2, 0.1):
            px = x + r * np.cos(rotation)
            py = y + r * np.sin(rotation)
            self.environment_mask[round(py), round(px)] = True
            px = x - r * np.cos(rotation)
            py = y - r * np.sin(rotation)
            self.environment_mask[round(py), round(px)] = True
        return np.any(self.environment_mask & ~self.environment) # check for overlaps with colliders

    @staticmethod
    def degree_to_radian(degree):
        return degree * (np.pi / 180)
