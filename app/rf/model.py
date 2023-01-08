import numpy as np

from .maze import Maze

class MazeModel:
    def __init__(self, maze: Maze):
        self._maze = maze
        self.start_state = self._maze.start
        self._rows, self._cols = maze.shape
        self.state_action_lookup_table = dict()
        self.action_map = ((-1, 0), (0, 1), (1, 0), (0, -1)) # ("up", "right", "down", "left")


    def reset(self):
        self.state_action_lookup_table.clear()

    def get_blank_policy(self):
        """Return a blank policy with dimensions (maze_rows, maze_cols)
        """
        return np.zeros((self._rows, self._cols))

    def get_blank_value_function(self):
        rows, cols = self._maze.shape
        return np.zeros((rows, cols, len(self.action_map)))

    def update(self, state, action, reward, next_state):
        self.state_action_lookup_table[state, action] = reward, next_state

    def sample(self, state, action):
        """Sample a random state/action that has been visited before and return the experience"""
        reward, next_state = self.state_action_lookup_table[state, action]
        return reward, next_state
