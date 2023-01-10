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

    def get_predicted_neighbours(self, target_state):
        """Returns the predicted state, actions that lead to the provided state"""
        neighbours = []
        for (state, action), (reward, next_state) in self.state_action_lookup_table.items():
            if next_state == target_state:
                neighbours.append((state, action))
        return neighbours

class DynaQPlusMazeModel(MazeModel):
    def __init__(self, maze: Maze):
        super().__init__(maze)
        self.init_lookup_table()

    def init_lookup_table(self):
        """Init the lookup table so that unknown state, action pairs lead back to the same state with zero reward."""
        for i in range(self._rows):
            for j in range(self._cols):
                state = i, j
                for action in range(len(self.action_map)):
                    self.state_action_lookup_table[state, action] = (0, state)

    def reset(self):
        self.init_lookup_table()