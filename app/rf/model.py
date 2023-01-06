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

    def get_random_state(self):
        state = np.random.choice(self._rows), np.random.choice(self._cols)
        while not self.is_valid_state(state):
            state = np.random.choice(self._rows), np.random.choice(self._cols)
        return state

    def get_start_state(self):
        return self._start_state if self._start_state is not None else self.get_random_state()

    def get_random_action(self):
        return np.random.choice(len(self.action_map))

    def is_valid_state(self, state):
        """Return whether a given state is valid. In this case it check whether a state is a wall"""
        return self._maze[state] > 0

    def is_terminal_state(self, state):
        return state in self._maze.goal_states

    def execute_action(self, state, action):
        """Execute an action in the maze and return the reward and next state as a tuple
        
        If an action leads to an invalid state, next_state will be the current state
        """
        # state: (x, y)
        #action: ("up", "right", "down", "left") = (0, 1, 2, 3)
        s_x, s_y = state
        a_x, a_y = self.action_map[action]
        next_state = max(0, min(s_x + a_x, self._rows - 1)), max(0, min(s_y + a_y, self._cols - 1)) # do_nothing if invalid next state
        if self._maze[next_state] == 0: # check if it's a wall
            next_state = state
        
        return self._maze.rewards[next_state], next_state

    def update(self, state, action, reward, next_state):
        self.state_action_lookup_table[state, action] = reward, next_state

    def sample(self, state, action):
        """Sample a random state/action that has been visited before and return the experience"""
        reward, next_state = self.state_action_lookup_table[state, action]
        return reward, next_state
