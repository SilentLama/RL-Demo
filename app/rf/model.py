import numpy as np

from .maze import Maze

class MazeModel:
    def __init__(self, maze: Maze, learning_rate, discount_factor, start_state = None):
        self._maze = maze
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
        self._start_state = start_state
        self._rows, self._cols = maze.shape
        self._q_table = np.repeat(np.full((4,), 0.0), self._rows * self._cols).reshape((self._rows, self._cols, 4)) # init all actions with 0
        self._visited_states = np.full(self._maze.shape, False)
        self.action_map = ((-1, 0), (0, 1), (1, 0), (0, -1)) # ("up", "right", "down", "left")

    @property
    def state_reward_table(self):
        """Displays the reward at a given state"""
        return self._q_table.max(axis = 2) # return the max reward at that state (greedy action)

    @property
    def q_table(self):
        """Return the q-learning table which has (maze_rows, maze_cols, actions) as dimensions

        """
        return self._q_table

    @property
    def visited_states(self):
        return self._visited_states

    def reset(self):
        self._q_table [:] = 0
        self._visited_states[:] = False

    def get_blank_policy(self):
        """Return a blank policy with dimensions (maze_rows, maze_cols)
        """
        return np.zeros((self._rows, self._cols))

    def get_blank_value_function(self):
        return np.zeros(self._q_table.shape)

    def get_random_state(self):
        state = np.random.choice(self._rows), np.random.choice(self._cols)
        while not self.is_valid_state(state):
            state = np.random.choice(self._rows), np.random.choice(self._cols)
        return state

    def get_random_visited_state(self):
        rows, cols = np.where(self.visited_states)
        state_idx = np.random.choice(len(rows))
        return rows[state_idx], cols[state_idx]

    def get_start_state(self):
        return self._start_state if self._start_state is not None else self.get_random_state()

    def get_random_action(self):
        return np.random.choice(len(self.action_map))

    def is_valid_state(self, state):
        """Return whether a given state is valid. In this case it check whether a state is a wall"""
        return self._maze[state] > 0

    def get_greedy_action(self, state):
        """Return the greedy action (i.e action with highest reward)
        Ties are solved uniformly

        :param state: _description_
        :type state: _type_
        :return: _description_
        :rtype: _type_
        """

        return np.random.choice(np.where(self._q_table[state] == self._q_table[state].max())[0])

    def generate_policy(self):
        return np.argmax(self._q_table, axis = 2)

    def is_terminal_state(self, state):
        return state in self._maze.goal_states

    def execute_action(self, state, action, sample = False):
        """Execute an action in the maze and return the reward and next state as a tuple
        
        If an action leads to an invalid state, next_state will be the current state
        """
        # state: (x, y)
        #action: ("up", "right", "down", "left") = (0, 1, 2, 3)
        if not sample:
            self._visited_states[state] = True
        s_x, s_y = state
        a_x, a_y = self.action_map[action]
        next_state = max(0, min(s_x + a_x, self._rows - 1)), max(0, min(s_y + a_y, self._cols - 1))
        if self._maze[next_state] == 0: # check if it's a wall
            next_state = state
        

        return self._maze.rewards[next_state], next_state

    def update(self, state, action, reward, next_state):
        self._q_table[state][action] += self._learning_rate * self.get_temporal_difference(state, action, reward, next_state)

    def get_temporal_difference(self, state, action, reward, next_state):
        return reward + self._discount_factor * self._q_table[next_state].max() - self._q_table[state][action]

    def sample(self):
        """Sample a random state/action that has been visited before and return the experience"""
        state = self.get_random_visited_state()
        action = self.get_random_action()

        reward, next_state = self.execute_action(state, action, sample = True)
        reward = self._maze.rewards[next_state]

        return state, action, reward, next_state
