import numpy as np
from time import sleep

class AgentVisualizer:
    def __init__(self, agent, color = (255, 0, 0)):
        self.agent = agent
        self.color = color

    @property
    def coordinates(self):
        return self.agent.state

class DynaQAgent:
    def __init__(self, model, learning_rate, discount_factor, epsilon, max_steps_per_episode, planning_steps, pause = None):
        self.model = model
        self.value_function = model.get_blank_value_function()
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.max_steps_per_episode = max_steps_per_episode
        self.planning_steps = planning_steps
        self.state = self.model.start_state
        self.visited_state_actions = np.full(self.value_function.shape, False)
        self.episode_reward = 0
        self.episode_rewards = []
        self.episode_step = 0
        self.steps_per_episode = []
        self.episode = 0
        self.step = 0
        self.pause = None

    @property
    def state_reward_table(self):
        """Displays the reward at a given state"""
        return self.value_function.max(axis = 2) # return the max reward at that state (greedy action)

    def reset(self):
        self.state = self.model.start_state
        self.value_function = self.model.get_blank_value_function()
        self.visited_state_actions[:] = False
        self.episode_reward = 0
        self.episode_rewards.clear()
        self.episode_step = 0
        self.steps_per_episode.clear()
        self.episode = 0
        self.step = 0

    def dyna(self, model, state, planning_steps, epsilon = 0.9):
        if np.random.uniform() > epsilon:
            action = self.get_random_action()
        else:
            action = self.get_greedy_action(state)

        reward, next_state = model.execute_action(state, action)
        self.visited_state_actions[state][action] = True
        
        self.update_value_function(state, action, reward, next_state)
        model.update(state, action, reward, next_state)
        # planning
        for _ in range(planning_steps):
            state, action = self.get_random_visited_state_action()
            sample_reward, sample_next_state = model.sample(state, action)
            self.update_value_function(state, action, sample_reward, sample_next_state)
        return next_state, reward

    def train_steps(self, n, algorithm):
        for _ in range(n):
            self.state, reward = algorithm(self.model, self.state, self.planning_steps, epsilon = self.epsilon)
            self.episode_reward += reward
            self.step += 1
            self.episode_step += 1
            if self.model.is_terminal_state(self.state) or self.episode_step >= self.max_steps_per_episode:
                self.state = self.model.start_state
                self.steps_per_episode.append(self.episode_step)
                self.episode_rewards.append(self.episode_reward)
                self.episode += 1
                self.episode_step = 0
                self.episode_reward = 0
                break

            if self.pause is not None:
                sleep(self.pause)
    
    def train_episode(self, algorithm):
        return self.train_steps(self.max_steps_per_episode, algorithm)

    def execute_policy(self, model, max_steps):
        """Execute a run following just the trained policy"""
        self.state = self.model.start_state
        steps = 0
        while not model.is_terminal_state(self.state) and steps < max_steps:
            action = self.get_greedy_action(self.state)
            _, self.state = model.execute_action(self.state, action)
            steps += 1
            if self.pause is not None:
                sleep(self.pause)
        self.state = self.model.start_state

    def update_value_function(self, state, action, reward, next_state):
        self.value_function[state][action] += self.learning_rate * self.get_temporal_difference(state, action, reward, next_state)

    def get_temporal_difference(self, state, action, reward, next_state):
        return reward + self.discount_factor * self.value_function[next_state].max() - self.value_function[state][action]

    def get_greedy_action(self, state):
        """Return the greedy action (i.e action with highest reward)
        Ties are solved uniformly
        """
        return np.random.choice(np.where(self.value_function[state] == self.value_function[state].max())[0])

    def get_random_action(self):
        return np.random.choice(self.value_function.shape[2])

    def generate_policy(self):
        return np.argmax(self.value_function, axis = 2)

    def get_random_visited_state_action(self):
        rows, cols, actions = np.where(self.visited_state_actions)
        state_idx = np.random.choice(len(rows))
        return (rows[state_idx], cols[state_idx]), actions[state_idx]