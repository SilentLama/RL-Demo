import numpy as np
from time import sleep

from queue import PriorityQueue

class AgentVisualizer:
    def __init__(self, agent, color = (255, 0, 0)):
        self.agent = agent
        self.color = color

    @property
    def coordinates(self):
        return self.agent.state

class DynaQAgent:
    def __init__(self, environment, model, learning_rate, discount_factor, epsilon, max_steps_per_episode, planning_steps, pause = None, use_prioritized_sweeping = False, theta = 0.0001):
        self.environment = environment
        self.model = model
        self.value_function = model.get_blank_value_function()
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.max_steps_per_episode = max_steps_per_episode
        self.planning_steps = planning_steps
        self.state = self.model.start_state
        self.visited_state_actions = np.full(self.value_function.shape, False)
        self.step_rewards = []
        self._cumulated_step_rewards = [0]
        self.episode_reward = 0
        self.episode_rewards = []
        self._cumulated_episode_rewards = [0]
        self.episode_step = 0
        self.steps_per_episode = []
        self.episode = 0
        self.step = 0
        self.pause = pause
        self._use_prioritized_sweeping = use_prioritized_sweeping
        self.theta = theta
        self._prioritized_sweep_queue = PriorityQueue()

    @property
    def state_reward_table(self):
        """Displays the reward at a given state"""
        return self.value_function.max(axis = 2) # return the max reward at that state (greedy action)

    @property
    def cumulated_episode_rewards(self):
        return self._cumulated_rewards[1:]

    @property
    def cumulated_step_rewards(self):
        return self._cumulated_step_rewards[1:]

    def normal_planning(self, planning_steps):
        for _ in range(planning_steps):
            state, action = self.get_random_visited_state_action()
            reward, next_state = self.model.sample(state, action)
            self.update_value_function(state, action, reward, next_state)

    def reset(self):
        self.state = self.model.start_state
        self.value_function = self.model.get_blank_value_function()
        self.visited_state_actions[:] = False
        self.step_rewards.clear()
        self._cumulated_step_rewards = [0]
        self.episode_reward = 0
        self.episode_rewards.clear()
        self._cumulated_episode_rewards = [0]
        self.episode_step = 0
        self.steps_per_episode.clear()
        self.episode = 0
        self.step = 0
        self._prioritized_sweep_queue = PriorityQueue()

    def dyna(self, state, planning_steps, epsilon = 0.9):
        if np.random.uniform() > epsilon:
            action = self.get_random_action()
        else:
            action = self.get_greedy_action(state)

        reward, next_state = self.environment.execute_action(state, action)
        self.visited_state_actions[state][action] = True
        
        self.update_value_function(state, action, reward, next_state)        
        self.model.update(state, action, reward, next_state)
        # planning
        if self._use_prioritized_sweeping:
            value_update = abs(self.get_temporal_difference(state, action, reward, next_state))
            self.prioritized_sweeping(value_update, state, action, planning_steps)
        else:
            self.normal_planning(planning_steps)
        return next_state, reward

    def prioritized_sweeping(self, value_update, state, action, planning_steps):
        if value_update > self.theta:
            self._prioritized_sweep_queue.put((-value_update, state, action)) # by default queue is a minheap so we use the negative value_update
        for _ in range(planning_steps):
            if self._prioritized_sweep_queue.empty():
                break
            _, state, action = self._prioritized_sweep_queue.get()            
            reward, next_state = self.model.sample(state, action)
            self.update_value_function(state, action, reward, next_state)
            for p_state, p_action in self.model.get_predicted_neighbours(state):
                predicted_reward, _ = self.model.sample(p_state, p_action)
                value_update = abs(self.get_temporal_difference(p_state, p_action, predicted_reward, state))
                if value_update > self.theta:
                    self._prioritized_sweep_queue.put((-value_update, p_state, p_action))
            self._prioritized_sweep_queue.task_done()

    def train(self):
        return self.dyna(self.state, self.planning_steps, epsilon = self.epsilon)

    def train_steps(self, n):
        for _ in range(n):
            self.state, reward = self.train()
            self.step_rewards.append(reward)
            self._cumulated_step_rewards.append(self._cumulated_step_rewards[-1] + reward)
            self.episode_reward += reward
            self.step += 1
            self.episode_step += 1
            if self.environment.is_terminal_state(self.state) or self.episode_step >= self.max_steps_per_episode:
                self.state = self.model.start_state
                self.steps_per_episode.append(self.episode_step)
                self.episode_rewards.append(self.episode_reward)
                self._cumulated_episode_rewards.append(self._cumulated_episode_rewards[-1] + self.episode_reward)
                self.episode += 1
                self.episode_step = 0
                self.episode_reward = 0
                break

            if self.pause is not None:
                sleep(self.pause)
    
    def train_episode(self):
        return self.train_steps(self.max_steps_per_episode)

    def execute_policy(self, max_steps):
        """Execute a run following just the trained policy"""
        self.state = self.model.start_state
        for step in range(max_steps):
            if self.environment.is_terminal_state(self.state):
                break
            action = self.get_greedy_action(self.state)
            _, self.state = self.environment.execute_action(self.state, action)
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

class DynaQPlusAgent(DynaQAgent):
    def __init__(self, environment, model, learning_rate, discount_factor, epsilon, max_steps_per_episode, planning_steps, pause=None, k = 0.001):
        super().__init__(environment, model, learning_rate, discount_factor, epsilon, max_steps_per_episode, planning_steps, pause)
        self.visited_state_actions_bonus_table = np.zeros(self.visited_state_actions.shape)
        self.k = k

    def dyna_plus(self, state, planning_steps, epsilon = 0.9):
        if np.random.uniform() > epsilon:
            action = self.get_random_action()
        else:
            action = self.get_greedy_action(state)

        reward, next_state = self.environment.execute_action(state, action)
        self.visited_state_actions[state][action] = True
        


        self.visited_state_actions_bonus_table[self.visited_state_actions] += 1
        self.visited_state_actions_bonus_table[state][action] = 0
        
        self.update_value_function(state, action, reward, next_state)
        self.model.update(state, action, reward, next_state)
        # planning
        self.normal_planning(planning_steps)
        return next_state, reward

    def normal_planning(self, planning_steps):
        for _ in range(planning_steps):
            state, action = self.environment.get_random_state(), self.environment.get_random_action()
            sample_reward, sample_next_state = self.model.sample(state, action)
            sample_reward += self.k * np.sqrt(self.visited_state_actions_bonus_table[state][action])
            self.update_value_function(state, action, sample_reward, sample_next_state)

    def reset(self):
        super().reset()
        self.visited_state_actions_bonus_table[:] = 0

    def train(self):
        return self.dyna_plus(self.state, self.planning_steps, epsilon = self.epsilon)