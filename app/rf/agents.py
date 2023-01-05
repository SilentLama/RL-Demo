import numpy as np
from time import sleep

class DynaQAgent:
    def __init__(self, model):
        self.model = model
        self.value_function = model.get_blank_value_function()

    def train(self, max_steps, planning_steps, epsilon = 0.95, learning_rate = 0.1, discount_factor = 0.9, episodes = 1):
        for episode in range(episodes):
            self.dyna(self.model, self.value_function, max_steps, planning_steps, 
                        epsilon = epsilon, learning_rate = learning_rate, discount_factor = discount_factor)


    def dyna(self, model, state, planning_steps, epsilon = 0.9):
        if np.random.uniform() > epsilon:
            action = model.get_random_action()
        else:
            action = model.get_greedy_action(state)

        reward, next_state = model.execute_action(state, action)
        
        #value_function[state][action] += learning_rate * (reward + discount_factor * (np.argmax(value_function[next_state]) - value_function[state][action]))
        model.update(state, action, reward, next_state)
        state = next_state
        # planning
        for _ in range(planning_steps):
            p_state, p_action, p_reward, p_next_state = model.sample()
            model.update(p_state, p_action, p_reward, p_next_state)
        return state, reward

    def train_steps(self, n, algorithm, model, state, planning_steps, epsilon = 0.9, 
                    pause = None, update_state = None):
        cumulated_reward = 0
        for _ in range(n):
            state, reward = algorithm(model, state, planning_steps, epsilon = epsilon)
            cumulated_reward += reward
            if update_state is not None:
                update_state(state)
            if model.is_terminal_state(state):
                break
            if pause is not None:
                sleep(pause)

        return state, cumulated_reward

    def train_episode(self, algorithm, max_steps, model, state, planning_steps, epsilon = 0.9, 
                        pause = None, update_state = None):
        return self.train_steps(max_steps, algorithm, model, state, planning_steps, epsilon = epsilon, 
                                pause = pause, update_state = update_state)

    def execute_policy(self, model, start_state, pause, update_state):
        """Execute a run following just the trained policy"""
        state = start_state
        while not model.is_terminal_state(state):
            action = model.get_greedy_action(state)
            _, state = model.execute_action(state, action)
            update_state(state)
            sleep(pause)
        return state