from abc import ABC, abstractmethod
from threading import Thread
import matplotlib.pyplot as plt

from . import Window
from .game_objects import (Button, TableVisualizer, MazeVisualizer, TextInput, HeatMapVisualizer, 
                            FPSDisplay, MatplotlibPlotDisplay, MazePolicyVisualizer, VariableDisplay)

from app.rf.maze import Maze
from app.rf.model import MazeModel
from app.rf.agents import DynaQAgent


class Scene(ABC):

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def close(self):
        pass

class DynaQMazeScene(Scene):
    PADDING = 10 # px
    BUTTON_WIDTH = 200
    BUTTON_HEIGHT = 50
    BUTTON_KWARGS = {
        "background_color": (255, 128, 128)
    }

    def policy_overlay_button_function(self):
        if self.is_in_render_pipeline(self.policy_visualizer):
            self.remove_from_render_pipeline(self.policy_visualizer)
            self.policy_overlay_button.text = "Enable P-Overlay"
        else:
            self.add_to_render_pipeline(self.policy_visualizer)
            self.policy_overlay_button.text = "Disable P-Overlay"

    def heatmap_overlay_button_function(self):
        if self.is_in_render_pipeline(self.heatmap):
            self.remove_from_render_pipeline(self.heatmap)
            self.policy_overlay_button.text = "Enable Heatmap"
        else:
            self.add_to_render_pipeline(self.heatmap)
            self.policy_overlay_button.text = "Disable Heatmap"

    def reset_button_function(self):
        self.maze_model.reset()
        self.state = self.start_state
        self.maze_visualizer.player_coords = self.state
        self.episode = 0
        self.step = 0
        self.episode_reward = 0

    def execute_policy_button_function(self):
        thread = Thread(target=self.agent.execute_policy, args = [self.maze_model, self.state, self.pause / 1000, self.update_state_function])
        thread.daemon = True
        thread.start()

    def update_state_function(self, state):
        self.step += 1
        self.episode_step += 1
        self.state = state
        self.maze_visualizer.player_coords = state
        # if self.maze_model.is_terminal_state(self.state):
        #     self.finish_episode()

    def train_steps_agent_thread_function(self, n):
        _, cumulated_reward = self.agent.train_steps(n, self.agent.dyna, self.maze_model, self.state, self.planning_steps, 
                            epsilon = self.epsilon, pause = self.pause / 1000, update_state = self.update_state_function)
        self.episode_reward += cumulated_reward
        if self.maze_model.is_terminal_state(self.state):
            self.finish_episode()


    def train_episode_agent_thread_function(self, n):
        for _ in range(n):
            _, cumulated_reward =  self.agent.train_episode(self.agent.dyna, self.max_steps_per_episode, self.maze_model, self.state, self.planning_steps, 
                                epsilon = self.epsilon, pause = self.pause / 1000, update_state = self.update_state_function)
            self.episode_reward += cumulated_reward
            # if not self.maze_model.is_terminal_state(self.state) and self.state != self.start_state and self.episode_step > 0:
            self.finish_episode()

    def finish_episode(self):
        self.state = self.start_state
        self.maze_visualizer.player_coords = self.state
        self.steps_per_episode.append(self.episode_step)
        self.episode_rewards.append(self.episode_reward)
        self.update_plots()
        self.episode += 1
        self.episode_step = 0
        self.episode_reward = 0

    def update_plots(self):
        X = [i for i in range(self.episode + 1)]
        self.steps_per_episode_line.set_xdata(X)
        self.steps_per_episode_line.set_ydata(self.steps_per_episode)
        self.steps_per_episode_line.axes.set_xlim(0, self.episode)
        self.steps_per_episode_line.axes.set_ylim(0, self.max_steps_per_episode + 1)
        
        self.reward_per_episode_line.set_xdata(X)
        self.reward_per_episode_line.set_ydata(self.episode_rewards)
        self.reward_per_episode_line.axes.set_xlim(0, self.episode)
        self.reward_per_episode_line.axes.set_ylim(min(self.episode_rewards) - 1, max(self.episode_rewards) + 1)


    def __init__(self, width, height, title, max_fps = 60):
        self.episode_reward = 0
        self.episode_rewards = []
        self.episode_step = 0
        self.steps_per_episode = []

        self.episode = 0
        self.step = 0
        self.learning_rate = 0.1
        self.epsilon = 0.9
        self.discount_factor = 0.9
        self.start_state = (2, 0)
        self.state = self.start_state
        self.planning_steps = 50
        self.max_steps_per_episode = 100
        self.pause = 50 # ms
        self.window = Window(width, height, title, max_fps)
        # BUTTON BAR
        self.episode_display = VariableDisplay(0, 0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Episode", 
                                            lambda: self.episode, background_color=(255, 128, 128))
        self.one_episode_button = Button(self.episode_display.x + self.episode_display.width  + self.PADDING,
                                        0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "1", lambda: self.train_episode(1), **self.BUTTON_KWARGS)
        self.ten_episode_button = Button(self.one_episode_button.x + self.one_episode_button.width  + self.PADDING,
                                        0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "10", lambda: self.train_episode(10), **self.BUTTON_KWARGS)
        self.hundred_episode_button = Button(self.ten_episode_button.x + self.ten_episode_button.width  + self.PADDING,
                                        0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "100", lambda: self.train_episode(100), **self.BUTTON_KWARGS)

        self.step_display = VariableDisplay(self.hundred_episode_button.x + self.hundred_episode_button.width + self.PADDING, 
                                            0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Step", 
                                            lambda: self.step, background_color=(255, 128, 128))
        self.one_step_button = Button(self.step_display.x + self.step_display.width  + self.PADDING,
                                        0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "1", lambda: self.train_step(1), **self.BUTTON_KWARGS)
        self.ten_step_button = Button(self.one_step_button.x + self.one_step_button.width  + self.PADDING,
                                        0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "10", lambda: self.train_step(10), **self.BUTTON_KWARGS)
        self.hundred_step_button = Button(self.ten_step_button.x + self.ten_step_button.width  + self.PADDING,
                                        0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "100", lambda: self.train_step(100), **self.BUTTON_KWARGS)
    
        self.learning_rate_display = VariableDisplay(0, self.episode_display.height + self.PADDING, self.BUTTON_WIDTH, self.BUTTON_HEIGHT,
                                                "Learning rate", lambda: self.learning_rate, **self.BUTTON_KWARGS)
        self.epsilon_display = VariableDisplay(self.learning_rate_display.width + self.PADDING, self.learning_rate_display.y, 
                                                self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Epsilon", lambda: self.epsilon, 
                                                **self.BUTTON_KWARGS)
        self.discount_factor_display = VariableDisplay(self.epsilon_display.x + self.epsilon_display.width + self.PADDING, self.epsilon_display.y, 
                                                self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Discount factor", lambda: self.discount_factor, 
                                                **self.BUTTON_KWARGS)
        self.policy_overlay_button = Button(self.discount_factor_display.x + self.discount_factor_display.width  + self.PADDING,
                                        self.discount_factor_display.y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Enable P-Overlay", 
                                        self.policy_overlay_button_function, **self.BUTTON_KWARGS)
        self.heatmap_overlay_button = Button(self.policy_overlay_button.x + self.policy_overlay_button.width  + self.PADDING,
                                        self.policy_overlay_button.y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Enable Heatmap", 
                                        self.heatmap_overlay_button_function, **self.BUTTON_KWARGS)
        self.execute_policy_button = Button(self.heatmap_overlay_button.x + self.heatmap_overlay_button.width  + self.PADDING,
                                        self.heatmap_overlay_button.y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Execute Policy", 
                                        self.execute_policy_button_function, **self.BUTTON_KWARGS)
        self.reset_button = Button(self.execute_policy_button.x + self.execute_policy_button.width  + self.PADDING,
                                        self.execute_policy_button.y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Reset", 
                                        self.reset_button_function, **self.BUTTON_KWARGS)


        item_width = self.window.width // 2
        item_height = (self.window.height - (self.BUTTON_HEIGHT + self.PADDING) * 2) // 2
        self.maze = Maze.load_from_numpy_array("./mazes/first_example_maze.npy")
        self.maze_model = MazeModel(self.maze, self.learning_rate, self.discount_factor, start_state = self.state)
        self.agent = DynaQAgent(self.maze_model)

        self.maze_visualizer = MazeVisualizer(0, self.discount_factor_display.y + self.discount_factor_display.height + self.PADDING, 
                                        self.maze.walls, item_width, item_height, self.state, self.maze.goal_states, wall_color=(128, 128, 128), 
                                        path_color=(255, 255, 255), player_color=(255, 0, 0), player_coords=self.state)
        self.policy_visualizer = MazePolicyVisualizer(self.maze_visualizer.x, self.maze_visualizer.y, self.agent.model.generate_policy, 
                                            self.maze_visualizer.width, self.maze_visualizer.height, cell_color=None)
        self.table = TableVisualizer(self.maze_visualizer.x, self.maze_visualizer.y + self.maze_visualizer.height, item_width, item_height, 
                                    lambda: self.agent.model.state_reward_table, grid_color=(0, 0, 0))
        self.heatmap = HeatMapVisualizer(self.table.x, self.table.y, lambda: self.agent.model.state_reward_table, self.table.width, self.table.height)

        self.steps_per_episode_figure, self.reward_per_episode_figure = self.init_mpl_plots()

        self.steps_per_episode_plot = MatplotlibPlotDisplay(self.maze_visualizer.x + self.maze_visualizer.width, self.maze_visualizer.y, 
                                                            self.steps_per_episode_figure, item_width, item_height)
        self.reward_per_episode_plot = MatplotlibPlotDisplay(self.steps_per_episode_plot.x, self.steps_per_episode_plot.y + self.steps_per_episode_plot.height, 
                                                            self.reward_per_episode_figure, item_width, item_height)



        fps_display = FPSDisplay(self.window.width - self.BUTTON_WIDTH, 0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, self.window)

        for object_ in (self.episode_display, self.one_episode_button, self.ten_episode_button, self.hundred_episode_button,
                        self.step_display, self.one_step_button, self.ten_step_button, self.hundred_step_button, 
                        self.learning_rate_display, self.epsilon_display, self.discount_factor_display, self.policy_overlay_button,
                        self.heatmap_overlay_button, self.execute_policy_button, self.reset_button,

                        self.maze_visualizer, self.table, self.steps_per_episode_plot, self.reward_per_episode_plot,

                        fps_display):
            self.window.add_game_object(object_)

    def remove_from_render_pipeline(self, object_):
        if self.is_in_render_pipeline(object_):
            self.window.game_objects.pop(self.window.game_objects.index(object_))

    def add_to_render_pipeline(self, object_):
        if not self.is_in_render_pipeline(object_):
            self.window.game_objects.append(object_)

    def is_in_render_pipeline(self, object_):
        return object_ in self.window.game_objects

    def train_episode(self, n):
        # train n episodes
        thread = Thread(target=self.train_episode_agent_thread_function, args = [n])
        thread.daemon = True
        thread.start()

    def train_step(self, n):
        # train n steps
        thread = Thread(target=self.train_steps_agent_thread_function, args = [n])
        thread.daemon = True
        thread.start()

    def init_mpl_plots(self):
        steps_per_episode_figure, ax = plt.subplots(1, 1)
        self.steps_per_episode_line, = ax.plot([], [])
        self.steps_per_episode_line.set_xdata(self.steps_per_episode_line.get_xdata().tolist())
        self.steps_per_episode_line.set_ydata(self.steps_per_episode_line.get_ydata().tolist())
        ax.set_title("STEPS/EPISODE")
        ax.set_xlabel("Episodes")
        # ax.autoscale(enable=False, axis="both")
        ax.autoscale(enable=True, axis="both")

        reward_per_episode_figure, ax = plt.subplots(1, 1)
        self.reward_per_episode_line, = ax.plot([], [])
        self.reward_per_episode_line.set_xdata(self.reward_per_episode_line.get_xdata().tolist())
        self.reward_per_episode_line.set_ydata(self.reward_per_episode_line.get_ydata().tolist())
        ax.set_title("REWARD/EPISODE")
        ax.set_xlabel("Episodes")
        # ax.autoscale(enable=False, axis="both")
        ax.autoscale(enable=True, axis="both")
        return steps_per_episode_figure, reward_per_episode_figure

    def run(self):
        self.window.run()

    def close(self):
        pass