from abc import ABC, abstractmethod
from threading import Thread, Event
import matplotlib.pyplot as plt
import numpy as np

from . import Window
from .game_objects import (Button, TableVisualizer, MazeVisualizer, TextInput, HeatMapVisualizer, 
                            FPSDisplay, MatplotlibPlotDisplay, MazePolicyVisualizer, VariableDisplay,
                            Slider, ObstacleEnvironmentVisualizer)

from app.rf.maze import Maze
from app.rf.obstacle_course import ObstacleEnvironment, Obstacle
from app.rf.model import MazeModel, DynaQPlusMazeModel, Model
from app.rf.agents import DynaQAgent, AgentVisualizer, DynaQPlusAgent, ObstacleCourseAgentVisualizer

def rgb_to_mpl_colors(color):
    return tuple([c / 255 for c in color])

class Scene(ABC):

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def close(self):
        pass

class DynaQMazeScene(Scene):
    

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
            self.add_to_render_pipeline(self.table)
            self.policy_overlay_button.text = "Enable Heatmap"
        else:
            self.add_to_render_pipeline(self.heatmap)
            self.remove_from_render_pipeline(self.table)
            self.policy_overlay_button.text = "Disable Heatmap"

    def reset_button_function(self):
        self.agent.reset()
        self.maze_model.reset()
        self.update_plots()

    def execute_policy_thread_function(self):
        self.agent.execute_policy(50)

    def pause_slider_function(self, pause_value):
        pause_value = round(pause_value) / 1000
        self.agent.pause = pause_value if pause_value > 0 else None

    def planning_steps_slider_function(self, planning_steps, agent):
        agent.planning_steps = round(planning_steps)

    def train_steps_agent_thread_function(self, n):
        self.agent.train_steps(n)
        self.update_plots()

    def train_episode_agent_thread_function(self, n):
        for _ in range(n):
            self.agent.train_episode()
            self.update_plots()

    def update_plots(self):
        X = [i for i in range(self.agent.episode)]
        self.steps_per_episode_line.set_xdata(X)
        self.steps_per_episode_line.set_ydata(self.agent.steps_per_episode)
        self.steps_per_episode_line.axes.set_xlim(0, self.agent.episode - 1)
        self.steps_per_episode_line.axes.set_ylim(0, max(self.agent.steps_per_episode) if len(self.agent.steps_per_episode) else 1)

        self.reward_per_episode_line.set_xdata(X)
        self.reward_per_episode_line.set_ydata(self.agent.episode_rewards)
        self.reward_per_episode_line.axes.set_xlim(0, self.agent.episode - 1)
        if len(self.agent.episode_rewards) > 0:
            self.reward_per_episode_line.axes.set_ylim(min(self.agent.episode_rewards) - 1, max(self.agent.episode_rewards) + 1)
        self.steps_per_episode_plot.update_next_frame()
        self.reward_per_episode_plot.update_next_frame()


    def __init__(self, width, height, title, maze, max_fps = 60):
        self.window = Window(width, height, title, max_fps)
        self.window.add_size_change_callback(self.create_widgets)

        self.learning_rate = 0.1
        self.epsilon = 0.9
        self.discount_factor = 0.95
        self.planning_steps = 0
        self.max_steps_per_episode = 10000
        self.pause = 0 # ms

        self.maze = maze
        self.maze_model = MazeModel(self.maze)
        self.agent = DynaQAgent(self.maze, self.maze_model, self.learning_rate, self.discount_factor, self.epsilon, self.max_steps_per_episode, self.planning_steps)
        self.agent_visualizer = AgentVisualizer(self.agent)

        self.create_widgets()


    def create_widgets(self):
        self.clear_render_pipeline()
        width_ratio = (self.window.width / 1920)
        self.PADDING = int(10 * width_ratio) # px
        self.BUTTON_WIDTH = self.window.width / 9 - self.PADDING
        self.BUTTON_HEIGHT = int(50 * width_ratio)
        self.FONT_SIZE = int(32 * width_ratio)
        self.BUTTON_KWARGS = {
            "background_color": (255, 128, 128),
            "font_size": self.FONT_SIZE,
        }

        # BUTTON BAR
        self.episode_display = VariableDisplay(0, 0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Episode", 
                                            lambda: self.agent.episode, **self.BUTTON_KWARGS)
        self.one_episode_button = Button(self.episode_display.x + self.episode_display.width  + self.PADDING,
                                        0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "1", lambda: self.train_episode(1), **self.BUTTON_KWARGS)
        self.ten_episode_button = Button(self.one_episode_button.x + self.one_episode_button.width  + self.PADDING,
                                        0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "10", lambda: self.train_episode(10), **self.BUTTON_KWARGS)
        self.hundred_episode_button = Button(self.ten_episode_button.x + self.ten_episode_button.width  + self.PADDING,
                                        0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "100", lambda: self.train_episode(100), **self.BUTTON_KWARGS)

        self.step_display = VariableDisplay(self.hundred_episode_button.x + self.hundred_episode_button.width + self.PADDING, 
                                            0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Step", 
                                            lambda: self.agent.step, **self.BUTTON_KWARGS)
        self.one_step_button = Button(self.step_display.x + self.step_display.width  + self.PADDING,
                                        0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "1", lambda: self.train_step(1), **self.BUTTON_KWARGS)
        self.ten_step_button = Button(self.one_step_button.x + self.one_step_button.width  + self.PADDING,
                                        0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "10", lambda: self.train_step(10), **self.BUTTON_KWARGS)
        self.hundred_step_button = Button(self.ten_step_button.x + self.ten_step_button.width  + self.PADDING,
                                        0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "100", lambda: self.train_step(100), **self.BUTTON_KWARGS)
    
        self.learning_rate_display = VariableDisplay(0, self.episode_display.height + self.PADDING, self.BUTTON_WIDTH, self.BUTTON_HEIGHT,
                                                "Learning rate", lambda: self.agent.learning_rate, **self.BUTTON_KWARGS)
        self.epsilon_display = VariableDisplay(self.learning_rate_display.width + self.PADDING, self.learning_rate_display.y, 
                                                self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Epsilon", lambda: self.agent.epsilon, 
                                                **self.BUTTON_KWARGS)
        self.discount_factor_display = VariableDisplay(self.epsilon_display.x + self.epsilon_display.width + self.PADDING, self.epsilon_display.y, 
                                                self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Discount factor", lambda: self.agent.discount_factor, 
                                                **self.BUTTON_KWARGS)
        self.pause_display = VariableDisplay(self.discount_factor_display.x + self.discount_factor_display.width + self.PADDING, self.discount_factor_display.y, 
                                                self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Pause [ms]", lambda: self.agent.pause * 1000 if self.agent.pause is not None else 0, 
                                                **self.BUTTON_KWARGS)
        self.plan_steps_display = VariableDisplay(self.pause_display.x + self.pause_display.width + self.PADDING, self.pause_display.y, 
                                                self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Plan steps", lambda: self.agent.planning_steps, 
                                                **self.BUTTON_KWARGS)
        self.policy_overlay_button = Button(self.plan_steps_display.x + self.plan_steps_display.width  + self.PADDING,
                                        self.plan_steps_display.y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Enable P-Overlay", 
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
        

        self.maze_visualizer = MazeVisualizer(0, self.discount_factor_display.y + self.discount_factor_display.height + self.PADDING, 
                                        self.maze.paths, item_width, item_height, self.maze.start, self.maze.goal_states, [self.agent_visualizer], 
                                        wall_color=(128, 128, 128), path_color=(255, 255, 255), player_color=(255, 0, 0))
        self.policy_visualizer = MazePolicyVisualizer(self.maze_visualizer.x, self.maze_visualizer.y, self.agent.generate_policy, 
                                            self.maze_visualizer.width, self.maze_visualizer.height, cell_color=None, 
                                            policy_mask_function = lambda: self.agent.state_reward_table != 0)
        self.table = TableVisualizer(self.maze_visualizer.x, self.maze_visualizer.y + self.maze_visualizer.height, item_width, item_height, 
                                    lambda: self.agent.state_reward_table, grid_color=(0, 0, 0), font_size=self.FONT_SIZE)
        self.heatmap = HeatMapVisualizer(self.table.x, self.table.y, lambda: self.agent.state_reward_table, self.table.width, self.table.height)

        
        
        
        
        self.steps_per_episode_figure, self.reward_per_episode_figure = self.init_mpl_plots()
        plot_width_offset = 150 * (width_ratio)
        self.steps_per_episode_plot = MatplotlibPlotDisplay(self.maze_visualizer.x + self.maze.paths.shape[1] * self.maze_visualizer.cell_size + self.PADDING, 
                                                            self.maze_visualizer.y, self.steps_per_episode_figure, item_width - plot_width_offset, item_height)
        self.reward_per_episode_plot = MatplotlibPlotDisplay(self.steps_per_episode_plot.x, self.steps_per_episode_plot.y + self.steps_per_episode_plot.height, 
                                                            self.reward_per_episode_figure, item_width - plot_width_offset, item_height)

        slider_width = (self.window.width - (self.steps_per_episode_plot.x + self.steps_per_episode_plot.width)) / 2 - self.PADDING * 2#150 * (width_ratio)
        handle_radius = int(20 * width_ratio)
        bar_width = int(10 * width_ratio)
        self.pause_length_slider = Slider(self.steps_per_episode_plot.x + self.steps_per_episode_plot.width + (self.window.width - self.steps_per_episode_plot.x - self.steps_per_episode_plot.width) // 3 - slider_width // 2,
                                        self.maze_visualizer.y, slider_width, item_height * 2 - 50, "Pause [ms]", 0, 250, bar_width, handle_radius, self.pause_slider_function,
                                        handle_color=(255, 128, 128), bar_color=(128, 64, 64), label_color=(255, 128, 128), font_size=self.FONT_SIZE)
        self.plan_steps_slider = Slider(self.pause_length_slider.x + self.pause_length_slider.width + self.PADDING,
                                        self.pause_length_slider.y, slider_width, item_height * 2 - 50, "Plan steps", 0, 100, bar_width, handle_radius, lambda plan_steps: self.planning_steps_slider_function(plan_steps, self.agent),
                                        handle_color=(255, 128, 128), bar_color=(128, 64, 64), label_color=(255, 128, 128), font_size=self.FONT_SIZE)


        fps_display = FPSDisplay(self.window.width - self.BUTTON_WIDTH, 0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, self.window, font_size=self.FONT_SIZE)

        for object_ in (self.episode_display, self.one_episode_button, self.ten_episode_button, self.hundred_episode_button,
                        self.step_display, self.one_step_button, self.ten_step_button, self.hundred_step_button, 
                        self.learning_rate_display, self.epsilon_display, self.discount_factor_display, self.pause_display, self.plan_steps_display,
                        self.policy_overlay_button, self.heatmap_overlay_button, self.execute_policy_button, self.reset_button,
                         

                        self.maze_visualizer, self.table, self.steps_per_episode_plot, self.reward_per_episode_plot, 
                        self.pause_length_slider, self.plan_steps_slider,

                        fps_display):
            self.window.add_game_object(object_)

    def clear_render_pipeline(self):
        self.window.game_objects.clear()

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

    def execute_policy_button_function(self):
        thread = Thread(target=self.execute_policy_thread_function, daemon = True)
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

class DynaQMazeMultiAgentScene(Scene):
    

    def policy_overlay_button_function(self, agent, agent_visualizer):
        if self.selected_policy_agent == agent:
            self.remove_from_render_pipeline(self.policy_visualizer)
            self.table.data_function = lambda: np.zeros(agent.state_reward_table.shape)
            self.heatmap.data_function = lambda: np.zeros(agent.state_reward_table.shape)
            self.selected_policy_agent = None
            return
        self.add_to_render_pipeline(self.policy_visualizer)
        self.policy_visualizer.arrow_color = agent_visualizer.color
        self.policy_visualizer.policy_function = agent.generate_policy
        self.policy_visualizer.policy_mask_function = lambda: agent.state_reward_table != 0
        self.table.data_function = lambda: agent.state_reward_table
        self.heatmap.data_function = lambda: agent.state_reward_table
        self.selected_policy_agent = agent

    def heatmap_overlay_button_function(self):
        if self.is_in_render_pipeline(self.heatmap):
            self.remove_from_render_pipeline(self.heatmap)
            self.add_to_render_pipeline(self.table)
            self.heatmap_overlay_button.text = "Enable Heatmap"
        else:
            self.add_to_render_pipeline(self.heatmap)
            self.remove_from_render_pipeline(self.table)
            self.heatmap_overlay_button.text = "Disable Heatmap"

    def reset_button_function(self):
        for agent in self.agents:
            agent.reset()
        self.maze_model.reset()
        self.update_plots()

    def execute_policy_thread_function(self, agent):
        agent.execute_policy(50)

    def pause_slider_function(self, pause_value):
        pause_value = round(pause_value) / 1000
        for agent in self.agents:
            agent.pause = pause_value if pause_value > 0 else None

    def planning_steps_slider_function(self, planning_steps, agent):
        agent.planning_steps = round(planning_steps)

    def train_steps_agent_thread_function(self, n, agent):
        agent.train_steps(n)
        self.update_plots()

    def train_episode_agent_thread_function(self, n, agent):
        for _ in range(n):
            agent.train_episode()
            self.update_plots()

    def update_plots(self):        
        for agent, line in zip(self.agents, 
                        (self.agent_one_steps_per_episode_line, 
                        self.agent_two_steps_per_episode_line, 
                        self.agent_three_steps_per_episode_line)):
            line.set_xdata([i for i in range(agent.episode)])
            line.set_ydata(agent.steps_per_episode)
        self.agent_one_steps_per_episode_line.axes.set_xlim(0, max([a.episode for a in self.agents]))
        self.agent_one_steps_per_episode_line.axes.set_ylim(0, max([max(a.steps_per_episode) if len(a.steps_per_episode) else 0 for a in self.agents ]))
        
        self.steps_per_episode_plot.update_next_frame()


    def __init__(self, width, height, title, maze, max_fps = 60):
        self.window = Window(width, height, title, max_fps)
        self.window.add_size_change_callback(self.create_widgets)

        self.learning_rate = 0.1
        self.epsilon = 0.9
        self.discount_factor = 0.95
        self.planning_steps = 0
        self.max_steps_per_episode = 100000
        self.pause = 0 # ms

        self.maze = maze
        self.maze_model = MazeModel(self.maze)
        self.agent_one = DynaQAgent(self.maze, self.maze_model, self.learning_rate, self.discount_factor, self.epsilon, self.max_steps_per_episode, self.planning_steps)        
        self.agent_two = DynaQAgent(self.maze, self.maze_model, self.learning_rate, self.discount_factor, self.epsilon, self.max_steps_per_episode, self.planning_steps)        
        self.agent_three = DynaQAgent(self.maze, self.maze_model, self.learning_rate, self.discount_factor, self.epsilon, self.max_steps_per_episode, self.planning_steps)
        self.agent_one_visualizer = AgentVisualizer(self.agent_one)
        self.agent_two_visualizer = AgentVisualizer(self.agent_two, color = (255, 0, 255))
        self.agent_three_visualizer = AgentVisualizer(self.agent_three, color = (0, 0, 255))
        self.agents = (self.agent_one, self.agent_two, self.agent_three)
        self.selected_policy_agent = None # keep track of the currently selected policy to display

        self.create_widgets()


    def create_widgets(self):
        self.clear_render_pipeline()
        width_ratio = (self.window.width / 1920)
        self.PADDING = int(10 * width_ratio) # px
        self.BUTTON_WIDTH = self.window.width / 9 - self.PADDING
        self.BUTTON_HEIGHT = int(50 * width_ratio)
        self.FONT_SIZE = int(32 * width_ratio)
        self.BUTTON_KWARGS = {
            "background_color": (255, 128, 128),
            "font_size": self.FONT_SIZE,
        }

        # BUTTON BAR
        self.episode_display = VariableDisplay(0, 0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Episode", 
                                            lambda:"X", **self.BUTTON_KWARGS)
        self.one_episode_button = Button(self.episode_display.x + self.episode_display.width  + self.PADDING,
                                        0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "1", lambda: self.train_episode(1), **self.BUTTON_KWARGS)
        self.ten_episode_button = Button(self.one_episode_button.x + self.one_episode_button.width  + self.PADDING,
                                        0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "10", lambda: self.train_episode(10), **self.BUTTON_KWARGS)
        self.hundred_episode_button = Button(self.ten_episode_button.x + self.ten_episode_button.width  + self.PADDING,
                                        0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "100", lambda: self.train_episode(100), **self.BUTTON_KWARGS)

        self.step_display = VariableDisplay(self.hundred_episode_button.x + self.hundred_episode_button.width + self.PADDING, 
                                            0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Step", 
                                            lambda: "X", **self.BUTTON_KWARGS)
        self.one_step_button = Button(self.step_display.x + self.step_display.width  + self.PADDING,
                                        0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "1", lambda: self.train_step(1), **self.BUTTON_KWARGS)
        self.ten_step_button = Button(self.one_step_button.x + self.one_step_button.width  + self.PADDING,
                                        0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "10", lambda: self.train_step(10), **self.BUTTON_KWARGS)
        self.hundred_step_button = Button(self.ten_step_button.x + self.ten_step_button.width  + self.PADDING,
                                        0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "100", lambda: self.train_step(100), **self.BUTTON_KWARGS)
        self.reset_button = Button(self.hundred_step_button.x + self.hundred_step_button.width  + self.PADDING,
                                        self.hundred_step_button.y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Reset", 
                                        self.reset_button_function, **self.BUTTON_KWARGS)                                        
    

        self.learning_rate_display = VariableDisplay(0, self.episode_display.height + self.PADDING, self.BUTTON_WIDTH, self.BUTTON_HEIGHT,
                                                "Learning rate", lambda: self.agent_one.learning_rate, **self.BUTTON_KWARGS)
        self.epsilon_display = VariableDisplay(self.learning_rate_display.width + self.PADDING, self.learning_rate_display.y, 
                                                self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Epsilon", lambda: self.agent_one.epsilon, 
                                                **self.BUTTON_KWARGS)
        self.discount_factor_display = VariableDisplay(self.epsilon_display.x + self.epsilon_display.width + self.PADDING, self.epsilon_display.y, 
                                                self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Discount factor", lambda: self.agent_one.discount_factor, 
                                                **self.BUTTON_KWARGS)
        self.pause_display = VariableDisplay(self.discount_factor_display.x + self.discount_factor_display.width + self.PADDING, self.discount_factor_display.y, 
                                                self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Pause [ms]", lambda: self.agent_one.pause * 1000 if self.agent_one.pause is not None else 0, 
                                                **self.BUTTON_KWARGS)
        self.heatmap_overlay_button = Button(self.pause_display.x + self.pause_display.width + self.PADDING, self.pause_display.y, 
                                                self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Heatmap Overlay", self.heatmap_overlay_button_function, 
                                                **self.BUTTON_KWARGS)
        self.policy_overlay_agent_one_button = Button(self.heatmap_overlay_button.x + self.heatmap_overlay_button.width  + self.PADDING,
                                        self.heatmap_overlay_button.y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "P-Overlay 1", 
                                        lambda: self.policy_overlay_button_function(self.agent_one, self.agent_one_visualizer), **self.BUTTON_KWARGS)
        self.policy_overlay_agent_two_button = Button(self.policy_overlay_agent_one_button.x + self.policy_overlay_agent_one_button.width  + self.PADDING,
                                        self.policy_overlay_agent_one_button.y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "P-Overlay 2", 
                                        lambda: self.policy_overlay_button_function(self.agent_two, self.agent_two_visualizer), **self.BUTTON_KWARGS)
        self.policy_overlay_agent_three_button = Button(self.policy_overlay_agent_two_button.x + self.policy_overlay_agent_two_button.width  + self.PADDING,
                                        self.policy_overlay_agent_two_button.y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "P-Overlay 3", 
                                        lambda: self.policy_overlay_button_function(self.agent_three, self.agent_three_visualizer), **self.BUTTON_KWARGS)
        self.execute_policy_button = Button(self.policy_overlay_agent_three_button.x + self.policy_overlay_agent_three_button.width  + self.PADDING,
                                        self.policy_overlay_agent_three_button.y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Execute Policy", 
                                        self.execute_policy_button_function, **self.BUTTON_KWARGS)
        

        


        item_width = self.window.width // 2
        item_height = (self.window.height - (self.BUTTON_HEIGHT + self.PADDING) * 2) // 2
        

        self.maze_visualizer = MazeVisualizer(0, self.discount_factor_display.y + self.discount_factor_display.height + self.PADDING, 
                                        self.maze.paths, item_width, item_height, self.maze.start, self.maze.goal_states, 
                                        [self.agent_one_visualizer, self.agent_two_visualizer, self.agent_three_visualizer], 
                                        wall_color=(128, 128, 128), path_color=(255, 255, 255), player_color=(255, 0, 0))
        self.policy_visualizer = MazePolicyVisualizer(self.maze_visualizer.x, self.maze_visualizer.y, lambda: np.zeros(self.maze_model.get_blank_policy().shape), # return an empty list to display nothing 
                                            self.maze_visualizer.width, self.maze_visualizer.height, cell_color=None, policy_mask_function=lambda: np.full(False, self.maze_model.get_blank_policy().shape))
        self.table = TableVisualizer(self.maze_visualizer.x, self.maze_visualizer.y + self.maze_visualizer.height, item_width, item_height, 
                                    lambda: np.zeros(self.maze_model.get_blank_policy().shape), grid_color=(0, 0, 0), font_size=self.FONT_SIZE)
        self.heatmap = HeatMapVisualizer(self.table.x, self.table.y, lambda: np.zeros(self.maze_model.get_blank_policy().shape), self.table.width, self.table.height)
        
        
        
        self.steps_per_episode_figure = self.init_mpl_plots()
        plot_width_offset = 150 * (width_ratio)
        self.steps_per_episode_plot = MatplotlibPlotDisplay(self.maze_visualizer.x + self.maze.paths.shape[1] * self.maze_visualizer.cell_size + self.PADDING, 
                                                            self.maze_visualizer.y, self.steps_per_episode_figure, item_width - plot_width_offset, item_height)

        slider_width = (self.window.width - (self.steps_per_episode_plot.x + self.steps_per_episode_plot.width)) / 2 - self.PADDING * 2#150 * (width_ratio)
        handle_radius = int(20 * width_ratio)
        bar_width = int(10 * width_ratio)
        self.pause_length_slider = Slider(self.steps_per_episode_plot.x + self.steps_per_episode_plot.width + (self.window.width - self.steps_per_episode_plot.x - self.steps_per_episode_plot.width) // 3 - slider_width // 2,
                                        self.maze_visualizer.y, slider_width, item_height * 2 - 50, "Pause [ms]", 0, 250, bar_width, handle_radius, self.pause_slider_function,
                                        handle_color=(255, 128, 128), bar_color=(128, 64, 64), label_color=(255, 128, 128), font_size=self.FONT_SIZE)

        plan_step_slider_width = self.steps_per_episode_plot.width // 3 - self.PADDING
        agent_variable_display_width = plan_step_slider_width // 2 - self.PADDING // 2
        self.agent_one_plan_steps_display = VariableDisplay(self.steps_per_episode_plot.x, self.table.y, 
                                                plan_step_slider_width, self.BUTTON_HEIGHT, "Plan steps", lambda: self.agent_one.planning_steps, 
                                                **self.BUTTON_KWARGS)
        self.agent_one_episode_display = VariableDisplay(self.agent_one_plan_steps_display.x, self.agent_one_plan_steps_display.y + self.agent_one_plan_steps_display.height + self.PADDING, 
                                                agent_variable_display_width, self.BUTTON_HEIGHT, "Episode", lambda: self.agent_one.episode, 
                                                **self.BUTTON_KWARGS)
        self.agent_one_step_display = VariableDisplay(self.agent_one_episode_display.x + self.agent_one_episode_display.width + self.PADDING, self.agent_one_episode_display.y, 
                                                agent_variable_display_width, self.BUTTON_HEIGHT, "Step", lambda: self.agent_one.step, 
                                                **self.BUTTON_KWARGS)

        self.agent_two_plan_steps_display = VariableDisplay(self.agent_one_plan_steps_display.x + self.agent_one_plan_steps_display.width + self.PADDING, self.agent_one_plan_steps_display.y, 
                                                plan_step_slider_width, self.BUTTON_HEIGHT, "Plan steps", lambda: self.agent_two.planning_steps, 
                                                **self.BUTTON_KWARGS)
        self.agent_two_episode_display = VariableDisplay(self.agent_two_plan_steps_display.x, self.agent_two_plan_steps_display.y + self.agent_two_plan_steps_display.height + self.PADDING, 
                                                agent_variable_display_width, self.BUTTON_HEIGHT, "Episode", lambda: self.agent_two.episode, 
                                                **self.BUTTON_KWARGS)
        self.agent_two_step_display = VariableDisplay(self.agent_two_episode_display.x + self.agent_two_episode_display.width + self.PADDING, self.agent_two_episode_display.y, 
                                                agent_variable_display_width, self.BUTTON_HEIGHT, "Step", lambda: self.agent_two.step, 
                                                **self.BUTTON_KWARGS)
                                                
        self.agent_three_plan_steps_display = VariableDisplay(self.agent_two_plan_steps_display.x + self.agent_two_plan_steps_display.width + self.PADDING, self.agent_two_plan_steps_display.y, 
                                                plan_step_slider_width, self.BUTTON_HEIGHT, "Plan steps", lambda: self.agent_three.planning_steps, 
                                                **self.BUTTON_KWARGS)
        self.agent_three_episode_display = VariableDisplay(self.agent_three_plan_steps_display.x, self.agent_three_plan_steps_display.y + self.agent_three_plan_steps_display.height + self.PADDING, 
                                                agent_variable_display_width, self.BUTTON_HEIGHT, "Episode", lambda: self.agent_three.episode, 
                                                **self.BUTTON_KWARGS)
        self.agent_three_step_display = VariableDisplay(self.agent_three_episode_display.x + self.agent_three_episode_display.width + self.PADDING, self.agent_three_episode_display.y, 
                                                agent_variable_display_width, self.BUTTON_HEIGHT, "Step", lambda: self.agent_three.step, 
                                                **self.BUTTON_KWARGS)

        slider_height = self.table.height - self.BUTTON_HEIGHT * 2 - self.PADDING - 50
        self.agent_one_plan_steps_slider = Slider(self.agent_one_plan_steps_display.x,
                                        self.agent_one_episode_display.y + self.agent_one_episode_display.height, plan_step_slider_width, slider_height, "Agent 1 P-Steps", 0, 100, bar_width, handle_radius, lambda p_steps: self.planning_steps_slider_function(p_steps, self.agent_one),
                                        handle_color=(255, 128, 128), bar_color=(128, 64, 64), label_color=self.agent_one_visualizer.color, font_size=self.FONT_SIZE)
        
        self.agent_two_plan_steps_slider = Slider(self.agent_one_plan_steps_slider.x + self.agent_one_plan_steps_slider.width + self.PADDING,
                                        self.agent_one_plan_steps_slider.y, plan_step_slider_width, slider_height, "Agent 2 P-Steps", 0, 100, bar_width, handle_radius, lambda p_steps: self.planning_steps_slider_function(p_steps, self.agent_two),
                                        handle_color=(255, 128, 128), bar_color=(128, 64, 64), label_color=self.agent_two_visualizer.color, font_size=self.FONT_SIZE)
        self.agent_three_plan_steps_slider = Slider(self.agent_two_plan_steps_slider.x + self.agent_two_plan_steps_slider.width + self.PADDING,
                                        self.agent_two_plan_steps_slider.y, plan_step_slider_width, slider_height, "Agent 3 P-Steps", 0, 100, bar_width, handle_radius, lambda p_steps: self.planning_steps_slider_function(p_steps, self.agent_three),
                                        handle_color=(255, 128, 128), bar_color=(128, 64, 64), label_color=self.agent_three_visualizer.color, font_size=self.FONT_SIZE)

        fps_display = FPSDisplay(self.window.width - self.BUTTON_WIDTH // 2, 0, self.BUTTON_WIDTH // 2, self.BUTTON_HEIGHT // 2, self.window, font_size=self.FONT_SIZE // 2)



        for object_ in (self.episode_display, self.one_episode_button, self.ten_episode_button, self.hundred_episode_button,
                        self.step_display, self.one_step_button, self.ten_step_button, self.hundred_step_button, 
                        self.learning_rate_display, self.epsilon_display, self.discount_factor_display, self.pause_display, self.heatmap_overlay_button,
                        self.policy_overlay_agent_one_button, self.policy_overlay_agent_two_button, self.execute_policy_button, self.policy_overlay_agent_three_button, 
                        self.reset_button,
                         

                        self.maze_visualizer, self.table, self.steps_per_episode_plot,
                        self.agent_one_plan_steps_display, self.agent_two_plan_steps_display, self.agent_three_plan_steps_display,
                        self.agent_one_episode_display, self.agent_one_step_display, self.agent_two_episode_display, self.agent_two_step_display, self.agent_three_episode_display, self.agent_three_step_display,
                        self.pause_length_slider, self.agent_one_plan_steps_slider, self.agent_two_plan_steps_slider, self.agent_three_plan_steps_slider,

                        fps_display):
            self.window.add_game_object(object_)

    def clear_render_pipeline(self):
        self.window.game_objects.clear()

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
        for agent in self.agents:
            thread = Thread(target=self.train_episode_agent_thread_function, args = [n, agent], daemon=True)
            thread.start()

    def train_step(self, n):
        # train n steps
        for agent in self.agents:
            thread = Thread(target=self.train_steps_agent_thread_function, args = [n, agent], daemon=True)
            thread.start()

    def execute_policy_button_function(self):
        for agent in self.agents:
            thread = Thread(target=self.execute_policy_thread_function, args = [agent], daemon = True)
            thread.start()

    def init_mpl_plots(self):
        steps_per_episode_figure, ax = plt.subplots(1, 1)
        self.agent_one_steps_per_episode_line, = ax.plot([], [], color = rgb_to_mpl_colors(self.agent_one_visualizer.color))
        self.agent_two_steps_per_episode_line, = ax.plot([], [], color = rgb_to_mpl_colors(self.agent_two_visualizer.color))
        self.agent_three_steps_per_episode_line, = ax.plot([], [], color = rgb_to_mpl_colors(self.agent_three_visualizer.color))
        ax.set_title("STEPS/EPISODE")
        ax.set_xlabel("Episodes")
        # ax.autoscale(enable=False, axis="both")
        ax.autoscale(enable=True, axis="both")

        return steps_per_episode_figure

    def run(self):
        self.window.run()

    def close(self):
        pass

class DynaQPlusMazeMultiAgentScene(Scene):
    

    def policy_overlay_button_function(self, agent, agent_visualizer):
        if self.selected_policy_agent == agent:
            self.remove_from_render_pipeline(self.policy_visualizer)
            self.table.data_function = lambda: np.zeros(agent.state_reward_table.shape)
            self.heatmap.data_function = lambda: np.zeros(agent.state_reward_table.shape)
            self.selected_policy_agent = None
            return
        self.add_to_render_pipeline(self.policy_visualizer)
        self.policy_visualizer.arrow_color = agent_visualizer.color
        self.policy_visualizer.policy_function = agent.generate_policy
        self.policy_visualizer.policy_mask_function = lambda: agent.state_reward_table != 0
        self.table.data_function = lambda: agent.state_reward_table
        self.heatmap.data_function = lambda: agent.state_reward_table
        self.selected_policy_agent = agent

    def heatmap_overlay_button_function(self):
        if self.is_in_render_pipeline(self.heatmap):
            self.remove_from_render_pipeline(self.heatmap)
            self.add_to_render_pipeline(self.table)
            self.heatmap_overlay_button.text = "Enable Heatmap"
        else:
            self.add_to_render_pipeline(self.heatmap)
            self.remove_from_render_pipeline(self.table)
            self.heatmap_overlay_button.text = "Disable Heatmap"

    def reset_button_function(self):
        for agent in self.agents:
            agent.reset()
        self.maze_model.reset()
        self.dynaq_plus_maze_model.reset()
        self.update_plots()

    def start_button_function(self):
        self.stop_event = Event()
        thread = Thread(target=self.train_endless, args = [self.stop_event], daemon=True)
        thread.start()

    def stop_button_function(self):
        self.stop_event.set()

    def execute_policy_thread_function(self, agent):
        agent.execute_policy(50)

    def pause_slider_function(self, pause_value):
        pause_value = round(pause_value) / 1000
        for agent in self.agents:
            agent.pause = pause_value if pause_value > 0 else None
    
    def k_lider_function(self, agent, k):
        agent.k = round(k, 5)

    def planning_steps_slider_function(self, planning_steps, agent):
        agent.planning_steps = round(planning_steps)

    def train_endless(self, stop_event):
        while not stop_event.is_set():
            for agent in self.agents:
                agent.train_steps(1)
            self.update_plots()

    def train_steps_agent_thread_function(self, n, agent):
        for _ in range(n):
            agent.train_steps(1)
            self.update_plots()

    def train_episode_agent_thread_function(self, n, agent):
        for _ in range(n):
            agent.train_episode()
            self.update_plots()

    def update_plots(self):
        for agent, line in zip(self.agents, 
                        (self.agent_one_steps_per_episode_line, 
                        self.agent_two_steps_per_episode_line, 
                        self.agent_three_steps_per_episode_line)):
            line.set_xdata([i for i in range(agent.step)])
            line.set_ydata(agent.cumulated_step_rewards)
        self.agent_one_steps_per_episode_line.axes.set_xlim(0, max([a.step for a in self.agents]))
        self.agent_one_steps_per_episode_line.axes.set_ylim(0, max([a.cumulated_step_rewards[-1] if len(a.cumulated_step_rewards) > 0 else 0 for a in self.agents]))
        
        self.steps_per_episode_plot.update_next_frame()


    def __init__(self, width, height, title, maze, max_fps = 60):
        self.window = Window(width, height, title, max_fps)
        self.window.add_size_change_callback(self.create_widgets)

        self.learning_rate = 0.25
        self.epsilon = 0.9
        self.discount_factor = 0.95
        self.planning_steps = 0
        self.max_steps_per_episode = 10000
        self.pause = 1 / 1000 # ms

        self.maze = maze
        self.maze_model = MazeModel(self.maze)
        self.dynaq_plus_maze_model = DynaQPlusMazeModel(self.maze)
        self.agent_one = DynaQPlusAgent(self.maze, self.dynaq_plus_maze_model, self.learning_rate, self.discount_factor, self.epsilon, self.max_steps_per_episode, self.planning_steps, k = 0.001, pause = self.pause)        
        self.agent_two = DynaQAgent(self.maze, self.maze_model, self.learning_rate, self.discount_factor, self.epsilon, self.max_steps_per_episode, self.planning_steps, pause = self.pause, use_prioritized_sweeping=True)        
        self.agent_three = DynaQAgent(self.maze, self.maze_model, self.learning_rate, self.discount_factor, self.epsilon, self.max_steps_per_episode, self.planning_steps, pause = self.pause)
        self.agent_one_visualizer = AgentVisualizer(self.agent_one)
        self.agent_two_visualizer = AgentVisualizer(self.agent_two, color = (255, 0, 255))
        self.agent_three_visualizer = AgentVisualizer(self.agent_three, color = (0, 0, 255))
        self.agents = (self.agent_one, self.agent_two, self.agent_three)
        self.selected_policy_agent = None # keep track of the currently selected policy to display

        self.create_widgets()


    def create_widgets(self):
        self.clear_render_pipeline()
        width_ratio = (self.window.width / 1920)
        self.PADDING = int(10 * width_ratio) # px
        self.BUTTON_WIDTH = self.window.width / 9 - self.PADDING
        self.BUTTON_HEIGHT = int(50 * width_ratio)
        self.FONT_SIZE = int(32 * width_ratio)
        self.BUTTON_KWARGS = {
            "background_color": (255, 128, 128),
            "font_size": self.FONT_SIZE,
        }

        # BUTTON BAR
        self.start_button = Button(0, 0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Start", 
                                            self.start_button_function, **self.BUTTON_KWARGS)
        self.stop_button = Button(self.start_button.x + self.start_button.width + self.PADDING, self.start_button.y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Stop", 
                                            self.stop_button_function, **self.BUTTON_KWARGS)
        self.one_episode_button = Button(self.stop_button.x + self.stop_button.width  + self.PADDING,
                                        0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "1 Episode", lambda: self.train_episode(1), **self.BUTTON_KWARGS)
        self.ten_episode_button = Button(self.one_episode_button.x + self.one_episode_button.width  + self.PADDING,
                                        0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "10 Episodes", lambda: self.train_episode(10), **self.BUTTON_KWARGS)
        self.hundred_episode_button = Button(self.ten_episode_button.x + self.ten_episode_button.width  + self.PADDING,
                                        0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "100 Episodes", lambda: self.train_episode(100), **self.BUTTON_KWARGS)

        
        self.one_step_button = Button(self.hundred_episode_button.x + self.hundred_episode_button.width  + self.PADDING,
                                        0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "1 Step", lambda: self.train_step(1), **self.BUTTON_KWARGS)
        self.ten_step_button = Button(self.one_step_button.x + self.one_step_button.width  + self.PADDING,
                                        0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "10 Steps", lambda: self.train_step(10), **self.BUTTON_KWARGS)
        self.hundred_step_button = Button(self.ten_step_button.x + self.ten_step_button.width  + self.PADDING,
                                        0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "100 Steps", lambda: self.train_step(100), **self.BUTTON_KWARGS)
        self.reset_button = Button(self.hundred_step_button.x + self.hundred_step_button.width  + self.PADDING,
                                        self.hundred_step_button.y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Reset", 
                                        self.reset_button_function, **self.BUTTON_KWARGS)                                        
    

        self.learning_rate_display = VariableDisplay(0, self.start_button.height + self.PADDING, self.BUTTON_WIDTH, self.BUTTON_HEIGHT,
                                                "Learning rate", lambda: self.agent_one.learning_rate, **self.BUTTON_KWARGS)
        self.epsilon_display = VariableDisplay(self.learning_rate_display.width + self.PADDING, self.learning_rate_display.y, 
                                                self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Epsilon", lambda: self.agent_one.epsilon, 
                                                **self.BUTTON_KWARGS)
        self.discount_factor_display = VariableDisplay(self.epsilon_display.x + self.epsilon_display.width + self.PADDING, self.epsilon_display.y, 
                                                self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Discount factor", lambda: self.agent_one.discount_factor, 
                                                **self.BUTTON_KWARGS)
        self.pause_display = VariableDisplay(self.discount_factor_display.x + self.discount_factor_display.width + self.PADDING, self.discount_factor_display.y, 
                                                self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Pause [ms]", lambda: self.agent_one.pause * 1000 if self.agent_one.pause is not None else 0, 
                                                **self.BUTTON_KWARGS)
        self.heatmap_overlay_button = Button(self.pause_display.x + self.pause_display.width + self.PADDING, self.pause_display.y, 
                                                self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Heatmap Overlay", self.heatmap_overlay_button_function, 
                                                **self.BUTTON_KWARGS)
        self.policy_overlay_agent_one_button = Button(self.heatmap_overlay_button.x + self.heatmap_overlay_button.width  + self.PADDING,
                                        self.heatmap_overlay_button.y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "P-Overlay 1", 
                                        lambda: self.policy_overlay_button_function(self.agent_one, self.agent_one_visualizer), **self.BUTTON_KWARGS)
        self.policy_overlay_agent_two_button = Button(self.policy_overlay_agent_one_button.x + self.policy_overlay_agent_one_button.width  + self.PADDING,
                                        self.policy_overlay_agent_one_button.y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "P-Overlay 2", 
                                        lambda: self.policy_overlay_button_function(self.agent_two, self.agent_two_visualizer), **self.BUTTON_KWARGS)
        self.policy_overlay_agent_three_button = Button(self.policy_overlay_agent_two_button.x + self.policy_overlay_agent_two_button.width  + self.PADDING,
                                        self.policy_overlay_agent_two_button.y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "P-Overlay 3", 
                                        lambda: self.policy_overlay_button_function(self.agent_three, self.agent_three_visualizer), **self.BUTTON_KWARGS)
        self.execute_policy_button = Button(self.policy_overlay_agent_three_button.x + self.policy_overlay_agent_three_button.width  + self.PADDING,
                                        self.policy_overlay_agent_three_button.y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Execute Policy", 
                                        self.execute_policy_button_function, **self.BUTTON_KWARGS)
        self.k_display = VariableDisplay(self.execute_policy_button.x, self.execute_policy_button.y + self.execute_policy_button.height + self.PADDING, 
                                        self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "K", lambda: self.agent_one.k, **self.BUTTON_KWARGS)
        

        


        item_width = self.window.width // 2
        item_height = (self.window.height - (self.BUTTON_HEIGHT + self.PADDING) * 2) // 2
        

        self.maze_visualizer = MazeVisualizer(0, self.discount_factor_display.y + self.discount_factor_display.height + self.PADDING, 
                                        self.maze.paths, item_width, item_height, self.maze.start, self.maze.goal_states, 
                                        [self.agent_one_visualizer, self.agent_two_visualizer, self.agent_three_visualizer], 
                                        wall_color=(128, 128, 128), path_color=(255, 255, 255), player_color=(255, 0, 0))
        self.policy_visualizer = MazePolicyVisualizer(self.maze_visualizer.x, self.maze_visualizer.y, lambda: np.zeros(self.maze_model.get_blank_policy().shape), # return an empty list to display nothing 
                                            self.maze_visualizer.width, self.maze_visualizer.height, cell_color=None, policy_mask_function=lambda: np.full(False, self.maze_model.get_blank_policy().shape))
        self.table = TableVisualizer(self.maze_visualizer.x, self.maze_visualizer.y + self.maze_visualizer.height, item_width, item_height, 
                                    lambda: np.zeros(self.maze_model.get_blank_policy().shape), grid_color=(0, 0, 0), font_size=self.FONT_SIZE)
        self.heatmap = HeatMapVisualizer(self.table.x, self.table.y, lambda: np.zeros(self.maze_model.get_blank_policy().shape), self.table.width, self.table.height)
        
        
        
        self.steps_per_episode_figure = self.init_mpl_plots()
        plot_width_offset = 150 * (width_ratio)
        self.steps_per_episode_plot = MatplotlibPlotDisplay(self.maze_visualizer.x + self.maze.paths.shape[1] * self.maze_visualizer.cell_size + self.PADDING, 
                                                            self.maze_visualizer.y, self.steps_per_episode_figure, item_width - plot_width_offset, item_height)

        slider_width = (self.window.width - (self.steps_per_episode_plot.x + self.steps_per_episode_plot.width)) / 2 - self.PADDING * 2#150 * (width_ratio)
        handle_radius = int(20 * width_ratio)
        bar_width = int(10 * width_ratio)
        self.pause_length_slider = Slider(self.steps_per_episode_plot.x + self.steps_per_episode_plot.width + (self.window.width - self.steps_per_episode_plot.x - self.steps_per_episode_plot.width) // 3 - slider_width // 2,
                                        self.maze_visualizer.y, slider_width, item_height * 2 - 50, "Pause [ms]", 1, 250, bar_width, handle_radius, self.pause_slider_function,
                                        handle_color=(255, 128, 128), bar_color=(128, 64, 64), label_color=(255, 128, 128), font_size=self.FONT_SIZE)
        self.k_slider = Slider(self.k_display.x, self.k_display.y + self.k_display.height, self.k_display.width, self.window.height - (self.k_display.y + self.k_display.height) - 50,
                                "K", 0, 0.002, bar_width, handle_radius, lambda val: self.k_lider_function(self.agent_one, val),
                                handle_color=(255, 128, 128), bar_color=(128, 64, 64), label_color=(255, 128, 128), font_size=self.FONT_SIZE)

        plan_step_slider_width = self.steps_per_episode_plot.width // 3 - self.PADDING
        agent_variable_display_width = plan_step_slider_width // 2 - self.PADDING // 2
        self.agent_one_plan_steps_display = VariableDisplay(self.steps_per_episode_plot.x, self.table.y, 
                                                plan_step_slider_width, self.BUTTON_HEIGHT, "Plan steps", lambda: self.agent_one.planning_steps, 
                                                **self.BUTTON_KWARGS)
        self.agent_one_episode_display = VariableDisplay(self.agent_one_plan_steps_display.x, self.agent_one_plan_steps_display.y + self.agent_one_plan_steps_display.height + self.PADDING, 
                                                agent_variable_display_width, self.BUTTON_HEIGHT, "Episode", lambda: self.agent_one.episode, 
                                                **self.BUTTON_KWARGS)
        self.agent_one_step_display = VariableDisplay(self.agent_one_episode_display.x + self.agent_one_episode_display.width + self.PADDING, self.agent_one_episode_display.y, 
                                                agent_variable_display_width, self.BUTTON_HEIGHT, "Step", lambda: self.agent_one.step, 
                                                **self.BUTTON_KWARGS)

        self.agent_two_plan_steps_display = VariableDisplay(self.agent_one_plan_steps_display.x + self.agent_one_plan_steps_display.width + self.PADDING, self.agent_one_plan_steps_display.y, 
                                                plan_step_slider_width, self.BUTTON_HEIGHT, "Plan steps", lambda: self.agent_two.planning_steps, 
                                                **self.BUTTON_KWARGS)
        self.agent_two_episode_display = VariableDisplay(self.agent_two_plan_steps_display.x, self.agent_two_plan_steps_display.y + self.agent_two_plan_steps_display.height + self.PADDING, 
                                                agent_variable_display_width, self.BUTTON_HEIGHT, "Episode", lambda: self.agent_two.episode, 
                                                **self.BUTTON_KWARGS)
        self.agent_two_step_display = VariableDisplay(self.agent_two_episode_display.x + self.agent_two_episode_display.width + self.PADDING, self.agent_two_episode_display.y, 
                                                agent_variable_display_width, self.BUTTON_HEIGHT, "Step", lambda: self.agent_two.step, 
                                                **self.BUTTON_KWARGS)
                                                
        self.agent_three_plan_steps_display = VariableDisplay(self.agent_two_plan_steps_display.x + self.agent_two_plan_steps_display.width + self.PADDING, self.agent_two_plan_steps_display.y, 
                                                plan_step_slider_width, self.BUTTON_HEIGHT, "Plan steps", lambda: self.agent_three.planning_steps, 
                                                **self.BUTTON_KWARGS)
        self.agent_three_episode_display = VariableDisplay(self.agent_three_plan_steps_display.x, self.agent_three_plan_steps_display.y + self.agent_three_plan_steps_display.height + self.PADDING, 
                                                agent_variable_display_width, self.BUTTON_HEIGHT, "Episode", lambda: self.agent_three.episode, 
                                                **self.BUTTON_KWARGS)
        self.agent_three_step_display = VariableDisplay(self.agent_three_episode_display.x + self.agent_three_episode_display.width + self.PADDING, self.agent_three_episode_display.y, 
                                                agent_variable_display_width, self.BUTTON_HEIGHT, "Step", lambda: self.agent_three.step, 
                                                **self.BUTTON_KWARGS)

        slider_height = self.table.height - self.BUTTON_HEIGHT * 2 - self.PADDING - 50
        self.agent_one_plan_steps_slider = Slider(self.agent_one_plan_steps_display.x,
                                        self.agent_one_episode_display.y + self.agent_one_episode_display.height, plan_step_slider_width, slider_height, "Agent 1(+) P-Steps", 0, 100, bar_width, handle_radius, lambda p_steps: self.planning_steps_slider_function(p_steps, self.agent_one),
                                        handle_color=(255, 128, 128), bar_color=(128, 64, 64), label_color=self.agent_one_visualizer.color, font_size=self.FONT_SIZE)
        
        self.agent_two_plan_steps_slider = Slider(self.agent_one_plan_steps_slider.x + self.agent_one_plan_steps_slider.width + self.PADDING,
                                        self.agent_one_plan_steps_slider.y, plan_step_slider_width, slider_height, "Agent 2 P-Steps", 0, 100, bar_width, handle_radius, lambda p_steps: self.planning_steps_slider_function(p_steps, self.agent_two),
                                        handle_color=(255, 128, 128), bar_color=(128, 64, 64), label_color=self.agent_two_visualizer.color, font_size=self.FONT_SIZE)
        self.agent_three_plan_steps_slider = Slider(self.agent_two_plan_steps_slider.x + self.agent_two_plan_steps_slider.width + self.PADDING,
                                        self.agent_two_plan_steps_slider.y, plan_step_slider_width, slider_height, "Agent 3 P-Steps", 0, 100, bar_width, handle_radius, lambda p_steps: self.planning_steps_slider_function(p_steps, self.agent_three),
                                        handle_color=(255, 128, 128), bar_color=(128, 64, 64), label_color=self.agent_three_visualizer.color, font_size=self.FONT_SIZE)

        fps_display = FPSDisplay(self.window.width - self.BUTTON_WIDTH // 2, 0, self.BUTTON_WIDTH // 2, self.BUTTON_HEIGHT // 2, self.window, font_size=self.FONT_SIZE // 2)



        for object_ in (self.stop_button, self.one_episode_button, self.ten_episode_button, self.hundred_episode_button,
                        self.start_button, self.one_step_button, self.ten_step_button, self.hundred_step_button, 
                        self.learning_rate_display, self.epsilon_display, self.discount_factor_display, self.pause_display, self.heatmap_overlay_button,
                        self.policy_overlay_agent_one_button, self.policy_overlay_agent_two_button, self.execute_policy_button, self.policy_overlay_agent_three_button, 
                        self.reset_button, self.k_display, self.k_slider,
                         

                        self.maze_visualizer, self.table, self.steps_per_episode_plot,
                        self.agent_one_plan_steps_display, self.agent_two_plan_steps_display, self.agent_three_plan_steps_display,
                        self.agent_one_episode_display, self.agent_one_step_display, self.agent_two_episode_display, self.agent_two_step_display, self.agent_three_episode_display, self.agent_three_step_display,
                        self.pause_length_slider, self.agent_one_plan_steps_slider, self.agent_two_plan_steps_slider, self.agent_three_plan_steps_slider,

                        fps_display):
            self.window.add_game_object(object_)

    def clear_render_pipeline(self):
        self.window.game_objects.clear()

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
        for agent in self.agents:
            thread = Thread(target=self.train_episode_agent_thread_function, args = [n, agent], daemon=True)
            thread.start()

    def train_step(self, n):
        # train n steps
        for agent in self.agents:
            thread = Thread(target=self.train_steps_agent_thread_function, args = [n, agent], daemon=True)
            thread.start()

    def execute_policy_button_function(self):
        for agent in self.agents:
            thread = Thread(target=self.execute_policy_thread_function, args = [agent], daemon = True)
            thread.start()

    def init_mpl_plots(self):
        steps_per_episode_figure, ax = plt.subplots(1, 1)
        self.agent_one_steps_per_episode_line, = ax.plot([], [], color = rgb_to_mpl_colors(self.agent_one_visualizer.color))
        self.agent_two_steps_per_episode_line, = ax.plot([], [], color = rgb_to_mpl_colors(self.agent_two_visualizer.color))
        self.agent_three_steps_per_episode_line, = ax.plot([], [], color = rgb_to_mpl_colors(self.agent_three_visualizer.color))
        ax.set_title("Cumulated Rewards")
        ax.set_xlabel("Steps")
        # ax.autoscale(enable=False, axis="both")
        ax.autoscale(enable=True, axis="both")

        return steps_per_episode_figure

    def run(self):
        self.window.run()

    def close(self):
        pass

class MixedAgentsScene(Scene):
    

    def policy_overlay_button_function(self, agent, agent_visualizer):
        if self.selected_policy_agent == agent and self.is_in_render_pipeline(self.policy_visualizer):
            self.remove_from_render_pipeline(self.policy_visualizer)
            self.selected_policy_agent = None
            return
        self.remove_from_render_pipeline(self.table)
        self.remove_from_render_pipeline(self.heatmap)
        self.add_to_render_pipeline(self.policy_visualizer)
        self.policy_visualizer.arrow_color = agent_visualizer.color
        self.policy_visualizer.policy_function = agent.generate_policy
        self.policy_visualizer.policy_mask_function = lambda: agent.state_reward_table != 0
        self.selected_policy_agent = agent

    def table_display_button_function(self):
        if self.selected_policy_agent is None:
            return
        if self.is_in_render_pipeline(self.table):
            self.remove_from_render_pipeline(self.table)
            self.add_to_render_pipeline(self.policy_visualizer)
            return
        self.remove_from_render_pipeline(self.policy_visualizer)
        self.remove_from_render_pipeline(self.heatmap)
        self.add_to_render_pipeline(self.table)
        self.table.data_function = lambda: self.selected_policy_agent.state_reward_table

    def heatmap_display_button_function(self):
        if self.selected_policy_agent is None:
            return
        if self.is_in_render_pipeline(self.heatmap):
            self.remove_from_render_pipeline(self.heatmap)
            self.add_to_render_pipeline(self.policy_visualizer)
            return
        self.remove_from_render_pipeline(self.policy_visualizer)
        self.remove_from_render_pipeline(self.table)
        self.add_to_render_pipeline(self.heatmap)
        self.heatmap.data_function = lambda: self.selected_policy_agent.state_reward_table

    def reset_button_function(self):
        for agent in self.agents:
            agent.reset()
        self.maze_model.reset()
        self.dynaq_plus_maze_model.reset()

    def start_button_function(self):
        self.stop_event = Event()
        thread = Thread(target=self.train_endless, args = [self.stop_event], daemon=True)
        thread.start()

    def stop_button_function(self):
        self.stop_event.set()

    def execute_policy_thread_function(self, agent):
        agent.execute_policy(50)

    def pause_slider_function(self, pause_value):
        pause_value = round(pause_value) / 1000
        for agent in self.agents:
            agent.pause = pause_value if pause_value > 0 else None
    
    def k_lider_function(self, agent, k):
        agent.k = round(k, 5)

    def planning_steps_slider_function(self, planning_steps, agent):
        agent.planning_steps = round(planning_steps)

    def train_endless(self, stop_event):
        while not stop_event.is_set():
            for agent in self.agents:
                agent.train_steps(1)

    def train_steps_agent_thread_function(self, n, agent):
        for _ in range(n):
            agent.train_steps(1)

    def train_episode_agent_thread_function(self, n, agent):
        for _ in range(n):
            agent.train_episode()


    def __init__(self, width, height, title, maze, agents, max_fps = 60):
        self.window = Window(width, height, title, max_fps)
        self.window.add_size_change_callback(self.create_widgets)

        self.planning_steps = 5
        self.max_steps_per_episode = 1000000
        self.pause = None # ms

        self.maze = maze
        self.maze_model = MazeModel(self.maze) # just there for compatibility
        self.dynaq_plus_maze_model = DynaQPlusMazeModel(self.maze) # just there for compatibility
        self.agent_one, self.agent_two, self.agent_three = agents
        self.agent_one.k = 0.0001
        self.agent_one_visualizer = AgentVisualizer(self.agent_one)
        self.agent_two_visualizer = AgentVisualizer(self.agent_two, color = (255, 0, 255))
        self.agent_three_visualizer = AgentVisualizer(self.agent_three, color = (0, 0, 255))
        self.agents = (self.agent_one, self.agent_two, self.agent_three)
        for agent in self.agents:
            agent.planning_steps = self.planning_steps
        self.selected_policy_agent = None # keep track of the currently selected policy to display

        self.create_widgets()


    def create_widgets(self):
        self.clear_render_pipeline()
        width_ratio = (self.window.width / 1920)
        self.PADDING = int(10 * width_ratio) # px
        self.BUTTON_WIDTH = self.window.width / 9 - self.PADDING
        self.BUTTON_HEIGHT = int(50 * width_ratio)
        self.FONT_SIZE = int(24 * width_ratio)
        self.BUTTON_KWARGS = {
            "background_color": (255, 128, 128),
            "font_size": self.FONT_SIZE,
        }

        # BUTTON BAR
        self.start_button = Button(0, 0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Start", 
                                            self.start_button_function, **self.BUTTON_KWARGS)
        self.stop_button = Button(self.start_button.x + self.start_button.width + self.PADDING, self.start_button.y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Stop", 
                                            self.stop_button_function, **self.BUTTON_KWARGS)
        self.steps_display = VariableDisplay(self.stop_button.x + self.stop_button.width + self.PADDING, self.stop_button.y, 
                                                self.BUTTON_WIDTH, self.BUTTON_HEIGHT,
                                                "Steps", lambda: self.agent_one.step, **self.BUTTON_KWARGS)
        self.table_display_button = Button(self.steps_display.x + self.steps_display.width + self.PADDING, self.steps_display.y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Show Table", 
                                            self.table_display_button_function, **self.BUTTON_KWARGS)
        self.heatmap_display_button = Button(self.table_display_button.x + self.table_display_button.width + self.PADDING, self.table_display_button.y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Show Heatmap", 
                                            self.heatmap_display_button_function, **self.BUTTON_KWARGS)


        self.learning_rate_display = VariableDisplay(0, self.start_button.height + self.PADDING, self.BUTTON_WIDTH, self.BUTTON_HEIGHT,
                                                "Learning rate", lambda: self.agent_one.learning_rate, **self.BUTTON_KWARGS)
        self.epsilon_display = VariableDisplay(self.learning_rate_display.width + self.PADDING, self.learning_rate_display.y, 
                                                self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Epsilon", lambda: self.agent_one.epsilon, 
                                                **self.BUTTON_KWARGS)
        self.discount_factor_display = VariableDisplay(self.epsilon_display.x + self.epsilon_display.width + self.PADDING, self.epsilon_display.y, 
                                                self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Discount factor", lambda: self.agent_one.discount_factor, 
                                                **self.BUTTON_KWARGS)
        self.pause_display = VariableDisplay(self.discount_factor_display.x + self.discount_factor_display.width + self.PADDING, self.discount_factor_display.y, 
                                                self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Pause [ms]", lambda: self.agent_one.pause * 1000 if self.agent_one.pause is not None else 0, 
                                                **self.BUTTON_KWARGS)
        self.policy_overlay_agent_one_button = Button(self.pause_display.x + self.pause_display.width  + self.PADDING,
                                        self.pause_display.y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "P-Overlay 1", 
                                        lambda: self.policy_overlay_button_function(self.agent_one, self.agent_one_visualizer), **self.BUTTON_KWARGS)
        self.policy_overlay_agent_two_button = Button(self.policy_overlay_agent_one_button.x + self.policy_overlay_agent_one_button.width  + self.PADDING,
                                        self.policy_overlay_agent_one_button.y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "P-Overlay 2", 
                                        lambda: self.policy_overlay_button_function(self.agent_two, self.agent_two_visualizer), **self.BUTTON_KWARGS)
        self.policy_overlay_agent_three_button = Button(self.policy_overlay_agent_two_button.x + self.policy_overlay_agent_two_button.width  + self.PADDING,
                                        self.policy_overlay_agent_two_button.y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "P-Overlay 3", 
                                        lambda: self.policy_overlay_button_function(self.agent_three, self.agent_three_visualizer), **self.BUTTON_KWARGS)
        self.execute_policy_button = Button(self.policy_overlay_agent_three_button.x + self.policy_overlay_agent_three_button.width  + self.PADDING,
                                        self.policy_overlay_agent_three_button.y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Execute Policy", 
                                        self.execute_policy_button_function, **self.BUTTON_KWARGS)
        self.reset_button = Button(self.policy_overlay_agent_three_button.x + self.policy_overlay_agent_three_button.width  + self.PADDING,
                                        self.policy_overlay_agent_three_button.y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Reset", 
                                        self.reset_button_function, **self.BUTTON_KWARGS)  
        

        


        item_width = self.window.width * 0.8
        item_height = (self.window.height - (self.BUTTON_HEIGHT + self.PADDING) * 2) 
        

        self.maze_visualizer = MazeVisualizer(0, self.discount_factor_display.y + self.discount_factor_display.height + self.PADDING, 
                                        self.maze.paths, item_width, item_height, self.maze.start, self.maze.goal_states, 
                                        [self.agent_one_visualizer, self.agent_two_visualizer, self.agent_three_visualizer], 
                                        wall_color=(128, 128, 128), path_color=(255, 255, 255), player_color=(255, 0, 0))
        self.policy_visualizer = MazePolicyVisualizer(self.maze_visualizer.x, self.maze_visualizer.y, lambda: np.zeros(self.maze_model.get_blank_policy().shape), # return an empty list to display nothing 
                                            self.maze_visualizer.width, self.maze_visualizer.height, cell_color=None, policy_mask_function=lambda: np.full(False, self.maze_model.get_blank_policy().shape))
        self.table = TableVisualizer(self.maze_visualizer.x, self.maze_visualizer.y, self.maze_visualizer.width, self.maze_visualizer.height, 
                                    lambda: np.zeros(self.maze_model.get_blank_policy().shape), grid_color=(0, 0, 0), font_size=self.FONT_SIZE)
        self.heatmap = HeatMapVisualizer(self.maze_visualizer.x, self.maze_visualizer.y, lambda: np.zeros(self.maze_model.get_blank_policy().shape), self.maze_visualizer.width, self.maze_visualizer.height)


        handle_radius = int(20 * width_ratio)
        bar_width = int(10 * width_ratio)
        slider_width = 100
        self.pause_length_slider = Slider(self.maze_visualizer.x + self.maze_visualizer.width + (self.window.width - self.maze_visualizer.x - self.maze_visualizer.width) // 3 - slider_width // 2,
                                        self.maze_visualizer.y, slider_width, item_height - 50, "Pause [ms]", 0, 250, bar_width, handle_radius, self.pause_slider_function,
                                        handle_color=(255, 128, 128), bar_color=(128, 64, 64), label_color=(255, 128, 128), font_size=self.FONT_SIZE)

        fps_display = FPSDisplay(self.window.width - self.BUTTON_WIDTH // 2, 0, self.BUTTON_WIDTH // 2, self.BUTTON_HEIGHT // 2, self.window, font_size=self.FONT_SIZE // 2)



        for object_ in (self.stop_button, self.start_button, self.learning_rate_display, self.epsilon_display, self.discount_factor_display, self.pause_display, 
                        self.policy_overlay_agent_one_button, self.policy_overlay_agent_two_button, self.execute_policy_button, self.policy_overlay_agent_three_button, 
                        self.reset_button, self.steps_display, self.table_display_button, self.heatmap_display_button,
                         

                        self.maze_visualizer, 
                        self.pause_length_slider,

                        fps_display):
            self.window.add_game_object(object_)

    def clear_render_pipeline(self):
        self.window.game_objects.clear()

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
        for agent in self.agents:
            thread = Thread(target=self.train_episode_agent_thread_function, args = [n, agent], daemon=True)
            thread.start()

    def train_step(self, n):
        # train n steps
        for agent in self.agents:
            thread = Thread(target=self.train_steps_agent_thread_function, args = [n, agent], daemon=True)
            thread.start()

    def execute_policy_button_function(self):
        for agent in self.agents:
            thread = Thread(target=self.execute_policy_thread_function, args = [agent], daemon = True)
            thread.start()

    def run(self):
        self.window.run()

    def close(self):
        pass

class ObstacleCourseScene(Scene):
    

    def policy_overlay_button_function(self, agent, agent_visualizer):
        if self.selected_policy_agent == agent and self.is_in_render_pipeline(self.policy_visualizer):
            self.remove_from_render_pipeline(self.policy_visualizer)
            self.selected_policy_agent = None
            return
        self.remove_from_render_pipeline(self.table)
        self.remove_from_render_pipeline(self.heatmap)
        self.add_to_render_pipeline(self.policy_visualizer)
        self.policy_visualizer.arrow_color = agent_visualizer.color
        self.policy_visualizer.policy_function = agent.generate_policy
        self.policy_visualizer.policy_mask_function = lambda: agent.state_reward_table != 0
        self.selected_policy_agent = agent

    def table_display_button_function(self):
        if self.selected_policy_agent is None:
            return
        if self.is_in_render_pipeline(self.table):
            self.remove_from_render_pipeline(self.table)
            self.add_to_render_pipeline(self.policy_visualizer)
            return
        self.remove_from_render_pipeline(self.policy_visualizer)
        self.remove_from_render_pipeline(self.heatmap)
        self.add_to_render_pipeline(self.table)
        self.table.data_function = lambda: self.selected_policy_agent.state_reward_table

    def heatmap_display_button_function(self):
        if self.selected_policy_agent is None:
            return
        if self.is_in_render_pipeline(self.heatmap):
            self.remove_from_render_pipeline(self.heatmap)
            self.add_to_render_pipeline(self.policy_visualizer)
            return
        self.remove_from_render_pipeline(self.policy_visualizer)
        self.remove_from_render_pipeline(self.table)
        self.add_to_render_pipeline(self.heatmap)
        self.heatmap.data_function = lambda: self.selected_policy_agent.state_reward_table

    def reset_button_function(self):
        for agent in self.agents:
            agent.reset()
        self.maze_model.reset()
        self.dynaq_plus_maze_model.reset()

    def start_button_function(self):
        self.stop_event = Event()
        thread = Thread(target=self.train_endless, args = [self.stop_event], daemon=True)
        thread.start()

    def stop_button_function(self):
        self.stop_event.set()

    def execute_policy_thread_function(self, agent):
        agent.execute_policy(50)

    def pause_slider_function(self, pause_value):
        pause_value = round(pause_value) / 1000
        for agent in self.agents:
            agent.pause = pause_value if pause_value > 0 else None
    
    def k_lider_function(self, agent, k):
        agent.k = round(k, 5)

    def planning_steps_slider_function(self, planning_steps, agent):
        agent.planning_steps = round(planning_steps)

    def train_endless(self, stop_event):
        while not stop_event.is_set():
            for agent in self.agents:
                agent.train_steps(1)

    def train_steps_agent_thread_function(self, n, agent):
        for _ in range(n):
            agent.train_steps(1)

    def train_episode_agent_thread_function(self, n, agent):
        for _ in range(n):
            agent.train_episode()


    def __init__(self, width, height, title, environment, agents, max_fps = 60):
        self.window = Window(width, height, title, max_fps)
        self.window.add_size_change_callback(self.create_widgets)

        self.planning_steps = 5
        self.max_steps_per_episode = 1000000
        self.pause = None # ms

        self.environment = environment
        self.agent_one, self.agent_two = agents
        self.agent_one_visualizer = ObstacleCourseAgentVisualizer(self.agent_one)
        self.agent_two_visualizer = ObstacleCourseAgentVisualizer(self.agent_two, color = (255, 0, 255))
        self.agents = (self.agent_one, self.agent_two)
        for agent in self.agents:
            agent.planning_steps = self.planning_steps
        self.selected_policy_agent = None # keep track of the currently selected policy to display

        self.create_widgets()


    def create_widgets(self):
        self.clear_render_pipeline()
        width_ratio = (self.window.width / 1920)
        self.PADDING = int(10 * width_ratio) # px
        self.BUTTON_WIDTH = self.window.width / 9 - self.PADDING
        self.BUTTON_HEIGHT = int(50 * width_ratio)
        self.FONT_SIZE = int(24 * width_ratio)
        self.BUTTON_KWARGS = {
            "background_color": (255, 128, 128),
            "font_size": self.FONT_SIZE,
        }

        # BUTTON BAR
        self.start_button = Button(0, 0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Start", 
                                            self.start_button_function, **self.BUTTON_KWARGS)
        self.stop_button = Button(self.start_button.x + self.start_button.width + self.PADDING, self.start_button.y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Stop", 
                                            self.stop_button_function, **self.BUTTON_KWARGS)
        self.steps_display = VariableDisplay(self.stop_button.x + self.stop_button.width + self.PADDING, self.stop_button.y, 
                                                self.BUTTON_WIDTH, self.BUTTON_HEIGHT,
                                                "Steps", lambda: self.agent_one.step, **self.BUTTON_KWARGS)
        self.table_display_button = Button(self.steps_display.x + self.steps_display.width + self.PADDING, self.steps_display.y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Show Table", 
                                            self.table_display_button_function, **self.BUTTON_KWARGS)
        self.heatmap_display_button = Button(self.table_display_button.x + self.table_display_button.width + self.PADDING, self.table_display_button.y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Show Heatmap", 
                                            self.heatmap_display_button_function, **self.BUTTON_KWARGS)


        self.learning_rate_display = VariableDisplay(0, self.start_button.height + self.PADDING, self.BUTTON_WIDTH, self.BUTTON_HEIGHT,
                                                "Learning rate", lambda: self.agent_one.learning_rate, **self.BUTTON_KWARGS)
        self.epsilon_display = VariableDisplay(self.learning_rate_display.width + self.PADDING, self.learning_rate_display.y, 
                                                self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Epsilon", lambda: self.agent_one.epsilon, 
                                                **self.BUTTON_KWARGS)
        self.discount_factor_display = VariableDisplay(self.epsilon_display.x + self.epsilon_display.width + self.PADDING, self.epsilon_display.y, 
                                                self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Discount factor", lambda: self.agent_one.discount_factor, 
                                                **self.BUTTON_KWARGS)
        self.pause_display = VariableDisplay(self.discount_factor_display.x + self.discount_factor_display.width + self.PADDING, self.discount_factor_display.y, 
                                                self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Pause [ms]", lambda: self.agent_one.pause * 1000 if self.agent_one.pause is not None else 0, 
                                                **self.BUTTON_KWARGS)
        self.policy_overlay_agent_one_button = Button(self.pause_display.x + self.pause_display.width  + self.PADDING,
                                        self.pause_display.y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "P-Overlay 1", 
                                        lambda: self.policy_overlay_button_function(self.agent_one, self.agent_one_visualizer), **self.BUTTON_KWARGS)
        self.policy_overlay_agent_two_button = Button(self.policy_overlay_agent_one_button.x + self.policy_overlay_agent_one_button.width  + self.PADDING,
                                        self.policy_overlay_agent_one_button.y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "P-Overlay 2", 
                                        lambda: self.policy_overlay_button_function(self.agent_two, self.agent_two_visualizer), **self.BUTTON_KWARGS)
        self.policy_overlay_agent_three_button = Button(self.policy_overlay_agent_two_button.x + self.policy_overlay_agent_two_button.width  + self.PADDING,
                                        self.policy_overlay_agent_two_button.y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "P-Overlay 3", 
                                        lambda: self.policy_overlay_button_function(self.agent_three, self.agent_three_visualizer), **self.BUTTON_KWARGS)
        self.execute_policy_button = Button(self.policy_overlay_agent_three_button.x + self.policy_overlay_agent_three_button.width  + self.PADDING,
                                        self.policy_overlay_agent_three_button.y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Execute Policy", 
                                        self.execute_policy_button_function, **self.BUTTON_KWARGS)
        self.reset_button = Button(self.policy_overlay_agent_three_button.x + self.policy_overlay_agent_three_button.width  + self.PADDING,
                                        self.policy_overlay_agent_three_button.y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Reset", 
                                        self.reset_button_function, **self.BUTTON_KWARGS)  
        

        

        item_width = self.window.width * 0.8
        item_height = (self.window.height - (self.BUTTON_HEIGHT + self.PADDING) * 2) 
        
        self.obstacle_course_visualizer = ObstacleEnvironmentVisualizer(0, self.reset_button.y + self.reset_button.height + self.PADDING,
                                                                        item_width, item_height, self.environment, (self.agent_one_visualizer, self.agent_two_visualizer))

        

        handle_radius = int(20 * width_ratio)
        bar_width = int(10 * width_ratio)
        slider_width = 100
        self.pause_length_slider = Slider(self.obstacle_course_visualizer.x + self.obstacle_course_visualizer.width + (self.window.width - self.obstacle_course_visualizer.x - self.obstacle_course_visualizer.width) // 3 - slider_width // 2,
                                        self.obstacle_course_visualizer.y, slider_width, item_height - 50, "Pause [ms]", 0, 250, bar_width, handle_radius, self.pause_slider_function,
                                        handle_color=(255, 128, 128), bar_color=(128, 64, 64), label_color=(255, 128, 128), font_size=self.FONT_SIZE)

        fps_display = FPSDisplay(self.window.width - self.BUTTON_WIDTH // 2, 0, self.BUTTON_WIDTH // 2, self.BUTTON_HEIGHT // 2, self.window, font_size=self.FONT_SIZE // 2)



        for object_ in (self.stop_button, self.start_button, self.learning_rate_display, self.epsilon_display, self.discount_factor_display, self.pause_display, 
                        self.policy_overlay_agent_one_button, self.policy_overlay_agent_two_button, self.execute_policy_button, self.policy_overlay_agent_three_button, 
                        self.reset_button, self.steps_display, self.table_display_button, self.heatmap_display_button,
                         

                        self.obstacle_course_visualizer, self.pause_length_slider,

                        fps_display):
            self.window.add_game_object(object_)

    def clear_render_pipeline(self):
        self.window.game_objects.clear()

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
        for agent in self.agents:
            thread = Thread(target=self.train_episode_agent_thread_function, args = [n, agent], daemon=True)
            thread.start()

    def train_step(self, n):
        # train n steps
        for agent in self.agents:
            thread = Thread(target=self.train_steps_agent_thread_function, args = [n, agent], daemon=True)
            thread.start()

    def execute_policy_button_function(self):
        for agent in self.agents:
            thread = Thread(target=self.execute_policy_thread_function, args = [agent], daemon = True)
            thread.start()

    def run(self):
        self.window.run()

    def close(self):
        pass