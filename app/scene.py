from abc import ABC, abstractmethod
from threading import Thread
import matplotlib.pyplot as plt

from . import Window
from .game_objects import (Button, TableVisualizer, MazeVisualizer, TextInput, HeatMapVisualizer, 
                            FPSDisplay, MatplotlibPlotDisplay, MazePolicyVisualizer, VariableDisplay,
                            Slider)

from app.rf.maze import Maze
from app.rf.model import MazeModel
from app.rf.agents import DynaQAgent, AgentVisualizer


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
        self.agent.reset()
        self.maze_model.reset()
        self.update_plots()

    def execute_policy_thread_function(self):
        self.agent.execute_policy(self.maze_model, 50)
        self.state = self.start_state
        self.maze_visualizer.player_coords = self.state

    def pause_slider_function(self, pause_value):
        pause_value = round(pause_value) / 1000
        self.agent.pause = pause_value if pause_value > 0 else None

    def planning_steps_slider_function(self, planning_steps):
        self.agent.planning_steps = round(planning_steps)

    def train_steps_agent_thread_function(self, n):
        self.agent.train_steps(n, self.agent.dyna)
        self.update_plots()

    def train_episode_agent_thread_function(self, n):
        for _ in range(n):
            self.agent.train_episode(self.agent.dyna)
            self.update_plots()

    def update_plots(self):
        X = [i for i in range(self.agent.episode)]
        self.steps_per_episode_line.set_xdata(X)
        self.steps_per_episode_line.set_ydata(self.agent.steps_per_episode)
        self.steps_per_episode_line.axes.set_xlim(0, self.agent.episode)
        self.steps_per_episode_line.axes.set_ylim(0, self.agent.max_steps_per_episode + 1)
        
        self.reward_per_episode_line.set_xdata(X)
        self.reward_per_episode_line.set_ydata(self.agent.episode_rewards)
        self.reward_per_episode_line.axes.set_xlim(0, self.agent.episode)
        if len(self.agent.episode_rewards) > 0:
            self.reward_per_episode_line.axes.set_ylim(min(self.agent.episode_rewards) - 1, max(self.agent.episode_rewards) + 1)
        self.steps_per_episode_plot.update_next_frame()
        self.reward_per_episode_plot.update_next_frame()


    def __init__(self, width, height, title, maze_data, max_fps = 60):
        self.learning_rate = 0.1
        self.epsilon = 0.9
        self.discount_factor = 0.95
        self.start_state = (2, 0)
        self.planning_steps = 0
        self.max_steps_per_episode = 1000
        self.pause = 0 # ms
        self.window = Window(width, height, title, max_fps)
        # BUTTON BAR
        self.episode_display = VariableDisplay(0, 0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Episode", 
                                            lambda: self.agent.episode, background_color=(255, 128, 128))
        self.one_episode_button = Button(self.episode_display.x + self.episode_display.width  + self.PADDING,
                                        0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "1", lambda: self.train_episode(1), **self.BUTTON_KWARGS)
        self.ten_episode_button = Button(self.one_episode_button.x + self.one_episode_button.width  + self.PADDING,
                                        0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "10", lambda: self.train_episode(10), **self.BUTTON_KWARGS)
        self.hundred_episode_button = Button(self.ten_episode_button.x + self.ten_episode_button.width  + self.PADDING,
                                        0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "100", lambda: self.train_episode(100), **self.BUTTON_KWARGS)

        self.step_display = VariableDisplay(self.hundred_episode_button.x + self.hundred_episode_button.width + self.PADDING, 
                                            0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, "Step", 
                                            lambda: self.agent.step, background_color=(255, 128, 128))
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
        self.maze = Maze.load_from_numpy_array(maze_data)
        self.maze_model = MazeModel(self.maze, start_state = self.start_state)
        self.agent = DynaQAgent(self.maze_model, self.learning_rate, self.discount_factor, self.epsilon, self.max_steps_per_episode, self.planning_steps)
        self.agent_visualizer = AgentVisualizer(self.agent)

        self.maze_visualizer = MazeVisualizer(0, self.discount_factor_display.y + self.discount_factor_display.height + self.PADDING, 
                                        self.maze.walls, item_width, item_height, self.start_state, self.maze.goal_states, [self.agent_visualizer], 
                                        wall_color=(128, 128, 128), path_color=(255, 255, 255), player_color=(255, 0, 0))
        self.policy_visualizer = MazePolicyVisualizer(self.maze_visualizer.x, self.maze_visualizer.y, self.agent.generate_policy, 
                                            self.maze_visualizer.width, self.maze_visualizer.height, cell_color=None)
        self.table = TableVisualizer(self.maze_visualizer.x, self.maze_visualizer.y + self.maze_visualizer.height, item_width, item_height, 
                                    lambda: self.agent.state_reward_table, grid_color=(0, 0, 0))
        self.heatmap = HeatMapVisualizer(self.table.x, self.table.y, lambda: self.agent.state_reward_table, self.table.width, self.table.height)

        
        
        
        
        self.steps_per_episode_figure, self.reward_per_episode_figure = self.init_mpl_plots()
        plot_width_offset = 150
        self.steps_per_episode_plot = MatplotlibPlotDisplay(self.maze_visualizer.x + self.maze.walls.shape[1] * self.maze_visualizer.cell_size + self.PADDING, 
                                                            self.maze_visualizer.y, self.steps_per_episode_figure, item_width - plot_width_offset, item_height)
        self.reward_per_episode_plot = MatplotlibPlotDisplay(self.steps_per_episode_plot.x, self.steps_per_episode_plot.y + self.steps_per_episode_plot.height, 
                                                            self.reward_per_episode_figure, item_width - plot_width_offset, item_height)

        slider_width = 150
        self.pause_length_slider = Slider(self.steps_per_episode_plot.x + self.steps_per_episode_plot.width + (self.window.width - self.steps_per_episode_plot.x - self.steps_per_episode_plot.width) // 3 - slider_width // 2,
                                        self.maze_visualizer.y, slider_width, item_height * 2 - 50, "Pause [ms]", 0, 250, 10, 20, self.pause_slider_function,
                                        handle_color=(255, 128, 128), bar_color=(128, 64, 64), label_color=(255, 128, 128))
        self.plan_steps_slider = Slider(self.pause_length_slider.x + self.pause_length_slider.width + self.PADDING,
                                        self.pause_length_slider.y, slider_width, item_height * 2 - 50, "Plan steps", 0, 100, 10, 20, self.planning_steps_slider_function,
                                        handle_color=(255, 128, 128), bar_color=(128, 64, 64), label_color=(255, 128, 128))


        fps_display = FPSDisplay(self.window.width - self.BUTTON_WIDTH, 0, self.BUTTON_WIDTH, self.BUTTON_HEIGHT, self.window)

        for object_ in (self.episode_display, self.one_episode_button, self.ten_episode_button, self.hundred_episode_button,
                        self.step_display, self.one_step_button, self.ten_step_button, self.hundred_step_button, 
                        self.learning_rate_display, self.epsilon_display, self.discount_factor_display, self.pause_display, self.plan_steps_display,
                        self.policy_overlay_button, self.heatmap_overlay_button, self.execute_policy_button, self.reset_button,
                         

                        self.maze_visualizer, self.table, self.steps_per_episode_plot, self.reward_per_episode_plot, 
                        self.pause_length_slider, self.plan_steps_slider,

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