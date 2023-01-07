import pygame
import numpy as np

from .game_object import GameObject

class HeatMapVisualizer(GameObject):
    def __init__(self, x, y, data_function, width, height, high_color=(255, 0, 0), low_color=(0, 0, 255), threshold=0, **kwargs):
        """A heat map visualizer that can be rendered on the screen.

        The heat map visualizer is rendered using the `x` and `y` coordinates, and has a `width` and `height` that determines the size of each cell. It uses a `data` array to determine the color of each cell, with cells above the `threshold` parameter being colored using the `high_color` and cells below the `threshold` being colored using the `low_color`. The intensity of the color is determined by the value of the cell in comparison to the maximum value in the `data` array. The heat map visualizer is rendered on the `UI` layer by default, but this can be changed using the `layer` parameter.

        :param int x: The x coordinate of the top left corner of the heat map visualizer.
        :param int y: The y coordinate of the top left corner of the heat map visualizer.
        :param function data: A function that returns the data array to be visualized as a heat map.
        :param int width: The width of the heat map visualizer, in pixels.
        :param int height: The height of the heat map visualizer, in pixels.
        :param tuple high_color: The color to be used for cells above the `threshold`, as a tuple of three integers in the range 0-255.
        :param tuple low_color: The color to be used for cells below the `threshold`, as a tuple of three integers in the range 0-255.
        :param float threshold: The threshold value to determine which cells are colored using the `high_color` and which cells are colored using the `low_color`.
        :param Layer layer: The layer that the heat map visualizer should be rendered on.
        :param kwargs: keyword arguments forwarded to the GameObject base class
        :type kwargs: dict
        """
        super().__init__(x, y, **kwargs)
        self.data_function = data_function
        self.width = width
        self.height = height
        self.high_color = high_color
        self.low_color = low_color
        self.threshold = threshold
        # Calculate the cell size
        data = data_function()
        self.rows, self.cols = data.shape
        self.cell_size = min(self.width // self.cols, self.height // self.rows)
        self.cell_width = self.cell_size
        self.cell_height = self.cell_size

    def update(self, events):
        # No updates needed for the table visualizer
        pass

    def draw(self, screen):
        # Iterate over the data array and draw the cells
        data = self.data_function()
        low_colors, high_colors = self.calculate_colors(data)
        for i in range(self.rows):
            for j in range(self.cols):
                # Determine the color of the cell
                if data[i, j] >= self.threshold:
                    color = tuple(high_colors[i, j])
                else:
                    color = tuple(low_colors[i, j])
                # Draw the cell
                pygame.draw.rect(screen, color, (self.x + j * self.cell_width, self.y + i * self.cell_height, self.cell_width, self.cell_height))
    
    def calculate_colors(self, data):
        data_RGB= np.abs(np.stack((data.copy(), data.copy(), data.copy()), axis = 2))
        high_color_array = np.array(self.high_color)
        low_color_array = np.array(self.high_color)
        
        low_rgb_colors = data_RGB / data.min()
        high_rgb_colors = data_RGB / data.max()
        np.nan_to_num(low_rgb_colors, copy = False, posinf = 0)
        np.nan_to_num(high_rgb_colors, copy = False, posinf = 0)

        np.multiply(low_rgb_colors, low_color_array, out = low_rgb_colors)
        np.multiply(high_rgb_colors, high_color_array, out = high_rgb_colors)
        return low_rgb_colors, high_rgb_colors
