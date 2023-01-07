import pygame
import numpy as np

from .game_object import GameObject
from ..enums import RenderLayer


class MazePolicyVisualizer(GameObject):
    def __init__(self, x, y, policy_function, width, height, cell_color=(255, 255, 255), grid_color=(0, 0, 0), arrow_color=(0, 0, 0), policy_mask_function = None, layer=RenderLayer.GAME):
        """A maze policy visualizer that can be rendered on the screen.

        The maze policy visualizer is rendered using the `x` and `y` coordinates, and has a `width` and `height` that determines the size of each cell. It uses a two dimensional numpy array `policy` to determine which direction should be displayed in each cell, and colors the arrows using the `arrow_color` attribute. The cells are colored using the `cell_color` attribute. The maze policy visualizer is rendered on the `GAME` layer by default, but this can be changed using the `layer` parameter.

        :param int x: The x coordinate of the top left corner of the maze policy visualizer.
        :param int y: The y coordinate of the top left corner of the maze policy visualizer.
        :param function policy: A function that returns a two dimensional numpy array representing the policy, where 0 indicates an upwards arrow, 1 indicates a rightwards arrow, 2 indicates a downwards arrow, and 3 indicates a leftwards arrow.
        :param int width: The width of each cell in the maze policy visualizer, in pixels.
        :param int height: The height of each cell in the maze policy visualizer, in pixels.
        :param tuple cell_color: The color of the cells in the maze policy visualizer, as a tuple of three integers in the range 0-255.
        :param tuple arrow_color: The color of the arrows in the maze policy visualizer, as a tuple of three integers in the range 0-255.
        :param Layer layer: The layer that the maze policy visualizer should be rendered on.
        """
        super().__init__(x, y, layer)
        self.policy_function = policy_function
        self.width = width
        self.height = height
        self.cell_color = cell_color
        self.grid_color = grid_color
        self.arrow_color = arrow_color
        self.policy_mask_function = policy_mask_function # where the policy should be displayed
        self.rows, self.cols = self.policy_function().shape
        self.cell_size = min(self.width // self.cols, self.height // self.rows)

    def update(self, events):
        pass

    def draw(self, screen):
        # Loop through the rows and columns of the policy
        policy = self.policy_function()
        policy_mask = self.policy_mask_function() if self.policy_mask_function is not None else None
        for i, row in enumerate(policy):
            for j, cell in enumerate(row):
                # Calculate the x and y coordinates of the cell
                x = self.x + j * self.cell_size
                y = self.y + i * self.cell_size
                # Draw the cell
                if self.cell_color is not None:
                    pygame.draw.rect(screen, self.cell_color, (x, y, self.cell_size, self.cell_size))
                if policy_mask is not None and not policy_mask[i, j]:
                    continue
                arrow_x = x + self.cell_size / 2
                arrow_y = y + self.cell_size / 2
                # Draw the arrow
                if cell == 0:
                    pygame.draw.polygon(screen, self.arrow_color, [(arrow_x, arrow_y - self.cell_size / 4), (arrow_x - self.cell_size / 4, arrow_y), (arrow_x + self.cell_size / 4, arrow_y)])
                elif cell == 1:
                    pygame.draw.polygon(screen, self.arrow_color, [(arrow_x - self.cell_size / 4, arrow_y - self.cell_size / 4), (arrow_x - self.cell_size / 4, arrow_y + self.cell_size / 4), (arrow_x, arrow_y)])
                elif cell == 2:
                    pygame.draw.polygon(screen, self.arrow_color, [(arrow_x, arrow_y + self.cell_size / 4), (arrow_x - self.cell_size / 4, arrow_y), (arrow_x + self.cell_size / 4, arrow_y)])
                elif cell == 3:
                    pygame.draw.polygon(screen, self.arrow_color, [(arrow_x + self.cell_size / 4, arrow_y - self.cell_size / 4), (arrow_x + self.cell_size / 4, arrow_y + self.cell_size / 4), (arrow_x, arrow_y)])
        # Draw the grid lines
        if self.grid_color is not None:
            for i in range(self.rows + 1):
                pygame.draw.line(screen, self.grid_color, (self.x, self.y + i * self.cell_size), (self.x + self.cols * self.cell_size, self.y + i * self.cell_size), 2)
            for j in range(self.cols + 1):
                pygame.draw.line(screen, self.grid_color, (self.x + j * self.cell_size, self.y), (self.x + j * self.cell_size, self.y + self.rows * self.cell_size), 2)