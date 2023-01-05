import pygame
import numpy as np

from .game_object import GameObject

class TableVisualizer(GameObject):
    def __init__(self, x, y, width, height, data_function, font_size=32, font_color=(0, 0, 0), cell_color=(255, 255, 255), grid_color=(0, 0, 0), **kwargs):
        """A visualizer for a two-dimensional NumPy array as a table.

        The table is rendered using the `x` and `y` coordinates inherited from the `GameObject` base class, and the `width` and `height` properties specific to the `TableVisualizer` class. The table cells can display text and can be customized with a font size, font color, cell color, and grid color. The cell size is calculated from the `width` and `height` parameters.

        :param int x: The x coordinate of the top left corner of the table.
        :param int y: The y coordinate of the top left corner of the table.
        :param function data_function: A function that returns the table data as a 2D numpy array
        :param int width: The width of the table, in pixels.
        :param int height: The height of the table, in pixels.
        :param int font_size: The size of the font used to display text in the table cells.
        :param tuple font_color: The color of the text in the table cells, as a tuple of three integers in the range 0-255.
        :param tuple cell_color: The color of the table cells, as a tuple of three integers in the range 0-255.
        :param tuple grid_color: The color of the grid lines in the table, as a tuple of three integers in the range 0-255.
        :param kwargs: keyword arguments forwarded to the GameObject base class
        :type kwargs: dict
        """
        super().__init__(x, y, **kwargs)      
        self.width = width
        self.height = height
        self.data_function = data_function
        self.font_size = font_size
        self.font_color = font_color
        self.cell_color = cell_color
        self.grid_color = grid_color
        self.font = pygame.font.Font(None, self.font_size)
        self.rows, self.cols = self.data_function().shape
        self.cell_size = min(self.width // self.cols, self.height //  self.rows)

    def update(self, events):
        # No updates needed for the table visualizer
        pass

    def draw(self, screen):
        # Draw the table cells
        data = self.data_function()
        for i in range(self.rows):
            for j in range(self.cols):
                cell_rect = pygame.Rect(self.x + j * self.cell_size, self.y + i * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(screen, self.cell_color, cell_rect)
                cell_text = self.font.render(str(round(data[i, j], 2)), True, self.font_color)
                screen.blit(cell_text, (self.x + j * self.cell_size + (self.cell_size - cell_text.get_width()) // 2, self.y + i * self.cell_size + (self.cell_size - cell_text.get_height()) // 2))

        # Draw the grid lines
        for i in range(self.rows + 1):
            pygame.draw.line(screen, self.grid_color, (self.x, self.y + i * self.cell_size), (self.x + self.cols * self.cell_size, self.y + i * self.cell_size), 2)
        for j in range(self.cols + 1):
            pygame.draw.line(screen, self.grid_color, (self.x + j * self.cell_size, self.y), (self.x + j * self.cell_size, self.y + self.rows * self.cell_size), 2)


