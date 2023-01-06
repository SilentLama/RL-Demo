import pygame
import numpy as np

from .game_object import GameObject

class MazeVisualizer(GameObject):
    def __init__(self, x, y, data, width, height, start_coordinates, goal_coordinates, players, wall_color=(0, 0, 0), path_color=(255, 255, 255), 
                player_color=(255, 0, 0), start_color = (0, 255, 0), goal_color = (0, 255, 255), **kwargs):
        """A visualizer for a boolean NumPy array as a maze.

        The maze is rendered using the `x` and `y` coordinates inherited from the `GameObject` base class, and the `width` and `height` properties specific to the `MazeVisualizer` class. The maze cells can be customized with colors for the walls and paths, and the player position can be visualized in a different color. The cell size is calculated from the `width` and `height` parameters.

        :param int x: The x coordinate of the top left corner of the maze.
        :param int y: The y coordinate of the top left corner of the maze.
        :param np.ndarray data: The boolean NumPy array to visualize as a maze.
        :param int width: The width of the maze, in pixels.
        :param int height: The height of the maze, in pixels.
        :param tuple start_coordinates:
        :param tuple goal_coordinates:
        :param list players: A list of AgentVisualizers that implement `coordinates` and `color` .The coordinates of the player in the maze, as a tuple of two integers.
        :param tuple wall_color: The color of the walls in the maze, as a tuple of three integers in the range 0-255.
        :param tuple path_color: The color of the paths in the maze, as a tuple of three integers in the range 0-255.
        :param tuple player_color: The color of the player position in the maze, as a tuple of three integers in the range 0-255.
        :param kwargs: keyword arguments forwarded to the GameObject base class
        :type kwargs: dict
        """
        super().__init__(x, y, **kwargs)
        self.data = data
        self.width = width
        self.height = height
        self.wall_color = wall_color
        self.path_color = path_color
        self.player_color = player_color
        self.players = players
        self.start_color = start_color
        self.start_coordinates = start_coordinates
        self.goal_color = goal_color
        self.goal_coordinates = goal_coordinates
        self.cell_size = min(self.width // self.data.shape[1], self.height // self.data.shape[0])
        self.mouse_down = False

    def update(self, events):
        # Check if the mouse button is released
        mouse_up = pygame.mouse.get_pressed()[0] == 0
        if self.mouse_down and mouse_up:
            # Mouse button was released, toggle the value of the cell under the mouse
            mouse_x, mouse_y = pygame.mouse.get_pos()
            i, j = (mouse_y - self.y) // self.cell_size, (mouse_x - self.x) // self.cell_size
            if i >= 0 and i < self.data.shape[0] and j >= 0 and j < self.data.shape[1]:
                self.data[i, j] = not self.data[i, j]
        self.mouse_down = pygame.mouse.get_pressed()[0] == 1

    def draw(self, screen):
        # Draw the maze cells
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                cell_rect = pygame.Rect(self.x + j * self.cell_size, self.y + i * self.cell_size, self.cell_size, self.cell_size)
                if self.data[i, j]:
                    pygame.draw.rect(screen, self.path_color, cell_rect)
                else:
                    pygame.draw.rect(screen, self.wall_color, cell_rect)
        # Draw the start
        start_rect = pygame.Rect(self.x + self.start_coordinates[1] * self.cell_size, self.y + self.start_coordinates[0] * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(screen, self.start_color, start_rect)
        # Draw the goals
        for i, j in self.goal_coordinates:
            goal_rect = pygame.Rect(self.x + j * self.cell_size, self.y + i * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(screen, self.goal_color, goal_rect)
        # Draw the player position
        for player in self.players:
            row, col = player.coordinates
            player_rect = pygame.Rect(self.x + col * self.cell_size, self.y + row * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(screen, player.color, player_rect)

