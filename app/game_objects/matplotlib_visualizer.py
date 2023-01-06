import matplotlib.pyplot as plt
import pygame
# import threading

from .game_object import GameObject
from ..enums import RenderLayer

import matplotlib
def change_backend(backend = "agg"):
    matplotlib.use(backend)

change_backend()

class MatplotlibPlotDisplay(GameObject):
    def __init__(self, x, y, figure, width, height, layer=RenderLayer.GAME):
        """A matplotlib plot display that can be rendered on the screen.

        The matplotlib plot display is rendered using the `x` and `y` coordinates, and has a `width` and `height` that determines the size of the display. It uses a `figure` object from matplotlib to render the plot. The matplotlib plot display is rendered on the `UI` layer by default, but this can be changed using the `layer` parameter.

        :param int x: The x coordinate of the top left corner of the matplotlib plot display.
        :param int y: The y coordinate of the top left corner of the matplotlib plot display.
        :param matplotlib.figure.Figure figure: The matplotlib figure object to be displayed.
        :param int width: The width of the matplotlib plot display, in pixels.
        :param int height: The height of the matplotlib plot display, in pixels.
        :param Layer layer: The layer that the matplotlib plot display should be rendered on.
        """
        super().__init__(x, y, layer)
        self.figure = figure
        self.width = width
        self.height = height
        # Create a canvas to draw the figure on
        self.update_image()
        # self.update_semaphore = threading.Semaphore(1)
        # self.worker_thread = threading.Thread(target=self.update_image, daemon=True)
        # self.worker_thread.start()
        self.update_flag = False

    def update_next_frame(self):
        self.update_flag = True
    
    def update_image(self):
        self.figure.canvas.draw()          
        self.image = self.figure.canvas.tostring_rgb()
        self.image_size = self.figure.canvas.get_width_height()
        self.image = pygame.image.fromstring(self.image, self.image_size, "RGB")
        self.image = pygame.transform.scale(self.image, (self.width, self.height))

    def update(self, events):
        """Update the matplotlib plot display with the latest data from the figure.

        :param pygame.display window: The window to update the plot display for.
        """
        if self.update_flag:
            self.update_image()
            self.update_flag = False
    
    def draw(self, screen):
        # Draw the image
        screen.blit(self.image, (self.x, self.y))
