import pygame

from .game_object import GameObject
from ..enums import RenderLayer

class FPSDisplay(GameObject):
    def __init__(self, x, y, width, height, window, background_color=None, font_color=(255, 255, 255), font_size=20, layer=RenderLayer.UI):
        """A FPS display that can be rendered on the screen.

        The FPS display is rendered using the `x` and `y` coordinates, and has a `width` and `height` that determines the size of the display. It is colored using the `background_color` and has a font with the specified `font_color` and `font_size`. The FPS display is rendered on the `UI` layer by default, but this can be changed using the `layer` parameter.

        :param int x: The x coordinate of the top left corner of the FPS display.
        :param int y: The y coordinate of the top left corner of the FPS display.
        :param int width: The width of the FPS display, in pixels.
        :param int height: The height of the FPS display, in pixels.
        :param tuple background_color: The color of the background of the FPS display, as a tuple of three integers in the range 0-255.
        :param tuple font_color: The color of the font in the FPS display, as a tuple of three integers in the range 0-255.
        :param int font_size: The size of the font in the FPS display, in pixels.
        :param RenderLayer layer: The RenderLayer that the FPS display should be rendered on.
        """
        super().__init__(x, y, layer)
        self.width = width
        self.height = height
        self.window = window
        self.background_color = background_color
        self.font_color = font_color
        self.font_size = font_size
        self.font = pygame.font.Font(None, self.font_size)
        self.fps = 0
        
    
    def update(self, events):
        """Update the FPS display with the current FPS of the window.
        """
        self.fps = self.window.fps
    
    def draw(self, screen):
        # Draw the background
        if self.background_color is not None:
            pygame.draw.rect(screen, self.background_color, (self.x, self.y, self.width, self.height))
        # Render the FPS text
        text = self.font.render(f"FPS: {self.fps:.2f}", True, self.font_color)
        # Calculate the text position
        text_rect = text.get_rect()
        text_rect.center = (self.x + self.width // 2, self.y + self.height // 2)
        # Draw the text
        screen.blit(text, text_rect)
