import pygame

from .game_object import GameObject

import pygame

class Button(GameObject):
    def __init__(self, x, y, width, height, text, callback, background_color=(0, 0, 0), font_color=(255, 255, 255), font_size = 32, **kwargs):
        """A button that can be clicked to trigger a callback function.

        The button is rendered using the `x` and `y` coordinates inherited from the `GameObject` base class, and the `width` and `height` properties specific to the `Button` class. The callback function is triggered when the button is clicked and released. The button can also display text and can be customized with a background color and a font color.

        :param int x: The x coordinate of the button.
        :param int y: The y coordinate of the button.
        :param int width: The width of the button.
        :param int height: The height of the button.
        :param str text: The text to display on the button.
        :param function callback: The callback function to be triggered when the button is clicked and released.
        :param tuple background_color: The background color of the button, as a tuple of three integers in the range 0-255.
        :param tuple font_color: The color of the text on the button, as a tuple of three integers in the range 0-255.
        :param kwargs: keyword arguments forwarded to the GameObject base class
        :type kwargs: dict
        """
        super().__init__(x, y, **kwargs)
        self.width = width
        self.height = height
        self.text = text
        self.callback = callback
        self.background_color = background_color
        self.font_color = font_color
        self.font = pygame.font.Font(None, font_size)
        self.image = self.font.render(self.text, True, self.font_color)
        self.clicked = False

    def update(self, events):
        # Check for mouse clicks on the button
        mouse_pos = pygame.mouse.get_pos()
        if self.x < mouse_pos[0] < self.x + self.width and self.y < mouse_pos[1] < self.y + self.height:
            if pygame.mouse.get_pressed()[0]:
                self.clicked = True
            elif self.clicked:
                self.callback()
                self.clicked = False

    def draw(self, screen):
        # Draw the button to the screen
        pygame.draw.rect(screen, self.background_color, (self.x, self.y, self.width, self.height))
        screen.blit(self.image, (self.x + (self.width - self.image.get_width()) // 2, self.y + (self.height - self.image.get_height()) // 2))
