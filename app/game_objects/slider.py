import pygame

from .game_object import GameObject

class Slider(GameObject):
    """A class for creating a slider in pygame.

    This class creates a vertical slider with a draggable handle in pygame. The
    slider has a label that is displayed underneath it and the color of the
    slider bar and handle, as well as the label are adjustable. The slider
    works with numeric values within a specified range and has a callback that
    is executed with the new slider value whenever the value changes.

    :param x: The x-coordinate of the lower left corner of the slider.
    :type x: int
    :param y: The y-coordinate of the lower left corner of the slider.
    :type y: int
    :param width: The width of the slider in pixels.
    :type width: int
    :param height: The height of the slider in pixels.
    :type height: int
    :param label: The label to display underneath the slider.
    :type label: str
    :param handle_color: The color of the handle as a tuple of (R, G, B) values.
    :type handle_color: tuple
    :param bar_color: The color of the bar as a tuple of (R, G, B) values.
    :type bar_color: tuple
    :param label_color: The color of the label as a tuple of (R, G, B) values.
    :type label_color: tuple
    :param from_: The minimum value of the slider.
    :type from_: int or float
    :param to: The maximum value of the slider.
    :type to: int or float
    :param callback: The callback function to execute with the new value when the value changes.
    :type callback: function
    """

    def __init__(self, x, y, width, height, label, from_, to, bar_width, handle_radius, callback, font_size = 32,
                handle_color = (255, 255, 255), bar_color = (128, 128, 128), label_color = (255, 255, 255)):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.label = label
        self.from_ = from_
        self.to = to
        self.bar_width = bar_width
        self.handle_radius = handle_radius
        self.callback = callback
        self.handle_color = handle_color
        self.bar_color = bar_color
        self.label_color = label_color
        self.value = from_
        self.handle_x = x + self.width // 2
        self.handle_y = y + self.height
        self.handle_rect = pygame.Rect(self.handle_x - self.handle_radius, self.handle_y - self.handle_radius, self.handle_radius * 2, self.handle_radius * 2)
        self.is_dragging = False
        self.font = pygame.font.Font(None, font_size)
        self.label_surface = self.font.render(self.label, True, self.label_color)
        self.label_rect = self.label_surface.get_rect()
        self.label_rect.center = (x + width // 2, y + height + self.label_rect.height + 5)

    def update(self, events):
        """Updates the state of the slider.

        :param events: A list of pygame events.
        :type events: list
        """
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.handle_rect.collidepoint(event.pos):
                    self.is_dragging = True
            elif event.type == pygame.MOUSEBUTTONUP:
                self.is_dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if self.is_dragging:
                    mouse_y = event.pos[1]
                    if mouse_y < self.y:
                        self.handle_y = self.y
                    elif mouse_y > self.y + self.height:
                        self.handle_y = self.y + self.height
                    else:
                        self.handle_y = mouse_y
                    self.value = self.from_ + (self.to - self.from_) * (self.y + self.height - self.handle_y) / self.height
                    self.callback(self.value)
            self.handle_rect = pygame.Rect(self.handle_x - self.handle_radius, self.handle_y - self.handle_radius, self.handle_radius * 2, self.handle_radius * 2)
        
    def draw(self, screen):
        """Draws the slider on the screen.

        :param screen: The surface to draw on.
        :type screen: pygame.Surface
        """
        # Draw the bar
        pygame.draw.rect(screen, self.bar_color, (self.x + (self.width - self.bar_width) // 2, self.y, self.bar_width, self.height))
        # Draw the handle
        pygame.draw.circle(screen, self.handle_color, (self.handle_x, self.handle_y), self.handle_radius)
        # Draw the label
        screen.blit(self.label_surface, self.label_rect)