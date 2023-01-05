import pygame

from .game_object import GameObject

class VariableDisplay(GameObject):
    """
    This class uses the observer pattern to observe an object and display
    its state variable as text in pygame. The text is displayed behind a label
    and the display can be customized with various colors and font sizes.

    :param width: The width of the display in pixels.
    :type width: int
    :param height: The height of the display in pixels.
    :type height: int
    :param label: The label to display in front of the variable value.
    :type label: str
    :param get_value: A function that retrieves the desired variable
    :type get_value: function
    :param text_color: The color of the text as a tuple of (R, G, B) values.
    :type text_color: tuple
    :param background_color: The color of the background as a tuple of (R, G, B) values.
    :type background_color: tuple
    :param border_color: The color of the border as a tuple of (R, G, B) values.
    :type border_color: tuple
    :param font_size: The size of the font in pixels.
    :type font_size: int
    """

    def __init__(self, x, y, width, height, label, get_value, text_color = (0, 0, 0), background_color = (255, 255, 255), 
                border_color = (0, 0, 0), font_size = 32):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.label = label
        self.get_value = get_value
        self.text_color = text_color
        self.background_color = background_color
        self.border_color = border_color
        self.font_size = font_size
        

    def update(self, events):
        pass

    def draw(self, screen):
        """Draws the display on the screen.

        :param screen: The surface to draw on.
        :type screen: pygame.Surface
        """
        # Draw the border
        pygame.draw.rect(screen, self.border_color, (self.x, self.y, self.width, self.height), 1)
        # Draw the background
        pygame.draw.rect(screen, self.background_color, (self.x + 1, self.y + 1, self.width - 2, self.height - 2))
        # Render the text
        font = pygame.font.Font(None, self.font_size)
        text = font.render(f"{self.label}: {self.get_value()}", True, self.text_color)
        # Calculate the position of the text
        text_rect = text.get_rect()
        text_rect.center = (self.x + self.width // 2, self.y + self.height // 2)
        # Draw the text on the screen
        screen.blit(text, text_rect)
