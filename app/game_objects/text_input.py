import pygame

from .game_object import GameObject

class TextInput(GameObject):
    def __init__(self, x, y, width, height, on_text_changed=None, on_enter=None, background_color=(255, 255, 255), font_color=(0, 0, 0), cursor_color=(0, 0, 0), cursor_width=2, **kwargs):
        """A text input field that can be rendered on the screen.

        The text input field is rendered using the `x` and `y` coordinates, and has a `width` and `height` that determines its size. It has a `background_color` and a `font_color` that can be customized, as well as a `cursor_color` and a `cursor_width` that determines the appearance of the cursor indicator. The text input field is rendered on the `UI` layer by default, but this can be changed using the `layer` parameter.

        :param int x: The x coordinate of the top left corner of the text input field.
        :param int y: The y coordinate of the top left corner of the text input field.
        :param int width: The width of the text input field, in pixels.
        :param int height: The height of the text input field, in pixels.
        :param tuple background_color: The color of the background of the text input field, as a tuple of three integers in the range 0-255.
        :param tuple font_color: The color of the text in the text input field, as a tuple of three integers in the range 0-255.
        :param tuple cursor_color: The color of the cursor indicator in the text input field, as a tuple of three integers in the range 0-255.
        :param int cursor_width: The width of the cursor indicator, in pixels.
        :param kwargs: keyword arguments forwarded to the GameObject base class
        :type kwargs: dict
        """
        super().__init__(x, y, **kwargs)
        self.width = width
        self.height = height
        self.background_color = background_color
        self.font_color = font_color
        self.cursor_color = cursor_color
        self.cursor_width = cursor_width
        self.font = pygame.font.Font(None, self.height)
        self.text = ''
        self.active = False
        self.cursor_visible = False
        self.cursor_time = 0
        self.on_text_changed = on_text_changed
        self.on_enter = on_enter
        self.last_text = ''

    def update(self, events):
        # Update the cursor time
        self.cursor_time += 1
        if self.cursor_time >= 30:
            self.cursor_time = 0
            self.cursor_visible = not self.cursor_visible
        # Check if the mouse was released inside the text input field
        if pygame.mouse.get_pressed()[0] == 0:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if (self.x <= mouse_x <= self.x + self.width) and (self.y <= mouse_y <= self.y + self.height):
                self.active = True
            else:
                self.active = False
        # Process keyboard events
        if self.active:
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_BACKSPACE:
                        self.text = self.text[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        self.active = False
                    elif event.key == pygame.K_RETURN:
                        if self.on_enter:
                            self.on_enter()
                    elif event.unicode.isprintable():
                        self.text += event.unicode
            # Execute the on_text_changed callback if the text has changed
            if self.text != self.last_text and self.on_text_changed:
                self.on_text_changed(self.text)
                self.last_text = self.text

    def draw(self, screen):
        # Draw the background
        pygame.draw.rect(screen, self.background_color, (self.x, self.y, self.width, self.height))
        # Draw the text
        text_surface = self.font.render(self.text, True, self.font_color)
        text_width, text_height = self.font.size(self.text)
        text_x = self.x
        text_y = self.y
        # Adjust the text position if it would be rendered outside the bounds of the text input field
        if text_x + text_width > self.x + self.width:
            text_x = self.x + self.width - text_width
        if text_y + text_height > self.y + self.height:
            text_y = self.y + self.height - text_height
        screen.blit(text_surface, (text_x, text_y))
        # Draw the cursor
        if self.active and self.cursor_visible:
            cursor_x = text_x + text_width
            cursor_y = text_y
            cursor_height = text_height
            # Adjust the cursor position if it would be rendered outside the bounds of the text input field
            if cursor_x + self.cursor_width > self.x + self.width:
                cursor_x = self.x + self.width - self.cursor_width
            if cursor_y + cursor_height > self.y + self.height:
                cursor_y = self.y + self.height - cursor_height
            pygame.draw.rect(screen, self.cursor_color, (cursor_x, cursor_y, self.cursor_width, cursor_height))
