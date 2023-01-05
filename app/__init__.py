import pygame

from .game_objects import GameObject 

pygame.init()

class Window:
    def __init__(self, width, height, title, max_fps = 60):
        self.width = width
        self.height = height
        self.title = title
        self._max_fps = max_fps
        self.game_objects = []

        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption(self.title)
        self.clock = pygame.time.Clock()

    def add_game_object(self, game_object):
        if not isinstance(game_object, GameObject):
            raise TypeError("Game object must be of type GameObject")
        self.game_objects.append(game_object)
        # self.game_objects.sort(key=lambda game_object: game_object.layer)

    @property
    def fps(self):
        return self.clock.get_fps()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                # print(event)
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.resize(event.w, event.h)

            self.screen.fill((0, 0, 0))  # Clear the screen

            for game_object in self.game_objects:
                game_object.update()  # Update the game object
                game_object.draw(self.screen)  # Draw the game object to the screen
            self.clock.tick(self._max_fps)
            pygame.display.flip()  # Update the display

        pygame.quit()

    def resize(self, width, height):
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)


