import pygame

from .game_object import GameObject

class ObstacleEnvironmentVisualizer(GameObject):
    def __init__(self, x, y, width, height, obstacles, actors, background_color=(255, 255, 255)):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.obstacles = obstacles
        self.actors = actors
        self.background_color = background_color

    def update(self):
        pass

    def draw(self, screen):
        pygame.draw.rect(screen, self.background_color, (self.x, self.y, self.width, self.height))
        for obstacle in self.obstacles:
            obstacle_coordinates = [(x + self.x, y + self.y) for x, y in obstacle.coordinates]
            pygame.draw.polygon(screen, obstacle.color, obstacle_coordinates)

        for actor in self.actors:
            actor_center = (actor.center[0] + self.x, actor.center[1] + self.y)
            actor_end = (actor_center[0] + actor.length[0], actor_center[1] + actor.length[1])
            pygame.draw.line(screen, actor.color, actor_center, actor_end, actor.rotation)

