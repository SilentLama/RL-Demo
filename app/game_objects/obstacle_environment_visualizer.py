import pygame
import numpy as np
from .game_object import GameObject

class ObstacleEnvironmentVisualizer(GameObject):
    def __init__(self, x, y, width, height, obstacle_environment, actors, background_color=(255, 255, 255)):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.obstacle_environment = obstacle_environment
        self.actors = actors
        self.background_color = background_color

        self.width_ratio = width / self.obstacle_environment.cols
        self.height_ratio = height / self.obstacle_environment.rows

        self.world_actor_lengths_x = [actor.length * self.width_ratio for actor in actors]
        self.world_actor_lengths_y = [actor.length * self.height_ratio for actor in actors]

    def update(self, events):
        pass

    def draw(self, screen):
        pygame.draw.rect(screen, self.background_color, (self.x, self.y, self.width, self.height))
        for obstacle in self.obstacle_environment.obstacles:
            obstacle_x, obstacle_y = obstacle.coordinates
            top_left = (self.x + obstacle_x * self.width_ratio, self.y + obstacle_y * self.height_ratio)
            top_right = top_left[0] + obstacle.width * self.width_ratio, top_left[1]
            bottom_right = top_right[0], top_right[1] + obstacle.height * self.height_ratio
            bottom_left = top_left[0], top_left[1] + obstacle.height * self.height_ratio
            pygame.draw.polygon(screen, obstacle.color, [top_left, top_right, bottom_right, bottom_left])

        for actor, length_x, length_y in zip(self.actors, self.world_actor_lengths_x, self.world_actor_lengths_y):
            # actor_center = (actor.coordinates[0] + self.x, actor.coordinates[1] + self.y)
            actor_x, actor_y = actor.coordinates # this is the center
            actor_start = (self.x + actor_x * self.width_ratio + (length_x / 2) * np.cos(actor.rotation), 
                            self.y + actor_y * self.height_ratio + (length_y / 2) * np.sin(actor.rotation))
            actor_end = (self.x + (actor_x * self.width_ratio) - length_x / 2 * np.cos(actor.rotation), 
                        self.y + (actor_y * self.height_ratio) - length_y * np.sin(actor.rotation))
            pygame.draw.line(screen, actor.color, actor_start, actor_end, actor.line_thickness)

