import pygame
from pygame.math import Vector2
from Obstacle import Obstacle, WHITE

class Boundary:
    def __init__(self, screen_width, screen_height, offset):
        self.points = [Vector2(offset, offset),
                       Vector2(screen_width - offset, offset),
                       Vector2(screen_width - offset, screen_height - offset),
                       Vector2(offset, screen_height - offset)]
        self.x = self.points[0].x
        self.y = self.points[0].y
        self.width = screen_width - 2*offset
        self.height = screen_height - 2*offset

    def draw(self, screen):
        pygame.draw.polygon(screen, WHITE, self.points)

    def get_edges(self):
        return [[self.points[i], self.points[(i + 1) % len(self.points)]]
                for i in range(len(self.points))]

    def get_points(self):
        return self.points
