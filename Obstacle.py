import pygame
from pygame.math import Vector2
import random
import math

# Define colors.
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Obstacle class: a rectangular barrier.
class Obstacle:
    def __init__(self, x, y, width, height):
        self.x = x  # Top-left x
        self.y = y  # Top-left y
        self.width = width
        self.height = height
        self.points = [Vector2(self.x, self.y), Vector2(self.x + self.width, self.y), Vector2(self.x + self.width, self.y + self.height), Vector2(self.x, self.y + self.height)]

    def draw(self, screen):
        pygame.draw.rect(screen, GRAY, pygame.Rect(self.x, self.y, self.width, self.height))

    def get_points(self):
        return self.points

    def get_edges(self):
        # Returns list of [point1, point2] edges
        return [[self.points[i], self.points[(i + 1) % len(self.points)]]
                for i in range(len(self.points))]
