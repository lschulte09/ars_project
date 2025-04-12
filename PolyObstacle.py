import pygame
from pygame.math import Vector2
import math
import random
from Obstacle import GRAY

class PolyObstacle:
    def __init__(self, points):
        self.points = points

    def get_edges(self):
        # Returns list of [point1, point2] edges
        return [[self.points[i], self.points[(i + 1) % len(self.points)]]
                for i in range(len(self.points))]

    def get_points(self):
        return self.points

    def draw(self, screen):
        pygame.draw.polygon(screen, GRAY, self.points)
