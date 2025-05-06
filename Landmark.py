import pygame
from pygame.math import Vector2
from Obstacle import BLUE
from Robot import points_distance

class Landmark:
    def __init__(self, x, y, signature):
        self.x = x
        self.y = y
        self.pos = Vector2(self.x, self.y)
        self.sign = signature
        self.id = signature

    def get_pos(self):
        return self.pos

    def draw(self, screen, robot):
        pygame.draw.circle(screen, BLUE, (int(self.x), int(self.y)), 5)

        if points_distance(self.pos, robot.pos) < robot.lm_range:
            pygame.draw.line(screen, (200, 50, 0), self.pos, robot.pos, width = 2)