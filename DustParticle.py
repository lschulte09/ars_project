import pygame

from Obstacle import YELLOW


class DustParticle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 3

    def draw(self, screen):
        pygame.draw.circle(screen, YELLOW, (int(self.x), int(self.y)), self.radius)