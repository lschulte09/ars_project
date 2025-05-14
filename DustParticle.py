import pygame
from pygame.math import Vector2



class DustParticle:

    def __init__(self, x, y):
        self.pos = Vector2(x, y)
        self.radius = 2


    def draw(self, screen):
        pygame.draw.circle(screen, (255,165,0), self.pos, self.radius)
    pass