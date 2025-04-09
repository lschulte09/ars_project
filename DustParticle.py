import pygame

from Obstacle import Obstacle, RED


class DustParticle(Obstacle):

    def draw(self, screen):
        pygame.draw.rect(screen, RED, pygame.Rect(self.x, self.y, self.width, self.height))
    pass