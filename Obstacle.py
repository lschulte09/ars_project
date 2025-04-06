import pygame
import random
import math

# Define colors.
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)

# Obstacle class: a rectangular barrier.
class Obstacle:
    def __init__(self, x, y, width, height):
        self.x = x  # Top-left x
        self.y = y  # Top-left y
        self.width = width
        self.height = height

    def draw(self, screen):
        pygame.draw.rect(screen, GRAY, pygame.Rect(self.x, self.y, self.width, self.height))
