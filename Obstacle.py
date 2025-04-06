import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

class Obstacle:
    def __init__(self, x, y, width, height):
        self.x = x              # x-coordinate of the bottom-left corner
        self.y = y              # y-coordinate of the bottom-left corner
        self.width = width
        self.height = height

    def get_patch(self):
        # Return a matplotlib rectangle patch for visualization.
        return Rectangle((self.x, self.y), self.width, self.height, fc='gray', ec='black')
