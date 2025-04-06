import random

from DustParticle import DustParticle
from Obstacle import Obstacle
from Robot import Robot


class MapEnvironment:
    def __init__(self, width, height, num_obstacles=5, num_dust=10):
        self.width = width
        self.height = height
        self.obstacles = []
        self.dust_particles = []
        self.robot = None

        self.generate_obstacles(num_obstacles)
        self.generate_dust(num_dust)

    def generate_obstacles(self, num):
        for _ in range(num):
            # Define obstacle dimensions within a range suitable for the window.
            obs_width = random.uniform(30, 100)
            obs_height = random.uniform(30, 100)
            x = random.uniform(0, self.width - obs_width)
            y = random.uniform(0, self.height - obs_height)
            self.obstacles.append(Obstacle(x, y, obs_width, obs_height))

    def generate_dust(self, num):
        for _ in range(num):
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)
            self.dust_particles.append(DustParticle(x, y))

    def place_robot(self, x, y, theta):
        self.robot = Robot(x, y, theta)

    def draw(self, screen):
        # Draw obstacles.
        for obs in self.obstacles:
            obs.draw(screen)
        # Draw dust particles.
        for dust in self.dust_particles:
            dust.draw(screen)
        # Draw the robot.
        if self.robot:
            self.robot.draw(screen)