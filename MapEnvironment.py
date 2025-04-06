import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
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
            # Random dimensions for each obstacle
            obs_width = random.uniform(5, 20)
            obs_height = random.uniform(5, 20)
            # Ensure obstacles are within the map bounds
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

    def plot(self):
        plt.figure(figsize=(8, 8))
        ax = plt.gca()

        # Draw obstacles as rectangles
        for obs in self.obstacles:
            ax.add_patch(obs.get_patch())

        # Draw dust particles as yellow dots
        for dust in self.dust_particles:
            plt.plot(dust.x, dust.y, 'yo', markersize=5)

        # Draw the robot (if placed) as a blue circle with an arrow showing orientation.
        if self.robot:
            # Robot body as a circle
            robot_circle = Circle((self.robot.x, self.robot.y), 3, fc='blue', ec='black')
            ax.add_patch(robot_circle)
            # Draw an arrow indicating the heading direction
            dx = 5 * np.cos(self.robot.theta)
            dy = 5 * np.sin(self.robot.theta)
            plt.arrow(self.robot.x, self.robot.y, dx, dy, head_width=1.5, head_length=2, fc='blue', ec='blue')

        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        plt.title('Simulated Environment')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.grid(True)
        plt.show()