import math

import pygame

from Obstacle import BLUE, BLACK, GREEN
from Sensor import Sensor


class Robot:
    def __init__(self, x, y, theta):
        self.x = x  # x-coordinate
        self.y = y  # y-coordinate
        self.theta = theta  # orientation in radians
        self.radius = 10
        # Create 12 sensors evenly distributed (every 30°).
        self.sensors = [Sensor(math.radians(angle), 100) for angle in range(0, 360, 30)]
        self.linear_velocity = 0  # Units per second.
        self.angular_velocity = 0  # Radians per second.

    def update(self, dt):
        # Update the robot's orientation and position using differential drive kinematics.
        self.theta += self.angular_velocity * dt
        self.x += self.linear_velocity * math.cos(self.theta) * dt
        self.y += self.linear_velocity * math.sin(self.theta) * dt

    def draw(self, screen):
        # Draw the robot as a circle.
        pygame.draw.circle(screen, BLUE, (int(self.x), int(self.y)), self.radius)
        # Draw a line to indicate the robot's heading.
        end_x = self.x + self.radius * math.cos(self.theta)
        end_y = self.y + self.radius * math.sin(self.theta)
        pygame.draw.line(screen, BLACK, (int(self.x), int(self.y)), (int(end_x), int(end_y)), 2)
        # Optionally, draw sensor lines (in green).
        for sensor in self.sensors:
            end_point = sensor.get_end_point(self)
            pygame.draw.line(screen, GREEN, (int(self.x), int(self.y)), (int(end_point[0]), int(end_point[1])), 1)
