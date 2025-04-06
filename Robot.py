import numpy as np
import pygame
import math
from Sensor import Sensor


class Robot:
    def __init__(self, x, y, theta):
        self.x = x  # position x-coordinate
        self.y = y  # position y-coordinate
        self.v_left = 0
        self.v_right = 0
        self.theta = theta  # orientation in radians
        self.radius = 30

        # Create 12 sensors placed every 30 degrees (360°/12)
        self.sensors = [Sensor(np.deg2rad(angle), 200+self.radius) for angle in range(0, 360, 30)]


    def draw(self, screen):
        # draw circle with line at the front

        pygame.draw.circle(screen, (255, 0, 0), (self.x, self.y), self.radius, 0)

        line_x = self.x + self.radius * math.cos(self.x)
        line_y = self.y+self.radius*math.sin(self.y)
        pygame.draw.line(screen, (0, 0, 0), (self.x, self.y), (line_x, line_y), 3)
        # draw sensors with numbers


    def move(self, linear_velocity, angular_velocity, dt=0.1):
        """
        Updates the robot's pose using simple differential drive kinematics.
        """
        self.theta += angular_velocity * dt
        self.x += linear_velocity * np.cos(self.theta) * dt
        self.y += linear_velocity * np.sin(self.theta) * dt

    def get_pose(self):
        return (self.x, self.y, self.theta)
