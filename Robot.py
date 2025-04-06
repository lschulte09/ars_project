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
        self.v = (self.v_right+self.v_left)/2
        self.theta = theta  # orientation in radians
        self.radius = 30

        # Create 12 sensors placed every 30 degrees (360°/12)
        self.sensors = [Sensor(np.deg2rad(angle), 200+self.radius) for angle in range(0, 360, 30)]


    def draw(self, screen):
        # draw circle with line at the front

        pygame.draw.circle(screen, (255, 0, 0), (self.x, self.y), self.radius, 0)

        line_x = self.x + self.radius * math.cos(self.theta)
        line_y = self.y + self.radius * math.sin(self.theta)
        pygame.draw.line(screen, (0, 0, 0), (self.x, self.y), (line_x, line_y), 3)

        left_angle = self.theta + math.pi / 2
        right_angle = self.theta - math.pi / 2

        text_dist = self.radius/2

        left_x = self.x + text_dist * math.cos(left_angle)
        left_y = self.y - text_dist * math.sin(left_angle)

        right_x = self.x + text_dist * math.cos(right_angle)
        right_y = self.y - text_dist * math.sin(right_angle)

        font = pygame.font.SysFont('Arial', 12)
        v_left_text = font.render(f"{self.v_left:.2f}", True, (0, 0, 0))
        v_right_text = font.render(f"{self.v_right:.2f}", True, (0, 0, 0))

        screen.blit(v_left_text, (left_x - v_left_text.get_width() / 2, left_y - v_left_text.get_height() / 2))
        screen.blit(v_right_text, (right_x - v_right_text.get_width() / 2, right_y - v_right_text.get_height() / 2))

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
