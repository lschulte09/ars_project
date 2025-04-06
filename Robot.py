import numpy as np

from Sensor import Sensor


class Robot:
    def __init__(self, x, y, theta):
        self.x = x  # position x-coordinate
        self.y = y  # position y-coordinate
        self.theta = theta  # orientation in radians

        # Create 12 sensors placed every 30 degrees (360°/12)
        self.sensors = [Sensor(np.deg2rad(angle), 100) for angle in range(0, 360, 30)]

    def move(self, linear_velocity, angular_velocity, dt=0.1):
        """
        Updates the robot's pose using simple differential drive kinematics.
        """
        self.theta += angular_velocity * dt
        self.x += linear_velocity * np.cos(self.theta) * dt
        self.y += linear_velocity * np.sin(self.theta) * dt

    def get_pose(self):
        return (self.x, self.y, self.theta)
