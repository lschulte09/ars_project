import numpy as np
import pygame
import math
from Sensor import Sensor

class Robot:
    def __init__(self, x, y, theta):
        self.x = x  # position x-coordinate
        self.y = y  # position y-coordinate
        self.v_left = 0.0  # left wheel velocity
        self.v_right = 0.0  # right wheel velocity
        self.radius = 30  # robot radius
        self.wheel_radius = 5.0  # wheel radius
        self.wheel_distance = self.radius*2  # distance between wheels
        self.v = (self.v_right+self.v_left)/2
        self.theta = theta  # orientation in radians
        self.max_speed = 20
        
        # Create 12 sensors placed every 30 degrees (360°/12)
        self.sensors = [Sensor(np.deg2rad(angle), 200) for angle in range(0, 360, 30)]
        
        # Collision flag
        self.collision = False
        
        # Store last valid position in case of collision
        self.last_valid_x = x
        self.last_valid_y = y
        self.last_valid_theta = theta

    def update_sensors(self, obstacles):
        """
        Update all sensor readings.
        """
        for sensor in self.sensors:
            sensor.read_distance(self, obstacles)
            
    def check_collision(self, obstacles):
        """
        Check if the robot is colliding with any obstacle.
        """
        # Check if any sensor detects an obstacle too close
        collision_threshold = self.radius + 1  # Add small buffer
        
        for sensor in self.sensors:
            if sensor.current_distance < collision_threshold:
                return True
                
        # Direct collision detection with obstacles
        for obstacle in obstacles:
            # Simple rectangular-circular collision detection
            # Find closest point on rectangle to circle center
            closest_x = max(obstacle.x, min(self.x, obstacle.x + obstacle.width))
            closest_y = max(obstacle.y, min(self.y, obstacle.y + obstacle.height))
            
            # Calculate distance between closest point and circle center
            distance = math.sqrt((self.x - closest_x)**2 + (self.y - closest_y)**2)
            
            # If distance is less than robot radius, we have a collision
            if distance < self.radius:
                return True
                
        return False
    
    def set_wheel_velocities(self, v_left, v_right):
        """
        Set the velocities of the wheels.
        """
        self.v_left = v_left
        self.v_right = v_right
    
    def calculate_velocities(self):
        """
        Calculate linear and angular velocities from wheel velocities.
        """
        linear_velocity = (self.v_right + self.v_left) / 2.0
        angular_velocity = (self.v_right - self.v_left) / self.wheel_distance
        return linear_velocity, angular_velocity

    def move(self, dt=0.1, obstacles=None):
        """
        Updates the robot's pose using differential drive kinematics.
        Handles collision detection if obstacles are provided.
        """
        # Save current position as the last valid position
        self.last_valid_x = self.x
        self.last_valid_y = self.y
        self.last_valid_theta = self.theta
        
        # Calculate velocities from wheel velocities
        linear_velocity, angular_velocity = self.calculate_velocities()
        
        # Update position and orientation
        self.theta += angular_velocity * dt
        self.x += linear_velocity * np.cos(self.theta) * dt
        self.y += linear_velocity * np.sin(self.theta) * dt
        
        # Normalize angle to keep it within [0, 2π]
        self.theta = self.theta % (2 * math.pi)
        
        # Check for collisions if obstacles are provided
        # need to implement sliding, use vector2 maybe
        if obstacles:
            if self.check_collision(obstacles):
                # Collision detected, revert to last valid position
                self.x = self.last_valid_x
                self.y = self.last_valid_y
                self.theta = self.last_valid_theta
                self.collision = True
                return False  # Movement failed due to collision
            else:
                self.collision = False
                
        return True  # Movement successful

    def get_pose(self):
        return (self.x, self.y, self.theta)

    def draw(self, screen):
        # Draw the robot body
        pygame.draw.circle(screen, (255, 0, 0), (int(self.x), int(self.y)), self.radius, 0)
        
        # Draw a line indicating the robot's orientation
        line_x = self.x + self.radius * math.cos(self.theta)
        line_y = self.y + self.radius * math.sin(self.theta)
        pygame.draw.line(screen, (0, 0, 0), (int(self.x), int(self.y)), (int(line_x), int(line_y)), 3)
        
        # Draw sensors
        for i, sensor in enumerate(self.sensors):
            sensor.draw(screen, self)
            
            # Draw sensor number at the end point
            if sensor.current_distance > self.radius:  # Only if sensor ray extends beyond robot body
                # Calculate position for the sensor number
                sensor_angle = self.theta + sensor.relative_angle
                text_x = self.x + (self.radius + 15) * math.cos(sensor_angle)
                text_y = self.y + (self.radius + 15) * math.sin(sensor_angle)
                
                # Render the sensor number
                font = pygame.font.SysFont(None, 20)
                text = font.render(str(int(sensor.current_distance)), True, (0, 0, 255))
                screen.blit(text, (int(text_x), int(text_y)))