import math
import pygame

class Sensor:
    def __init__(self, relative_angle, max_range):
        self.relative_angle = relative_angle  # Angle relative to the robot's heading (in radians)
        self.max_range = max_range
        self.current_distance = max_range  # Current reading, initialized to max_range

    def read_distance(self, robot, obstacles):
        """
        Calculate the distance from the robot to the nearest obstacle in the sensor's direction.
        Returns the distance to the obstacle, or max_range if no obstacle is detected.
        """
        # Calculate the absolute angle of the sensor in world coordinates
        sensor_angle = robot.theta + self.relative_angle
        
        # Starting point (robot's position)
        start_x, start_y = robot.x, robot.y
        
        # Initialize with max range
        min_distance = self.max_range
        
        # Check for collisions with rectangular obstacles
        for obstacle in obstacles:
            # Find intersection with rectangular obstacle
            distance = self._check_rect_intersection(start_x, start_y, sensor_angle, obstacle)
            if distance < min_distance:
                min_distance = distance
        
        # Update the current distance reading
        self.current_distance = min_distance
        return min_distance

    def _check_rect_intersection(self, x, y, angle, obstacle):
        """
        Check if the sensor ray intersects with a rectangular obstacle.
        Returns the distance to the intersection point, or max_range if no intersection.
        """
        # Direction vector
        dx = math.cos(angle)
        dy = math.sin(angle)
        
        # Define rectangle edges
        left = obstacle.x
        right = obstacle.x + obstacle.width
        top = obstacle.y
        bottom = obstacle.y + obstacle.height
        
        # Time to hit each edge
        t_min = float('inf')
        
        # Check intersection with vertical edges (left and right)
        if dx != 0:
            t1 = (left - x) / dx
            t2 = (right - x) / dx
            
            if t1 > 0 and t1 < t_min and y + t1 * dy >= top and y + t1 * dy <= bottom:
                t_min = t1
            if t2 > 0 and t2 < t_min and y + t2 * dy >= top and y + t2 * dy <= bottom:
                t_min = t2
        
        # Check intersection with horizontal edges (top and bottom)
        if dy != 0:
            t3 = (top - y) / dy
            t4 = (bottom - y) / dy
            
            if t3 > 0 and t3 < t_min and x + t3 * dx >= left and x + t3 * dx <= right:
                t_min = t3
            if t4 > 0 and t4 < t_min and x + t4 * dx >= left and x + t4 * dx <= right:
                t_min = t4
        
        # If we found an intersection, calculate the distance
        if t_min != float('inf'):
            distance = t_min
            if distance < self.max_range:
                return distance
        
        return self.max_range

    def get_end_point(self, robot):
        """
        Calculate the end point of the sensor ray based on current reading.
        """
        sensor_angle = robot.theta + self.relative_angle
        end_x = robot.x + self.current_distance * math.cos(sensor_angle)
        end_y = robot.y + self.current_distance * math.sin(sensor_angle)
        return (end_x, end_y)
    
    def draw(self, screen, robot, color=(0, 255, 0)):
        """
        Draw the sensor ray on the screen.
        """
        sensor_angle = robot.theta + self.relative_angle
        start_x, start_y = robot.x, robot.y
        end_x = start_x + self.current_distance * math.cos(sensor_angle)
        end_y = start_y + self.current_distance * math.sin(sensor_angle)
        
        pygame.draw.line(screen, color, (int(start_x), int(start_y)), (int(end_x), int(end_y)), 1)
        # Draw a small circle at the end point if it's not at max range (indicating detection)
        if self.current_distance < self.max_range:
            pygame.draw.circle(screen, (255, 0, 0), (int(end_x), int(end_y)), 3)