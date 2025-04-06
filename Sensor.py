import math


class Sensor:
    def __init__(self, relative_angle, max_range):
        self.relative_angle = relative_angle  # Angle relative to the robot's heading (in radians)
        self.max_range = max_range

    def read_distance(self, robot, obstacles):
        """
        Placeholder for sensor reading.
        In a full implementation, this method would compute the distance from the robot along the sensor's ray
        until an obstacle is detected.
        """
        return self.max_range

    def get_end_point(self, robot):
        # Calculate the end point of the sensor ray.
        sensor_angle = robot.theta + self.relative_angle
        end_x = robot.x + self.max_range * math.cos(sensor_angle)
        end_y = robot.y + self.max_range * math.sin(sensor_angle)
        return (end_x, end_y)
