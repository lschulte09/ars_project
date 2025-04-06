class Sensor:
    def __init__(self, relative_angle, max_range):
        self.relative_angle = relative_angle  # angle relative to robot's heading (radians)
        self.max_range = max_range

    def read_distance(self, robot, obstacles):
        """
        A placeholder method for sensor reading.
        In a full implementation, this would calculate the distance from the robot (in its local coordinate system)
        along the sensor's direction until an obstacle is detected.
        """
        # For now, return max_range indicating no obstacle detected.
        return self.max_range