import numpy as np
import pygame
import math

class OccupancyGrid:
    def __init__(self, width, height, resolution=10):
        """
        Initialize an occupancy grid map.
        
        Args:
            width (int): Width of the environment in pixels
            height (int): Height of the environment in pixels
            resolution (int): Grid resolution in pixels per cell (default: 10)
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        
        # Calculate grid dimensions
        self.grid_width = int(math.ceil(width / resolution))
        self.grid_height = int(math.ceil(height / resolution))
        
        # Initialize grid with 0.5 (unknown)
        # Values close to 0 mean free space, values close to 1 mean occupied
        self.grid = np.ones((self.grid_width, self.grid_height)) * 0.5
        
        # Log odds representation for more efficient updates
        self.log_odds = np.zeros((self.grid_width, self.grid_height))
        
        # Parameters for the update
        self.p_occ = 0.65  # Probability that cell is occupied if sensor hit
        self.p_free = 0.35  # Probability that cell is occupied if sensor missed
        
        # Precompute log odds values for efficiency
        self.l_occ = math.log(self.p_occ / (1 - self.p_occ))
        self.l_free = math.log(self.p_free / (1 - self.p_free))
        
        # For visualization
        # Create a surface to draw the occupancy grid
        self.grid_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        
    def world_to_grid(self, x, y):
        """Convert world coordinates to grid cell indices."""
        grid_x = int(x / self.resolution)
        grid_y = int(y / self.resolution)
        
        # Clamp values to grid bounds
        grid_x = max(0, min(grid_x, self.grid_width - 1))
        grid_y = max(0, min(grid_y, self.grid_height - 1))
        
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x, grid_y):
        """Convert grid cell indices to world coordinates (center of cell)."""
        world_x = (grid_x + 0.5) * self.resolution
        world_y = (grid_y + 0.5) * self.resolution
        return world_x, world_y
    
    def update_cell(self, grid_x, grid_y, is_occupied):
        """
        Update a single cell's occupancy probability using log-odds.
        
        Args:
            grid_x, grid_y: Grid cell indices
            is_occupied: True if sensor detected obstacle, False otherwise
        """
        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
            # Update in log-odds form
            if is_occupied:
                self.log_odds[grid_x, grid_y] += self.l_occ
            else:
                self.log_odds[grid_x, grid_y] += self.l_free
            
            self.log_odds[grid_x, grid_y] = np.clip(self.log_odds[grid_x, grid_y],
                                        -10.0, 10.0)

            # Convert back to probability for the grid
            odds = math.exp(self.log_odds[grid_x, grid_y])
            self.grid[grid_x, grid_y] = odds / (1 + odds)
    
    def bresenham_line(self, x0, y0, x1, y1):
        """
        Bresenham's line algorithm to get all cells along a line.
        Returns list of (x, y) grid coordinates.
        """
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            cells.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                if x0 == x1:
                    break
                err -= dy
                x0 += sx
            if e2 < dx:
                if y0 == y1:
                    break
                err += dx
                y0 += sy
                
        return cells
    
    def update_from_sensor(self, robot_pos, sensor_endpoint, hit):
        """
        Update the occupancy grid based on a single sensor reading.
        
        Args:
            robot_pos: (x, y) tuple of robot's position
            sensor_endpoint: (x, y) tuple of sensor endpoint
            hit: Whether the sensor detected an obstacle
        """
        robot_grid_x, robot_grid_y = self.world_to_grid(robot_pos[0], robot_pos[1])
        sensor_grid_x, sensor_grid_y = self.world_to_grid(sensor_endpoint[0], sensor_endpoint[1])
        
        # Get all cells along the sensor ray
        cells = self.bresenham_line(robot_grid_x, robot_grid_y, sensor_grid_x, sensor_grid_y)
        
        # Mark all cells except the last as free
        for i, (grid_x, grid_y) in enumerate(cells):
            if i == len(cells) - 1 and hit:
                # Last cell is occupied if sensor detected obstacle
                self.update_cell(grid_x, grid_y, True)
            else:
                # All other cells are free
                self.update_cell(grid_x, grid_y, False)
    
    def update_from_sensors(self, robot):
        """
        Update the occupancy grid based on all sensor readings from the robot.
        
        Args:
            robot: Robot instance with sensors
        """
        robot_pos = (robot.x, robot.y)
        
        for sensor in robot.sensors:
            # Get sensor endpoint
            endpoint = sensor.get_end_point(robot)
            # Check if sensor hit something (distance less than max range)
            hit = sensor.current_distance < sensor.max_range
            
            # Update grid based on this sensor reading
            self.update_from_sensor(robot_pos, endpoint, hit)
    
    def draw(self, screen, x_offset=0, y_offset=0):
        """
        Draw the occupancy grid on the screen.
        
        Args:
            screen: PyGame screen to draw on
            x_offset, y_offset: Offset for drawing (useful for UI)
        """
        # Clear the grid surface
        self.grid_surface.fill((0, 0, 0, 0))
        
        # Draw each cell
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                # Calculate cell position and size
                cell_x = x * self.resolution
                cell_y = y * self.resolution
                
                # Determine color based on occupancy probability
                # 0 = free (white), 0.5 = unknown (gray), 1 = occupied (black)
                p = self.grid[x, y]
                if p < 0.3:  # Free space
                    color = (255, 255, 255, 180)
                elif p > 0.7:  # Occupied
                    color = (0, 0, 0, 180)
                else:  # Unknown
                    color = (128, 128, 128, 100)
                
                # Draw the cell
                pygame.draw.rect(self.grid_surface, color, 
                                 (cell_x, cell_y, self.resolution, self.resolution))
        
        # Blit the grid surface onto the screen
        screen.blit(self.grid_surface, (x_offset, y_offset))
    
    def export_map(self):
        """
        Export the current map as a NumPy array.
        Returns a binary map where 1 = occupied, 0 = free.
        """
        binary_map = np.zeros_like(self.grid, dtype=np.uint8)
        binary_map[self.grid > 0.65] = 1  # Threshold for considering a cell occupied
        return binary_map