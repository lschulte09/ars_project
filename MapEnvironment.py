import math
import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import pygame
import json
from pygame.math import Vector2
from Boundary import Boundary
from DustParticle import DustParticle
from Landmark import Landmark
from Obstacle import Obstacle
from PolyObstacle import PolyObstacle
from Robot import Robot
from OccupancyGrid import OccupancyGrid


def generate_polygon(center, radius, num_vertices):
    angles = sorted([random.uniform(0, 2 * math.pi) for _ in range(num_vertices)])
    points = []
    for angle in angles:
        r = int(radius * (0.8 + 0.2 * random.random()))  # Slight variation
        x = center.x + r * math.cos(angle)
        y = center.y + r * math.sin(angle)
        points.append(Vector2(x, y))
    return points

class MapEnvironment:
    def __init__(self, width, height, num_obstacles=5, max_obstacle_size=100, num_landmarks=0, random_bots = 0,
                 draw_kalman=False, obstacle_type='poly', draw_occupancy_grid=True, slam_enabled=False, make_dust = False,
                 landmark_dist = 'even'):
        self.width = width
        self.height = height
        self.obstacles = []
        self.dust_particles = []
        self.dust_collected = 0
        self.landmarks = []
        self.landmark_dist = landmark_dist
        self.boundary = Boundary(self.width, self.height, 20)
        self.obstacles_boundary = [Obstacle(self.boundary.x, self.boundary.y, self.boundary.width, self.boundary.height)]
        self.poly_obstacles = [PolyObstacle(self.boundary.get_points())]
        self.robot = None
        self.num_random_bots = random_bots
        self.random_bots = []
        self.all_robots = []
        self.draw_kalman = draw_kalman
        self.draw_occupancy_grid = draw_occupancy_grid
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Robot Simulation with Mapping")
        
        # Create occupancy grid
        self.grid_resolution = 10  # pixels per cell
        self.occupancy_grid = OccupancyGrid(self.width, self.height, self.grid_resolution)

        self.obstacle_magnitude = max_obstacle_size
        if obstacle_type == 'rect':
            self.generate_obstacles(num_obstacles)
        if obstacle_type == 'poly':
            self.generate_poly_obstacles(num_obstacles)
        if make_dust:
            self.dust_density = 4
            self.generate_dust()
        self.num_landmarks = num_landmarks
        self.generate_landmarks()
        # SLAM logging setup
        self.slam_enabled = slam_enabled
        self.gt_landmarks = [(lm.x, lm.y) for lm in self.landmarks]
        self.gt_poses = []
        self.est_poses = []
        self.est_landmarks_hist = []
        self.nis_history = []
        # continue
        self.font = pygame.font.SysFont('Arial', 24)

        # Control params
        self.v_left = 0.0
        self.v_right = 0.0
        self.step_size = 1.0  # Velocity increment
        
        # UI options
        self.show_map = True  # Toggle for showing/hiding the occupancy grid

    def generate_obstacles(self, num):
        for _ in range(num):
            # Random dimensions for each obstacle
            obs_width = random.uniform(30, 80)
            obs_height = random.uniform(30, 80)
            # Ensure obstacles are within the map bounds
            x = random.uniform(0, self.width - obs_width)
            y = random.uniform(0, self.height - obs_height)
            obs = Obstacle(x, y, obs_width, obs_height)
            self.obstacles.append(obs)
            self.obstacles_boundary.append(obs)

    def generate_poly_obstacles(self, num):
        for _ in range(num):
            pos = Vector2(random.uniform(0, self.width), random.uniform(0, self.height))
            radius = int(random.uniform(40, self.obstacle_magnitude))
            n_vert = int(random.uniform(3, 8))
            points = generate_polygon(pos, radius, n_vert)
            obs = PolyObstacle(points)
            self.obstacles.append(obs)
            self.poly_obstacles.append(obs)

    def generate_landmarks(self):

        if self.num_landmarks <= 0:
            return

        if self.landmark_dist == 'random':
            for i in range(self.num_landmarks):
                x = random.uniform(0, self.width)
                y = random.uniform(0, self.height)
                # numerate landmarks for signatures
                self.landmarks.append(Landmark(x, y, i))

        elif self.landmark_dist == 'even':
            grid_size = max(1, math.ceil(math.sqrt(self.num_landmarks)))
            cell_width = self.width / grid_size
            cell_height = self.height / grid_size

            idx = 0
            for row in range(grid_size):
                for col in range(grid_size):
                    if idx >= self.num_landmarks:
                        return
                    # Random point within the cell
                    x = random.uniform(col * cell_width, (col + 1) * cell_width)
                    y = random.uniform(row * cell_height, (row + 1) * cell_height)
                    self.landmarks.append(Landmark(x, y, idx))
                    idx += 1


    def generate_dust(self):
        for x in range(int(self.boundary.x), int(self.boundary.x + self.boundary.width), int(self.dust_density)):
            for y in range(int(self.boundary.y), int(self.boundary.y + self.boundary.height), int(self.dust_density)):
                self.dust_particles.append(DustParticle(x, y))

    def collect_dust(self):
        for dust in self.dust_particles:
            if (self.robot.pos - dust.pos).length() < self.robot.radius:
                self.dust_collected += 1
                self.dust_particles.remove(dust)

    def place_bots(self):
        for _ in range(self.num_random_bots):
            rand_x_robot = random.uniform(50, self.width-50)
            rand_y_robot = random.uniform(50, self.height-50)
            rand_theta = random.uniform(0, 2 * math.pi)
            self.random_bots.append(Robot(rand_x_robot, rand_y_robot, rand_theta, draw_trail=False, draw_ghost=False, slam_enabled=False, control='RANDOM'))
            self.all_robots.append(self.random_bots[-1])


    def place_robot(self):
        rand_x_robot = random.uniform(50, self.width-50)
        rand_y_robot = random.uniform(50, self.height-50)
        rand_theta = random.uniform(0, 2 * math.pi)
        self.robot = Robot(rand_x_robot, rand_y_robot, rand_theta, draw_trail=self.draw_kalman, draw_ghost=self.draw_kalman, slam_enabled=self.slam_enabled, control='MANUAL')
        self.all_robots.append(self.robot)
        # Initial sensor update
        self.robot.update_sensors(self.poly_obstacles, self.all_robots)
        if self.robot.slam_enabled:
            self.robot.initialize_slam(self.num_landmarks)

    def handle_input(self, event):
        """
        Handle keyboard input for controlling the robot.
        """
        if event.key == pygame.K_SPACE:
            self.v_left = 0.0
            self.v_right = 0.0

        if event.key == pygame.K_q:  # Increase left wheel velocity
            self.v_left += self.step_size
        if event.key == pygame.K_a:  # Decrease left wheel velocity
            self.v_left -= self.step_size

        if event.key == pygame.K_e:  # Increase right wheel velocity
            self.v_right += self.step_size
        if event.key == pygame.K_d:  # Decrease right wheel velocity
            self.v_right -= self.step_size

        if event.key == pygame.K_w:
            self.v_left += self.step_size
            self.v_right += self.step_size

        if event.key == pygame.K_s:
            self.v_left -= self.step_size
            self.v_right -= self.step_size
            
        # Toggle map visibility with 'M' key
        if event.key == pygame.K_m:
            self.show_map = not self.show_map
            
        # Apply velocity limits
        self.v_left = max(-self.robot.max_speed, min(self.robot.max_speed, self.v_left))
        self.v_right = max(-self.robot.max_speed, min(self.robot.max_speed, self.v_right))
        
        # Update robot wheel velocities
        self.robot.set_wheel_velocities(self.v_left, self.v_right)

    def update(self):
        """
        Update the environment state.
        """
        if self.robot:
            underlying_square_length = self.robot.radius * 1.41
            
            # Move the robot
            self.robot.move(dt=0.1, obstacles=self.poly_obstacles, landmarks=self.landmarks, robots=self.all_robots)

            # SLAM logging: ground-truth vs EKF‐SLAM estimate
            # 1) ground truth
            self.gt_poses.append((self.robot.x, self.robot.y, self.robot.theta))
             # 2) EKF‐SLAM estimate
            if self.robot.slam_enabled and self.robot.mu is not None:
                mu = self.robot.mu.flatten()
                # pose
                self.est_poses.append((mu[0], mu[1], mu[2]))
                # landmark positions
                L = (len(mu) - 3) // 2
                lm_est = [(mu[3 + 2 * i], mu[3 + 2 * i + 1]) for i in range(L)]
                self.est_landmarks_hist.append(lm_est)
                # NIS
                if hasattr(self.robot, 'last_nis'):
                    self.nis_history.append(self.robot.last_nis)
            
            # Check for dust collection
            self.collect_dust()
            
            # Update sensor readings
            self.robot.update_sensors(self.poly_obstacles, self.all_robots)
            
            # Update occupancy grid based on sensor readings
            if self.draw_occupancy_grid:
                self.occupancy_grid.update_from_sensors(self.robot)

        if self.random_bots:
            for bot in self.random_bots:
                bot.move(dt=0.1, obstacles=self.poly_obstacles, landmarks=self.landmarks, robots=self.all_robots)

    def update_bot_controls(self):
        for bot in self.random_bots:
            bot.random_move()

    def draw_screen(self):
        self.screen.fill('gray')

        # Draw the occupancy grid if enabled
        if self.draw_occupancy_grid and self.show_map:
            self.occupancy_grid.draw(self.screen)

        self.boundary.draw(self.screen)
        
        # Draw obstacles
        for obstacle in self.poly_obstacles[1:]:
            obstacle.draw(self.screen)
            
        # Draw dust particles
        for dust in self.dust_particles:
            dust.draw(self.screen)

        # Draw robot
        # if self.robot:
        #     self.robot.draw(self.screen)

        for bot in self.all_robots:
            bot.draw(self.screen)

        for landmark in self.landmarks:
            landmark.draw(self.screen, self.robot)
            
        # Draw control information
        text_y = 10
        wheel_text = f"Left: {self.v_left:.1f}, Right: {self.v_right:.1f}"
        text_surface = self.font.render(wheel_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, text_y))
        text_y += 30
        
        # Draw collision status
        if self.robot and (self.robot.obs_collisions or self.robot.bot_collisions):
            collision_text = "COLLISION DETECTED!"
            text_surface = self.font.render(collision_text, True, (255, 0, 0))
            self.screen.blit(text_surface, (10, text_y))
            
        # Draw control instructions
        instructions = [
            "Controls:",
            "W/S - Forward/Backward",
            "Q/A - Left Wheel +/-",
            "E/D - Right Wheel +/-",
            "SPACE - Stop",
            "M - Toggle Map"
        ]
        
        for i, instruction in enumerate(instructions):
            text_surface = self.font.render(instruction, True, (0, 0, 0))
            self.screen.blit(text_surface, (self.width - 200, 10 + 25 * i))

    def save_map(self, filename="occupancy_map.npy"):
        """Save the occupancy grid map to a file"""
        binary_map = self.occupancy_grid.export_map()
        np.save(filename, binary_map)
        print(f"Map saved to {filename}")
        
    def plot_map(self):
        """Plot the occupancy grid map using matplotlib"""
        binary_map = self.occupancy_grid.export_map()
        plt.figure(figsize=(10, 8))
        plt.imshow(binary_map.T, cmap='gray_r', origin='lower')
        plt.title('Occupancy Grid Map')
        plt.colorbar(label='Occupancy (0=free, 1=occupied)')
        plt.show()

    def save_env(self, directory):
        """Save the environment to a directory"""
        if not os.path.exists(directory):
            os.makedirs(directory)

        data = {
            'width': self.width,
            'height': self.height,
            'boundary': self.boundary,
            'poly_obstacles': self.poly_obstacles,
            'landmarks': self.landmarks,
            'all_robots': self.all_robots,
            'random_bots': self.random_bots,
            'robot': self.robot,
            'dust_particles': self.dust_particles
        }

        with open(os.path.join(directory, f'environment_{random.randint(0, 1000)}.pkl'), 'wb') as f:
            pickle.dump(data, f)

    def load_env(self, filename):
        """Load the environment from a file"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        self.width = data['width']
        self.height = data['height']
        self.boundary = data['boundary']
        self.poly_obstacles = data['poly_obstacles']
        self.landmarks = data['landmarks']
        self.all_robots = data['all_robots']
        self.random_bots = data['random_bots']
        self.robot = data['robot']
        self.dust_particles = data['dust_particles']