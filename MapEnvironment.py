import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pygame
from matplotlib.patches import Rectangle, Circle
from pygame.math import Vector2
from Boundary import Boundary
from DustParticle import DustParticle
from Landmark import Landmark
from Obstacle import Obstacle
from PolyObstacle import PolyObstacle
from Robot import Robot


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
    def __init__(self, width, height, num_obstacles=5, num_dust=20, num_landmarks = 0, draw_kalman = False, obstacle_type = 'poly'):
        self.width = width
        self.height = height
        self.obstacles = []
        self.dust_particles = []
        self.landmarks = []
        self.boundary = Boundary(self.width, self.height, 20)
        self.obstacles_boundary = [Obstacle(self.boundary.x, self.boundary.y, self.boundary.width, self.boundary.height)]
        self.poly_obstacles = [PolyObstacle(self.boundary.get_points())]
        self.robot = None
        self.draw_kalman = draw_kalman
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Robot Simulation")
        if obstacle_type == 'rect':
            self.generate_obstacles(num_obstacles)
        if obstacle_type == 'poly':
            self.generate_poly_obstacles(num_obstacles)
        self.generate_dust(num_dust)
        self.num_landmarks = num_landmarks
        self.generate_landmarks()
        self.font = pygame.font.SysFont('Arial', 24)

        # Control params
        self.v_left = 0.0
        self.v_right = 0.0
        self.step_size = 1.0  # Velocity increment

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
            radius = int(random.uniform(40, 100))
            n_vert = int(random.uniform(3, 8))
            points = generate_polygon(pos, radius, n_vert)
            obs = PolyObstacle(points)
            self.obstacles.append(obs)
            self.poly_obstacles.append(obs)


    def generate_landmarks(self):
        for i in range(self.num_landmarks):
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)
            # numerate landmarks for signatures
            self.landmarks.append(Landmark(x, y, i))

    def generate_dust(self, num):
        for _ in range(num):
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)
            self.dust_particles.append(DustParticle(x, y, width=4, height=4))

    def place_robot(self):
        rand_x_robot = random.uniform(0, self.width)
        rand_y_robot = random.uniform(0, self.height)
        # make sure robot doesn't spawn in an obstacle
        boo = 0
        # while boo < len(self.obstacles):
        #     boo = 0
        #     for obstacle in self.obstacles:
        #         if not (obstacle.x <= rand_x_robot <= obstacle.x + obstacle.width
        #                 and
        #                 obstacle.y <= rand_y_robot < obstacle.y + obstacle.height):
        #             boo += 1
        #     rand_x_robot = random.uniform(50, self.width-50)
        #     rand_y_robot = random.uniform(50, self.height-50)



        rand_theta = random.uniform(0, 2 * math.pi)
        self.robot = Robot(rand_x_robot, rand_y_robot, rand_theta, draw_trail=self.draw_kalman, draw_ghost=self.draw_kalman)
        # Initial sensor update
        self.robot.update_sensors(self.poly_obstacles)

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

        # keys = pygame.key.get_pressed()
        #
        # # Control left wheel velocity
        # if keys[pygame.K_q]:  # Increase left wheel velocity
        #     self.v_left += self.step_size
        # if keys[pygame.K_a]:  # Decrease left wheel velocity
        #     self.v_left -= self.step_size
        #
        # # Control right wheel velocity
        # if keys[pygame.K_e]:  # Increase right wheel velocity
        #     self.v_right += self.step_size
        # if keys[pygame.K_d]:  # Decrease right wheel velocity
        #     self.v_right -= self.step_size
        #
        # # Set both wheels to the same velocity (forward)
        # if keys[pygame.K_w]:
        #     self.v_left += self.step_size
        #     self.v_right += self.step_size
        #
        # # Set both wheels to the same velocity (backward)
        # if keys[pygame.K_s]:
        #     self.v_left -= self.step_size
        #     self.v_right -= self.step_size
        #
        # # Stop the robot
        # if keys[pygame.K_SPACE]:
        #     self.v_left = 0.0
        #     self.v_right = 0.0
            
        # Apply velocity limits
        self.v_left = max(-self.robot.max_speed, min(self.robot.max_speed, self.v_left))
        self.v_right = max(-self.robot.max_speed, min(self.robot.max_speed, self.v_right))
        
        # Update robot wheel velocities
        self.robot.set_wheel_velocities(self.v_left, self.v_right)

    def update(self):
        underlying_square_length = self.robot.radius * 1.41
        """
        Update the environment state.
        """
        if self.robot:
            # Move the robot
            self.robot.move(dt=0.1, obstacles=self.poly_obstacles, landmarks = self.landmarks)
            # lets assume for now that there is a square under the vacuum that sucks it up to ease computational burden
            # and lets make it suck it up the dust particles
            for i in range(len(self.dust_particles)):
                dust_particle = self.dust_particles[i]
                if (    self.robot.x - underlying_square_length/2 < dust_particle.x < self.robot.x + underlying_square_length / 2
                        and
                        self.robot.y - underlying_square_length/2 < dust_particle.y < self.robot.y + underlying_square_length / 2):
                    self.dust_particles.pop(i)
                    break
            # Update sensor readings
            self.robot.update_sensors(self.poly_obstacles) # + dust_particles

    def draw_screen(self):
        self.screen.fill('gray')

        self.boundary.draw(self.screen)
        
        # Draw obstacles
        # for obstacle in self.obstacles:
            # obstacle.draw(self.screen)

        for obstacle in self.poly_obstacles[1:]:
            obstacle.draw(self.screen)
            
        # Draw dust particles
        for dust in self.dust_particles:
            dust.draw(self.screen)

        # Draw robot
        if self.robot:
            self.robot.draw(self.screen)

        for landmark in self.landmarks:
            landmark.draw(self.screen, self.robot)


            
        # Draw control information
        text_y = 10
        wheel_text = f"Left: {self.v_left:.1f}, Right: {self.v_right:.1f}"
        text_surface = self.font.render(wheel_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, text_y))
        text_y += 30
        
        # Draw collision status
        if self.robot and self.robot.collision:
            collision_text = "COLLISION DETECTED!"
            text_surface = self.font.render(collision_text, True, (255, 0, 0))
            self.screen.blit(text_surface, (10, text_y))
            
        # Draw control instructions
        instructions = [
            "Controls:",
            "W/S - Forward/Backward",
            "Q/A - Left Wheel +/-",
            "E/D - Right Wheel +/-",
            "SPACE - Stop"
        ]
        
        for i, instruction in enumerate(instructions):
            text_surface = self.font.render(instruction, True, (0, 0, 0))
            self.screen.blit(text_surface, (self.width - 200, 10 + 25 * i))

    def plot(self):
        plt.figure(figsize=(8, 8))
        ax = plt.gca()

        # Draw obstacles as rectangles
        for obs in self.obstacles:
            rect = Rectangle((obs.x, obs.y), obs.width, obs.height, color='gray')
            ax.add_patch(rect)

        # Draw dust particles as yellow dots
        for dust in self.dust_particles:
            plt.plot(dust.x, dust.y, 'yo', markersize=5)

        # Draw the robot (if placed) as a blue circle with an arrow showing orientation.
        if self.robot:
            # Robot body as a circle
            robot_circle = Circle((self.robot.x, self.robot.y), self.robot.radius, fc='red', ec='black')
            ax.add_patch(robot_circle)
            
            # Draw an arrow indicating the heading direction
            dx = self.robot.radius * np.cos(self.robot.theta)
            dy = self.robot.radius * np.sin(self.robot.theta)
            plt.arrow(self.robot.x, self.robot.y, dx, dy, head_width=5, head_length=10, fc='black', ec='black')
            
            # Draw sensor rays
            for sensor in self.robot.sensors:
                end_x, end_y = sensor.get_end_point(self.robot)
                plt.plot([self.robot.x, end_x], [self.robot.y, end_y], 'g-', linewidth=1)

        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        plt.title('Simulated Environment')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.grid(True)
        plt.show()