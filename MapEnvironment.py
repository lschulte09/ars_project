import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pygame
from matplotlib.patches import Rectangle, Circle

from Boundary import Boundary
from DustParticle import DustParticle
from Landmark import Landmark
from Obstacle import Obstacle
from Robot import Robot




class MapEnvironment:
    def __init__(self, width, height, num_obstacles=5, num_dust=20, num_landmarks = 0, draw_bearings = False):
        self.width = width
        self.height = height
        self.obstacles = []
        self.dust_particles = []
        self.landmarks = []
        self.boundary = Boundary(self.width, self.height, 20)
        self.obstacles_boundary = [Obstacle(self.boundary.x, self.boundary.y, self.boundary.width, self.boundary.height)]
        self.robot = None
        self.draw_bearings = draw_bearings
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Robot Simulation")
        self.generate_obstacles(num_obstacles)
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


    def generate_landmarks(self):
        for _ in range(self.num_landmarks):
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)
            self.landmarks.append(Landmark(x, y, self.draw_bearings))

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
        while boo < len(self.obstacles):
            boo = 0
            for obstacle in self.obstacles:
                if not (obstacle.x <= rand_x_robot <= obstacle.x + obstacle.width
                        and
                        obstacle.y <= rand_y_robot < obstacle.y + obstacle.height):
                    boo += 1
            rand_x_robot = random.uniform(0, self.width)
            rand_y_robot = random.uniform(0, self.height)



        rand_theta = random.uniform(0, 2 * math.pi)
        self.robot = Robot(rand_x_robot, rand_y_robot, rand_theta)
        # Initial sensor update
        self.robot.update_sensors(self.obstacles_boundary)

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
        """
        Update the environment state.
        """
        if self.robot:
            # Move the robot
            self.robot.move(dt=0.1, obstacles=self.obstacles_boundary, landmarks = self.landmarks)
            # Update sensor readings
            self.robot.update_sensors(self.obstacles_boundary + self.dust_particles)

    def draw_screen(self):
        self.screen.fill('gray')

        self.boundary.draw(self.screen)
        
        # Draw obstacles
        for obstacle in self.obstacles:
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