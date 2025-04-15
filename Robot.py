import random
import numpy as np
import pygame
import math
from pygame.math import Vector2
from Sensor import Sensor


# idk if this is fine or will make problems later
def points_distance(p1, p2):
    dist = (p2-p1).length()
    return dist

def points_line_dist(p1, l1, l2):
    line = l2 - l1
    ap = p1 - l1
    project = ap.dot(line) / line.length_squared()
    project = max(0, min(1, project))
    closest = l1 + line * project
    dist = (p1 - closest).length()
    return dist, closest

def points_line_dist_norm(p1, l1, l2):
    line = l2 - l1
    ap = p1 - l1
    project = ap.dot(line) / line.length_squared()
    project = max(0, min(1, project))
    closest = l1 + line * project
    dist = (p1 - closest).length()

    if project == 0 or project == 1:
        end = l1 if project == 0 else l2
        end_vec = (end - p1).normalize()
        norm = end_vec.rotate(90)
        return norm, dist

    else:
        norm = line.normalize()
        return norm, dist

def trilaterate(p1, d1, p2, d2, p3, d3):
        ex = (p2 - p1).normalize()
        i = ex.dot(p3 - p1)
        ey = (p3 - p1 - i * ex).normalize()
        d = (p2 - p1).length()
        j = ey.dot(p3 - p1)

        # Calculate coordinates
        x = (d1 ** 2 - d2 ** 2 + d ** 2) / (2 * d)
        y = (d1 ** 2 - d3 ** 2 + i ** 2 + j ** 2 - 2 * i * x) / (2 * j)

        result = p1 + x * ex + y * ey
        return result



class Robot:
    def __init__(self, x, y, theta, lm_range = 200, sensor_range = 200, draw_trail = False, draw_ghost = False):
        self.x = x  # position x-coordinate
        self.y = y  # position y-coordinate
        self.pos = Vector2(x, y)
        self.v_left = 0.0  # left wheel velocity
        self.v_right = 0.0  # right wheel velocity
        self.radius = 30  # robot radius
        self.wheel_radius = 5.0  # wheel radius - is this ever used?
        self.wheel_distance = self.radius*2  # distance between wheels
        self.v = (self.v_right+self.v_left)/2
        self.theta = theta  # orientation in radians
        self.max_speed = 20
        self.eps_dist = 5 # distance sensor noise
        self.eps_ang = 0.1 # angle sensor noise
        self.move_vec = Vector2(0, 0)
        self.lm_range = lm_range
        self.sensor_range = sensor_range
        self.draw_ghost = draw_ghost
        self.draw_trail = draw_trail
        if draw_trail:
            self.trail = []
            if draw_ghost:
                self.ghost_trail = []
        self.ghost = None
        self.collided = False
        self.vis_landmarks = {}
        # init estimate to initial position (assume robot knows where it starts)
        self.mu = np.array([[self.pos.x], [self.pos.y], [self.theta]])
        self.Sigma = np.identity(3)
        # Covariance matrix for motion error
        self.R = np.array([[0.5, 0, 0],
                           [0, 0.5, 0],
                           [0, 0, 0.05]])
        self.Q = np.array([[2, 0, 0],
                           [0, 2, 0],
                           [0, 0, 0.05]])
        self.pos_est = Vector2(self.pos.x, self.pos.y)
        
        # Create 12 sensors placed every 30 degrees (360°/12)
        self.sensors = [Sensor(np.deg2rad(angle), sensor_range) for angle in range(0, 360, 30)]
        
        # Collision flag
        self.collision = False
        
        # Store last valid position in case of collision
        self.last_valid_x = x
        self.last_valid_y = y
        self.last_valid_theta = theta

    def kalman_localisation(self, v, w, dt=0.1):
        # A: nxn matrix, identity (no control independent changes)
        # B: control vector? matrix? nxl
        # C: observation matrix? vector? kxn
        # eps, sig: random var (normal) with covar R, Q
        # mu: state (pos, theta)
        # u: robot control -> B
        # v, w: linear, angular velocity
        # Sigma: covariance matrix, init with small values
        mu_old = self.mu
        Sigma_old = self.Sigma
        theta_old = mu_old[2, 0]
        A = np.identity(3)
        B = np.array([[dt*math.cos(theta_old), 0],
                      [dt*math.sin(theta_old), 0],
                      [0, dt]])
        u = np.array([[v], [w]])
        C = np.identity(3)

        # initial prediction
        mu_guess = A@mu_old + B@u
        Sigma_guess = A@Sigma_old@A.T + self.R

        pos_est = Vector2(float(mu_guess[0, 0]), float(mu_guess[1, 0]))

        # measurement update (only necessary if there is any sensor data)
        if len(self.vis_landmarks) > 0:

        # triangulation check
            landmark_list = []
            for key in self.vis_landmarks.keys():
                landmark_list.append(key)
            if len(self.vis_landmarks) == 3:
                p1 = Vector2(landmark_list[0])
                d1 = self.vis_landmarks[landmark_list[0]][0]
                p2 = Vector2(landmark_list[1])
                d2 = self.vis_landmarks[landmark_list[1]][0]
                p3 = Vector2(landmark_list[2])
                d3 = self.vis_landmarks[landmark_list[2]][0]
                guess = trilaterate(p1, d1, p2, d2, p3, d3)
                # simulate sensor noise
                pos_est = self.add_noise_to_point(guess)
            if len(self.vis_landmarks) > 3:
                pos_guesses = []

                for index, i in enumerate(landmark_list):
                    p1 = Vector2(i)
                    d1 = self.vis_landmarks[i][0]
                    p2 = Vector2(landmark_list[index+1])
                    d2 = self.vis_landmarks[landmark_list[index+1]][0]
                    p3 = Vector2(landmark_list[index+2])
                    d3 = self.vis_landmarks[landmark_list[index+1]][0]

                    guess = trilaterate(p1, d1, p2, d2, p3, d3)
                    # simulate sensor noise
                    pos_guesses.append(self.add_noise_to_point(guess))

                array = np.array([v.xy for v in pos_guesses])
                mean = np.mean(array, axis=0)
                pos_est = Vector2(mean)

            theta_guess = float(self.mu[2, 0])
            theta_guesses = np.array([])
            for lm_tuple, vals in self.vis_landmarks.items():
                lm = Vector2(lm_tuple)
                theta_guess = math.atan2(lm.y - self.pos.y, lm.x - self.pos.x) - vals[1]
                theta_guess += random.uniform(-self.eps_ang, self.eps_ang)
                theta_guess = theta_guess % (2 * math.pi)
                np.append(theta_guesses, theta_guess)

            if theta_guesses.size > 0:
                mean = np.mean(theta_guesses, axis=0)
                theta_guess = mean

            z = np.array([[pos_est.x], [pos_est.y], [theta_guess]])

            K = Sigma_guess@C.T@np.linalg.inv(C@Sigma_guess@C.T + self.Q)
            mu_guess = mu_guess + K@(z - C@mu_guess)
            Sigma_guess = (np.identity(3) - K@C)@Sigma_guess
        self.mu = mu_guess
        self.Sigma = Sigma_guess

        if self.draw_trail:
            self.ghost_trail.append(Vector2(float(self.mu[0, 0]), float(self.mu[1, 0])))
            if len(self.ghost_trail) > 1000:
                self.ghost_trail.pop(0)

    def update_sensors(self, obstacles):
        """
        Update all sensor readings for obstacles.
        """
        for sensor in self.sensors:
            sensor.read_distance(self, obstacles, type = 'poly')
            
    def check_collision(self, obstacles, dust_particles):
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

    def move(self, dt=0.1, obstacles=None, landmarks=None):
        """
        Updates the robot's pose using differential drive kinematics.
        Handles collision detection if obstacles are provided.
        """
        # Save current position as the last valid position
        #self.last_valid_x = self.x
        #self.last_valid_y = self.y
        #self.last_valid_theta = self.theta

        if self.draw_trail:
            self.trail.append(self.pos)
            if len(self.trail) > 1000:
                self.trail.pop(0)

        # Calculate velocities from wheel velocities
        linear_velocity, angular_velocity = self.calculate_velocities()

        self.kalman_localisation(linear_velocity, angular_velocity)
        
        # Update position and orientation
        self.theta -= angular_velocity * dt
        self.x += linear_velocity * np.cos(self.theta) * dt
        self.y += linear_velocity * np.sin(self.theta) * dt

        dir_vec = Vector2(1, 0).rotate_rad(self.theta)
        self.move_vec = dir_vec * linear_velocity * dt
        
        # Normalize angle to keep it within [0, 2π]
        self.theta = self.theta % (2 * math.pi)

        new_pos = self.pos + self.move_vec

        # check collision
        for obstacle in obstacles:
            points = obstacle.get_points()
            # might have to change this for non convex obstacles (also see backwards check), create get_lines in obstacle?
            lines = [[points[i], points[i + 1]] for i in range(len(points) - 1)]
            lines.append([points[-1], points[0]])
            for line in lines:
                norm, dist = points_line_dist_norm(new_pos, line[0], line[1])

                if dist <= self.radius:
                    self.move_vec = self.move_vec.dot(norm) * norm

        # self.move_vec = check_collision_vec(obstacles)
        new_pos = self.pos + self.move_vec

        # backwards check for illegal move, "bump out of wall"
        for obstacle in obstacles:
            points = obstacle.get_points()
            lines = [[points[i], points[i + 1]] for i in range(len(points) - 1)]
            lines.append([points[-1], points[0]])
            for line in lines:
                dist, closest_point = points_line_dist(new_pos, line[0], line[1])

                if dist < self.radius:
                    diff = self.radius - dist
                    dir = (new_pos - closest_point).normalize()
                    new_pos = new_pos + dir * diff

        self.pos = new_pos
        self.x = self.pos.x
        self.y = self.pos.y

        self.vis_landmarks = {}
        for landmark in landmarks:
            lm_pos = landmark.get_pos()
            lm_dist = points_distance(self.pos, lm_pos)
            if lm_dist < self.lm_range:
                r_vec = (lm_pos - self.pos).normalize()
                bearing = math.radians(dir_vec.angle_to(r_vec))
                self.vis_landmarks[tuple(landmark.pos)] = [lm_dist, bearing]

        self.kalman_localisation(linear_velocity, angular_velocity)



        # Check for collisions if obstacles are provided
        # need to implement sliding, use vector2 maybe
        # if obstacles:
        #     if self.check_collision(obstacles):
        #         # Collision detected, revert to last valid position
        #         self.x = self.last_valid_x
        #         self.y = self.last_valid_y
        #         self.theta = self.last_valid_theta
        #         self.collision = True
        #         return False  # Movement failed due to collision
        #     else:
        #         self.collision = False
                
        # return True  # Movement successful

    def get_pose(self):
        return self.x, self.y, self.theta

    def check_collision_vec(self, obstacles, move_vec):
        for obstacle in obstacles:
            points = obstacle.get_points()
            lines = [[points[i], points[i+1]] for i in range(len(points)-1)]
            lines.append([points[-1], points[0]])
            for line in lines:
                norm, dist = points_line_dist_norm(self.pos, line[0], line[1])

                if dist <= self.radius:
                    move_vec = move_vec.dot(norm)*norm

        return move_vec

    def add_noise_to_point(self,p):
        return p + Vector2(np.random.normal(0, self.eps_dist), np.random.normal(0, self.eps_dist))


    def draw(self, screen):
        # Draw the robot body
        pygame.draw.circle(screen, (255, 0, 0), self.pos, self.radius, 0)
        
        # Draw a line indicating the robot's orientation
        line_x = self.x + self.radius * math.cos(self.theta)
        line_y = self.y + self.radius * math.sin(self.theta)
        line_end = self.pos + Vector2(self.radius, 0).rotate_rad(self.theta)
        pygame.draw.line(screen, (0, 0, 0), self.pos, line_end, 3)
        
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

        if self.draw_trail:
            for i in self.trail:
                pygame.draw.circle(screen, (0, 0, 0), (int(i[0]), int(i[1])), 1)


        if self.draw_ghost:
            ghost_pos = Vector2(float(self.mu[0, 0]), float(self.mu[1, 0]))
            ghost_theta = self.mu[2, 0]
            pygame.draw.circle(screen, (0, 0, 255), ghost_pos, self.radius, 2)

            # Draw a line indicating the robot's orientation
            line_end = ghost_pos + Vector2(self.radius, 0).rotate_rad(ghost_theta)
            pygame.draw.line(screen, (0, 0, 255), ghost_pos, line_end, 3)

            if self.draw_trail:
                for i in self.ghost_trail:
                    pygame.draw.circle(screen, (0, 0, 255), (int(i[0]), int(i[1])), 1)








