import random
import numpy as np
import pygame
import math
import numpy.linalg as LA
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

def two_landmark_localize(p1, d1, b1, p2, d2, b2):
    # Step 1: Local vectors (landmarks in robot frame)
    v1_local = Vector2(d1 * math.cos(b1), d1 * math.sin(b1))
    v2_local = Vector2(d2 * math.cos(b2), d2 * math.sin(b2))

    v_local = v2_local - v1_local

    # Step 2: World vectors (difference between landmarks)
    v_world = Vector2(p2) - Vector2(p1)

    # Step 3: Compute θ
    theta = -math.radians(v_world.angle_to(v_local))  # convert to radians

    # Step 4: Solve for robot position using one landmark
    v1_rotated = Vector2(
        v1_local.x * math.cos(theta) - v1_local.y * math.sin(theta),
        v1_local.x * math.sin(theta) + v1_local.y * math.cos(theta)
    )
    pos = Vector2(p1) - v1_rotated

    return pos

def normalize_angle(angle):
    """
    Normalize angle to be within [-pi, pi].
    """
    return (angle + math.pi) % (2 * math.pi) - math.pi

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
        distance = math.sqrt((self.x - closest_x) ** 2 + (self.y - closest_y) ** 2)

        # If distance is less than robot radius, we have a collision
        if distance < self.radius:
            return True

    return False

class Robot:
    def __init__(self, x, y, theta, lm_range=200, sensor_range=200, draw_trail=False, draw_ghost=False, slam_enabled=False, control = 'MANUAL'):
        self.x = x  # position x-coordinate
        self.y = y  # position y-coordinate
        self.pos = Vector2(x, y)
        self.v_left = 0.0  # left wheel velocity
        self.v_right = 0.0  # right wheel velocity
        self.radius = 30  # robot radius
        self.wheel_radius = 5.0  # wheel radius - is this ever used?
        self.wheel_distance = self.radius * 2  # distance between wheels
        self.v = (self.v_right + self.v_left) / 2
        self.theta = theta  # orientation in radians
        self.max_speed = 30
        self.eps_dist = 5  # distance sensor noise
        self.eps_ang = 0.1  # angle sensor noise
        self.sensor_noise_range = self.eps_dist
        self.sensor_noise_bearing = self.eps_ang
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
        # control should be either "MANUAL", "RANDOM", or "AUTO"
        self.type = control
        self.vis_landmarks = {}
        # Covariance matrix for motion error
        self.R = np.array([[10, 0, 0],
                           [0, 10, 0],
                           [0, 0, 1]])
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

        # SLAM
        self.slam_enabled = slam_enabled
        self.num_landmarks = 0
        self.mu = None        # EKF-SLAM state vector
        self.Sigma = None     # EKF-SLAM covariance matrix

        if not slam_enabled:
            # give our EKF‐only code a valid 3×1, 3×3
            self.mu = np.array([[x], [y], [theta]])
            self.Sigma = np.eye(3) * 0.1

        # Localization-only (Kalman localisation)
        self.loc_mu = np.array([[x], [y], [theta]])
        self.loc_Sigma = np.eye(3) * 0.1

        # Visualization
        self.draw_trail = draw_trail
        self.trail = []
        self.draw_ghost = draw_ghost
        self.ghost_trail = []

        # Storage for latest measurements: {id: (range, bearing)}
        self.measurements = {}

    def initialize_slam(self, num_landmarks):
        self.num_landmarks = num_landmarks
        dim = 3 + 2 * num_landmarks
        self.mu = np.zeros((dim, 1))
        self.mu[:3, 0] = [self.x, self.y, self.theta]
        self.Sigma = np.eye(dim) * 1e3
        self.Sigma[:3, :3] = np.eye(3) * 0.1


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
        u = np.array([[v], [-w]])
        C = np.identity(3)

        # initial prediction
        mu_guess = A@mu_old + B@u
        Sigma_guess = A@Sigma_old@A.T + self.R

        pos_est = Vector2(float(mu_guess[0, 0]), float(mu_guess[1, 0]))

        # measurement update (only necessary if there is any sensor data)
        # triangulation check
        if len(self.vis_landmarks) > 0:
            landmark_list = []
            for key in self.vis_landmarks.keys():
                landmark_list.append(key)

            if len(self.vis_landmarks) == 2:
                p1 = Vector2(landmark_list[0])
                d1 = self.vis_landmarks[landmark_list[0]][0]
                b1 = self.vis_landmarks[landmark_list[0]][1]
                p2 = Vector2(landmark_list[1])
                d2 = self.vis_landmarks[landmark_list[1]][0]
                b2 = self.vis_landmarks[landmark_list[1]][1]
                guess = two_landmark_localize(p1, d1, b1, p2, d2, b2)
                # simulate sensor noise
                pos_est = self.add_noise_to_point(guess)

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
                    p2 = Vector2(landmark_list[(index+1) % len(landmark_list)])
                    d2 = self.vis_landmarks[landmark_list[(index+1) % len(landmark_list)]][0]
                    p3 = Vector2(landmark_list[(index+2) % len(landmark_list)])
                    d3 = self.vis_landmarks[landmark_list[(index+2) % len(landmark_list)]][0]

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

    def simple_localisation(self, v, w, dt=0.1):
        mu, S = self.loc_mu.copy(), self.loc_Sigma.copy()
        th = mu[2,0]
        mu[0,0] += v*math.cos(th)*dt
        mu[1,0] += v*math.sin(th)*dt
        mu[2,0]  = normalize_angle(th - w*dt)
        # predict cov
        G = np.eye(3)
        G[0,2], G[1,2] = -v*math.sin(th)*dt, v*math.cos(th)*dt
        alpha = [0.1]*4
        R_ctrl = np.array([[alpha[0]*v*v+alpha[1]*w*w,0],[0,alpha[2]*v*v+alpha[3]*w*w]])
        V = np.zeros((3,2)); V[0,0],V[1,0],V[2,1]=math.cos(th)*dt,math.sin(th)*dt,dt
        self.loc_mu, self.loc_Sigma = mu, G.dot(S).dot(G.T)+V.dot(R_ctrl).dot(V.T)

    def update_sensors(self, obstacles, robots):
        """
        Update all sensor readings for obstacles.
        """
        for sensor in self.sensors:
            sensor.read_distance(self, obstacles, robots, type = 'poly')
            
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

    def random_move(self):
        """
        Move the robot with random wheel velocities.
        """
        l_vel = random.uniform(-self.max_speed, self.max_speed)
        r_vel = int(np.random.normal(l_vel, 10))
        self.set_wheel_velocities(l_vel, r_vel)

    def calculate_velocities(self):
        """
        Calculate linear and angular velocities from wheel velocities.
        """
        linear_velocity = (self.v_right + self.v_left) / 2.0
        angular_velocity = (self.v_right - self.v_left) / self.wheel_distance
        return linear_velocity, angular_velocity


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

    def move_2(self, dt=0.1, obstacles=None, landmarks=None, robots=None):
        # 1. True motion
        v, w = self.calculate_velocities()
        self.theta = (self.theta - w * dt) % (2 * math.pi)
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt
        self.pos = Vector2(self.x, self.y)
        # 2. Intended move vector
        dir_vec = Vector2(1, 0).rotate_rad(self.theta)
        self.move_vec = dir_vec * v * dt
        # 3. Collision sliding + push-out
        new_pos = self.pos + self.move_vec
        if obstacles:
            # slide
            for obs in obstacles:
                pts = obs.get_points()
                lines = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)] + [(pts[-1], pts[0])]
                for l1, l2 in lines:
                    norm, dist = points_line_dist_norm(new_pos, l1, l2)
                    if dist <= self.radius:
                        self.move_vec = self.move_vec.dot(norm) * norm
        if robots:
            for bot in robots:
                if bot is self:
                    continue
                if points_distance(new_pos, bot.pos) <= bot.radius + self.radius:
                    offset_norm = (bot.pos - self.pos).normalize()
                    tangent = Vector2(-offset_norm.y, offset_norm.x)
                    self.move_vec = self.move_vec.dot(tangent) * tangent
            new_pos = self.pos + self.move_vec
        # push out
        if obstacles:
            for obs in obstacles:
                pts = obs.get_points()
                lines = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)] + [(pts[-1], pts[0])]
                for l1, l2 in lines:
                    dist, closest = points_line_dist(new_pos, l1, l2)
                    if dist < self.radius:
                        push_dir = (new_pos - closest).normalize()
                        new_pos += push_dir * (self.radius - dist)

        if robots:
            for bot in robots:
                if bot is self:
                    continue
                dist = points_distance(new_pos, bot.pos)
                if dist < bot.radius + self.radius:
                    push_dir = (new_pos - bot.pos).normalize()
                    new_pos += push_dir * (bot.radius + self.radius - dist)
        # 4. Apply
        self.pos = new_pos
        self.x, self.y = new_pos.x, new_pos.y
        # 5. Sense
        self.update_sensors(obstacles, robots)
        # 6. Build SLAM measurements
        self.measurements.clear()
        for lm in (landmarks or []):
            r = (Vector2(lm.x, lm.y) - self.pos).length()
            if r <= self.lm_range:
                b = math.atan2(lm.y - self.y, lm.x - self.x) - self.theta
                self.measurements[lm.id] = (
                    r + np.random.normal(0, self.sensor_noise_range),
                    normalize_angle(b + np.random.normal(0, self.sensor_noise_bearing))
                )
        # 7. EKF-SLAM predict&update
        if self.slam_enabled:
            self._ekf_predict(v, w, dt)
            self._ekf_update()
        else:
            # fallback to simple localisation
            self.simple_localisation(v, w, dt)
        # 8. Ghost trail
        if self.draw_ghost and self.mu is not None:
            ex, ey = float(self.mu[0, 0]), float(self.mu[1, 0])
            self.ghost_trail.append(Vector2(ex, ey))
            if len(self.ghost_trail) > 1000:
                self.ghost_trail.pop(0)

    def move(self, dt=0.1, obstacles=None, landmarks=None, robots=None):
        if self.slam_enabled:
            self.move_2(dt=dt, obstacles=obstacles, landmarks=landmarks, robots=robots)
            return

        # Save current position as the last valid position
        #self.last_valid_x = self.x
        #self.last_valid_y = self.y
        #self.last_valid_theta = self.theta

        if self.draw_trail:
            self.trail.append(self.pos)
            if len(self.trail) > 1000:
                self.trail.pop(0)

        if self.draw_ghost and self.mu is not None:
            gp = Vector2(float(self.mu[0, 0]), float(self.mu[1, 0]))
            self.ghost_trail.append(gp)
        if len(self.ghost_trail) > 1000:
            self.ghost_trail.pop(0)
            
        # Calculate velocities from wheel velocities
        linear_velocity, angular_velocity = self.calculate_velocities()


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
        for bot in robots:
            if bot is self:
                continue
            if points_distance(new_pos, bot.pos) <= bot.radius + self.radius:
                offset_norm = (bot.pos - self.pos).normalize()
                tangent = Vector2(-offset_norm.y, offset_norm.x)
                self.move_vec = self.move_vec.dot(tangent) * tangent

        # update new position
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

        if robots:
            for bot in robots:
                if bot is self:
                    continue
                dist = points_distance(new_pos, bot.pos)
                if dist < bot.radius + self.radius:
                    push_dir = (new_pos - bot.pos).normalize()
                    new_pos += push_dir * (bot.radius + self.radius - dist)

        self.pos = new_pos
        self.x = self.pos.x
        self.y = self.pos.y
        self.update_sensors(obstacles, robots)


        if self.type != "RANDOM":
            self.vis_landmarks = {}
            for landmark in landmarks:
                lm_pos = landmark.get_pos()
                lm_dist = points_distance(self.pos, lm_pos)
                if lm_dist < self.lm_range:
                    r_vec = (lm_pos - self.pos).normalize()
                    bearing = math.radians(dir_vec.angle_to(r_vec))
                    self.vis_landmarks[tuple(landmark.pos)] = [lm_dist, bearing]

            self.kalman_localisation(linear_velocity, angular_velocity)

    def _ekf_predict(self, v: float, w: float, dt: float):
        """
        EKF prediction: propagate mean and covariance through motion model.
        """
        mu, Sigma = self.mu, self.Sigma
        theta = mu[2, 0]
        # State prediction
        mu[0, 0] += v * math.cos(theta) * dt
        mu[1, 0] += v * math.sin(theta) * dt
        mu[2, 0] = normalize_angle(theta - w * dt)
        # Jacobian G
        dim = mu.shape[0]
        G = np.eye(dim)
        G[0, 2] = -v * math.sin(theta) * dt
        G[1, 2] = v * math.cos(theta) * dt
        # Control noise, a1, 3 are for speed, a2, 4 are for angle
        alpha1, alpha2, alpha3, alpha4 = 0.1, 0.1, 0.1, 0.1
        R_control = np.array([[alpha1 * v ** 2 + alpha2 * w ** 2, 0],
                              [0, alpha3 * v ** 2 + alpha4 * w ** 2]])
        V = np.zeros((dim, 2))
        V[0, 0] = math.cos(theta) * dt
        V[1, 0] = math.sin(theta) * dt
        V[2, 1] = dt
        Sigma[...] = G.dot(Sigma).dot(G.T) + V.dot(R_control).dot(V.T)
        self.mu, self.Sigma = mu, Sigma

    def _ekf_update(self):
        """
        EKF update: incorporate range-bearing measurements for each observed landmark.
        """
        mu, Sigma = self.mu, self.Sigma
        Q = np.diag([self.sensor_noise_range ** 2, self.sensor_noise_bearing ** 2])
        dim = mu.shape[0]
        for lm_id, (r_meas, b_meas) in self.measurements.items():
            idx = 3 + 2 * lm_id
            # Initialize landmark if first seen
            if Sigma[idx, idx] > 1e2:
                mu[idx, 0] = mu[0, 0] + r_meas * math.cos(b_meas + mu[2, 0])
                mu[idx + 1, 0] = mu[1, 0] + r_meas * math.sin(b_meas + mu[2, 0])
            dx = mu[idx, 0] - mu[0, 0]
            dy = mu[idx + 1, 0] - mu[1, 0]
            q = dx * dx + dy * dy
            sqrt_q = math.sqrt(q)
            z_hat = np.array([[sqrt_q], [normalize_angle(math.atan2(dy, dx) - mu[2, 0])]])
            H = np.zeros((2, dim))
            H[0, 0] = -dx / sqrt_q
            H[0, 1] = -dy / sqrt_q
            H[1, 0] = dy / q
            H[1, 1] = -dx / q
            H[1, 2] = -1
            H[0, idx] = dx / sqrt_q
            H[0, idx + 1] = dy / sqrt_q
            H[1, idx] = -dy / q
            H[1, idx + 1] = dx / q
            z = np.array([[r_meas], [normalize_angle(b_meas)]])
            y = z - z_hat
            y[1, 0] = normalize_angle(y[1, 0])
            S = H.dot(Sigma).dot(H.T) + Q
            K = Sigma.dot(H.T).dot(np.linalg.inv(S))
            mu += K.dot(y)
            mu[2, 0] = normalize_angle(mu[2, 0])
            Sigma = (np.eye(dim) - K.dot(H)).dot(Sigma)
        self.mu, self.Sigma = mu, Sigma

    def draw(self, screen):
        # Draw the robot body
        pygame.draw.circle(screen, (255, 0, 0), self.pos, self.radius, 0)
        
        # Draw a line indicating the robot's orientation
        line_x = self.x + self.radius * math.cos(self.theta)
        line_y = self.y + self.radius * math.sin(self.theta)
        line_end = self.pos + Vector2(self.radius, 0).rotate_rad(self.theta)
        pygame.draw.line(screen, (0, 0, 0), self.pos, line_end, 3)

        if self.type != 'RANDOM':
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
                line_end = ghost_pos + Vector2(self.radius, 0).rotate_rad(-ghost_theta)
                pygame.draw.line(screen, (0, 0, 255), ghost_pos, line_end, 3)

                pos_cov = self.Sigma[:2, :2]
                eigenvalues, eigenvectors = LA.eig(pos_cov)
                order = eigenvalues.argsort()[::-1]
                eigenvalues = eigenvalues[order]
                eigenvectors = eigenvectors[:, order]

                angle = math.atan2(eigenvectors[1, 0], eigenvectors[0, 0])  # Orientation of ellipse
                angle_deg = math.degrees(angle)

                std_devs = 2.0 * np.sqrt(eigenvalues)  # 2-sigma ellipse
                ellipse_width = std_devs[0] * 2
                ellipse_height = std_devs[1] * 2
                ellipse_surf = pygame.Surface((ellipse_width, ellipse_height), pygame.SRCALPHA)
                pygame.draw.ellipse(ellipse_surf, (0, 0, 255, 50), (0, 0, ellipse_width, ellipse_height))

                rotated_ellipse = pygame.transform.rotate(ellipse_surf, -angle_deg)  # Pygame uses clockwise rotation
                ellipse_rect = rotated_ellipse.get_rect(center=(ghost_pos.x, ghost_pos.y))

                screen.blit(rotated_ellipse, ellipse_rect)

                if self.draw_trail:
                    for i in self.ghost_trail:
                        pygame.draw.circle(screen, (0, 0, 255), (int(i[0]), int(i[1])), 1)

    def get_sensor_data(self):
        sensor_data = []
        for sensor in self.sensors:
        # Calculate absolute angle of the sensor in world coordinates
            sensor_angle = self.theta + sensor.relative_angle
        # Check if sensor detected an obstacle
            hit = sensor.current_distance < sensor.max_range
        # Get endpoint coordinates
            endpoint = sensor.get_end_point(self)
        
            sensor_data.append((sensor_angle, sensor.current_distance, hit, endpoint))
    
        return sensor_data








