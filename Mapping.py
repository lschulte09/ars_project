# mapping.py
import math
import numpy as np
from abc import ABC, abstractmethod

class Mapping(ABC):
    """
    Abstract base class for map representations
    """
    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def draw(self, screen):
        pass

class OccupancyGridMapping(Mapping):
    """
    Concrete occupancy grid mapping using log-odds.
    """
    def __init__(self, width, height, resolution, p0=0.5, p_occ=0.7, p_free=0.3):
        # World size and discretization
        self.width = width
        self.height = height
        self.resolution = resolution
        self.cols = int(math.ceil(width / resolution))
        self.rows = int(math.ceil(height / resolution))

        # Log-odds parameters
        self.l0 = self._logit(p0)
        self.l_occ = self._logit(p_occ)
        self.l_free = self._logit(p_free)

        # Initialize grid to prior
        self.log_odds = np.ones((self.rows, self.cols), dtype=float) * self.l0

    def _logit(self, p):
        return math.log(p / (1 - p))

    def _inv_logit(self, l):
        return 1.0 - 1.0/(1.0 + np.exp(l))

    def _world_to_grid(self, x, y):
        c = int(x / self.resolution)
        r = int(y / self.resolution)
        return r, c

    def update(self, pose, sensor_readings):
        x, y, theta = pose
        for rel_angle, dist in sensor_readings:
            alpha = theta + rel_angle
            # free-space along ray
            num_steps = int(dist / self.resolution)
            for step in range(num_steps):
                px = x + step * self.resolution * math.cos(alpha)
                py = y + step * self.resolution * math.sin(alpha)
                r, c = self._world_to_grid(px, py)
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    self.log_odds[r, c] += (self.l_free - self.l0)
            # occupied at end
            ox = x + dist * math.cos(alpha)
            oy = y + dist * math.sin(alpha)
            r, c = self._world_to_grid(ox, oy)
            if 0 <= r < self.rows and 0 <= c < self.cols:
                self.log_odds[r, c] += (self.l_occ - self.l0)

    def draw(self, screen):
        import pygame
        for r in range(self.rows):
            for c in range(self.cols):
                p = self._inv_logit(self.log_odds[r, c])
                shade = int((1.0 - p) * 255)
                rect = pygame.Rect(c*self.resolution, r*self.resolution,
                                   self.resolution, self.resolution)
                pygame.draw.rect(screen, (shade, shade, shade), rect)

class EKFSLAMMapping(Mapping):
    """
    Simple EKF-SLAM: state includes robot pose + landmarks up to max_landmarks.
    """
    def __init__(self, initial_pose, Q, R, max_landmarks=50):
        # 3 state for robot + 2*max_landmarks for landmarks
        self.max_landmarks = max_landmarks
        dim = 3 + 2*max_landmarks
        self.mu = np.zeros(dim)
        self.mu[0:3] = initial_pose
        self.Sigma = np.eye(dim) * 1e6
        self.Sigma[0:3,0:3] = np.eye(3) * 1e-3
        self.Q = Q  # motion noise (3x3)
        self.R = R  # measurement noise (2x2)
        self.num_landmarks = 0

    def predict(self, u):
        """
        Prediction step with control u = (v, omega, dt).
        """
        x, y, theta = self.mu[0], self.mu[1], self.mu[2]
        v, omega, dt = u
        if abs(omega) > 1e-3:
            x_new = x + -v/omega * math.sin(theta) + v/omega * math.sin(theta + omega*dt)
            y_new = y + v/omega * math.cos(theta) - v/omega * math.cos(theta + omega*dt)
        else:
            x_new = x + v * dt * math.cos(theta)
            y_new = y + v * dt * math.sin(theta)
        theta_new = theta + omega*dt
        # State transition Jacobian approx identity for all but robot
        dim = len(self.mu)
        G = np.eye(dim)
        # robot noise mapping
        Fxr = np.hstack((np.eye(3), np.zeros((3, dim-3))))
        # update
        self.mu[0:3] = [x_new, y_new, theta_new]
        self.Sigma = G.dot(self.Sigma).dot(G.T) + Fxr.T.dot(self.Q).dot(Fxr)

    def update(self, u, sensor_readings):
        """
        Full SLAM update: first predict, then incorporate sensor_readings:
        sensor_readings: list of (rng, bearing) measurements, indexed by order.
        New landmarks initialized in sequence.
        """
        # 1) predict
        self.predict(u)
        # 2) update for each measurement
        for idx, (rng, bearing) in enumerate(sensor_readings):
            # assign landmark id = idx
            lm_id = idx
            # init if new
            if lm_id >= self.num_landmarks and self.num_landmarks < self.max_landmarks:
                # global bearing
                b = self.mu[2] + bearing
                lx = self.mu[0] + rng * math.cos(b)
                ly = self.mu[1] + rng * math.sin(b)
                self.mu[3+2*self.num_landmarks:3+2*self.num_landmarks+2] = [lx, ly]
                self.num_landmarks += 1
            # compute expected measurement
            id_ = lm_id
            lm = self.mu[3+2*id_:3+2*id_+2]
            dx = lm[0] - self.mu[0]
            dy = lm[1] - self.mu[1]
            q = dx**2 + dy**2
            z_hat = np.array([math.sqrt(q), math.atan2(dy, dx) - self.mu[2]])
            # Jacobian H
            dim = len(self.mu)
            Fxj = np.zeros((5, dim))
            Fxj[0:3,0:3] = np.eye(3)
            Fxj[3:5,3+2*id_:3+2*id_+2] = np.eye(2)
            # measurement Jacobian simplified
            H_low = np.array([[-dx/math.sqrt(q), -dy/math.sqrt(q), 0, dx/math.sqrt(q), dy/math.sqrt(q)],
                              [dy/q, -dx/q, -1, -dy/q, dx/q]])
            H = H_low.dot(Fxj)
            S = H.dot(self.Sigma).dot(H.T) + self.R
            K = self.Sigma.dot(H.T).dot(np.linalg.inv(S))
            z = np.array([rng, bearing])
            self.mu = self.mu + K.dot(z - z_hat)
            self.Sigma = (np.eye(dim) - K.dot(H)).dot(self.Sigma)

    def draw(self, screen):
        """
        Draw robot pose and landmarks.
        """
        import pygame
        # landmarks
        for i in range(self.num_landmarks):
            x_l, y_l = self.mu[3+2*i], self.mu[3+2*i+1]
            pygame.draw.circle(screen, (255, 0, 0), (int(x_l), int(y_l)), 4)
        # robot
        x, y, theta = self.mu[0], self.mu[1], self.mu[2]
        end = (x + 10*math.cos(theta), y + 10*math.sin(theta))
        pygame.draw.line(screen, (0, 255, 0), (int(x), int(y)), (int(end[0]), int(end[1])), 2)
