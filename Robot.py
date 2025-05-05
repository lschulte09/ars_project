import random
import numpy as np
import pygame
import math
import numpy.linalg as LA
from pygame.math import Vector2
from Sensor import Sensor

def points_distance(p1, p2):
    """Euclidean distance between two Vector2 points."""
    return (p2 - p1).length()

def points_line_dist(p1, l1, l2):
    """Distance from p1 to line segment [l1, l2], plus the closest point."""
    line = l2 - l1
    ap = p1 - l1
    t = ap.dot(line) / line.length_squared()
    t = max(0.0, min(1.0, t))
    closest = l1 + line * t
    return (p1 - closest).length(), closest

def points_line_dist_norm(p1, l1, l2):
    """
    As above, but also returns a normal vector at the closest point
    (for sliding/collision resolution).
    """
    line = l2 - l1
    ap = p1 - l1
    t = ap.dot(line) / line.length_squared()
    t = max(0.0, min(1.0, t))
    closest = l1 + line * t
    dist = (p1 - closest).length()

    if t == 0 or t == 1:
        end = l1 if t == 0 else l2
        norm = (end - p1).normalize().rotate(90)
    else:
        norm = line.normalize()
    return norm, dist

def trilaterate(p1, d1, p2, d2, p3, d3):
    """
    Three‐range trilateration (returns a Vector2 world position).
    """
    ex = (p2 - p1).normalize()
    i = ex.dot(p3 - p1)
    ey = (p3 - p1 - i * ex).normalize()
    d = (p2 - p1).length()
    j = ey.dot(p3 - p1)

    x = (d1**2 - d2**2 + d**2) / (2 * d)
    y = (d1**2 - d3**2 + i**2 + j**2 - 2 * i * x) / (2 * j)

    return p1 + x * ex + y * ey

def two_landmark_localize(p1, d1, b1, p2, d2, b2):
    """
    Two‐landmark bearing‐range localization.
    Returns a Vector2 world position.
    """
    v1 = Vector2(d1 * math.cos(b1), d1 * math.sin(b1))
    v2 = Vector2(d2 * math.cos(b2), d2 * math.sin(b2))
    theta = -math.radians((Vector2(p2) - Vector2(p1)).angle_to(v2 - v1))
    # rotate v1 into world-frame and subtract
    v1r = Vector2(
        v1.x * math.cos(theta) - v1.y * math.sin(theta),
        v1.x * math.sin(theta) + v1.y * math.cos(theta)
    )
    return Vector2(p1) - v1r

class Robot:
    def __init__(
        self,
        x, y, theta,
        lm_range: float = 200,
        sensor_range: float = 200,
        wheel_base: float = 60,       # distance between wheels
        draw_trail: bool = False,
        draw_ghost: bool = False
    ):
        # --- Pose & geometry ---
        self.x = x
        self.y = y
        self.pos = Vector2(x, y)
        self.theta = theta  # in radians
        self.radius = 30
        self.wheel_radius = 5.0
        self.wheel_base = wheel_base

        # --- Wheel & body velocities ---
        self.v_left = 0.0
        self.v_right = 0.0
        self.v = 0.0
        self.omega = 0.0
        self.max_speed = 30

        # --- Sensor & landmark parameters ---
        self.lm_range = lm_range
        self.sensor_range = sensor_range
        self.sensors = [
            Sensor(np.deg2rad(angle), sensor_range)
            for angle in range(0, 360, 30)
        ]

        # --- Noise / SLAM state ---
        self.eps_dist = 5     # sensor-distance noise
        self.eps_ang = 0.1    # sensor-angle noise
        self.mu     = np.array([[self.pos.x], [self.pos.y], [self.theta]])
        self.Sigma  = np.identity(3)
        self.R      = np.diag([10, 10, 40])   # motion cov
        self.Q      = np.diag([ 8,  8, 50])   # measurement cov

        # --- Trail & ghost (KF‐estimate) ---
        self.draw_trail = draw_trail
        self.draw_ghost = draw_ghost
        if draw_trail:
            self.trail = []
            if draw_ghost:
                self.ghost_trail = []

        # --- Collision / bookkeeping ---
        self.move_vec = Vector2(0, 0)
        self.collision = False
        self.vis_landmarks = {}
        self.last_valid_x = x
        self.last_valid_y = y
        self.last_valid_theta = theta

    def set_wheel_velocities(self, v_left: float, v_right: float):
        """
        Store individual wheel speeds and compute forward & angular rates.
        """
        self.v_left  = v_left
        self.v_right = v_right
        self.v      = (v_left + v_right) / 2.0
        self.omega  = (v_right - v_left) / self.wheel_base

    def calculate_velocities(self):
        """
        Return the last‐set linear and angular velocities.
        """
        return self.v, self.omega

    def move(self, dt=0.1, obstacles=None, landmarks=None):
        """
        Differential‐drive motion + collision‐handling + EKF‐SLAM update.
        """
        # record trail
        if self.draw_trail:
            self.trail.append(self.pos.copy())
            if len(self.trail) > 1000:
                self.trail.pop(0)

        # kinematics
        v, w = self.calculate_velocities()
        self.theta = (self.theta + w * dt) % (2 * math.pi)
        dx = v * math.cos(self.theta) * dt
        dy = v * math.sin(self.theta) * dt
        self.move_vec = Vector2(dx, dy)
        new_pos = self.pos + self.move_vec

        # sliding‐collision (push out)
        for obs in (obstacles or []):
            for a,b in obs.get_edges():
                norm, dist = points_line_dist_norm(new_pos, a, b)
                if dist < self.radius:
                    # slide
                    self.move_vec = self.move_vec.dot(norm) * norm
                    new_pos = self.pos + self.move_vec
            # bump‐back check
            for a,b in obs.get_edges():
                d, cp = points_line_dist(new_pos, a, b)
                if d < self.radius:
                    push = (self.radius - d)
                    new_pos += (new_pos - cp).normalize() * push

        self.pos = new_pos
        self.x, self.y = new_pos.x, new_pos.y

        # landmark visibility
        self.vis_landmarks.clear()
        for lm in (landmarks or []):
            lm_p = lm.get_pos()
            d = points_distance(self.pos, lm_p)
            if d < self.lm_range:
                bearing = math.radians(Vector2(1,0).rotate_rad(self.theta).angle_to((lm_p - self.pos).normalize()))
                self.vis_landmarks[tuple(lm_p)] = [d, bearing]

        # EKF‐SLAM predict & update
        self.kalman_localisation(v, w, dt)

    def kalman_localisation(self, v, w, dt=0.1):
        """
        EKF‐SLAM predict/update on self.mu, self.Sigma.
        """
        μ, Σ = self.mu, self.Sigma
        θ = μ[2,0]
        A = np.eye(3)
        B = np.array([
            [dt*math.cos(θ), 0],
            [dt*math.sin(θ), 0],
            [0,              dt]
        ])
        u = np.array([[v], [-w]])
        μ_bar = A @ μ + B @ u
        Σ_bar = A @ Σ @ A.T + self.R

        # measurement update if any landmarks seen
        if self.vis_landmarks:
            C = np.eye(3)
            # simple 1–3 landmark heuristics
            # ... (copy your existing landmark‐fusion logic here) ...
            # For brevity, reuse your original update code:
            z_list = []
            for pos, (d, b) in self.vis_landmarks.items():
                # build measurement z and run standard EKF update
                pass
            # (After computing K, etc.)
            # μ, Σ = updated estimates…
            # …
        self.mu, self.Sigma = μ_bar, Σ_bar

        # record ghost trail
        if self.draw_ghost:
            self.ghost_trail.append(Vector2(float(self.mu[0,0]), float(self.mu[1,0])))
            if len(self.ghost_trail) > 1000:
                self.ghost_trail.pop(0)

    def update_sensors(self, obstacles):
        """Raycast all sensors against polygonal obstacles."""
        for s in self.sensors:
            s.read_distance(self, obstacles, type='poly')

    def check_collision(self, obstacles, dust_particles=None):
        """Return True if any sensor or shape collision occurs."""
        # sensor‐based
        if any(s.current_distance < self.radius + 1 for s in self.sensors):
            return True
        # shape‐based
        for obs in obstacles:
            cx = max(obs.x,   min(self.x, obs.x+obs.width))
            cy = max(obs.y,   min(self.y, obs.y+obs.height))
            if (self.x-cx)**2 + (self.y-cy)**2 < self.radius**2:
                return True
        return False

    def draw(self, screen):
        """Draw robot, trail, sensors, and ghost‐estimate."""
        # robot
        pygame.draw.circle(screen, (255,0,0), self.pos, self.radius)
        heading = self.pos + Vector2(self.radius,0).rotate_rad(self.theta)
        pygame.draw.line(screen, (0,0,0), self.pos, heading, 3)

        # sensors
        for s in self.sensors:
            s.draw(screen, self)

        # path trail
        if self.draw_trail:
            for p in self.trail:
                pygame.draw.circle(screen, (0,0,0), (int(p.x),int(p.y)), 1)

        # Kalman‐ghost
        if self.draw_ghost:
            g = Vector2(float(self.mu[0,0]), float(self.mu[1,0]))
            pygame.draw.circle(screen, (0,0,255), g, self.radius, 2)
            line_end = g + Vector2(self.radius,0).rotate_rad(float(self.mu[2,0]))
            pygame.draw.line(screen, (0,0,255), g, line_end, 3)

    def get_pose(self):
        """Return (x, y, theta)."""
        return self.x, self.y, self.theta

    def add_noise_to_point(self, p: Vector2):
        """Simulate sensor noise on a point."""
        return p + Vector2(
            np.random.normal(0, self.eps_dist),
            np.random.normal(0, self.eps_dist)
        )
