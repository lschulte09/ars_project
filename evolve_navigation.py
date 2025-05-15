"""
evolve_navigation.py
====================

Self‑contained script that evolves a neural‑network controller for the
existing PyGame robot‑mapping simulator.  Run it from the project root:

    $ python evolve_navigation.py --train      # evolve on random mazes
    $ python evolve_navigation.py --demo best.npy   # watch best policy

You need numpy and pygame (already used by the simulator).  No other
dependencies.
"""

import os
#os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')  # headless training speed‑up

import argparse
import math
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pygame

from MapEnvironment import MapEnvironment


# ----------------------------------------------------------------------
# Policy (1‑hidden‑layer MLP → wheel velocities)
# ----------------------------------------------------------------------

class EvoPolicy:
    """Feed‑forward neural net evolved via a Genetic Algorithm."""

    def __init__(self, chromosome, n_inputs=12, n_hidden=8,
                 wheel_speed=30.0):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.wheel_speed = wheel_speed

        # decode chromosome -> weight matrices
        w1_len = (n_inputs + 1) * n_hidden                 # incl. bias
        self.w1 = chromosome[:w1_len].reshape(n_hidden, n_inputs + 1)
        self.w2 = chromosome[w1_len:].reshape(2, n_hidden + 1)

    # ------------------------------------------------------------------
    # static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def chrom_length(n_inputs=12, n_hidden=8):
        return (n_inputs + 1) * n_hidden + (n_hidden + 1) * 2

    @staticmethod
    def random_chrom(n_inputs=12, n_hidden=8):
        size = EvoPolicy.chrom_length(n_inputs, n_hidden)
        # Xavier‑style initialisation
        lim = math.sqrt(6 / (n_inputs + n_hidden))
        return np.random.uniform(-lim, lim, size).astype(np.float32)

    # ------------------------------------------------------------------
    # act
    # ------------------------------------------------------------------
    def act(self, sensor_inputs):
        """Convert 12 range readings [0,1] -> (v_left, v_right)."""
        x = np.append(sensor_inputs, 1.0)  # bias
        h = np.tanh(self.w1 @ x)
        h = np.append(h, 1.0)              # bias
        out = np.tanh(self.w2 @ h)         # ∈ [‑1,1]
        return out[0] * self.wheel_speed, out[1] * self.wheel_speed


# ----------------------------------------------------------------------
# Genetic Algorithm utilities
# ----------------------------------------------------------------------

def crossover(parent_a, parent_b):
    """1‑point crossover."""
    cut = random.randrange(1, len(parent_a) - 1)
    child = np.concatenate([parent_a[:cut], parent_b[cut:]]).copy()
    return child


def mutate(chrom, rate=0.05, strength=0.4):
    mask = np.random.rand(len(chrom)) < rate
    noise = np.random.normal(0, strength, mask.sum())
    chrom = chrom.copy()
    chrom[mask] += noise
    return chrom


def tournament(pop, fitnesses, k=3):
    """Return index of winner among *k* random contestants."""
    contestants = random.sample(range(len(pop)), k)
    best = max(contestants, key=lambda i: fitnesses[i])
    return best


# ----------------------------------------------------------------------
# Environment helpers
# ----------------------------------------------------------------------

def make_env(width=800, height=600):
    """Factory: new MapEnvironment with random poly obstacles."""
    env = MapEnvironment(
        width, height,
        num_obstacles=random.randint(5, 10),
        max_obstacle_size=random.randint(80, 140),
        num_landmarks=0,
        random_bots=0,
        draw_kalman=False,
        draw_occupancy_grid=True,
        obstacle_type='poly',
        slam_enabled=True
    )
    env.place_robot()
    # ensure auto control
    env.robot.type = 'AUTO'
    return env


def coverage(grid):
    """Fraction of cells that are either free or occupied
    (|p‑0.5| > 0.1)."""
    known = np.abs(grid - 0.5) > 0.1
    return known.mean()


def evaluate(chromosome, num_mazes=3, steps_per_episode=900):
    """Average fitness over *num_mazes* random environments."""
    total = 0.0
    for _ in range(num_mazes):
        env = make_env()
        policy = EvoPolicy(chromosome)
        robot = env.robot
        crashed = False

        for _ in range(steps_per_episode):
            # sensor vector normalised to [0,1]
            s_in = np.array([s.current_distance / s.max_range
                             for s in robot.sensors], dtype=np.float32)
            v_l, v_r = policy.act(s_in)
            robot.set_wheel_velocities(v_l, v_r)
            env.update()

            if getattr(robot, 'collided', False):
                crashed = True
                break

        cov = coverage(env.occupancy_grid.grid)
        fit = cov * (0.2 if crashed else 1.0)
        total += fit

        # tidy up pygame surfaces to avoid leak
        pygame.display.quit()
        pygame.display.init()

    return total / num_mazes


# ----------------------------------------------------------------------
# Main GA loop
# ----------------------------------------------------------------------

def run_ga(pop_size=40, generations=30,
           save_path='best.npy', rng_seed=None):
    if rng_seed is not None:
        random.seed(rng_seed)
        np.random.seed(rng_seed)

    n_inputs, n_hidden = 12, 8
    chrom_len = EvoPolicy.chrom_length(n_inputs, n_hidden)

    population = [EvoPolicy.random_chrom(n_inputs, n_hidden)
                  for _ in range(pop_size)]

    best_chrom = None
    best_score = -float('inf')

    for gen in range(generations):
        fitnesses = [evaluate(c) for c in population]

        # Record best
        idx_best = int(np.argmax(fitnesses))
        if fitnesses[idx_best] > best_score:
            best_score = fitnesses[idx_best]
            best_chrom = population[idx_best].copy()
            np.save(save_path, best_chrom)
            print(f'[Gen {gen}]  New best: {best_score:.3f}   '
                  f'→ saved to {save_path}')

        # Elitism: keep top 10 %
        num_elite = max(1, pop_size // 10)
        elite_indices = np.argsort(fitnesses)[-num_elite:]
        elite = [population[i] for i in elite_indices]

        # Build next generation
        next_pop = elite.copy()
        while len(next_pop) < pop_size:
            a = population[tournament(population, fitnesses)]
            b = population[tournament(population, fitnesses)]
            child = crossover(a, b)
            child = mutate(child)
            next_pop.append(child)

        population = next_pop

    print(f'Finished.  Best fitness = {best_score:.3f}')
    return best_chrom


# ----------------------------------------------------------------------
# Demo playback
# ----------------------------------------------------------------------

def demo(chrom_path):
    chrom = np.load(chrom_path)
    policy = EvoPolicy(chrom)
    env = make_env()
    env.draw_occupancy_grid = True  # visualise during demo
    robot = env.robot
    robot.type = 'AUTO'
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        s_in = np.array([s.current_distance / s.max_range
                         for s in robot.sensors], dtype=np.float32)
        v_l, v_r = policy.act(s_in)
        robot.set_wheel_velocities(v_l, v_r)

        env.update()
        env.draw_screen()
        pygame.display.set_caption('Evolved navigation demo '
                                   f'coverage: {coverage(env.occupancy_grid.grid):.2%}')
        pygame.display.flip() 
        clock.tick(60)

    pygame.quit()


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Evolve or demo robot navigation')
    parser.add_argument('--train', action='store_true',
                        help='run GA training')
    parser.add_argument('--demo', metavar='PATH',
                        help='play back a saved chromosome (.npy)')
    parser.add_argument('--pop', type=int, default=40,
                        help='population size (train)')
    parser.add_argument('--gen', type=int, default=30,
                        help='generations (train)')
    parser.add_argument('--out', default='best.npy',
                        help='where to save best chromosome (train)')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed')

    args = parser.parse_args()
    pygame.init()  # needed even in headless‑SDL mode

    if args.train:
        run_ga(pop_size=args.pop, generations=args.gen,
               save_path=args.out, rng_seed=args.seed)
    elif args.demo:
        demo(args.demo)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
