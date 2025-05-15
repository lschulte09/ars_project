import math
import random
import os
# os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame
import numpy as np
from Controller import Controller
from MapEnvironment import MapEnvironment


def calc_fitness(V, delta_v, i, collision_ratio, coverage):
    # Calculate fitness
    delta_v = min(delta_v, 1.0)
    phi = V*(1-math.sqrt(delta_v))*i

    fitness = (1 - collision_ratio) * coverage

    return fitness


def crossover(parent_1, parent_2):
    # Crossover parents by generating random values between both per gene
    genome_1 = parent_1['genome']
    genome_2 = parent_2['genome']

    child_genome = [np.random.uniform(min(genome_1[i], genome_2[i]), max(genome_1[i], genome_2[i])) for i in range(len(genome_1))]
    child_controller = Controller()
    child_controller.set_genome(child_genome)
    child = {
        'controller': Controller(),
        'genome': child_genome,
        'fitness': None
    }
    return child


class EvolutionaryAlgorithm:
    def __init__(self, population_size = 60, num_generations = 40, k = 10, elite = 5, mutation_rate = 0.1, mutation_strength = 0.1, environment = None):
        self.population_size = population_size
        self.num_generations = num_generations
        self.population = []
        self.best = None
        self.fitness = 0
        self.generation = 0
        self.init_population()

        self.k = k
        self.elite = elite
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength

        # if environment is none, the map will be randomly generated every simulation
        self.environment = environment

        self.best_dir = './best_individual'

    def init_population(self):
        for i in range(self.population_size):
            # Create individual
            controller = Controller()
            individual = {
                'controller': controller,
                'genome': controller.get_genome(),
                'fitness': None
            }
            self.population.append(individual)

    def evaluate_population(self):
        for individual in self.population:
            # Evaluate individual
            controller = individual['controller']
            V, delta_v, i, collision_ratio, coverage = self.simulate(controller)
            fitness = calc_fitness(V, delta_v, i, collision_ratio, coverage)

            individual['fitness'] = fitness


    def evolve(self):
        # do selection, crossover, mutation, get new population
        self.init_population()
        for generation in range(self.num_generations):
            print("Generation " + str(generation))
            self.evaluate_population()
            self.best = sorted(self.population, key=lambda ind: ind["fitness"], reverse=True)[0]
            print(f"Best fitness: {self.best['fitness']}")
            self.selection_crossover()
        self.evaluate_population()
        self.best = sorted(self.population, key=lambda ind: ind["fitness"], reverse=True)[0]
        print(f"Best fitness: {self.best['fitness']}")
        best_controller = self.best['controller']
        best_controller.save(self.best_dir, 'best_individual')

    def selection_crossover(self):
        # Select parents according to selection algorithm
        sorted_population = sorted(self.population, key=lambda ind: ind["fitness"], reverse=True)
        for individual in sorted_population:
            print(individual["fitness"])
        top_individuals = sorted_population[:self.elite]

        new_generation = top_individuals

        while len(new_generation) < self.population_size:
            parent_1 = self.tournament_selection()
            parent_2 = self.tournament_selection()

            child = crossover(parent_1, parent_2)
            mutated_child = self.mutation(child['controller'])
            child = {
                'controller': mutated_child,
                'genome': mutated_child.get_genome(),
                'fitness': None
            }
            new_generation.append(child)

        self.population = new_generation



    def tournament_selection(self):
        # Select k individuals randomly from pop, pick highest fitness one
        tournament = [random.choice(self.population) for _ in range(self.k)]
        sorted_tournament = sorted(tournament, key=lambda ind: ind["fitness"], reverse=True)
        return sorted_tournament[0]

    def mutation(self, controller):

        genome = controller.get_genome()
        for i in range(len(genome)):
            if random.random() < self.mutation_rate:
                genome[i] = np.random.normal(genome[i], self.mutation_strength)

        controller.set_genome(genome)

        return controller

    def simulate(self, controller, sim_length = 300):
        # Run one simulation
        pygame.init()

        env = MapEnvironment(1280, 720,
        num_obstacles=15,
        num_landmarks=7,
        random_bots=2,
        max_obstacle_size = 100,
        draw_kalman=False,
        draw_occupancy_grid=False,
        slam_enabled=False,
        make_dust=True
        )
        env.place_robot()
        env.place_bots()
        # if self.environment is not None:
        #     print(f"Loading environment: {self.environment}")
        #     env = env.load_env(self.environment)

        frames = 0
        running = True

        speed_avgs = []
        speed_diffs = []
        lowest_sens_reads = []
        V = 0
        delta_v = 0
        i = 0
        collision_ratio = 0
        coverage = 0

        env.update_bot_controls()
        while running:
            frames += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            env.update()
            readings = env.robot.readings
            v_l_old, v_r_old = env.robot.get_velocities()
            mu = env.robot.mu
            pos_x = mu[0,0]
            pos_y = mu[1,0]
            est_theta = mu[2,0]
            l_vel, r_vel = controller.out(readings, v_l_old, v_r_old, pos_x, pos_y, est_theta)

            # log fitness data
            speed_avgs.append((l_vel + r_vel) / 2)
            speed_diffs.append(l_vel - r_vel)
            lowest_sens_reads.append(min(readings))
            env.robot.set_wheel_velocities(l_vel, r_vel)

            if frames >= sim_length:
                obs_collisions = env.robot.obs_collisions
                bot_collisions = env.robot.bot_collisions
                collision_ratio = (obs_collisions + bot_collisions)/sim_length
                coverage = env.dust_collected/env.all_dust
                V = abs(sum(speed_avgs) / len(speed_avgs) / env.robot.max_speed)
                delta_v = abs(sum(speed_diffs) / len(speed_diffs)) / env.robot.max_speed
                i = (sum(lowest_sens_reads) / len(lowest_sens_reads)) / env.robot.sensor_range
                running = False

        pygame.quit()

        return V, delta_v, i, collision_ratio, coverage

if __name__ == "__main__":
    evolution = EvolutionaryAlgorithm(population_size=40, num_generations=30, elite=3, k=5, mutation_rate=0.01, mutation_strength=0.01)
    evolution.evolve()

