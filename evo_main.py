import sys

import pygame

from Controller import Controller
from EvolutionaryAlgorithm import EvolutionaryAlgorithm
from MapEnvironment import MapEnvironment

def main():

    pygame.init()
    clock = pygame.time.Clock()

    env = MapEnvironment(1280, 720,
                         num_obstacles=15,
                         num_landmarks=7,
                         random_bots=0,
                         max_obstacle_size=100,
                         draw_kalman=False,
                         draw_occupancy_grid=True,
                         slam_enabled=False,
                         make_dust=True
                         )
    env.place_robot()
    env.place_bots()

    env.load_env("maps/environment_244.pkl")
    controller = Controller()
    controller.load("best_individual/best_individual.npy")

    frames = 0
    running = True


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
        pos_x = mu[0, 0]
        pos_y = mu[1, 0]
        est_theta = mu[2, 0]
        l_vel, r_vel = controller.out(readings, v_l_old, v_r_old, pos_x, pos_y, est_theta)

        env.robot.set_wheel_velocities(l_vel, r_vel)

        env.draw_screen()
        pygame.display.flip()

        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()