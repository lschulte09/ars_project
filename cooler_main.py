import numpy as np
import pygame
from MapEnvironment import MapEnvironment

def main():

    pygame.init()
    clock = pygame.time.Clock()
    env = MapEnvironment(1280, 720, num_obstacles=8, num_dust=15)
    running = True

    env.place_robot()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        env.draw_screen()
        pygame.display.flip()
        clock.tick(60)

    # Place the robot at position (50, 50) with a 45° orientation.

    # Visualize the environment.
    # env.plot()


if __name__ == "__main__":
    main()