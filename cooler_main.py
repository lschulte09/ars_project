import numpy as np
import pygame
from MapEnvironment import MapEnvironment

def main():
    pygame.init()
    clock = pygame.time.Clock()
    env = MapEnvironment(1280, 720, num_obstacles=8, num_dust=15, num_landmarks=5)
    running = True

    env.place_robot()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                # Handle keyboard input
                env.handle_input(event)
        
        # Update the environment
        env.update()
        
        # Draw everything
        env.draw_screen()
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()