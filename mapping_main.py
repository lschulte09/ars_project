import pygame
import sys
from MapEnvironment import MapEnvironment

def main():
    pygame.init()
    clock = pygame.time.Clock()
    
    # Create environment with occupancy grid mapping enabled
    env = MapEnvironment(
        1280, 720,
        num_obstacles=8,
        num_landmarks=5,
        random_bots=2,
        draw_kalman=True,  # Enable Kalman filter visualization
        draw_occupancy_grid=True,  # Enable occupancy grid mapping
        slam_enabled=True,
        make_dust=True
    )
    
    running = True
    env.place_robot()
    env.place_bots()
    env.update_bot_controls()
    
    # For map saving functionality
    map_saved = False
    frames = 0
    while running:
        frames +=1
        if pygame.time.get_ticks() % 3000 == 0:
            env.update_bot_controls()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                # Handle keyboard input
                env.handle_input(event)
                
                # Save map with S key
                if event.key == pygame.K_p:
                    env.save_map("robot_map.npy")
                    map_saved = True
                    print("Map saved!")
                
                # Show map plot with P key
                if event.key == pygame.K_o:
                    env.plot_map()
        
        # Update the environment
        env.update()
        
        # Draw everything
        env.draw_screen()
        
        # Display "Map saved!" message if applicable
        if map_saved:
            font = pygame.font.SysFont('Arial', 24)
            text_surface = font.render("Map saved!", True, (0, 128, 0))
            env.screen.blit(text_surface, (env.width - 150, env.height - 30))
            
            # Reset the flag after a few seconds
            if frames % 300 == 0:
                map_saved = False
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()