import pygame

from EvolutionaryAlgorithm import EvolutionaryAlgorithm
from MapEnvironment import MapEnvironment



pygame.init()

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
    l_vel, r_vel = controller.out(readings, v_l_old, v_r_old)

    # log fitness data
    speed_avgs.append((l_vel + r_vel) / 2)
    speed_diffs.append(l_vel - r_vel)
    lowest_sens_reads.append(min(readings))
    env.robot.set_wheel_velocities(l_vel, r_vel)

    if frames >= sim_length:
        obs_collisions = env.robot.obs_collisions
        bot_collisions = env.robot.bot_collisions
        collision_ratio = (obs_collisions + bot_collisions) / sim_length
        coverage = env.dust_collected / env.all_dust
        V = sum(speed_avgs) / len(speed_avgs) / env.robot.max_speed
        delta_v = abs(sum(speed_diffs)) / env.robot.max_speed
        i = (sum(lowest_sens_reads) / len(lowest_sens_reads)) / env.robot.sensor_range
        running = False

pygame.quit()


