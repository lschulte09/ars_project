import sys

import pygame

from Controller import Controller
from EvolutionaryAlgorithm import EvolutionaryAlgorithm
from MapEnvironment import MapEnvironment
import heapq
import math

def a_star(og, start, goal):
    """
    Simple 4‐connected A* on an OccupancyGrid:
      - og.grid is a 2D array of occupancy probabilities
      - free if p < 0.3
      - start, goal are (i,j) grid‐indices
    Returns list of (i,j) from start→goal (inclusive), or [] if no path.
    """
    w, h = og.grid_width, og.grid_height
    def h_cost(a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    open_set = [(h_cost(start,goal), 0, start)]
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, cost, current = heapq.heappop(open_set)
        if current == goal:
            # reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nb = (current[0]+dx, current[1]+dy)
            if 0 <= nb[0] < w and 0 <= nb[1] < h:
                if og.grid[nb[0], nb[1]] < 0.3:
                    tentative = cost + 1
                    if tentative < g_score.get(nb, float('inf')):
                        g_score[nb] = tentative
                        priority = tentative + h_cost(nb, goal)
                        heapq.heappush(open_set, (priority, tentative, nb))
                        came_from[nb] = current

    return []  # no path found

def pd_control(x, y, th, tx, ty, L):
    """
    Goal→(tx,ty), current pose (x,y,th), wheel‐base L
    v = kρ·ρ,  ω = kα·α
    Returns (v, ω).
    """
    dx, dy = tx - x, ty - y
    p = math.hypot(dx, dy)
    dd = math.atan2(dy, dx)
    a = ((dd - th + math.pi) % (2*math.pi)) - math.pi

    kp = 0.8
    ka = 1.5

    v = kp * p
    w = ka * a
    return v, w

def main():
    pygame.init()
    env = MapEnvironment(1280, 720, num_obstacles=15, seed=42)
    controller = Controller()
    controller.load("best_individual/best_individual.npy")

    # record start pose for return‐home
    home_mu = env.robot.mu.copy()

    cleaning_done = False
    return_path = []
    wp_idx = 0

    running = True
    v_l_old, v_r_old = 0.0, 0.0

    while running:
        env.update_bot_controls()
        env.update()

        # check if all dust is cleaned
        if not cleaning_done and env.dust_collected >= env.all_dust:
            cleaning_done = True

            # plan a homeward path via A*
            start_cell = env.occupancy_grid.world_to_grid(
                env.robot.mu[0,0], env.robot.mu[1,0]
            )
            goal_cell = env.occupancy_grid.world_to_grid(
                home_mu[0,0], home_mu[1,0]
            )

            path_cells = a_star(
                env.occupancy_grid, start_cell, goal_cell
            )
            return_path = [
                env.occupancy_grid.grid_to_world(cx, cy)
                for (cx, cy) in path_cells
            ]
            wp_idx = 0

        # get sensor readings & pose-estimate
        readings   = env.robot.readings
        pos_x      = env.robot.mu[0,0]
        pos_y      = env.robot.mu[1,0]
        est_theta  = env.robot.mu[2,0]

        # choose control
        if not cleaning_done:
            # cleaning phase: use your EA‐evolved policy
            l_vel, r_vel = controller.out(
                readings, v_l_old, v_r_old, pos_x, pos_y, est_theta
            )
        else:
            # return-home phase: waypoint follower
            if wp_idx < len(return_path):
                tx, ty = return_path[wp_idx]
                v, w = pd_control(
                    pos_x, pos_y, est_theta, tx, ty,
                    env.robot.wheel_distance
                )
                # convert (v,ω) to left/right wheel speeds
                l_vel = v - 0.5 * w * env.robot.wheel_distance
                r_vel = v + 0.5 * w * env.robot.wheel_distance

                # advance waypoint if close
                if math.hypot(tx - pos_x, ty - pos_y) < env.robot.radius * 0.5:
                    wp_idx += 1
            else:
                # home reached → stop and exit
                l_vel, r_vel = 0.0, 0.0
                running = False

        # apply velocities
        env.robot.set_wheel_velocities(l_vel, r_vel)
        v_l_old, v_r_old = l_vel, r_vel

        # render
        env.render()
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()