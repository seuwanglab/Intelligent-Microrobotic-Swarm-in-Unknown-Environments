import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import random


def is_overlapping(obs1, obs2):
    left1, right1 = obs1["x"] - obs1["w"] / 2, obs1["x"] + obs1["w"] / 2
    bottom1, top1 = obs1["y"] - obs1["h"] / 2, obs1["y"] + obs1["h"] / 2
    left2, right2 = obs2["x"] - obs2["w"] / 2, obs2["x"] + obs2["w"] / 2
    bottom2, top2 = obs2["y"] - obs2["h"] / 2, obs2["y"] + obs2["h"] / 2
    return (left1 < right2 and right1 > left2) and (bottom1 < top2 and top1 > bottom2)


def calculate_overlap(obs1, obs2):
    overlap_x = min(obs1['x'] + obs1['w'] / 2, obs2['x'] + obs2['w'] / 2) - \
                max(obs1['x'] - obs1['w'] / 2, obs2['x'] - obs2['w'] / 2)
    overlap_y = min(obs1['y'] + obs1['h'] / 2, obs2['y'] + obs2['h'] / 2) - \
                max(obs1['y'] - obs1['h'] / 2, obs2['y'] - obs2['h'] / 2)
    if overlap_x > 0 and overlap_y > 0:
        if overlap_x < overlap_y:
            return (overlap_x if obs1['x'] < obs2['x'] else -overlap_x), 0
        else:
            return 0, (overlap_y if obs1['y'] < obs2['y'] else -overlap_y)
    return 0, 0


def handle_obstacle_collisions(moving_obstacles, static_obstacles, dt=0.2):
    predicted_positions = []
    for obs in moving_obstacles:
        predicted = obs.copy()
        predicted['x'] += predicted['vx'] * dt
        predicted['y'] += predicted['vy'] * dt
        predicted_positions.append(predicted)
    for i in range(len(moving_obstacles)):
        for j in range(i + 1, len(moving_obstacles)):
            obs1 = moving_obstacles[i]
            obs2 = moving_obstacles[j]
            pred1 = predicted_positions[i]
            pred2 = predicted_positions[j]
            if is_overlapping(pred1, pred2):
                dx, dy = calculate_overlap(obs1, obs2)
                if dx != 0 or dy != 0:
                    obs1['x'] -= dx / 2
                    obs2['x'] += dx / 2
                    obs1['y'] -= dy / 2
                    obs2['y'] += dy / 2
                obs1['vx'], obs2['vx'] = obs2['vx'], obs1['vx']
                obs1['vy'], obs2['vy'] = obs2['vy'], obs1['vy']
                rel = np.array([obs2['x'] - obs1['x'], obs2['y'] - obs1['y']])
                mag = np.linalg.norm(rel) or 0.001
                norm = rel / mag
                impulse = 0.01
                obs1['x'] -= norm[0] * impulse
                obs1['y'] -= norm[1] * impulse
                obs2['x'] += norm[0] * impulse
                obs2['y'] += norm[1] * impulse
    for i, moving_obs in enumerate(moving_obstacles):
        pred = predicted_positions[i]
        for static_obs in static_obstacles:
            if is_overlapping(pred, static_obs):
                dx, dy = calculate_overlap(moving_obs, static_obs)
                if dx != 0 or dy != 0:
                    moving_obs['x'] -= dx
                    moving_obs['y'] -= dy
                if abs(dx) > abs(dy):
                    moving_obs['vx'] *= -1
                else:
                    moving_obs['vy'] *= -1
                rel = np.array([moving_obs['x'] - static_obs['x'], moving_obs['y'] - static_obs['y']])
                mag = np.linalg.norm(rel) or 0.001
                norm = rel / mag
                impulse = 0.01
                moving_obs['x'] += norm[0] * impulse
                moving_obs['y'] += norm[1] * impulse


def is_too_close(obstacle, points, min_distance):
    obstacle_center = np.array([obstacle['x'], obstacle['y']])
    points_arr = np.array(points)
    distances = np.linalg.norm(points_arr - obstacle_center, axis=1)
    return np.any(distances < min_distance)


def is_angle_in_interval(angle, lower, upper):
    if lower <= -np.pi:
        upper1 = upper
        lower1 = -np.pi

        upper2 = np.pi
        lower2 = 2 * np.pi + lower
        return (lower1 <= angle <= upper1) or (lower2 <= angle <= upper2)
    elif upper >= np.pi:
        upper1 = np.pi
        lower1 = lower

        upper2 = upper - 2 * np.pi
        lower2 = -np.pi
        return (lower1 <= angle <= upper1) or (lower2 <= angle <= upper2)
    else:
        return lower <= angle <= upper


class DynamicObstacleEnvironment(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 100}

    def __init__(self, render_mode=None):
        super().__init__()
        self.stay_steps = 0
        self.width = self.height = 800
        self.fps = 100
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        """
        action_{t-1}: last action
        observation_pos[x, y]: observation position
        observation_size[w, h]: observation size
        observation_velocity[v_x, v_y], observation velocity
        target_pos[x, y]: target position
        (obstacle_pos[x, y] obstacle_size[w, h]) * 4: information of the 5 nearest obstacles
        """
        self.observation_range = 400
        low = np.full(35, -np.inf, dtype=np.float32)
        high = np.full(35, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.max_steps = 512
        self.goal_degree = self.observation_r = self.region_stay_steps = 0
        self.total_grid = self.current_step = 0
        self.last_action = np.zeros(2, dtype=np.float32)
        self.velocity_range = self.last_region_center = self.region_center = self.observation_position = None
        self.observation_width = self.observation_height = 0
        self.observation_velocity = np.zeros(2, dtype=np.float32)
        self.initial_distance = self.last_distance = self.current_distance = 0
        self.static_obstacles = []
        self.moving_obstacles = []
        self.observation_angle = 0
        self.reward = 0
        self.at_goal = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.observation_r = 0
        self.current_step = 0
        self.region_stay_steps = 0
        self.last_action = np.zeros(2)
        low = np.array([-np.inf] * 35, dtype=np.float32)
        high = np.array([np.inf] * 35, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self._generate_positions()
        self._generate_obstacles()
        self.last_distance = self.current_distance = self.initial_distance = np.linalg.norm(
            self.observation_position - self.goal_position)
        self.goal_degree = np.arctan2(self.goal_position[1] - self.observation_position[1],
                                      self.goal_position[0] - self.observation_position[0])
        self.last_region_center = self.region_center = self.observation_position.copy()
        self.at_goal = False
        self.observation_angle = 0
        observation_information, _ = self._get_observation()
        return observation_information, self._get_information()

    def _get_observation(self):
        observation = np.array([
            self.last_action[0],
            self.last_action[1],
            self.observation_position[0] * np.random.normal(1, 0.025) / 40,
            self.observation_position[1] * np.random.normal(1, 0.025) / 40,
            self.observation_width * np.random.normal(1, 0.025) / 40,
            self.observation_height * np.random.normal(1, 0.025) / 40,
            self.observation_velocity[0] * np.random.normal(1, 0.025) / 40,
            self.observation_velocity[1] * np.random.normal(1, 0.025) / 40,
            self.current_distance / 40,
            self.goal_degree
        ], dtype=np.float32)

        all_obstacles = self.static_obstacles + self.moving_obstacles
        if len(all_obstacles) > 0:
            obs_array = np.array([[obs['x'], obs['y'], obs['w'], obs['h']] for obs in all_obstacles])
            positions = obs_array[:, :2]
            dists = np.linalg.norm(self.observation_position - positions, axis=1)
            sorted_indices = np.argsort(dists)
            n_selected = min(5, len(sorted_indices))
            sorted_indices = sorted_indices[:n_selected]
            sorted_obstacles_all = [all_obstacles[i] for i in sorted_indices]
            sorted_positions = positions[sorted_indices]
            sorted_ws = obs_array[sorted_indices, 2]
            sorted_hs = obs_array[sorted_indices, 3]
            sorted_dists = dists[sorted_indices]
            vectors = sorted_positions - self.observation_position
            angles = np.arctan2(vectors[:, 1], vectors[:, 0])
            noise_distance = np.random.normal(1, 0.025, size=sorted_dists.shape)
            noise_angle = np.random.normal(1, 0.025, size=angles.shape)
            noise_w = np.random.normal(1, 0.025, size=sorted_ws.shape)
            noise_h = np.random.normal(1, 0.025, size=sorted_hs.shape)
            in_range = sorted_dists <= (self.observation_range / 2)
            obs_features = np.zeros((5, 5), dtype=np.float32)
            obs_features[in_range, 0] = 1.0
            obs_features[in_range, 1] = sorted_dists[in_range] * noise_distance[in_range] / 40
            obs_features[in_range, 2] = angles[in_range] * noise_angle[in_range]
            obs_features[in_range, 3] = sorted_ws[in_range] * noise_w[in_range] / 40
            obs_features[in_range, 4] = sorted_hs[in_range] * noise_h[in_range] / 40
            sorted_obstacles = [obs for obs, valid in zip(sorted_obstacles_all, in_range) if valid]
        else:
            sorted_obstacles = []
            obs_features = np.zeros((5, 5), dtype=np.float32)

        obstacles_in_range = obs_features.flatten()[:25]
        observation = np.concatenate([observation, obstacles_in_range])
        return observation, sorted_obstacles

    def _get_information(self):
        return {
            'current_step': self.current_step,
            'reward': self.reward,
            'observation_position': self.observation_position,
            'goal_position': self.goal_position,
            'success': self.at_goal
        }

    def _update_moving_obstacles(self):
        if len(self.moving_obstacles) == 0:
            return
        data = np.array([[ob['x'], ob['y'], ob['vx'], ob['vy'], ob['w'], ob['h']] for ob in self.moving_obstacles])
        x, y, vx, vy, w, h = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5]
        noise_x = np.random.normal(1, 0.2, size=x.shape)
        noise_y = np.random.normal(1, 0.2, size=y.shape)
        dx = np.clip(vx * 0.2 * noise_x, -2.8, 2.8)
        dy = np.clip(vy * 0.2 * noise_y, -2.8, 2.8)
        x_new = x + dx
        y_new = y + dy
        min_x = (w + 2) / 2
        max_x = self.width - (w + 2) / 2
        min_y = (h + 2) / 2
        max_y = self.height - (h + 2) / 2
        lower_x = x_new < min_x
        upper_x = x_new > max_x
        x_new[lower_x] = min_x[lower_x]
        vx[lower_x] = -vx[lower_x]
        x_new[upper_x] = max_x[upper_x]
        vx[upper_x] = -vx[upper_x]
        lower_y = y_new < min_y
        upper_y = y_new > max_y
        y_new[lower_y] = min_y[lower_y]
        vy[lower_y] = -vy[lower_y]
        y_new[upper_y] = max_y[upper_y]
        vy[upper_y] = -vy[upper_y]
        for i, ob in enumerate(self.moving_obstacles):
            ob['x'] = x_new[i]
            ob['y'] = y_new[i]
            ob['vx'] = vx[i]
            ob['vy'] = vy[i]

    def step(self, action):
        self.reward = 0
        terminated = False
        self.current_step += 1
        self.observation_angle = (self.observation_angle + np.random.uniform(30, 60)) % 360
        speed = (action[0] + 1) / 2
        angle = action[1] * np.pi
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)
        self.observation_velocity = np.array([vx, vy]) * self.velocity_range * np.random.normal(1, 0.025)
        self.observation_position += self.observation_velocity * 0.2
        self.current_distance = np.linalg.norm(self.observation_position - self.goal_position)
        distance_progress = self.last_distance - self.current_distance
        self.last_distance = self.current_distance
        self.goal_degree = np.arctan2(self.goal_position[1] - self.observation_position[1],
                                      self.goal_position[0] - self.observation_position[0])
        last_speed_norm = (self.last_action[0] + 1) / 2
        last_angle = self.last_action[1] * np.pi
        vx_last = last_speed_norm * np.cos(last_angle)
        vy_last = last_speed_norm * np.sin(last_angle)
        energy = np.hypot(vx - vx_last, vy - vy_last)
        self.last_action = action

        observation_x, observation_y = self.observation_position
        observation_info = {'x': observation_x, 'y': observation_y,
                            'w': self.observation_width, 'h': self.observation_height}

        self._update_moving_obstacles()
        handle_obstacle_collisions(self.moving_obstacles, self.static_obstacles)

        observation_information, sorted_obstacles = self._get_observation()

        distance_to_border_x = min(self.observation_position[0], self.width - self.observation_position[0])
        distance_to_border_y = min(self.observation_position[1], self.height - self.observation_position[1])
        boundary_distance = min(distance_to_border_x, distance_to_border_y)
        safe_distance = max(self.observation_width, self.observation_height)
        if boundary_distance >= safe_distance:
            pass
        elif boundary_distance <= 0:
            self.reward = -50
            terminated = True
        else:
            self.reward += (boundary_distance/safe_distance - 1)

        if not terminated:
            if sorted_obstacles:
                if is_overlapping(observation_info, sorted_obstacles[0]):
                    self.reward = -50
                    terminated = True
                else:
                    min_distance = np.linalg.norm([sorted_obstacles[0]['x'] - observation_x,
                                                   sorted_obstacles[0]['y'] - observation_y])
                    if min_distance <= 175:
                        velocity_angle = np.arctan2(vy, vx)
                        for i, obstacle in enumerate(sorted_obstacles):
                            dx = obstacle['x'] - observation_x
                            dy = obstacle['y'] - observation_y
                            distance = np.linalg.norm([dy, dx])
                            obstacle_angle = np.arctan2(dy, dx)
                            risk_factor = 1 if distance < 100 else (2 - distance / 100)
                            scale = min_distance / distance
                            r_total = (obstacle['r'] + self.observation_r) * risk_factor
                            cone_half = np.arctan2(r_total, distance)
                            lower_bound = obstacle_angle - cone_half
                            upper_bound = obstacle_angle + cone_half
                            if is_angle_in_interval(velocity_angle, lower_bound, upper_bound):
                                self.reward -= 2 * risk_factor * scale / np.sqrt(i + 1)
                            else:
                                self.reward += (1 - risk_factor / 2) * scale / np.sqrt(i + 1) / 2
                    else:
                        self.reward += 0.5
            else:
                self.reward += 0.5

        if not terminated:
            if not self.at_goal:
                region_threshold = 192
                if not hasattr(self, 'region_center'):
                    self.region_center = self.observation_position.copy()
                    self.last_region_center = self.observation_position.copy()
                    self.region_stay_steps = 0
                displacement = np.linalg.norm(self.observation_position - self.region_center)
                if displacement < region_threshold:
                    self.region_stay_steps += 1
                else:
                    if np.linalg.norm(self.observation_position - self.last_region_center) >= 96:
                        self.last_region_center = self.region_center.copy()
                        self.region_center = self.observation_position.copy()
                        self.region_stay_steps = 1
                    else:
                        self.region_stay_steps += 1
                if self.region_stay_steps <= 72:
                    self.reward += 0.5 * (72 - self.region_stay_steps)/72
                else:
                    self.reward -= np.clip((self.region_stay_steps - 72) * 0.1, 0, 2)
            else:
                self.reward += 0.25

            self.reward += -0.25 * energy

            if self.current_distance >= 30:
                self.reward += (1 - self.current_distance / self.initial_distance)/3
                if distance_progress >= 0:
                    self.reward += distance_progress/10
                else:
                    self.reward += -distance_progress/50
            else:
                self.reward += 0.5 + (1 - self.current_distance/30) * 5
                if not self.at_goal and self.current_distance < 5:
                    self.reward += 100
                    self.at_goal = True
        truncated = self.current_step >= self.max_steps
        return observation_information, self.reward/100., terminated, truncated, self._get_information()

    def render(self):
        if self.render_mode is None:
            return

        if not hasattr(self, 'record_trajectory'):
            self.record_trajectory = True
            self.trajectory_points = []

        if self.window is None:
            pygame.init()
            if self.render_mode == 'human':
                self.window = pygame.display.set_mode((self.width * 2, self.height))
            else:
                self.window = pygame.Surface((self.width * 2, self.height))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas_global = pygame.Surface((self.width, self.height))
        canvas_global.fill((255, 255, 255))

        def draw_obstacles(canvas, obstacles):
            for obs in obstacles:
                x, y, w, h = obs['x'], obs['y'], obs['w'], obs['h']
                shape = obs['shape']
                if shape < 0.5:
                    pygame.draw.ellipse(canvas, (173, 193, 207),
                                        pygame.Rect(x - w / 2, y - h / 2, w, h))
                else:
                    pygame.draw.rect(canvas, (173, 193, 207),
                                     pygame.Rect(x - w / 2, y - h / 2, w, h))

        draw_obstacles(canvas_global, self.static_obstacles)
        draw_obstacles(canvas_global, self.moving_obstacles)

        target_surface = pygame.Surface((self.observation_width, self.observation_height), pygame.SRCALPHA)
        pygame.draw.ellipse(target_surface, (100, 100, 100), target_surface.get_rect())
        rotated_target = pygame.transform.rotate(target_surface, self.observation_angle)
        target_rect = rotated_target.get_rect(center=self.observation_position.astype(int))
        canvas_global.blit(rotated_target, target_rect.topleft)

        if self.record_trajectory:
            self.trajectory_points.append(self.observation_position.astype(int))
            if len(self.trajectory_points) > 1:
                pygame.draw.lines(canvas_global, (0, 255, 0), False, self.trajectory_points, 2)

        mask = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        mask.fill((0, 0, 0, 128))
        pygame.draw.circle(mask, (0, 0, 0, 0),
                           self.observation_position.astype(int),
                           int(self.observation_range / 2))
        canvas_global.blit(mask, (0, 0))

        goal_radius = 30
        num_points = 360
        total_length = 20
        dash_length = 10
        points = []
        for i in range(num_points):
            angle = np.radians(i)
            x = self.goal_position[0] + goal_radius * np.cos(angle)
            y = self.goal_position[1] + goal_radius * np.sin(angle)

            if (i % total_length) < dash_length:
                points.append((x, y))
            else:
                if len(points) > 1:
                    pygame.draw.lines(canvas_global, (255, 165, 0), False, points, 2)
                points = []

        if len(points) > 1:
            pygame.draw.lines(canvas_global, (255, 165, 0), False, points, 2)

        local_view_size = int(self.observation_range)
        obs_center = self.observation_position.astype(int)

        local_view = pygame.Surface((local_view_size, local_view_size))
        local_view.fill((0, 0, 0))

        extract_x = max(0, obs_center[0] - local_view_size // 2)
        extract_y = max(0, obs_center[1] - local_view_size // 2)
        extract_width = min(local_view_size, self.width - extract_x)
        extract_height = min(local_view_size, self.height - extract_y)

        extract_area = pygame.Rect(extract_x, extract_y, extract_width, extract_height)
        local_pos_x = max(0, local_view_size // 2 - (obs_center[0] - extract_x))
        local_pos_y = max(0, local_view_size // 2 - (obs_center[1] - extract_y))

        local_view.blit(canvas_global.subsurface(extract_area), (local_pos_x, local_pos_y))

        observed_rect = pygame.Rect(local_pos_x, local_pos_y, extract_width, extract_height)
        pygame.draw.rect(local_view, (255, 165, 0), observed_rect, 4)
        scale_factor = self.height / local_view_size
        local_view_scaled = pygame.transform.scale(local_view,
                                                   (int(local_view_size * scale_factor),
                                                    int(local_view_size * scale_factor)))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                exit()

        self.window.blit(canvas_global, (0, 0))
        self.window.blit(local_view_scaled, (self.width, 0))

        pygame.draw.line(self.window, (0, 0, 0), (self.width, 0), (self.width, self.height), 5)

        pygame.display.flip()
        self.clock.tick(self.metadata['render_fps'])
        return None

    def _generate_obstacles(self):
        avoid_points = [self.observation_position, self.goal_position]
        scenario = np.random.rand()
        if self.observation_width <= 35:
            total_obstacles = np.random.randint(18, 23)
        elif self.observation_width <= 40:
            total_obstacles = np.random.randint(16, 21)
        elif self.observation_width <= 45:
            total_obstacles = np.random.randint(14, 19)
        else:
            total_obstacles = np.random.randint(12, 17)
        if scenario < 0.1:
            num_static = total_obstacles
            num_moving = 0
        elif scenario > 0.9:
            num_static = 0
            num_moving = total_obstacles
        else:
            factor = np.random.uniform(0.1, 0.9)
            num_moving = int(np.round(total_obstacles * factor))
            num_static = total_obstacles - num_moving
        self.static_obstacles = []
        static_attempts = 0
        max_static_attempts = num_static * 20
        batch_size = 10
        while len(self.static_obstacles) < num_static and static_attempts < max_static_attempts:
            candidate_count = batch_size
            w_candidates = np.random.randint(40, 80, size=candidate_count, dtype=np.int16)
            h_candidates = np.clip((w_candidates * np.random.uniform(0.75, 1.25, size=candidate_count)).astype(np.int16),
                                   40, 80)
            r_candidates = np.maximum(w_candidates, h_candidates) * 0.65
            x_candidates = np.random.randint(0, self.width, size=candidate_count)
            y_candidates = np.random.randint(0, self.height, size=candidate_count)
            shapes = (np.random.rand(candidate_count) < 0.5).astype(int)
            for i in range(candidate_count):
                new_obstacle = {
                    'x': int(x_candidates[i]),
                    'y': int(y_candidates[i]),
                    'w': int(w_candidates[i]),
                    'h': int(h_candidates[i]),
                    'r': float(r_candidates[i]),
                    'shape': int(shapes[i])
                }
                new_obstacle_comp = {
                    'x': new_obstacle['x'],
                    'y': new_obstacle['y'],
                    'w': new_obstacle['w'] * 0.25,
                    'h': new_obstacle['h'] * 0.25
                }
                if is_too_close(new_obstacle, avoid_points, min_distance=150):
                    continue
                overlap = False
                for obs in self.static_obstacles:
                    if is_overlapping(new_obstacle_comp, obs):
                        overlap = True
                        break
                if not overlap:
                    self.static_obstacles.append(new_obstacle)
                    if len(self.static_obstacles) >= num_static:
                        break
            static_attempts += candidate_count
        self.moving_obstacles = []
        moving_attempts = 0
        max_moving_attempts = num_moving * 20
        while len(self.moving_obstacles) < num_moving and moving_attempts < max_moving_attempts:
            candidate_count = batch_size
            w_candidates = np.random.randint(40, 80, size=candidate_count)
            h_candidates = np.clip((w_candidates * np.random.uniform(0.75, 1.25, size=candidate_count)).astype(np.int16),
                                   35, 70)
            r_candidates = np.maximum(w_candidates, h_candidates) * 0.65
            x_candidates = np.random.randint(w_candidates, self.width - w_candidates, size=candidate_count)
            y_candidates = np.random.randint(h_candidates, self.height - h_candidates, size=candidate_count)
            vx_candidates = np.random.uniform(-14, 14, size=candidate_count)
            vy_candidates = np.random.uniform(-14, 14, size=candidate_count)
            shapes = (np.random.rand(candidate_count) < 0.5).astype(int)
            for i in range(candidate_count):
                new_obstacle = {
                    'x': int(x_candidates[i]),
                    'y': int(y_candidates[i]),
                    'w': int(w_candidates[i]),
                    'h': int(h_candidates[i]),
                    'r': float(r_candidates[i]),
                    'vx': float(vx_candidates[i]),
                    'vy': float(vy_candidates[i]),
                    'shape': int(shapes[i])
                }
                overlap = False
                for obs in self.static_obstacles + self.moving_obstacles:
                    if is_overlapping(new_obstacle, obs):
                        overlap = True
                        break
                if is_too_close(new_obstacle, avoid_points, min_distance=120):
                    overlap = True
                if not overlap:
                    self.moving_obstacles.append(new_obstacle)
                    if len(self.moving_obstacles) >= num_moving:
                        break
            moving_attempts += candidate_count

    def _generate_positions(self):
        self.observation_width = np.random.uniform(30, 50)
        self.observation_height = self.observation_width * np.random.uniform(0.9, 1.1)
        self.observation_r = max(self.observation_width, self.observation_height) * 0.65
        self.velocity_range = np.clip(20 * np.random.normal(1, 0.1), 17.5, 22.5)
        self.observation_velocity = np.zeros(2)
        quadrants = [
            (0, self.width / 2, 0, self.height / 2),
            (self.width / 2, self.width, 0, self.height / 2),
            (0, self.width / 2, self.height / 2, self.height),
            (self.width / 2, self.width, self.height / 2, self.height)
        ]
        observation_quadrant = random.choice(quadrants)
        if observation_quadrant[1] <= 401 and observation_quadrant[3] <= 401:
            self.observation_position = np.array([
                np.random.uniform(observation_quadrant[0] + 75, observation_quadrant[0] + 225),
                np.random.uniform(observation_quadrant[2] + 75, observation_quadrant[2] + 225)
            ])
        elif observation_quadrant[1] >= 799 and observation_quadrant[3] <= 401:
            self.observation_position = np.array([
                np.random.uniform(observation_quadrant[1] - 225, observation_quadrant[1] - 75),
                np.random.uniform(observation_quadrant[2] + 75, observation_quadrant[2] + 225)
            ])
        elif observation_quadrant[1] <= 401 and observation_quadrant[3] >= 799:
            self.observation_position = np.array([
                np.random.uniform(observation_quadrant[0] + 75, observation_quadrant[0] + 225),
                np.random.uniform(observation_quadrant[3] - 225, observation_quadrant[3] - 75)
            ])
        else:
            self.observation_position = np.array([
                np.random.uniform(observation_quadrant[1] - 225, observation_quadrant[1] - 75),
                np.random.uniform(observation_quadrant[3] - 225, observation_quadrant[3] - 75)
            ])
        available_quadrants = [q for q in quadrants if q != observation_quadrant]
        attempt = 0
        max_attempts = 1000
        goal_distance = 600
        self.goal_position = np.array([400, 400])
        while attempt < max_attempts:
            goal_quadrant = random.choice(available_quadrants)
            goal_x = np.random.uniform(goal_quadrant[0] + 75, goal_quadrant[1] - 75)
            goal_y = np.random.uniform(goal_quadrant[2] + 75, goal_quadrant[3] - 75)
            goal_position = np.array([goal_x, goal_y])
            if np.linalg.norm(goal_position - self.observation_position) >= goal_distance:
                self.goal_position = goal_position
                break
            attempt += 1

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
