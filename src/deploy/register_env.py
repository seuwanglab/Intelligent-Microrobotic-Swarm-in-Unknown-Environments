import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id='DynamicObstacleEnv-v0',
    entry_point='environment.dynamic_obstacle_env:DynamicObstacleEnvironment',
    max_episode_steps=512
)