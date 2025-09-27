import time
import gymnasium as gym
import numpy as np
import register_env


class DynamicObstacleEnvWrapper:
    def __init__(self, render_mode=None):
        self._env = gym.make('DynamicObstacleEnv-v0', render_mode=render_mode) \
            if render_mode else gym.make('DynamicObstacleEnv-v0')
        self.max_episode_steps = self._env.spec.max_episode_steps

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    def get_max_episode_steps(self):
        return self.max_episode_steps

    def reset(self):
        self._reward = []
        _observation, _info_vector = self._env.reset()
        return _observation, _info_vector

    def step(self, _action):
        _observation, _reward, _terminated, _truncated, _info_vector = self._env.step(_action)

        self._reward.append(_reward)
        _done = _terminated or _truncated

        return _observation, _reward, _done, _info_vector

    def render(self):
        self._env.render()
        time.sleep(0.05)

    def close(self):
        self._env.close()


if __name__ == "__main__":
    import numpy as np
    env = DynamicObstacleEnvWrapper(render_mode='human')
    observation, info_vector = env.reset()
    print(f'Initial observation: {observation}')
    print(f'Observation space: {env.observation_space}')
    print(env.action_space.shape)
    print(f'Action space: {env.action_space}')
    print(f'Max episode steps: {env.max_episode_steps}')

    done = False
    info = None
    step_count = 0

    while not done:
        action = np.clip(env.action_space.sample(), -1, 1)
        print(action)
        a = np.array([-1, -1], dtype=np.float32)
        next_observation, reward, done, info_vector = env.step(action)
        step_count += 1

        print(f'Step {step_count}:')
        print(f'  Action: {action}')
        print(f'  Observation: {next_observation}')
        print(f'  Reward: {reward}')
        print(f'  Info vector: {info_vector}')

        env.render()
    env.close()
