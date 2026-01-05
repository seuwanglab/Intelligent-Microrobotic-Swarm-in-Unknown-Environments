import torch
import numpy as np
from environment import DynamicObstacleEnvWrapper


def create_env(config, render_mode=None):
    if config['type'] == 'DynamicObstacle':
        return DynamicObstacleEnvWrapper(render_mode=render_mode)


def cosine_annealing_decay(initial, end, total_steps, step):
    if step >= total_steps or initial == end:
        return end
    cosine_decay = 0.5 * (1 + np.cos(np.pi * step / total_steps))
    return end + (initial - end) * cosine_decay


def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


def process_episode_info(episode_info):
    result = {}
    if len(episode_info) > 0:
        keys = episode_info[0].keys()
        for key in keys:
            values = [episode[key] for episode in episode_info]
            result[f'{key}_mean'] = np.mean(values)
            result[f'{key}_std'] = np.std(values)
    return result



