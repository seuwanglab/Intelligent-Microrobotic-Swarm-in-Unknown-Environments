import numpy as np
import math
import torch
from torch import nn
import torch.nn.init as init
from .transformer import Transformer


@torch.jit.script
class _Normal:
    __constants__ = ['mean', 'std', 'LOG_2PI']

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.LOG_2PI = math.log(2 * math.pi)

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)

    def rsample(self):
        return self.mean + self.std * torch.randn_like(self.mean)

    def log_prob(self, value):
        return -0.5 * ((value - self.mean) / self.std).pow(2) - torch.log(self.std) - 0.5 * self.LOG_2PI

    def entropy(self):
        return 0.5 + 0.5 * (torch.log(self.std.pow(2)) + self.LOG_2PI)


def get_grad_norm(model):
    grads = {}
    grads['embedding_layer'] = calc_grad_norm(model.embedding_layer)
    transformer_blocks = model.transformer.transformer_blocks
    for i, block in enumerate(transformer_blocks):
        grads['transformer_block_' + str(i)] = calc_grad_norm(block)
    grads['policy_head'] = calc_grad_norm(model.policy_head)
    grads['value_head'] = calc_grad_norm(model.value_head)
    grads['model'] = calc_grad_norm(model)
    return grads


def calc_grad_norm(module):
    grads = []
    for name, parameter in module.named_parameters():
        if parameter.grad is not None:
            grads.append(parameter.grad.view(-1))
    return torch.linalg.norm(torch.cat(grads)).item() if len(grads) > 0 else None


class PPO(nn.Module):
    def __init__(self, config, observation_space, action_dim):
        super().__init__()
        self.hidden_size = config['transformer']['hidden_size']
        self.embedding_dim = config['transformer']['embedding_dim']
        self.observation_dim = observation_space.shape[0]
        self.action_dim = action_dim
        self.min_log_std = config['min_log_std']

        self.embedding_layer = nn.Sequential(
            nn.Linear(self.observation_dim, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.embedding_dim)
        )
        for layer in self.embedding_layer:
            if isinstance(layer, nn.Linear):
                init.orthogonal_(layer.weight, np.sqrt(2))

        self.transformer = Transformer(config['transformer'])

        self.policy_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.action_dim)
        )
        init.orthogonal_(self.policy_head[0].weight, np.sqrt(2))
        init.orthogonal_(self.policy_head[2].weight, np.sqrt(0.01))
        self.policy_log_std = nn.Parameter(torch.zeros(action_dim))

        self.value_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1)
        )
        init.orthogonal_(self.value_head[0].weight, np.sqrt(2))
        init.orthogonal_(self.value_head[2].weight, np.sqrt(1))

    def forward(self, observation, memory, memory_indices, mask):
        h = self.embedding_layer(observation)
        h, memory, attention_list = self.transformer(h, memory, memory_indices, mask)
        mean = self.policy_head(h)
        std = torch.exp(self.policy_log_std.clamp(min=self.min_log_std).expand_as(mean))
        pi = _Normal(mean, std)
        value = self.value_head(h).reshape(-1)
        return pi, value, memory, attention_list
