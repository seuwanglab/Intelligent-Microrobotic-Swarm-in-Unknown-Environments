import torch
import numpy as np


class Buffer:
    def __init__(self, config, observation_space, action_dim, device):
        self.device = device

        self.n_workers = config['n_workers']
        self.worker_steps = config['worker_steps']
        self.batch_size = self.n_workers * self.worker_steps

        self.n_mini_batch = config['n_mini_batch']
        self.mini_batch_size = self.batch_size // self.n_mini_batch

        self.num_blocks = config['transformer']['n_blocks']
        self.memory_length = config['transformer']['memory_length']

        self.rewards = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        self.dones = np.zeros((self.n_workers, self.worker_steps), dtype=bool)
        self.actions = torch.zeros((self.n_workers, self.worker_steps, action_dim), dtype=torch.float32)
        self.raw_actions = torch.zeros((self.n_workers, self.worker_steps, action_dim), dtype=torch.float32)
        self.observations = torch.zeros((self.n_workers, self.worker_steps) + observation_space.shape)
        self.log_probs = torch.zeros((self.n_workers, self.worker_steps))
        self.values = torch.zeros((self.n_workers, self.worker_steps))
        self.advantages = torch.zeros((self.n_workers, self.worker_steps))

        self.memory_mask = torch.zeros((self.n_workers, self.worker_steps, self.memory_length), dtype=torch.bool)
        self.memory_index = torch.zeros((self.n_workers, self.worker_steps), dtype=torch.long)
        self.memory_indices = torch.zeros((self.n_workers, self.worker_steps, self.memory_length), dtype=torch.long)

        self.memories = []
        self.flattened_samples = {}

    def prepare_batch_dict(self):
        samples = {
            'actions': self.actions,
            'raw_actions': self.raw_actions,
            'observation': self.observations,
            'log_probs': self.log_probs,
            'values': self.values,
            'advantages': self.advantages,
            'memory_mask': self.memory_mask,
            'memory_index': self.memory_index,
            'memory_indices': self.memory_indices
        }

        self.memories = torch.stack(self.memories, dim=0)

        self.flattened_samples = {}
        for key, value in samples.items():
            self.flattened_samples[key] = value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])

    def mini_batch_generator(self):
        indices = torch.randperm(self.batch_size)
        for start in range(0, self.batch_size, self.mini_batch_size):
            end = start + self.mini_batch_size
            mini_batch_indices = indices[start: end]
            mini_batch = {}
            for key, value in self.flattened_samples.items():
                if key == 'memory_index':
                    mini_batch['memories'] = self.memories[value[mini_batch_indices]]
                else:
                    mini_batch[key] = value[mini_batch_indices].to(self.device)
            yield mini_batch

    def compute_advantages(self, last_value, gamma, lamda):
        with torch.no_grad():
            last_advantage = 0
            mask = torch.tensor(self.dones).logical_not()
            rewards = torch.tensor(self.rewards)
            for t in reversed(range(self.worker_steps)):
                last_value = last_value * mask[:, t]
                last_advantage = last_advantage * mask[:, t]
                delta = rewards[:, t] + gamma * last_value - self.values[:, t]
                last_advantage = delta + gamma * lamda * last_advantage
                self.advantages[:, t] = last_advantage
                last_value = self.values[:, t]