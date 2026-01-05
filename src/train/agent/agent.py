import os
import time
import numpy as np
from collections import deque

import torch
import torch.optim as optim

from .ppo import PPO, get_grad_norm
from utils import Worker
from utils import Buffer
from utils import create_env, cosine_annealing_decay, process_episode_info, batched_index_select

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from torch.utils.tensorboard import SummaryWriter


class Agent:
    def __init__(self, config, model_id, device):
        self.config = config
        self.model_id = model_id
        self.device = device

        print('Step 1: Initialization')
        print('=================================================')
        self.n_updates = config['n_updates']
        self.n_workers = config['n_workers']
        self.n_blocks = config['transformer']['n_blocks']
        self.memory_length = config['transformer']['memory_length']
        self.embedding_dim = config['transformer']['embedding_dim']
        self.lr_schedule = config['learning_rate_schedule']
        self.cr_schedule = config['clip_range_schedule']
        self.ec_schedule = config['entropy_coef_schedule']
        self.vl_schedule = config['value_loss_coef_schedule']
        self.kl_schedule = config['kl_coef_schedule']
        self.k_epoch = config['epochs']

        if not os.path.exists('../logs'):
            os.makedirs('../logs')
        timestamp = time.strftime('/%Y%m%d-%H%M%S' + "/")
        self.writer = SummaryWriter('./logs/' + model_id + timestamp)

        env = create_env(self.config['environment'])
        self.observation_space = env.observation_space
        self.action_dim = env.action_space.shape[0]
        self.max_episode_steps = env.max_episode_steps
        env.close()

        self.buffer = Buffer(self.config, self.observation_space, self.action_dim, self.device)

        self.model = torch.jit.script(PPO(self.config, self.observation_space, self.action_dim).to(self.device))
        print('model structure: \n', self.model)
        print('=================================================')
        print('Number of parameters: \n', sum(p.numel() for p in self.model.parameters()))
        print('=================================================')
        self.model.train()
        self.optimizer = optim.AdamW([
            {'params': self.model.embedding_layer.parameters(), 'lr': self.lr_schedule['start']},
            {'params': [p for name, p in self.model.named_parameters() if 'embedding_layer' not in name],
             'lr': self.lr_schedule['start']}
        ], weight_decay=1e-2)

        self.workers = [Worker(self.config['environment']) for _ in range(self.n_workers)]
        self.worker_ids = range(self.n_workers)
        self.worker_current_episode_steps = torch.zeros((self.n_workers,), dtype=torch.long)
        self.observation = np.zeros((self.config['n_workers'],) + self.observation_space.shape, dtype=np.float32)

        for worker in self.workers:
            worker.child.send(('reset', None))
        for i, worker in enumerate(self.workers):
            self.observation[i] = worker.child.recv()

        repetitions = torch.repeat_interleave(torch.arange(0, self.memory_length).unsqueeze(0),
                                              self.memory_length - 1, dim=0).long()
        self.memory = torch.zeros((self.n_workers, self.max_episode_steps, self.n_blocks, self.embedding_dim),
                                  dtype=torch.float32)
        self.memory_mask = torch.tril(torch.ones((self.memory_length, self.memory_length)))

        self.memory_indices = torch.stack([torch.arange(i, i + self.memory_length)
                                           for i in range(self.max_episode_steps - self.memory_length + 1)]).long()
        self.memory_indices = torch.cat((repetitions, self.memory_indices))

    def train(self):
        print('Step 2: Start training using ' + str(self.device))
        print('=================================================')
        episode_info = deque(maxlen=100)
        for update in range(self.n_updates):
            pre_stage = int(self.n_updates * 0.875)
            if update < pre_stage:
                learning_rate = cosine_annealing_decay(self.lr_schedule['start'], self.lr_schedule['end'],
                                                       pre_stage, update)
                clip_range = cosine_annealing_decay(self.cr_schedule['start'], self.cr_schedule['end'],
                                                    pre_stage, update)
                entropy_coef = cosine_annealing_decay(self.ec_schedule['start'], self.ec_schedule['end'],
                                                      pre_stage, update)
                value_loss_coefficient = 1 - cosine_annealing_decay(self.vl_schedule['start'], self.vl_schedule['end'],
                                                                    pre_stage, update)
                kl_coef = cosine_annealing_decay(self.kl_schedule['start'], self.kl_schedule['end'],
                                                 pre_stage, update)

                for pg in self.optimizer.param_groups:
                    pg['lr'] = learning_rate

            else:
                self.k_epoch = 5
                learning_rate = self.lr_schedule['end']
                clip_range = self.cr_schedule['end']
                entropy_coef = self.ec_schedule['end']
                value_loss_coefficient = 1 - self.vl_schedule['end']
                kl_coef = self.kl_schedule['end']

                self.optimizer.param_groups[0]['lr'] = learning_rate / 5
                self.optimizer.param_groups[1]['lr'] = learning_rate

            print(f'Update {update + 1}/{self.n_updates}, lr: {learning_rate}, cr: {clip_range}, ec: {entropy_coef}, '
                  f'vl: {value_loss_coefficient}, kl: {kl_coef}')

            sampled_episode_info = self._sample_training_data()

            self.buffer.prepare_batch_dict()

            training_stats, grad_info = self._train_epochs(learning_rate, clip_range, entropy_coef,
                                                           value_loss_coefficient, kl_coef)
            training_stats = np.mean(training_stats, axis=0)

            episode_info.extend(sampled_episode_info)
            episode_result = process_episode_info(episode_info)

            result = (('reward={:.4f} std={:.4f} '
                       'length={:.4f} std={:.4f} \n'
                       'pi_loss={:4f} v_loss={:4f} loss={:4f} '
                       'entropy={:.4f} value={:.4f} advantage={:.4f} \n').
                      format(episode_result['reward_mean'], episode_result['reward_std'],
                             episode_result['length_mean'], episode_result['length_std'],
                             training_stats[0], training_stats[1], training_stats[2], training_stats[3],
                             torch.mean(self.buffer.values), torch.mean(self.buffer.advantages)))
            print(result)
            print('=================================================')
            self._write_gradient_summary(update, grad_info)
            self._write_training_summary(update, training_stats, episode_result)
            if update % 20 == 0:
                self._save_model(update=update)

        self._save_model()

    def _sample_training_data(self):
        episode_infos = []

        self.buffer.memories = [self.memory[w] for w in range(self.n_workers)]
        for w in range(self.n_workers):
            self.buffer.memory_index[w] = w

        for t in range(self.config['worker_steps']):
            with torch.no_grad():
                self.buffer.observations[:, t] = torch.tensor(self.observation)
                self.buffer.memory_mask[:, t] = self.memory_mask[
                    torch.clip(self.worker_current_episode_steps, 0, self.memory_length - 1)]
                self.buffer.memory_indices[:, t] = self.memory_indices[self.worker_current_episode_steps]

                sliced_memory = batched_index_select(self.memory, 1, self.buffer.memory_indices[:, t])

                policy, value, memory = self.model(torch.tensor(self.observation), sliced_memory,
                                                   self.buffer.memory_indices[:, t], self.buffer.memory_mask[:, t])
                raw_action = policy.rsample()
                action = torch.tanh(raw_action)
                log_prob = (policy.log_prob(raw_action) - torch.log(1 - action.pow(2) + 1e-5)).sum(dim=1)

                self.memory[self.worker_ids, self.worker_current_episode_steps] = memory
                self.buffer.actions[:, t] = action
                self.buffer.raw_actions[:, t] = raw_action
                self.buffer.log_probs[:, t] = log_prob
                self.buffer.values[:, t] = value

            for w, worker in enumerate(self.workers):
                worker.child.send(('step', self.buffer.actions[w, t].cpu().numpy()))

            for w, worker in enumerate(self.workers):
                observation, self.buffer.rewards[w, t], self.buffer.dones[w, t], info = worker.child.recv()
                if info:
                    self.worker_current_episode_steps[w] = 0
                    episode_infos.append(info)

                    worker.child.send(('reset', None))

                    observation = worker.child.recv()

                    mem_index = self.buffer.memory_index[w, t]

                    self.buffer.memories[mem_index] = self.buffer.memories[mem_index].clone()

                    self.memory[w] = torch.zeros((self.max_episode_steps, self.n_blocks, self.embedding_dim),
                                                 dtype=torch.float32)
                    if t < self.config['worker_steps'] - 1:
                        self.buffer.memories.append(self.memory[w])
                        self.buffer.memory_index[w, t + 1:] = len(self.buffer.memories) - 1
                else:
                    self.worker_current_episode_steps[w] += 1
                self.observation[w] = observation

        last_value = self.get_last_value()
        self.buffer.compute_advantages(last_value, self.config['gamma'], self.config['lamda'])
        return episode_infos

    def get_last_value(self):
        start = torch.clip(self.worker_current_episode_steps - self.memory_length, 0)
        end = torch.clip(self.worker_current_episode_steps, self.memory_length)
        indices = torch.stack([torch.arange(start[i], end[i]) for i in range(self.n_workers)]).long()
        sliced_memory = batched_index_select(self.memory, 1, indices)
        _, last_value, _ = self.model(torch.tensor(self.observation), sliced_memory,
                                      self.buffer.memory_indices[:, -1],
                                      self.memory_mask[torch.clip(self.worker_current_episode_steps, 0,
                                                                  self.memory_length - 1)].bool())
        return last_value

    def _train_epochs(self, learning_rate, clip_range, entropy_coef, value_loss_coefficient, kl_coef):
        train_info, grad_info = [], {}
        for _ in range(self.k_epoch):
            mini_batch_generator = self.buffer.mini_batch_generator()
            for mini_batch in mini_batch_generator:
                train_info.append(
                    self._train_mini_batch(mini_batch, learning_rate, clip_range, entropy_coef, value_loss_coefficient,
                                           kl_coef))
                for key, value in get_grad_norm(self.model).items():
                    grad_info.setdefault(key, []).append(value)
        return train_info, grad_info

    def _train_mini_batch(self, samples, learning_rate, clip_range, entropy_coef, value_loss_coefficient, kl_coef):
        memory = batched_index_select(samples['memories'], 1, samples['memory_indices'])
        policy, value, _ = self.model(samples['observation'], memory, samples['memory_indices'],
                                      samples['memory_mask'])

        actions = samples['actions']
        raw_actions = samples['raw_actions']

        log_probs = (policy.log_prob(raw_actions) - torch.log(1 - actions.pow(2) + 1e-5)).sum(dim=1)
        entropies = (policy.entropy() - torch.log(1 - actions.pow(2) + 1e-5)).sum(dim=1)

        advantages = samples['advantages']
        normalized_advantage = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        log_ratio = log_probs - samples['log_probs']
        ratio = torch.exp(log_ratio)

        surr1 = ratio * normalized_advantage
        surr2 = torch.clamp(ratio, 1. - clip_range, 1. + clip_range) * normalized_advantage
        policy_loss = torch.min(surr1, surr2).mean()
        sampled_return = samples['values'] + advantages
        clipped_value = samples['values'] + (value - samples['values']).clamp(-clip_range, clip_range)
        vf_loss1 = (value - sampled_return) ** 2
        vf_loss2 = (clipped_value - sampled_return) ** 2
        vf_loss = torch.max(vf_loss1, vf_loss2).mean()
        entropy_bonus = entropies.mean()

        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        loss = -(policy_loss - value_loss_coefficient * vf_loss + entropy_coef * entropy_bonus - kl_coef * approx_kl)

        for pg in self.optimizer.param_groups:
            pg['lr'] = learning_rate
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config["max_grad_norm"])
        self.optimizer.step()

        clip_fraction = (abs((ratio - 1.)) > clip_range).float().mean()

        return [policy_loss.cpu().data.numpy(),
                vf_loss.cpu().data.numpy(),
                loss.cpu().data.numpy(),
                entropy_bonus.cpu().data.numpy(),
                approx_kl.cpu().data.numpy(),
                clip_fraction.cpu().data.numpy()]

    def _write_training_summary(self, update, training_stats, episode_result):
        if episode_result:
            for key in episode_result:
                if 'std' not in key:
                    self.writer.add_scalar("episode/" + key, episode_result[key], update)
        self.writer.add_scalar('losses/loss', training_stats[2], update)
        self.writer.add_scalar('losses/policy_loss', training_stats[0], update)
        self.writer.add_scalar('losses/value_loss', training_stats[1], update)
        self.writer.add_scalar('losses/entropy', training_stats[3], update)
        self.writer.add_scalar('training/value_mean', torch.mean(self.buffer.values), update)
        self.writer.add_scalar('training/advantage_mean', torch.mean(self.buffer.advantages), update)
        self.writer.add_scalar('other/clip_fraction', training_stats[4], update)
        self.writer.add_scalar('other/kl', training_stats[5], update)

    def _write_gradient_summary(self, update, grad_info):
        for key, value in grad_info.items():
            self.writer.add_scalar('gradients/' + key, np.mean(value), update)

    def _save_model(self, update=None):
        if not os.path.exists('./models'):
            os.makedirs('./models')
        self.model.cpu()
        if update is not None:
            model_id = self.model_id + '-e' + str(update)
            torch.jit.save(self.model, './models/' + model_id + '.pt')
            self.model.to(self.device)
        else:
            torch.jit.save(self.model, './models/' + self.model_id + '.pt')
            print('Step 3: Model saved to ' + './models/' + self.model_id + '.pt')

    def close(self):
        try:
            self.env.close()
        except:
            pass

        try:
            self.writer.close()
        except:
            pass

        try:
            for worker in self.workers:
                worker.child.send(("close", None))
        except:
            pass

        time.sleep(1.0)
        exit(0)
