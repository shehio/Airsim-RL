import copy
import itertools
import numpy as np
import torch
from torch.distributions import Categorical
import os

from actor import Actor
from critic import Critic


class PPO:

    class Memory:
        def __init__(self):
            self.dlogps = []
            self.hidden_layers = []
            self.actual_rewards = []
            self.states = []

            # PPO additional fields
            self.entropies = []
            self.actions = []
            self.new_dlogps = []
            self.predicted_values = []
            self.episode_complete = []

    def __init__(self, actor: Actor, critic: Critic, action_space: list, episode_number: int = 0, gamma: float = 0.99,
                 eta: float = 0.2, c1: float = 0.5, c2: float = 0.01, batch_size: int = 10, save_interval: int = 50,
                 learning_rate: float = 0.002, decay_rate: float = 0.90, epochs: int = 4,
                 optimizer_file: str = 'optimizer_file', load: bool = True):
        self.actor = actor
        self.critic = critic
        self.action_space = action_space
        self.episode_number = episode_number

        self.gamma = gamma
        self.eta = eta
        self.c1 = c1
        self.c2 = c2
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epochs = epochs
        self.save_interval = save_interval
        self.optimizer_file = optimizer_file

        self.new_policy_network = copy.deepcopy(self.actor.policy_network)
        self.optimizer = self.__get_optimizers()
        self.memory = self.Memory()

        if load:
            self.load_optimizer_and_episode()

    def get_action(self, state: np.array):
        # from_numpy() automatically inherits input array dtype.
        # On the other hand, torch.Tensor is an alias for torch.FloatTensor.
        state = torch.Tensor(state)

        self.memory.states.append(state)

        with torch.no_grad():
            sampled_action_index, action, distribution = self.actor.act(state)

        self.memory.actions.append(sampled_action_index)
        self.memory.dlogps.append(distribution.log_prob(sampled_action_index))

        return action

    def reap_reward(self, reward):
        self.memory.actual_rewards.append(reward)

    def has_finished(self, done):
        self.memory.episode_complete.append(done)
        if done:
            self.make_episode_updates()

    def update_batch(self, batch_size):
        self.batch_size = batch_size

    def make_episode_updates(self):
        self.episode_number = self.episode_number + 1  # @Todo: This should probably be removed from the class.

        if self.episode_number % self.batch_size == 0:
            self.__evaluate_and_train_networks()
            self.__reset_actor_and_memory()

            if self.episode_number % self.save_interval == 0:
                self.save_model()

    def save_model(self):
        print(f'Saving the model at {self.episode_number}.')
        self.actor.save(self.episode_number)
        self.critic.save(self.episode_number)
        torch.save(
            {'optimizer_state_dict': self.optimizer.state_dict(), 'episode': self.episode_number},
            self.optimizer_file)

    def load_model(self):
        self.actor.load()
        self.critic.load()
        self.load_optimizer_and_episode()

    def load_optimizer_and_episode(self):
        if os.path.exists(self.optimizer_file):
            checkpoint = torch.load(self.optimizer_file)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.episode_number = checkpoint['episode']
            print(f'Loaded existing optimizer continuing from episode {self.episode_number}.')
        else:
            print('No optimizer found.')

    def __get_optimizers(self):
        params = [self.new_policy_network.parameters(), self.critic.value_network.parameters()]
        optimizer = torch.optim.Adam(itertools.chain(*params), lr=self.learning_rate, betas=(0.9, 0.999))

        # @Todo The model never converges using RMSProp for some reason, investigate!
        # optimizer = RMSprop(itertools.chain(*params), lr=self.learning_rate, weight_decay=self.decay_rate)
        return optimizer

    def __evaluate_and_train_networks(self):
        old_states = torch.stack(self.memory.states).detach()
        old_action_log_probabilities = torch.stack(self.memory.dlogps).flatten().detach()
        old_actions = torch.stack(self.memory.actions).detach()

        rewards = self.__discount_and_normalize_rewards(
            self.memory.actual_rewards,
            self.memory.episode_complete,
            self.gamma)

        for _ in range(self.epochs):
            action_probabilities = self.new_policy_network(old_states)
            distributions = Categorical(action_probabilities)
            new_action_log_probabilities = distributions.log_prob(old_actions)

            state_values = self.critic.evaluate(old_states).flatten()
            advantages = rewards - state_values.detach()

            policy_ratio = torch.exp(new_action_log_probabilities - old_action_log_probabilities)
            clipped_ratio = torch.clamp(policy_ratio, 1 - self.eta, 1 + self.eta)

            surrogate1 = policy_ratio * advantages
            surrogate2 = clipped_ratio * advantages
            action_loss = torch.min(surrogate1, surrogate2).mean()

            value_loss = ((state_values - rewards) ** 2).mean()
            entropy_loss = distributions.entropy().mean()

            # loss is the negative of the gain in the paper: https://arxiv.org/abs/1707.06347
            total_loss = - action_loss + self.c1 * value_loss - self.c2 * entropy_loss

            self.optimizer.zero_grad()
            total_loss.mean().backward()
            self.optimizer.step()

        print(f'Episode: {self.episode_number}. Total loss that we trained on: {total_loss} for {self.epochs} epochs.')

    def __reset_actor_and_memory(self):
        self.actor = Actor(self.new_policy_network, self.action_space, load=False)
        self.memory = self.Memory()

    @staticmethod
    def __discount_and_normalize_rewards(rewards, episode_completed, gamma, device=torch.device("cpu")):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            if episode_completed[t]:
                running_add = 0
            running_add = running_add * gamma + rewards[t]
            discounted_rewards[t] = running_add

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + + 1e-5)

        return discounted_rewards
