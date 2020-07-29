import torch
import torch.nn as nn
from torch.distributions import Categorical
import os


class Actor:
    def __init__(
            self,
            policy_network: nn.Module,
            action_space: list,
            actor_file: str = 'actor_file',
            load: bool = True):
        self.policy_network = policy_network
        self.action_space = action_space
        self.actor_file = actor_file

        if load:
            self.load()

    def act(self, state: torch.Tensor) -> (torch.Tensor, any, torch.distributions.distribution):
        action_probabilities = self.policy_network(state)
        distribution = Categorical(action_probabilities)
        sampled_action_index = distribution.sample()
        action = self.action_space[sampled_action_index.item()]

        return sampled_action_index, action, distribution

    def save(self, episode):
        print(f'Saving the actor at {episode}')
        torch.save({'policy_network_dict': self.policy_network.state_dict(), 'episode': episode}, self.actor_file)

    def load(self):
        if os.path.exists(self.actor_file):
            checkpoint = torch.load(self.actor_file)
            self.policy_network.load_state_dict(checkpoint['policy_network_dict'])
            episode = checkpoint['episode']
            print(f'Loaded existing actor continuing from episode {episode}.')
        else:
            print(f'No actor found.')
