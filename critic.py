import torch
import torch.nn as nn
import os


class Critic:
    def __init__(self, value_network: nn.Module, critic_file: str = 'critic_file', load: bool = True):
        self.value_network = value_network
        self.critic_file = critic_file

        if load:
            self.load()

    def evaluate(self, state: torch.Tensor) -> torch.Tensor:
        return self.value_network(state)

    def save(self, episode):
        print(f'Saving the critic at {episode}')
        torch.save({'value_network_dict': self.value_network.state_dict(), 'episode': episode}, self.critic_file)

    def load(self):
        if os.path.exists(self.critic_file):
            checkpoint = torch.load(self.critic_file)
            self.value_network.load_state_dict(checkpoint['value_network_dict'])
            episode = checkpoint['episode']
            print(f'Loaded existing critic continuing from episode {episode}.')
        else:
            print(f'No critic found.')
