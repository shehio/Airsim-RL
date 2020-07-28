import torch
import torch.nn as nn
import torch.optim.rmsprop as rmsprop
import torch.nn.functional as F


class CommonHelpers:
    @staticmethod
    def create_network(
            input_layer_neurons: int,
            hidden_layer_neurons: list,
            output_layer_neurons: int,
            drop_out_rate,
            tanh=False):

        network_layers = nn.ModuleList()
        CommonHelpers.__add_layer_to_network(network_layers, input_layer_neurons, hidden_layer_neurons[0], tanh)

        for i in range(len(hidden_layer_neurons) - 1):
            CommonHelpers.__add_layer_to_network(
                network_layers,
                hidden_layer_neurons[i],
                hidden_layer_neurons[i + 1],
                tanh)

        network_layers.append(nn.Dropout(drop_out_rate))
        network_layers.append(nn.Linear(hidden_layer_neurons[-1], output_layer_neurons))
        return network_layers

    @staticmethod
    def __add_layer_to_network(network_layers, input_layer_neurons, output_layer_neurons, tanh=False):
        current_layer = nn.Linear(input_layer_neurons, output_layer_neurons, bias=False)
        network_layers.append(current_layer)
        if tanh:
            network_layers.append(nn.Tanh())
        else:
            network_layers.append(nn.ReLU(inplace=True))


class ActorNetwork(nn.Module):
    def __init__(self, input_layer_neurons: int, hidden_layer_neurons: list[int], output_layer_neurons: int,
                 learning_rate=0.002, decay_rate=0.99, dropout_rate=0.0, tanh=False):
        super(ActorNetwork, self).__init__()
        self.layers = CommonHelpers.create_network(
            input_layer_neurons,
            hidden_layer_neurons,
            output_layer_neurons,
            dropout_rate,
            tanh)
        self.optimizer = rmsprop.RMSprop(self.parameters(), lr=learning_rate, weight_decay=decay_rate)

    def forward(self, _input: torch.tensor):
        _output = _input.float()
        for layer in self.layers:
            _output = layer(_output)

        return F.softmax(_output)


class CriticNetwork(nn.Module):
    def __init__(self, input_layer_neurons: int, hidden_layer_neurons: list[int], output_layer_neurons: int,
                 learning_rate=0.002, decay_rate=0.99, dropout_rate=0.0, tanh=False):
        super(CriticNetwork, self).__init__()
        self.layers = CommonHelpers.create_network(
            input_layer_neurons,
            hidden_layer_neurons,
            output_layer_neurons,
            dropout_rate,
            tanh)
        self.optimizer = rmsprop.RMSprop(self.parameters(), lr=learning_rate, weight_decay=decay_rate)

    def forward(self, _input):
        _output = _input
        for layer in self.layers:
            _output = layer(_output)
        return _output


class NetworkHelpers:

    @staticmethod
    def create_simple_actor_network(
            input_count: int,
            hidden_layers: list,
            output_count: int,
            dropout_rate: float = 0.0,
            tanh=False):
        """
        :return: A simple actor network with either ReLU or Tanh in the intermediate layers
        and softmax in the final layer
        """
        return ActorNetwork(
            input_count=input_count,
            hidden_layers=hidden_layers,
            output_count=output_count,
            dropout_rate=dropout_rate,
            tanh=tanh)

    @staticmethod
    def create_simple_critic_network(
            input_count: int,
            hidden_layers: list,
            output_count: int,
            dropout_rate: float = 0.0,
            tanh=False):
        """
        :return: A simple critic network with either ReLU or Tanh in the all layers
        """
        return CriticNetwork(
            input_count=input_count,
            hidden_layers=hidden_layers,
            output_count=output_count,
            dropout_rate=dropout_rate,
            tanh=tanh)
