from env import DroneEnv
from networkhelpers import NetworkHelpers
from actor import Actor
from critic import Critic
from ppo import PPO


def get_actor_critic_networks():
    _actor_network = NetworkHelpers.create_simple_actor_network(
        input_layer_neurons=84 * 84,
        hidden_layer_neurons=[200],
        output_layer_neurons=7,
        tanh=True)
    _critic_network = NetworkHelpers.create_simple_critic_network(
        input_layer_neurons=84 * 84,
        hidden_layer_neurons=[200],
        output_layer_neurons=1,
        tanh=True)
    return _actor_network, _critic_network


if __name__ == '__main__':
    env = DroneEnv()
    state = env.state

    actor_network, critic_network = get_actor_critic_networks()
    ppo_agent = PPO(Actor(actor_network, env.action_space), Critic(critic_network), env.action_space)

    while True:
        action = ppo_agent.get_action(state)
        state, reward, done = env.step(action)

        ppo_agent.reap_reward(reward)
        ppo_agent.has_finished(done)
