from env import DroneEnv
from networkhelpers import NetworkHelpers
from ppo import PPO


def get_actor_critic():
    _actor = NetworkHelpers.create_simple_actor_network(
        input_count=64 * 64,
        hidden_layers=[200],
        output_count=7,
        tanh=True)
    _critic = NetworkHelpers.create_simple_critic_network(
        input_count=64 * 64,
        hidden_layers=[200],
        output_count=1,
        tanh=True)
    return _actor, _critic


if __name__ == '__main__':
    env = DroneEnv()
    actor, critic = get_actor_critic()
    ppo_agent = PPO(actor, critic, env.action_space)

    while True:
        env.step(1)
