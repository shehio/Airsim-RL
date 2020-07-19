from env import DroneEnv
import airsim

# env = DroneEnv()


def move_quadcopter(client, offset: list):
    current_velocity = client.getMultirotorState().kinematics_estimated.linear_velocity
    x_velocity = current_velocity.x_val + offset[0]
    y_velocity = current_velocity.y_val + offset[1]
    z_velocity = current_velocity.z_val + offset[2]
    duration_in_seconds = 1
    client.moveByVelocityAsync(x_velocity, y_velocity, z_velocity, duration_in_seconds).join()


if __name__ == '__main__':
    airsim_client = airsim.MultirotorClient()
    airsim_client.confirmConnection()
    airsim_client.enableApiControl(True)
    airsim_client.armDisarm(True)
    airsim_client.takeoffAsync().join()

    # move_quadcopter(airsim_client, [2, 2, 2])

    while True:
        state = airsim_client.getMultirotorState().kinematics_estimated.position
        velocity = airsim_client.getMultirotorState().kinematics_estimated.linear_velocity

        print(f'Current Position: ({state.x_val}, {state.y_val}, {state.z_val})')
        print(f'Current Velocity: ({velocity.x_val}, {velocity.y_val}, {velocity.z_val})')

        move_quadcopter(airsim_client, [0, 0, -1])

        # env.step(1)
