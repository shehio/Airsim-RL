# Airsim demo: Hello Drone. To run it, please comment out all lines below the "Another demo" comment.
import airsim
import os

# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Async methods returns Future. Call join() to wait for task to complete.
client.takeoffAsync().join()
client.moveToPositionAsync(-10, 10, -10, 5).join()

client.simPrintLogMessage("Hello Shehio!")

# Get images
responses = client.simGetImages([
    airsim.ImageRequest("0", airsim.ImageType.DepthVis),
    airsim.ImageRequest("1", airsim.ImageType.DepthPlanner, True)])
print('Retrieved images: %d', len(responses))

# Process images
for response in responses:
    if response.pixels_as_float:
        print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
        airsim.write_pfm(os.path.normpath('./py1.pfm'), airsim.get_pfm_array(response))
    else:
        print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
        airsim.write_file(os.path.normpath('./py1.png'), response.image_data_uint8)


# Another demo I wrote to learn the env
def move_quadcopter(_client, offset: list):
    current_velocity = _client.getMultirotorState().kinematics_estimated.linear_velocity
    x_velocity = current_velocity.x_val + offset[0]
    y_velocity = current_velocity.y_val + offset[1]
    z_velocity = current_velocity.z_val + offset[2]
    duration_in_seconds = 1
    _client.moveByVelocityAsync(x_velocity, y_velocity, z_velocity, duration_in_seconds).join()


if __name__ == '__main__':
    airsim_client = airsim.MultirotorClient()
    airsim_client.confirmConnection()
    airsim_client.enableApiControl(True)
    airsim_client.armDisarm(True)
    airsim_client.takeoffAsync().join()

    while True:
        state = airsim_client.getMultirotorState().kinematics_estimated.position
        velocity = airsim_client.getMultirotorState().kinematics_estimated.linear_velocity

        print(f'Current Position: ({state.x_val}, {state.y_val}, {state.z_val})')
        print(f'Current Velocity: ({velocity.x_val}, {velocity.y_val}, {velocity.z_val})')

        move_quadcopter(airsim_client, [0, 0, -1])
