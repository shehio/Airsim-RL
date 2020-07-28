from PIL import Image
import numpy as np
import airsim
import matplotlib.pyplot as plt


class DroneEnv:
    """Drone environment class using AirSim python API"""
    client: object
    pose: object
    state: object
    quad_offset: tuple
    start_collision: str
    next_collision: str
    cnt_collision: int
    collision_change: bool

    def __init__(self, duration: int = 1):
        self.duration = duration
        self.episode = 0

        self.client = self.__get_client()
        self.pose = self.client.simGetVehiclePose()

        step_size = 5
        self.action_space = [(0, 0, 0), (step_size, 0, 0), (0, step_size, 0), (0, 0, step_size),
                             (-step_size, 0, 0), (0, -step_size, 0), (0, 0, -step_size)]

        self.reset()
        print(f'Initial position: ({self.state.x_val}, {self.state.y_val}, {self.state.z_val})\n')

    def reset(self):
        print('RESET\n\n')

        self.state = self.client.getMultirotorState().kinematics_estimated.position
        self.client.moveToPositionAsync(0, 0, -10, 5).join()

        self.quad_offset = (0, 0, 0)
        self.episode += 1

    def step(self, action_index):
        print("Taking a step.")
        self.quad_offset = self.__get_action_from_action_index(action_index)
        print("Quad offset: ", self.quad_offset)

        quad_state = self.client.getMultirotorState().kinematics_estimated.position
        print(f'Position Before: ({quad_state.x_val}, {quad_state.y_val}, {quad_state.z_val})')
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity

        print(f'Current Velocity: ({quad_vel.x_val}, {quad_vel.y_val}, {quad_vel.z_val})')

        self.__move_quadrotor(quad_vel)
        quad_state = self.client.getMultirotorState().kinematics_estimated.position
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        print(f'Position After: ({quad_state.x_val}, {quad_state.y_val}, {quad_state.z_val})\n\n')

        reward = self.__compute_reward(quad_state, quad_vel, self.client.simGetCollisionInfo())
        state = self.__get_observation()
        done = self.__is_done(reward)
        return state, reward, done

    @staticmethod
    def __get_client():
        client = airsim.MultirotorClient()
        client.confirmConnection()
        client.enableApiControl(True)
        client.armDisarm(True)
        client.takeoffAsync().join()
        return client

    # @Todo: Add a hover action so it stays where the desired destination is.
    def __get_action_from_action_index(self, action_index: int):
        assert 0 <= action_index <= 6
        self.quad_offset = self.action_space[action_index]
        return self.quad_offset

    def __move_quadrotor(self, quad_vel):
        x_velocity = quad_vel.x_val + self.quad_offset[0]
        y_velocity = quad_vel.y_val + self.quad_offset[1]
        z_velocity = quad_vel.z_val + self.quad_offset[2]
        self.client.simPrintLogMessage(f'Moving velocity: ({x_velocity}, {y_velocity}, {z_velocity})')
        self.client.moveByVelocityAsync(x_velocity, y_velocity, z_velocity, self.duration).join()

    def __compute_reward(self, quad_state, quad_vel, collision_info):
        bound = 100
        if collision_info.has_collided:
            reward = -1000
        elif quad_state.x_val > bound or quad_state.y_val > bound or quad_state.z_val > bound:
            reward = -1000
        else:
            reward = - self.__get_distance(quad_state)

        return reward

    def __get_observation(self):
        responses = self.client.simGetImages([airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])
        observation = self.__transform_input(responses)
        return observation

    def __is_done(self, reward):
        done = 0

        # There are three ways an episode can end: Collision, Going out of bounds,
        # and Reaching the desired destination. The first two cases are covered with reward < 1000.
        # The third case is covered by reward being close enough (hence the -1) to the destination.
        if reward < -100 or reward > -1:
            done = 1

        if done == 1:
            print("======DONE======\n")
            self.client.simPrintLogMessage(f'Episode {self.episode} done, reward is: {reward}, resetting.')
            self.reset()
        return done

    @staticmethod
    def __get_distance(quad_state):
        """Get distance between current state and goal state"""
        desired_destination = np.array([-10, 10, -10])
        quad_position = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))
        return np.linalg.norm(quad_position - desired_destination)

    @staticmethod
    def __transform_input(responses):
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        img2d = np.flipud(img_rgb)

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        # Uncomment to see the sample.
        # plt.imshow(im_final)

        return im_final
