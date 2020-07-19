#! /usr/bin/env python
import time
from PIL import Image
import numpy as np
import airsim


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
        self.reset()
        print(f'Initial position: ({self.state.x_val}, {self.state.y_val}, {self.state.z_val})\n')

    def reset(self):
        print('RESET\n\n')
        self.client = self.__get_client()
        self.pose = self.client.simGetVehiclePose()
        self.state = self.client.getMultirotorState().kinematics_estimated.position

        self.client.moveToPositionAsync(-10, 10, -10, 5).join()

        self.quad_offset = (0, 0, 0)
        self.start_collision = "Cube"
        self.next_collision = "Cube"
        self.cnt_collision = 0
        self.collision_change = False
        self.episode += 1

    def step(self, action):
        print("Taking a step.")
        self.quad_offset = self.__interpret_action(action)
        print("Quad offset: ", self.quad_offset)

        quad_state = self.client.getMultirotorState().kinematics_estimated.position
        print(f'Position Before: ({quad_state.x_val}, {quad_state.y_val}, {quad_state.z_val})')
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity

        print(f'Current Velocity: ({quad_vel.x_val}, {quad_vel.y_val}, {quad_vel.z_val})')
        self.__move_quadrotor(quad_vel)

        collision_info = self.client.simGetCollisionInfo()
        if self.next_collision != collision_info.object_name:
            self.collision_change = True

        self.__check_for_collision(collision_info)
        quad_state = self.client.getMultirotorState().kinematics_estimated.position
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        print(f'Position After: ({quad_state.x_val}, {quad_state.y_val}, {quad_state.z_val})\n\n')

        reward = self.__compute_reward(quad_state, quad_vel, collision_info)
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

    def __interpret_action(self, action: int):
        step_size = 5
        if action == 0:
            self.quad_offset = (0, 0, 0)
        elif action == 1:
            self.quad_offset = (step_size, 0, -1)
        elif action == 2:
            self.quad_offset = (0, step_size, 0)
        elif action == 3:
            self.quad_offset = (0, 0, step_size)
        elif action == 4:
            self.quad_offset = (-step_size, 0, 0)
        elif action == 5:
            self.quad_offset = (0, -step_size, 0)
        elif action == 6:
            self.quad_offset = (0, 0, -step_size)

        return self.quad_offset

    def __move_quadrotor(self, quad_vel):
        x_velocity = quad_vel.x_val + self.quad_offset[0]
        y_velocity = quad_vel.y_val + self.quad_offset[1]
        z_velocity = quad_vel.z_val + self.quad_offset[2]
        self.client.simPrintLogMessage(f'Moving ({x_velocity}, {y_velocity}, {z_velocity})')
        self.client.moveByVelocityAsync(x_velocity, y_velocity, z_velocity, self.duration).join()

    def __check_for_collision(self, collision_info):
        if collision_info.has_collided:
            if self.cnt_collision == 0:
                self.start_collision = collision_info.object_name
                self.next_collision = collision_info.object_name
                self.cnt_collision = 1
            else:
                self.next_collision = collision_info.object_name

    def __compute_reward(self, quad_state, quad_vel, collision_info):
        thresh_dist, max_dist, beta, z = 7, 500, 1, -10

        if self.collision_change and self.next_collision != self.start_collision and "Cube" in self.next_collision:
            reward = -1000
        else:
            reward = - self.__get_distance(quad_state)

        if quad_state.z_val < -280:
            reward = -100

        return reward

    def __get_observation(self):
        responses = self.client.simGetImages([airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])
        observation = self.__transform_input(responses)
        return observation

    def __is_done(self, reward):
        done = 0
        if reward <= -10:
            done = 1
        elif reward > 499:
            done = 1

        if done == 1:
            print("======DONE======\n")
            self.client.armDisarm(False)
            self.client.reset()
            self.client.enableApiControl(False)
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

        return im_final
