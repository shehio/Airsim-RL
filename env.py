#! /usr/bin/env python
"""Environment for Microsoft AirSim Unity Quadrotor using AirSim python API
- Author: Subin Yang
- Contact: subinlab.yang@gmail.com
- Date: 2019.06.20.
"""
import time
from PIL import Image
import numpy as np
import airsim


class DroneEnv(object):
    """Drone environment class using AirSim python API"""
    client: object
    pose: object
    state: object
    quad_offset: tuple
    start_collision: str
    next_collision: str
    cnt_collision: int
    collision_change: bool

    def __init__(self):
        self.reset()
        self.episode = 0

    def step(self, action):
        print("Taking a step.")
        self.quad_offset = self.__interpret_action(action)
        print("Quad offset: ", self.quad_offset)

        quad_state = self.client.getMultirotorState().kinematics_estimated.position
        print(f'Position Before: ({quad_state.x_val}, {quad_state.y_val}, {quad_state.z_val})')
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        self.__move_quadrotor(quad_vel)

        collision_info = self.client.simGetCollisionInfo()
        if self.next_collision != collision_info.object_name:
            self.collision_change = True

        self.__check_for_collision(collision_info)
        quad_state = self.client.getMultirotorState().kinematics_estimated.position
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        print(f'Position After: ({quad_state.x_val}, {quad_state.y_val}, {quad_state.z_val})')

        reward = self.__compute_reward(quad_state, quad_vel, collision_info)
        state = self.__get_observation()
        done = self.__is_done(reward)
        return state, reward, done

    def reset(self):
        self.client = self.__get_client()
        self.pose = self.client.simGetVehiclePose()
        self.state = self.client.getMultirotorState().kinematics_estimated.position

        print(f'Initial position: ({self.state.x_val}, {self.state.y_val}, {self.state.z_val})')

        self.quad_offset = (0, 0, 0)
        init_x = 162
        init_y = -320
        init_z = -150

        self.start_collision = "Cube"
        self.next_collision = "Cube"
        self.cnt_collision = 0
        self.collision_change = False

        self.client.takeoffAsync().join()
        print("take off moving position")
        self.client.moveToPositionAsync(init_x, init_y, init_z, 5).join()

    @staticmethod
    def __get_client(self):
        client = airsim.MultirotorClient()
        client.confirmConnection()
        client.enableApiControl(True)
        client.armDisarm(True)
        return client

    def __interpret_action(self, action):
        scaling_factor = 5
        if action.item() == 0:
            self.quad_offset = (0, 0, 0)
        elif action.item() == 1:
            self.quad_offset = (scaling_factor, 0, 0)
        elif action.item() == 2:
            self.quad_offset = (0, scaling_factor, 0)
        elif action.item() == 3:
            self.quad_offset = (0, 0, scaling_factor)
        elif action.item() == 4:
            self.quad_offset = (-scaling_factor, 0, 0)
        elif action.item() == 5:
            self.quad_offset = (0, -scaling_factor, 0)
        elif action.item() == 6:
            self.quad_offset = (0, 0, -scaling_factor)

        return self.quad_offset

    def __move_quadrotor(self, quad_vel):
        x_velocity = quad_vel.x_val + self.quad_offset[0]
        y_velocity = quad_vel.y_val + self.quad_offset[1]
        z_velocity = quad_vel.z_val + self.quad_offset[2]
        duration = 20  # @Todo: What's the unit here?
        self.client.moveByVelocityAsync(x_velocity, y_velocity, z_velocity, duration).join()
        time.sleep(0.5)  # @Todo: Why the sleep?

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

        if self.episode == 0:
            if self.collision_change and self.next_collision != self.start_collision:
                if "Cube" in self.next_collision:
                    dist = self.__get_distance(quad_state)
                    reward = 50000
                else:
                    reward = -100
            else:
                reward = 0
        else:
            if self.next_collision != self.start_collision:
                if "Cube" in self.next_collision:
                    dist = self.__get_distance(quad_state)
                    reward = 50000
                else:
                    reward = -100
            else:
                reward = 0

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

        self.client.armDisarm(False)
        self.client.reset()
        self.client.enableApiControl(False)
        time.sleep(1)  # @Todo: Why do we have this sleep here?
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
        img_rgba = img1d.reshape(response.height, response.width, 4)
        img2d = np.flipud(img_rgba)

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        return im_final
