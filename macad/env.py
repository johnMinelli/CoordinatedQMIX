import math
import socket
import traceback

import carla
import cv2
import numpy as np
import pygame
from PIL import Image
from macad_gym.carla.multi_env import MultiCarlaEnv, DEFAULT_MULTIENV_CONFIG, DISCRETE_ACTIONS, COMMAND_ORDINAL, sigmoid
from gym.spaces import Box, Discrete, Tuple, Dict
import json
from macad_gym.core.controllers.keyboard_control import KeyboardControl
from macad_gym.viz.render import multi_view_render

from macad.reward import CustomReward


def make_env(opt):
    configs = DEFAULT_MULTIENV_CONFIG
    configs["env"]["enable_planner"] = False
    configs["env"]["discrete_actions"] = False
    return UrbanSignalIntersection3Car()  # instead of the general class you can do a superclass down here

class UrbanSignalIntersection3Car(MultiCarlaEnv):
    """A 4-way signalized intersection Multi-Agent Carla-Gym environment"""
    def __init__(self):
        config_file = open("macad/urban_signal_intersection_3c.json")
        self.configs = json.load(config_file)
        super(UrbanSignalIntersection3Car, self).__init__(self.configs)

        self.non_auto_actors = [k for k, v in self.configs["actors"].items() if not (v["auto_control"] or v["manual_control"])]
        self.auto_actors = [k for k, v in self.configs["actors"].items() if v["auto_control"] or v["manual_control"]]
        if len(self.auto_actors) != 0: print("Warning: Be aware that the environment will output observation also for actors with autopilot. If this is not a wanted behaviour set `ignore_autonomous` in environment config file.")
        self.auto_filter = lambda dict_obj: dict(filter(lambda a: a[0] not in self.auto_actors, dict_obj.items()))

        self._image_space = Box(-1.0, 1.0, shape=(3, self._x_res, self._y_res))
        self.observation_space = Dict({actor_id: Dict({"img": self._image_space, "msr": Box(-10.0, 10.0, shape=(16,))}) for actor_id in self._actor_configs.keys()})
        self._cmpt_reward = CustomReward()

    def reset(self):
        obs_dict = super().reset()
        f = self.auto_filter
        return f(obs_dict) if self._env_config["ignore_autonomous"] else obs_dict

    def _get_relative_location_target(self, loc_x, loc_y, loc_yaw, target_x, target_y):

        veh_yaw = loc_yaw * np.pi / 180
        veh_dir_world = np.array([np.cos(veh_yaw), np.sin(veh_yaw)])
        veh_loc_world = np.array([loc_x, loc_y])
        target_loc_world = np.array([target_x, target_y])
        d_world = target_loc_world - veh_loc_world
        dot = np.dot(veh_dir_world, d_world)
        det = veh_dir_world[0]*d_world[1] - d_world[0]*veh_dir_world[1]
        rel_angle = np.arctan2(det, dot)
        target_location_rel_x = np.linalg.norm(d_world) * np.cos(rel_angle)
        target_location_rel_y = np.linalg.norm(d_world) * np.sin(rel_angle)

        return target_location_rel_x.item(), target_location_rel_y.item()

    def _encode_obs(self, actor_id, image, py_measurements):
        # image = cv2.resize(image, (self.h, self.w))
        image = np.transpose(image, (2, 0, 1))
        target_rel_x, target_rel_y = self._get_relative_location_target(py_measurements["x"], py_measurements["y"], py_measurements["yaw"], self._end_pos[actor_id][0], self._end_pos[actor_id][1])
        target_rel_norm = np.linalg.norm(np.array([target_rel_x, target_rel_y]))
        target_rel_x_unit = target_rel_x / target_rel_norm
        target_rel_y_unit = target_rel_y / target_rel_norm
        one_hot_command = [0]*len(COMMAND_ORDINAL)
        one_hot_command[COMMAND_ORDINAL[py_measurements["next_command"]]] += 1

        measures = np.array([
            py_measurements["forward_speed"],
            target_rel_x, target_rel_y, target_rel_x_unit, target_rel_y_unit,
            py_measurements["collision_vehicles"],
            py_measurements["collision_pedestrians"],
            py_measurements["collision_other"],
            py_measurements["intersection_otherlane"],
            py_measurements["intersection_offroad"],
            py_measurements["intersection_otherlane"]] + \
            one_hot_command
        )

        obs = {'img': image, 'msr': measures}
        return obs

    def _decode_obs(self, actor_id, obs):
        return np.transpose(obs["img"], (1, 2, 0))

    def step(self, action_dict):
        """"""
        if (not self._server_process) or (not self._client): raise RuntimeError("Cannot call step(...) before calling reset()")
        assert len( self._actors), ("No actors exist in the environment. Either the environment was not properly initialized using`reset()` or all the actors have exited. Cannot execute `step()`")
        if not isinstance(action_dict, dict): raise ValueError("`step(action_dict)` expected dict of actions. Got {}".format(type(action_dict)))
        # Make sure the action_dict contains actions only for actors that exist in the environment
        if not set(action_dict).issubset(set(self._actors)): raise ValueError("Cannot execute actions for non-existent actors. Received unexpected actor ids:{}".format(set(action_dict).difference(set(self._actors))))

        try:
            obs_dict = {}
            reward_dict = {}
            info_dict = {}

            for actor_id, action in action_dict.items():
                obs, reward, done, info = self._step(actor_id, action)
                obs_dict[actor_id] = obs
                reward_dict[actor_id] = reward
                self._done_dict[actor_id] = done
                if done:
                    self._dones.add(actor_id)
                info_dict[actor_id] = info
            self._done_dict["__all__"] = len(self._dones) == len(self._actors)

            # Find if any actor's config has render=True & render only for that actor. NOTE: with async server stepping, enabling rendering affects the step time & therefore MAX_STEPS needs adjustments
            render_required = [k for k, v in self._actor_configs.items() if v.get("render", False)]
            if render_required:
                # reformat the observations
                multi_view_render({k: self._decode_obs(k, v) for k, v in obs_dict.items()}, [self._x_res, self._y_res], self._actor_configs)

            f = self.auto_filter
            return (f(obs_dict), f(reward_dict), f(self._done_dict), f(info_dict)) if self._env_config["ignore_autonomous"] else obs_dict, reward_dict, self._done_dict, info_dict

        except Exception:
            print("Error during step, terminating episode early.", traceback.format_exc())
            self._clear_server_state()

    def get_actors_loc(self):
        return {k: np.array([v.get_location().x, v.get_location().y, v.get_location().z]) for k, v in self._actors.items() if k in self.non_auto_actors}

    @staticmethod
    def _get_tcp_port(port=9000):
        s = socket.socket()
        s.bind(("", port))  # Request the sys to provide a free port dynamically
        server_port = s.getsockname()[1]
        s.close()
        return server_port
    # image_size_net_chans = (160, 90, 3)


def process_image(queue):
    """
    Get the image from the buffer and process it. It's the state for vision-based systems
    """
    image = queue.get()
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image = Image.fromarray(array).convert('L')  # grayscale conversion
    image = np.array(image.resize((84, 84)))  # convert to numpy array
    image = np.reshape(image, (84, 84, 1))  # reshape image
    return image


# If instead you want to predict 5 separata values
def vehicle_control_to_action(vehicle_control, is_discrete):
    if vehicle_control.hand_brake:
        continuous_action = [-1.0, vehicle_control.steer]
    else:
        if vehicle_control.reverse:
            continuous_action = [
                vehicle_control.brake - vehicle_control.throttle,
                vehicle_control.steer
            ]
        else:
            continuous_action = [
                vehicle_control.throttle - vehicle_control.brake,
                vehicle_control.steer
            ]

    def dist(a, b):
        return math.sqrt((a[0] - b[0]) * (a[0] - b[0]) +
                         (a[1] - b[1]) * (a[1] - b[1]))

    if is_discrete:
        closest_action = 0
        shortest_action_distance = dist(continuous_action, DISCRETE_ACTIONS[0])

        for i in range(1, len(DISCRETE_ACTIONS)):
            d = dist(continuous_action, DISCRETE_ACTIONS[i])
            if d < shortest_action_distance:
                closest_action = i
                shortest_action_distance = d
        return closest_action

    return continuous_action


def compute_reward(vehicle, sensors):#, collision_sensor, lane_sensor):
    max_speed = 14
    min_speed = 2
    speed = vehicle.get_velocity()
    vehicle_speed = np.linalg.norm([speed.x, speed.y, speed.z])

    speed_reward = (abs(vehicle_speed) - min_speed) / (max_speed - min_speed)
    lane_reward = 0

    if (vehicle_speed > max_speed) or (vehicle_speed < min_speed):
        speed_reward = -0.05

    if sensors.lane_crossed:
        if sensors.lane_crossed_type == "'Broken'" or sensors.lane_crossed_type == "'NONE'":
            lane_reward = -0.5
            sensors.lane_crossed = False

    if sensors.collision_flag:
        return -1

    else:
        return speed_reward + lane_reward