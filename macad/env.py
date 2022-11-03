import logging
import math
import random
import socket
import sys
import time
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
from macad_gym.core.sensors.utils import preprocess_image
from macad_gym.viz.render import multi_view_render

from macad.reward import CustomReward
from macad.scenarios import CustomScenarios

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

N_ACTORS_ENV = 4
Z_CUBE = 40


def make_env(opt):
    return NonSignalizedIntersection4Car()


class NonSignalizedIntersection4Car(MultiCarlaEnv):
    """A 4-way signalized intersection Multi-Agent Carla-Gym environment"""
    def __init__(self):
        config_file = open("macad/non_signalized_intersection_4c.json")
        self.configs = json.load(config_file)
        super(NonSignalizedIntersection4Car, self).__init__(self.configs)

        self.non_auto_actors = [k for k, v in self.configs["actors"].items() if not (v["auto_control"] or v["manual_control"])]
        self.auto_actors = [k for k, v in self.configs["actors"].items() if v["auto_control"] or v["manual_control"]]
        if len(self.auto_actors) != 0: logger.warning("Warning: Be aware that the environment will output observation also for actors with autopilot. If this is not a wanted behaviour set `ignore_autonomous` in environment config file.")
        self.auto_filter = lambda dict_obj: dict(filter(lambda a: a[0] not in self.auto_actors, dict_obj.items()))
        self._previous_comm_value = {}
        self._zone = 0

        self.exclude_hard_vehicles = True  # Remove hard vehicles due to problems of camera position
        self._x_res = self._y_res = 84  # Match the img_obs with agents resized observation set in json config
        self._image_space = Box(-1.0, 1.0, shape=(3, self._x_res, self._y_res))
        # Set custom observation space
        self.observation_space = Dict({actor_id: Dict({"img": self._image_space, "msr": Box(-10.0, 10.0, shape=(15,))})
                                       for actor_id in self._actor_configs.keys()})
        # Set custom functionalities classes
        self._reward_policy = CustomReward()
        self._scenario_config = CustomScenarios.resolve_scenarios_parameter(self.configs["scenarios"])

    def reset(self):
        # Random pick another scenario
        prev_s = self._scenario_map
        obs_dict = super().reset()
        # Move the spectator camera in the area
        if prev_s != self._scenario_map:
            spectator = self.world.get_spectator()
            x = np.mean([a["start"][0] for a in self._scenario_map["actors"].values()])
            y = np.mean([a["start"][1] for a in self._scenario_map["actors"].values()])
            spectator.set_transform(carla.Transform(carla.Location(x, y, 56), carla.Rotation(pitch=-90)))

        f = self.auto_filter
        return f(obs_dict) if self._env_config["ignore_autonomous"] else obs_dict

    def _set_zone(self, value):
        self._zone = value
        # move spectator in zone of interest
        spectator = self.world.get_spectator()
        x = np.mean([a["start"][0] for a in list(self._scenario_map["actors"].values())[self._zone*N_ACTORS_ENV:self._zone*N_ACTORS_ENV+N_ACTORS_ENV]])
        y = np.mean([a["start"][1] for a in list(self._scenario_map["actors"].values())[self._zone*N_ACTORS_ENV:self._zone*N_ACTORS_ENV+N_ACTORS_ENV]])
        spectator.set_transform(carla.Transform(carla.Location(x, y, 56), carla.Rotation(pitch=-90)))
        # flash the path in the spectator view
        [self._path_trackers[k].draw() for k in list(self._actor_configs.keys())[self._zone*N_ACTORS_ENV:self._zone*N_ACTORS_ENV+N_ACTORS_ENV]]

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

        config = self._actor_configs[actor_id]
        image = preprocess_image(image, config)

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
            py_measurements["intersection_offroad"]] + \
            one_hot_command
        )

        obs = {'img': image, 'msr': measures}
        return obs

    def _decode_obs(self, actor_id, obs):
        img = np.transpose(obs["img"], (1, 2, 0))
        return img.swapaxes(0, 1) * 128 + 128

    def _register_input(self):
        if pygame.get_init():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Stop")
                    self._clear_server_state()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        self._set_zone(0)
                    if event.key == pygame.K_2:
                        if len(self._actors) > N_ACTORS_ENV: self._set_zone(1)
                    if event.key == pygame.K_3:
                        if len(self._actors) > N_ACTORS_ENV*2: self._set_zone(2)

    def step(self, action_dict, comm_dict):
        """Complete override of step function"""
        # ear the input
        self._register_input()
        # get spectator position
        x_cam, y_cam, z_cam = (lambda l: (l.x, l.y, l.z))(self.world.get_spectator().get_location())
        self._previous_comm_value = comm_dict

        try:
            obs_dict = {}
            reward_dict = {}
            info_dict = {}
            # do step for each actor
            for actor_id, action in action_dict.items():
                obs, reward, done, info = self._step(actor_id, action)
                obs_dict[actor_id] = obs
                reward_dict[actor_id] = reward
                if not self._done_dict.get(actor_id, False):
                    self._done_dict[actor_id] = done
                info_dict[actor_id] = info
            self._done_dict["__all__"] = sum(self._done_dict.values()) >= len(self._actors)
            # info drawing for debug
            for k, v in list(obs_dict.items())[self._zone*N_ACTORS_ENV:self._zone*N_ACTORS_ENV+N_ACTORS_ENV]:
                actor_coords = (lambda l: (l.x, l.y, l.z))(self._actors[k].get_location())
                if comm_dict[k] and not self._done_dict[k]:  # render a green square in the spectator view
                    self.world.debug.draw_point(self._cube_coords(x_cam, y_cam, z_cam, *actor_coords[:2], Z_CUBE), 0.15, life_time=0.2, color=carla.Color(0, 255, 0))
                elif self._done_dict[k]:  # render a red square in the spectator view
                    self.world.debug.draw_point(self._cube_coords(x_cam, y_cam, z_cam, *actor_coords[:2], Z_CUBE), 0.15, life_time=0.8, color=carla.Color(255, 0, 0))
            render_required = [k for k, v in self._actor_configs.items() if v.get("render", False)]
            # view rendering for actors
            if render_required:
                images = {}
                for k, v in list(obs_dict.items())[self._zone*N_ACTORS_ENV:self._zone*N_ACTORS_ENV+N_ACTORS_ENV]:
                    im = self._decode_obs(k, v)
                    if comm_dict[k] and not self._done_dict[k]:  # render a green circle in the view rendering
                        im[(np.arange(im.shape[0])[np.newaxis, :] - 16) ** 2 + (np.arange(im.shape[1])[:, np.newaxis] - 16) ** 2 < 4 ** 2] = [0,255,0]
                    images.update({k: im})
                multi_view_render(images, [400, 300], self._actor_configs)

            f = self.auto_filter
            return (f(obs_dict), f(reward_dict), f(self._done_dict), f(info_dict)) if self._env_config[
                "ignore_autonomous"] else obs_dict, reward_dict, self._done_dict, info_dict
        except Exception:
            print("Error during step, terminating episode early.",
                  traceback.format_exc())
            self._clear_server_state()

    def _cube_coords(self, x_cam, y_cam, z_cam, x_act, y_act, z):
        x_cube_persp = (((abs(x_cam - x_act) / 2) * z_cam) / z)
        x_cube = x_act + (x_cube_persp) if abs((x_act + x_cube_persp) - x_cam) < abs((x_act - x_cube_persp) - x_cam) else x_act - x_cube_persp
        y_cube_persp = (((abs(y_cam - y_act) / 2) * z_cam) /z)
        y_cube = y_act + y_cube_persp if abs((y_act + y_cube_persp) - y_cam) < abs((y_act - y_cube_persp) - y_cam) else y_act - y_cube_persp
        return carla.Location(x_cube, y_cube, z)

    def get_actors_loc(self):
        return {k: np.array([v.get_location().x, v.get_location().y, v.get_location().z]) for k, v in self._actors.items() if not self._env_config["ignore_autonomous"] or k in self.non_auto_actors }

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
