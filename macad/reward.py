import numpy as np
from macad_gym.carla.reward import Reward


class CustomReward(Reward):

    def compute_reward_custom(self):
        self.reward = 0.0
        cur_dist = self.curr["distance_to_goal"]
        prev_dist = self.prev["distance_to_goal"]
        collision = (self.curr["collision_vehicles"] + self.curr["collision_pedestrians"] + self.curr["collision_other"] - self.prev["collision_vehicles"] - self.prev["collision_pedestrians"] - self.prev["collision_other"])
        offroad = (self.curr["intersection_offroad"] - self.prev["intersection_offroad"])
        offlane = (self.curr["intersection_otherlane"] - self.prev["intersection_otherlane"])

        if self.curr["done"] and self.prev["done"]:
            # if already dead continue to give the same value for stats consistency
            self.reward = self.prev["reward"]
        elif self.curr["done"] and not self.prev["done"]:
            # if newly dead by collision or by time
            if collision: self.reward -= 100.0
            elif self.curr["next_command"] == "REACH_GOAL": self.reward += 100.0
            else: self.reward -= cur_dist
        else:
            # Distance travelled toward the goal in m
            self.reward += np.clip(prev_dist - cur_dist, -10.0, 10.0) * 2
            # Change in speed (km/h)
            self.reward += np.clip(self.curr["forward_speed"]*3.6, -10.0, 30.0) * 0.5
            # New collision damage (in case this is not set as terminal condition)
            if collision: self.reward -= 100.0
            # If is a unlawful driver
            if offroad: self.reward -= 10.0
            if offlane: self.reward -= 10.0

        return self.reward
