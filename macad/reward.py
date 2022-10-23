import numpy as np
from macad_gym.carla.reward import Reward


class CustomReward(Reward):

    def compute_reward_custom(self):
        self.reward = 0.0
        cur_dist = self.curr["distance_to_goal"]
        prev_dist = self.prev["distance_to_goal"]
        # Distance travelled toward the goal in m
        self.reward += np.clip(prev_dist - cur_dist, -10.0, 10.0)
        # Change in speed (km/h)
        self.reward += np.clip(self.curr["forward_speed"], -10.0, 30.0) * 0.1
        # New collision damage
        new_damage = (
            self.curr["collision_vehicles"] +
            self.curr["collision_pedestrians"] + self.curr["collision_other"] -
            self.prev["collision_vehicles"] -
            self.prev["collision_pedestrians"] - self.prev["collision_other"])
        if new_damage:
            self.reward -= 100.0

        self.reward -= self.curr["intersection_offroad"] * 0.05
        self.reward -= self.curr["intersection_otherlane"] * 0.05

        if self.curr["next_command"] == "REACH_GOAL":
            self.reward += 100

        return self.reward
