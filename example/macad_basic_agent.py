import math

import carla
from macad_gym.carla.multi_env import MultiCarlaEnv, DEFAULT_MULTIENV_CONFIG, DISCRETE_ACTIONS
from macad_gym.core.maps.nav_utils import get_next_waypoint
from external.navigation.basic_agent import BasicAgent

# multi agent, multi actor, custom env with planner, actors with hard coded behaviour rules for obstacles

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

if __name__ == "__main__":
    configs = DEFAULT_MULTIENV_CONFIG
    configs["env"]["enable_planner"] = True
    configs["env"]["discrete_actions"] = False
    env = MultiCarlaEnv(configs)

    env_config = configs["env"]
    actor_configs = configs["actors"]
    vehicle_dict = {}
    agent_dict = {}

    for ep in range(2):
        obs = env.reset()
        total_rewards = {k: 0.0 for k in actor_configs.keys()}
        for actor_id in actor_configs.keys():
            vehicle_dict[actor_id] = env._actors[actor_id]
            end_wp = env._end_pos[actor_id]
            # Set the goal for the planner to be 0.2 m after the destination
            # to avoid falling short & not triggering done
            dest_loc = get_next_waypoint(env.world, env._end_pos[actor_id], 0.2)
            # Use BasicAgent from Carla PythonAPI
            agent = BasicAgent(env._actors[actor_id], target_speed=40)
            agent.set_destination(carla.Location(*dest_loc))
            agent_dict[actor_id] = agent

        done = {"__all__": False}
        step = 0
        while not done["__all__"]:
            action_dict = {}
            for actor_id, agent in agent_dict.items():
                action_dict[actor_id] = vehicle_control_to_action(agent.run_step(), env._discrete_actions)

            observations, rewards, dones, info_dict = env.step(action_dict)

            print(":{}\n\t".join(["Step#", "rew", "ep_rew", "done{}"]).format(step, rewards, total_rewards, dones))
            step += 1
            for actor_id, reward in rewards.items():
                total_rewards[actor_id] += reward
