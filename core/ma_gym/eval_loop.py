from argparse import Namespace
from typing import Callable
import gym
import numpy as np
import torch

from core.base_agent import BaseAgent
from core.ma_gym.comix.comix_agent import CoordQMixGymAgent
from envs.wrappers import MaGymEnvWrap
from utils.logger import Logger

global _device
global best_result


def play_loop(opt: Namespace, env_fn: Callable[[], MaGymEnvWrap], agents_fn: Callable[[Namespace, MaGymEnvWrap], BaseAgent], logger: Logger, delay=0):
    # Initialize elements
    test_env = env_fn()
    agents = agents_fn(opt, test_env)
    agents_ids = test_env.agents_ids
    success = []

    logger.valid_start()
    for episode in range(opt.val_episodes):
        total_rewards = np.zeros(test_env.n_agents)
        max_delay = delay
        hxs = agents.init_hidden()
        comm = None
        delays = torch.zeros(test_env.n_agents,test_env.n_agents).to(_device)
        current_state_obs, _, dones, _ = test_env.reset(options={"test": True})
        current_state_obs = torch.Tensor(np.stack([current_state_obs[id] for id in agents_ids])).to(_device)
        dones = torch.Tensor(np.expand_dims(np.stack([dones[id] for id in agents_ids]), -1)).to(_device)
        while not dones.all():

            actions, hxs, add_in = agents.take_action(current_state_obs, hxs, dones, comm, delays)
            next_state_obs, rewards_dict, dones, info = test_env.step(list(map(lambda a: None if dones[a[0]] else a[1][0].cpu().numpy(), enumerate(actions))))
            if opt.render_mode == "human": test_env.render()
            total_rewards += np.array(list(rewards_dict.values()))
            next_state_obs = torch.Tensor(np.stack([next_state_obs[id] for id in agents_ids])).to(_device)
            dones = torch.Tensor(np.expand_dims(np.stack([dones[id] for id in agents_ids]), -1)).to(_device)

            current_state_obs = next_state_obs
            comm = add_in
            if delays[0][0] == max_delay:
                delays = torch.zeros(test_env.n_agents, test_env.n_agents).to(_device)
            else:
                delays += 1
            logger.valid_step(1, {})
        success.append(test_env.get_success_metric())
        # success.append(np.sum(total_rewards))
    logger.valid_stop({"rewards": {k: v for k,v in zip(test_env.agents_ids, total_rewards)}}, {"success": test_env.get_success_metric(), "avg_success": np.array(success).mean(), "std_success": np.array(success).std()})
    test_env.close()


def gym_loop(args: Namespace, device: torch.device, logger: Logger):
    global _device
    _device = device

    # Create the game environment
    def create_env_fn():
        # Create the game environment; the env name is filtered by parser
        if args.env == "switch_dev" or args.env == "CoMix_switch":
            gym.envs.register(id='CustomSwitch4-v0', entry_point='envs.switch:Switch', kwargs=
            {'n_agents': 4, 'full_observable': False, 'step_cost': 0.0, 'max_steps': 500})
            env = MaGymEnvWrap(gym.make("CustomSwitch4-v0"))
            args.hi, args.hc, args.hm, args.hs = 128, 128, 32, 1
        elif args.env == "predator_prey_dev" or "CoMix_predator_prey" in args.env:
            xy, n = (14, 8) if args.env == "CoMix_predator_prey_8" else (16, 16) if args.env == "CoMix_predator_prey_16" else (12, 4)
            gym.envs.register(id='CustomPredatorPrey-v0', entry_point='envs.predator_prey:PredatorPrey', kwargs=
            {'grid_shape': (xy, xy), 'n_agents': n, 'n_preys': 16, 'prey_move_probs': (0.2, 0.2, 0.2, 0.2, 0.2), 'full_observable': False, 'penalty': 0, 'step_cost': 0, 'max_steps': 500, 'agent_view_range': (5, 5)})
            env = MaGymEnvWrap(gym.make("CustomPredatorPrey-v0"))
            args.hi, args.hc, args.hm, args.hs = 128, 128, 32, 1
        elif args.env == "transport_dev" or "CoMix_transport" in args.env:
            n, h = (4, 2) if args.env == "CoMix_transport_2" else (8, 4) if args.env == "CoMix_transport_4" else (2, 1)
            gym.envs.register(id='CustomTransport-v0', entry_point='envs.transport:Transport', kwargs=
            {'grid_size': (16,10), 'n_agents': n, 'n_loads': h, 'full_observable': False, 'step_cost': 0, 'max_steps': 500, 'agent_view_range': (5, 5)})
            env = MaGymEnvWrap(gym.make("CustomTransport-v0"))
            args.hi, args.hc, args.hm, args.hs = 128, 64, 32, 1
        else:
            env = gym.make(args.env)
        return env

    def create_agent_fn(opt: Namespace, env):
        return CoordQMixGymAgent(opt, env, _device)

    # Play the game, using the agent
    play_loop(args, create_env_fn, create_agent_fn, logger)
