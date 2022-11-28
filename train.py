import os
from argparse import Namespace
from copy import deepcopy

import gym
import numpy as np
import pandas as pd
import pygame
import torch
import wandb
from tensorboardX import SummaryWriter

from core.carla.train_loop import carla_loop
from core.ma_gym.train_loop import gym_loop
from envs.macad.env import NonSignalizedIntersection4Car
from external.navigation.basic_agent import BasicAgent


from utils.logger import Logger
from utils.parser import TrainOptions
from utils.utils import fix_random


def main():
    global best_result, parser_config
    args = deepcopy(parser_config)

    # Init loggers
    if args.wandb:
        wandb.init(project="TrafficAD", entity="johnminelli")
        # Init sweep agent configuration
        if args.sweep_id is not None: args.__dict__.update(wandb.config)
        wandb.config = args
        wandb.log({"params": wandb.Table(data=pd.DataFrame({k: [v] for k, v in vars(args).items()}))})
    if args.tensorboard:
        tb_writer = SummaryWriter()
    else: tb_writer = None
    logger = Logger(valid=True, episodes=args.episodes, batch_size=args.batch_size, terminal_print_freq=args.print_freq, tensorboard=tb_writer, wand=args.wandb)

    # Set the seed
    fix_random(args.seed)

    # Setup training devices
    if args.gpu_ids[0] < 0 or not torch.cuda.is_available():
        print("%s on CPU" % ("Training" if args.isTrain else "Executing"))
        device = torch.device("cpu")
    else:
        print("%s on GPU" % ("Training" if args.isTrain else "Executing"))
        if len(args.gpu_ids) > 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)[1:-1]
        device = torch.device("cuda")

    # Call specific loop
    if args.env == "carla":
        return carla_loop(args, device, logger)
    else:
        return gym_loop(args, device, logger)


if __name__ == '__main__':
    # Get arguments
    global parser_config
    parser_config = TrainOptions().parse()

    if parser_config.sweep_id is not None:
        wandb.agent(parser_config.sweep_id, main)
    else:
        main()

    # To get a SWEEP ID:
    #   sweep_configuration = {
    #       "name": "my-awesome-sweep", "metric": {"name": "accuracy", "goal": "maximize"}, "method": "grid",
    #       "parameters": {"a": {"values": [1, 2, 3, 4]}}
    #   }
    #   print(wandb.sweep(sweep_configuration))
    #
    # Or from CommandLine:
    #   wandb sweep config.yaml
    #
    # Or from web interface


    # for actor_id in actor_configs.keys():
    #     vehicle_dict[actor_id] = env._actors[actor_id]
    #     end_wp = env._end_pos[actor_id]
    #     # Set the goal for the planner to be 0.2 m after the destination
    #     # to avoid falling short & not triggering done
    #     dest_loc = get_next_waypoint(env.world, env._end_pos[actor_id], 0.2)
    #     agent = BasicAgent(env._actors[actor_id], target_speed=40)
    #     agent.set_destination(dest_loc)
    #     agent_dict[actor_id] = agent