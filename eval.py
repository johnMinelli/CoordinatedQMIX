import os
from argparse import Namespace
from typing import Callable
import sys
import gym
import numpy as np
import torch
from tensorboardX import SummaryWriter
import wandb

from core.ma_gym.eval_loop import gym_loop
from utils.logger import Logger
from utils.parser import EvalOptions
from utils.utils import fix_random


def main(args):
    global best_result, _device

    # Init loggers
    if args.wandb:
        wandb.init(group="CoMix", project="TrafficAD", entity="johnminelli")
    if args.tensorboard:
        tb_writer = SummaryWriter()
    else: tb_writer = None
    logger = Logger(valid=True, episodes=args.episodes, batch_size=args.batch_size, terminal_print_freq=args.print_freq, tensorboard=tb_writer, wand=args.wandb)

    # Set the seed
    # fix_random(args.seed)

    # Setup training devices
    if args.gpu_ids[0] < 0 or not torch.cuda.is_available():
        print("%s on CPU" % ("Training" if args.isTrain else "Executing"))
        device = torch.device("cpu")
    else:
        print("%s on GPU" % ("Training" if args.isTrain else "Executing"))
        if len(args.gpu_ids) > 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)[1:-1]
        device = torch.device("cuda")

    gym_loop(args, device, logger)



if __name__ == '__main__':
    # Get arguments
    parser_config = EvalOptions().parse()

    main(parser_config)
