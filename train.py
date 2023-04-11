import os
from copy import deepcopy

import pandas as pd
import torch
import wandb
from tensorboardX import SummaryWriter
from core.ma_gym.train_loop import gym_loop
# import lovely_tensors as lt
# lt.monkey_patch()
# lt.repr_str.PRINT_OPTS.color=False

from utils.logger import Logger
from utils.parser import TrainOptions
from utils.utils import fix_random


def main():
    global best_result, parser_config
    args = deepcopy(parser_config)

    # Init loggers
    if args.wandb:
        wandb.init(group="CoMix", project="TrafficAD", entity="johnminelli")
        # Init sweep agent configuration
        if args.sweep_id is not None: args.__dict__.update(wandb.config)
        # wandb.config = args
        wandb.config.update(args.__dict__)
        wandb.log({"params": wandb.Table(data=pd.DataFrame({k: [v] for k, v in vars(args).items()}))})
    if args.tensorboard:
        tb_writer = SummaryWriter()
    else: tb_writer = None
    logger = Logger(valid=True, episodes=args.episodes, batch_size=args.batch_size, terminal_print_freq=args.print_freq, tensorboard=tb_writer, wand=args.wandb)

    # Set the seed
    if args.seed != -1:
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
    gym_loop(args, device, logger)


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
