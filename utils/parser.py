import os
import re
from argparse import ArgumentParser, ArgumentTypeError, ArgumentDefaultsHelpFormatter

import yaml

from utils.utils import mkdirs
from path import Path


def positive_int(value: any) -> int:
    """Checks if a value is a positive integer.

    :param value: the value to be checked.
    :return: the value if valid integer, otherwise raises an ArgumentTypeError.
    """
    int_value = int(value)

    if int_value <= 0:
        raise ArgumentTypeError("%s should be a positive integer value." % value)

    return int_value


def positive_float(value: any) -> float:
    """Checks if a value is a positive float.

    :param value: the value to be checked.
    :return: the value if valid float, otherwise raises an ArgumentTypeError.
    """
    float_value = float(value)

    if float_value <= 0:
        raise ArgumentTypeError("%s should be a positive float value." % value)

    return float_value


class BaseOptions():
    def __init__(self):
        self.parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # basic info
        self.parser.add_argument('--name', type=str, default='experiment_name', help='Name of the experiment. It decides where to store results and models')
        self.parser.add_argument('--seed', type=int, default=0, help='Seed for random functions, and network initialization')
        self.parser.add_argument('--gpu_ids', type=str, default='0,1', help='GPU ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--env', default=None, choices=['CoMix_switch', 'CoMix_predator_prey_4', 'CoMix_predator_prey_8', 'CoMix_predator_prey_16', 'CoMix_transport_1','CoMix_transport_2','CoMix_transport_4','CoMix_drones', 'switch_dev', 'predator_prey_dev', 'transport_dev'], help='Name of the environment (default: %(default)s)')
        self.parser.add_argument('--agent', type=str, required=False, default="comix", help='Label of a trained solver agent to be loaded from `load_path` (default %(default)s). -1 to load the last saved model in the folder.')

        # log
        self.parser.add_argument('-pf', '--print_freq', type=int, default=500, help='Frequency of showing training results on console')
        self.parser.add_argument('-lu', '--log_per_update', action='store_true', help='Log per weights update instead of per env step')
        self.parser.add_argument('-t', '--tensorboard', action='store_true', help='log stats on tensorboard local dashboard')
        self.parser.add_argument('--wandb', action='store_true', help='log stats on wandb dashboard')
        self.parser.add_argument('--yaml_params', type=str, required=False, default=None, help='File with parameters to load. If `None` it will attempted the load of a file named params_{env_name}.yaml')

        # output
        self.parser.add_argument('-op', '--out_path', type=str, default='./out', help='Results are saved here')
        self.parser.add_argument('-r', '--render_mode', default='human', required=False, type=str.lower, choices=['human', 'human_val', 'none'], help='Modality of rendering of the environment.')

        self.parser.add_argument('-ep', '--episodes', type=positive_int, default=2000, required=False, help='The episodes to run the training procedure (default %(default)s).')
        self.parser.add_argument('-ve', '--val_episodes', type=positive_int, default=1, required=False, help='The episodes to run the validation procedure (default %(default)s).')

        self.isTrain = False
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            self.opt.gpu_ids.append(id)

        if self.opt.render_mode == "none":
            self.opt.render_mode = None

        args = vars(self.opt)

        self._check_args_consistency()

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        return self.opt

    def _check_args_consistency(self):
        """ Checks the input arguments. """

        # Create the path to the files, if necessary.
        self.opt.results_path = Path(self.opt.out_path) / self.opt.name / "results/"
        mkdirs(self.opt.results_path)

        if self.isTrain:
            self.opt.backup_dir = Path(self.opt.save_path) / self.opt.name
            mkdirs(self.opt.backup_dir)
        else:
            self.opt.models_path = Path(self.opt.models_path) / self.opt.name
        # load mdoel parameters from yaml
        if self.opt.yaml_params is None:
            self.opt.yaml_params = "params_{0}.yaml".format(re.sub(r'^(.*?)_dev$|^CoMix_(.*?)_?\d*$', r'\1\2', self.opt.env))
        if os.path.exists(self.opt.yaml_params):
            with open(self.opt.yaml_params, 'r') as f:
                args_dict = yaml.safe_load(f)
            self.opt.__dict__.update(args_dict)


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--sweep_id', type=str, help='sweep id for wandb hyperparameters search e.g. user/project/sweep')

        self.parser.add_argument('-s', '--save_path', type=str, default='./save', help='Checkpoints are saved here')
        self.parser.add_argument('-as', '--agent_save_interval', type=positive_int, default=5, required=False, help='The save interval for the trained agent (default %(default)s), in episodes.')
        self.parser.add_argument('-vi', '--agent_valid_interval', type=positive_int, default=5, required=False, help='The eval interval for the trained agent (default %(default)s), in episodes.')
        self.parser.add_argument('--load_path', type=str, required=False, default=None, help='Path where to search the trained agents to be loaded (default `save_path`/models.')
        self.parser.add_argument('--continue_train', type=int, default=None, help='continue training: if set to -1 load the latest model from save_path')

        self.parser.add_argument('-bs', '--batch_size', type=positive_int, default=512, required=False, help='The batch size to be sampled from the memory for the training (default %(default)s).')
        self.parser.add_argument('-k', '--K_epochs', type=float, default=0.02, required=False, help='The number of epochs to run on the single batch (default %(default)s).')
        self.parser.add_argument('-ck', '--coord_K_epochs', type=float, default=0.02, required=False, help='The number of epochs to run on the single batch (default %(default)s).')
        self.parser.add_argument('-cs', '--chunk_size', type=positive_int, default=10, required=False, help='The size of the sequence for each sample (default %(default)s).')
        self.parser.add_argument('--min_buffer_len', type=positive_int, default=2000, required=False, help='The number of necessary samples in the buffer for training (default %(default)s).')
        self.parser.add_argument('--max_buffer_len', type=positive_int, default=20000, required=False, help='The maximum number of samples in the buffer for training (default %(default)s).')

        self.parser.add_argument('-opt', '--optimizer', type=str.lower, default='adam', required=False, choices=['adam', 'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adamax'], help='The optimizer to be used. (default %(default)s).')
        self.parser.add_argument('--lr_q', type=positive_float, default=0.0001, help='initial learning rate')
        self.parser.add_argument('--lr_co', type=positive_float, default=0.00005, help='initial learning rate')
        self.parser.add_argument('--lr_ae', type=positive_float, default=0.0001, help='initial learning rate')
        self.parser.add_argument('--lr_niter_frozen', type=int, default=1000, help='[lr_policy=lambda] # of iter at starting learning rate')
        self.parser.add_argument('--lr_niter_decay', type=int, default=3000, help='[lr_policy=lambda] # of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--lr_weight_decay', '-wd', type=float, default=0.001, required=False, help='The value of weight decay regularizer for Q optimizer (default %(default)s).')
        self.parser.add_argument('--lr_beta1', type=positive_float, default=0.9, required=False, help='The beta 1 for the optimizer (default %(default)s).')
        self.parser.add_argument('--lr_beta2', type=positive_float, default=0.999, required=False, help='The beta 2 for the optimizer (default %(default)s).')
        self.parser.add_argument('--lr_rho', type=positive_float, default=0.95, required=False, help='The rho for the optimizer (default %(default)s).')
        self.parser.add_argument('--lr_fuzz', type=positive_float, default=0.01, required=False, help='The fuzz factor for the "rmsprop" optimizer (default %(default)s).')
        self.parser.add_argument('--lr_momentum', type=positive_float, default=0.5, required=False, help='The momentum for the "sgd" optimizer and alpha parameter for "adam" (default %(default)s).')
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr_decay_every', type=int, default=100, help='[lr_policy=step] multiply by a gamma by lr_decay_every iterations')
        self.parser.add_argument('--gamma', type=positive_float, default=0.99, required=False, help='The discount factor of PPO advantage (default %(default)s).')  # Q LEARNING discount rate
        self.parser.add_argument('--update_target_interval', type=int, default=40000, required=False, help='Hard update the target network every many backprop steps (default %(default)s).')
        self.parser.add_argument('--tau', type=float, default=0.005, required=False, help='Soft update the target network at given rate (default %(default)s).')
        self.parser.add_argument('--cnn_input_proc', type=int, default=0, required=False, help='Use or not a CNN based feature extractor (default %(default)s).')
        self.parser.add_argument('--fine_tune', type=int, default=0, required=False, help='Train with a disrupted communication channel (default %(default)s).')
        self.parser.add_argument('--coord_mask_type', type=str.lower, default='optout', required=False, choices=['true', 'inverse', 'optout'], help='The coordination mask type to use in the loss (default %(default)s).')
        self.parser.add_argument('--ae_comm', type=int, default=0, required=False, help='Use the autoencoder for the message communication channel (default %(default)s).')
        self.parser.add_argument('--lambda_coord', type=positive_float, default=1, required=False, help='Weight for coordinator loss (default %(default)s).')
        self.parser.add_argument('--lambda_q', type=positive_float, default=1, required=False, help='Weight for Q network loss (default %(default)s).')
        self.parser.add_argument('--grad_clip_norm', type=int, default=5, required=False, help='Clip the gradient in norm 2 up to a certain value (default %(default)s).')
        self.parser.add_argument('--hi', type=int, default=128, required=False, help='Hidden size value input layer (default %(default)s).')
        self.parser.add_argument('--hs', type=int, default=1, required=False, help='Hidden type feature extractor (default %(default)s).')
        self.parser.add_argument('--hc', type=int, default=128, required=False, help='Hidden size coordinator (default %(default)s).')
        self.parser.add_argument('--hm', type=int, default=32, required=False, help='Hidden size mixer (default %(default)s).')

        self.isTrain = True


class EvalOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--models_path', type=str, required=True, default='./models_ckp/shapenet', help='path where models are stored')
        self.parser.add_argument('--model_epoch', type=int, required=True, default='-1', help='which epoch of the model to load from save_path. If set to -1 load the latest model')

        self.isTrain = False
