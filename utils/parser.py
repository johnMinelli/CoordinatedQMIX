from argparse import ArgumentParser, ArgumentTypeError, ArgumentDefaultsHelpFormatter
from utils.utils import mkdirs
from path import Path


def positive_int(value: any) -> int:
    """
    Checks if a value is a positive integer.

    :param value: the value to be checked.
    :return: the value if valid integer, otherwise raises an ArgumentTypeError.
    """
    int_value = int(value)

    if int_value <= 0:
        raise ArgumentTypeError("%s should be a positive integer value." % value)

    return int_value


def positive_float(value: any) -> float:
    """
    Checks if a value is a positive float.

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
        self.parser.add_argument('--env', default='pursuit', choices=['CoMix_pursuit4', 'CoMix_pursuit8',  'CoMix_pursuit16', 'pursuit_dev', 'CoMix_switch', 'switch_dev', 'checkers'], help='Name of the environment (default: %(default)s)')

        # model
        self.parser.add_argument('-s', '--save_path', type=str, default='./save', help='Checkpoints are saved here')
        self.parser.add_argument('-as', '--agent_save_interval', type=positive_int, default=5, required=False, help='The save interval for the trained agent (default %(default)s), in episodes.')
        self.parser.add_argument('-ae', '--agent_valid_interval', type=positive_int, default=5, required=False, help='The eval interval for the trained agent (default %(default)s), in episodes.')
        self.parser.add_argument('--load_path', type=str, required=False, default=None, help='Path where to search the trained agents to be loaded (default `save_path`/models.')
        self.parser.add_argument('--load_agent', type=int, required=False, default=None, help='Label of a trained solver agent to be loaded from `load_path` (default %(default)s). -1 to load the last saved model in the folder.')
        self.parser.add_argument('--agent', type=str, required=False, default="comix", help='Label of a trained solver agent to be loaded from `load_path` (default %(default)s). -1 to load the last saved model in the folder.')

        # output
        self.parser.add_argument('-op', '--out_path', type=str, default='./out', help='Results are saved here')
        self.parser.add_argument('-r', '--render_mode', default='human', required=False, type=str.lower, choices=['human', 'human_val', 'none'], help='Modality of rendering of the environment.')
        self.parser.add_argument('-rec', '--record', required=False, action='store_true', help='Whether the game should be recorded. Please note that you need to have ffmpeg in your path!')

        self.parser.add_argument('-wd', '--weight_decay', type=float, default=0.0001, required=False, help='The value of weight decay regularizer for Q optimizer (default %(default)s).')
        self.parser.add_argument('-ep', '--episodes', type=positive_int, default=1200, required=False, help='The episodes to run the training procedure (default %(default)s).')
        self.parser.add_argument('-vep', '--val_episodes', type=positive_int, default=1, required=False, help='The episodes to run the validation procedure (default %(default)s).')
        self.parser.add_argument('-bs', '--batch_size', type=positive_int, default=16, required=False, help='The batch size to be sampled from the memory for the training (default %(default)s).')
        self.parser.add_argument('-k', '--K_epochs', type=float, default=0.02, required=False, help='The number of epochs to run on the single batch (default %(default)s).')
        self.parser.add_argument('-ck', '--chunk_size', type=positive_int, default=250, required=False, help='The size of the sequence for each sample (default %(default)s).')
        self.parser.add_argument('--min_buffer_len', type=positive_int, default=3000, required=False, help='The number of necessary samples in the buffer for training (default %(default)s).')
        self.parser.add_argument('--max_buffer_len', type=positive_int, default=30000, required=False, help='The maximum number of samples in the buffer for training (default %(default)s).')
        self.parser.add_argument('--rollout_size', type=positive_float, default=200, required=False, help='Maximum number of samples to collect in replay storage (default %(default)s).')

        # cmd = "nvidia-docker run --rm -e NVIDIA_VISIBLE_DEVICES={} -p {}-{}:2000-2002 {} /bin/bash -c \"sed -i '5i sync' ./CarlaUE4.sh; ./CarlaUE4.sh /Game/Maps/Town01 -carla-server -benchmark -fps=10 -carla-settings=\"CarlaSettings.ini\"\"".format(
        #     gpu_id, port, port + 2, args.image_name)

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
        self.opt.models_path = Path(self.opt.save_path)/"models/"
        # self.opt.plots_path = Path(self.opt.out_path)/self.opt.name/"plots/"
        # self.opt.results_path = Path(self.opt.out_path)/self.opt.name/"results/"

        mkdirs(self.opt.models_path)
        # create_path(self.opt.plots_path)
        # create_path(self.opt.results_path)

        if self.opt.load_path is None:
            self.opt.load_path = self.opt.models_path


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--parallel_envs', default=1, type=int, help='Number of parallel environments to run (default: %(default)s)')

        self.parser.add_argument('-pf', '--print_freq', type=int, default=500, help='Frequency of showing training results on console')
        self.parser.add_argument('-lu', '--log_per_update', action='store_true', help='Log per weights update instead of per env step')
        self.parser.add_argument('-t', '--tensorboard', action='store_true', help='log stats on tensorboard local dashboard')
        self.parser.add_argument('--wandb', action='store_true', help='log stats on wandb dashboard')
        self.parser.add_argument('--sweep_id', type=str, help='sweep id for wandb hyperparameters search e.g. user/project/sweep')

        self.parser.add_argument('--continue_train', type=int, default=None, help='continue training: if set to -1 load the latest model from save_path')
        self.parser.add_argument('-opt', '--optimizer', type=str.lower, default='adam', required=False, choices=['adam', 'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adamax'], help='The optimizer to be used. (default %(default)s).')
        self.parser.add_argument('--lr', type=positive_float, default=0.0005, help='initial learning rate')
        self.parser.add_argument('--lr_c', type=positive_float, default=0.001, help='initial learning rate')
        self.parser.add_argument('--lr_q', type=positive_float, default=0.0007, help='initial learning rate')
        self.parser.add_argument('--lr_co', type=positive_float, default=0.00008, help='initial learning rate')
        self.parser.add_argument('--lr_ae', type=positive_float, default=0.001, help='initial learning rate')
        self.parser.add_argument('--lr_niter_frozen', type=int, default=1000, help='[lr_policy=lambda] # of iter at starting learning rate')
        self.parser.add_argument('--lr_niter_decay', type=int, default=1000, help='[lr_policy=lambda] # of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--lr_weight_decay', default=0.0001, type=float, help='Weight decay for "adam"')
        self.parser.add_argument('--lr_beta1', type=positive_float, default=0.9, required=False, help='The beta 1 for the optimizer (default %(default)s).')
        self.parser.add_argument('--lr_beta2', type=positive_float, default=0.999, required=False, help='The beta 2 for the optimizer (default %(default)s).')
        self.parser.add_argument('--lr_rho', type=positive_float, default=0.95, required=False, help='The rho for the optimizer (default %(default)s).')
        self.parser.add_argument('--lr_fuzz', type=positive_float, default=0.01, required=False, help='The fuzz factor for the "rmsprop" optimizer (default %(default)s).')
        self.parser.add_argument('--lr_momentum', type=positive_float, default=0.5, required=False, help='The momentum for the "sgd" optimizer and alpha parameter for "adam" (default %(default)s).')
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr_decay_every', type=int, default=100, help='[lr_policy=step] multiply by a gamma by lr_decay_every iterations')
        self.parser.add_argument('--gamma', type=positive_float, default=0.99, required=False, help='The discount factor of PPO advantage (default %(default)s).')  # Q LEARNING discount rate
        self.parser.add_argument('--min_epsilon', type=positive_float, default=0.15, required=False, help='Min temperature of randomness in acting while training. Reached at 60% of episodes done. (default %(default)s).')
        self.parser.add_argument('--decay_epsilon', type=positive_float, default=0.33, required=False, help='Min temperature of randomness in acting while training. Reached at 60% of episodes done. (default %(default)s).')
        self.parser.add_argument('--max_epsilon', type=positive_float, default=0.90, required=False, help='Max temperature of randomness in acting while training (default %(default)s).')
        self.parser.add_argument('--update_target_interval', type=int, default=1000, required=False, help='Hard update the target network every many backprop steps (default %(default)s).')
        self.parser.add_argument('--tau', type=float, default=0.005, required=False, help='Soft update the target network at given rate (default %(default)s).')
        self.parser.add_argument('--cnn_input_proc', type=int, default=0, required=False, help='Use or not a CNN based feature extractor (default %(default)s).')
        self.parser.add_argument('--coord_mask_type', type=str.lower, default='optout', required=False, choices=['true', 'inverse', 'optout'], help='The coordination mask type to use in the loss (default %(default)s).')
        self.parser.add_argument('--ae_comm', type=int, default=0, required=False, help='Use the autoencoder for the message communication channel (default %(default)s).')
        self.parser.add_argument('--lambda_distance', type=positive_float, default=0, required=False, help='Weight for coordinator loss over logits distance (default %(default)s).')
        self.parser.add_argument('--lambda_coord', type=positive_float, default=1, required=False, help='Weight for coordinator loss (default %(default)s).')
        self.parser.add_argument('--lambda_q', type=positive_float, default=1, required=False, help='Weight for Q network loss (default %(default)s).')
        self.parser.add_argument('--grad_clip_norm', type=int, default=5, required=False, help='Clip the gradient in norm 2 up to a certain value (default %(default)s).')
        self.parser.add_argument('--hi', type=int, default=32, required=False, help='Hidden size value input layer (default %(default)s).')
        self.parser.add_argument('--hs', type=int, default=0, required=False, help='Hidden size feature extractor  (default %(default)s).')
        self.parser.add_argument('--hc', type=int, default=64, required=False, help='Hidden size coordinator  (default %(default)s).')
        self.parser.add_argument('--hm', type=int, default=32, required=False, help='Hidden size mixer  (default %(default)s).')

        self.isTrain = True

class EvalOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--models_path', type=str, required=True, default='./models_ckp/shapenet', help='path where models are stored')
        self.parser.add_argument('--model_epoch', type=int, required=True, default='-1', help='which epoch of the model to load from save_path. If set to -1 load the latest model')
        self.parser.add_argument("--test_file", type=str, required=True, help='file with the pairs of the test split')
        self.isTrain = False

