import time

import numpy as np
import wandb
from progressbar import progressbar


class Logger(object):

    def __init__(self, valid, episodes, batch_size, terminal_print_freq=1, tensorboard=None, wand=None):
        self.valid = valid
        self.episodes = episodes
        self.batch_size = batch_size
        self.print_freq = terminal_print_freq
        self.tensorboard = tensorboard
        self.wandb = wand
        self.total_steps = 0

        s = 10
        e = 1  # episode bar position
        tr = 3  # train bar position
        ts = 6  # valid bar position
        h = 100

        self.progress_bar = None
        self.epoch_bar = None
        self.epoch = None
        self.t = None

        self.writer = Writer(self.t, (0, h - s + tr))
        self.bar_writer = Writer(self.t, (0, h - s + tr + 1))
        self.progress_bar = progressbar.ProgressBar(maxval=self.episodes, fd=Writer(self.t, (0, h - s + e)))
        [print('') for i in range(2)]
        self.progress_bar.start()

    def set_tensorboard(self, writer):
        self.tensorboard = writer

    def set_wandb(self, writer):
        self.wandb = writer

    def log(self, text):
        self.writer.write(text)

    def episode_start(self, episode):
        self.episode = episode
        self.steps = 0
        self.total_metrics = {}
        self.episode_start_time = time.time()
        return self

    def valid_start(self):
        assert self.valid, "Logger not initialized with validation settings."
        self.holding_episode_info = (self.episode, self.steps, self.total_metrics, self.episode_start_time)
        self.episode_start(0)

    def _step(self, steps, metrics):
        self.steps += steps
        self.total_steps += steps
        # losses error & metrics
        for k, v in metrics.items():
            self.total_metrics[k] = (self.total_metrics.get(k, []) + [v])
        avg_metrics = [np.mean(v) for v in metrics.values()]
        return avg_metrics

    def train_step(self, steps, metrics, lr):
        assert self.episode is not None, "You should call `episode_start` first."
        avg_metrics = self._step(steps, metrics)
        if len(avg_metrics) > 0:
            prefix = "train"
            if not steps%self.print_freq: self.log(' * ' + ', '.join(['Avg ' + str(k).capitalize() + ' : {:.5f}'.format(v) for k, v in zip(self.total_metrics.keys(), avg_metrics)]) + '\tLr: ' + str(lr))
            self._log_stats_to_dashboards(self.total_steps, prefix, {str(k).capitalize(): v for k, v in zip(self.total_metrics.keys(), avg_metrics)})
            self._log_stats_to_dashboards(self.total_steps, prefix, {"lr": lr})

    def valid_step(self, steps, metrics):
        assert self.episode is not None, "You should call `valid_start` first."
        avg_metrics = self._step(steps, metrics)
        if len(avg_metrics) > 0:
            prefix = "valid"
            if not steps%self.print_freq: self.log(' * ' + ', '.join(['Avg ' + str(k).capitalize() + ' : {:.5f}'.format(v) for k, v in zip(self.total_metrics.keys(), avg_metrics)]))
            self._log_stats_to_dashboards(self.total_steps, prefix, {str(k).capitalize(): v for k, v in zip(self.total_metrics.keys(), avg_metrics)})

    def episode_stop(self, env_stats, num_stats):
        assert self.episode is not None, "You should call `episode_start` first."
        assert "rewards" in env_stats and isinstance(env_stats["rewards"], dict) is not None, "The rewards:dict field is necessary"
        episode_time = time.time() - self.episode_start_time
        avg_time = episode_time / self.steps

        total_reward = env_stats["rewards"]
        tot_actors_reward = {"Tot_reward_"+k: r for k,r in total_reward.items()}
        avg_actors_reward = {"Avg_reward_"+k: r / self.steps for k,r in total_reward.items()}
        tot_reward = np.mean(np.array(list(tot_actors_reward.values())))
        avg_reward = np.mean(np.array(list(avg_actors_reward.values())))
        avg_reward_over_time = avg_reward / episode_time

        tot_env_stats = {k: sum([vi for vi in list(v.values())]) for k, v in env_stats.items() if k != "rewards"}

        avg_metrics = [np.mean(v) for v in list(self.total_metrics.values())] if len(self.total_metrics) > 0 else []

        self.log('Ep: %d / %d - Time: %d sec' % (self.episode, self.episodes, episode_time) + '\t' +
                 ' * Tot Reward : {:.5f}'.format(tot_reward) + ', Avg Reward : {:.5f}'.format(avg_reward) + ', Reward/time : {:.5f}'.format(avg_reward_over_time) +
                 (' - Avg Metrics : [' + ', '.join([str(l) for l in avg_metrics]) + ']' if len(avg_metrics)>0 else '') +
                 ' - Avg Time : {:.3f}'.format(avg_time))
        self._log_stats_to_dashboards(self.total_steps, "Train", {**tot_actors_reward, **avg_actors_reward, "Avg_reward": avg_reward, "Reward_over_time": avg_reward_over_time, "Avg_time": avg_time, **tot_env_stats, **num_stats})

        if self.progress_bar is not None:
            self.progress_bar.update(self.episode + 1)
            if self.episode + 1 == self.episodes:
                self.progress_bar.finish()

        return avg_time, avg_metrics, avg_reward

    def valid_stop(self, env_stats, num_stats):
        assert self.episode is not None, "You should call `episode_start` first."
        assert "rewards" in env_stats and isinstance(env_stats["rewards"], dict) is not None, "The rewards:dict field is necessary"
        episode_time = time.time() - self.episode_start_time

        total_reward = env_stats["rewards"]
        avg_actors_reward = {"Avg_reward_"+k: r / self.steps for k,r in total_reward.items()}
        avg_reward = np.mean(np.array(list(avg_actors_reward.values())))
        avg_reward_over_time = avg_reward / episode_time

        self.log('Ep: %d / %d - Time: %d sec' % (self.episode, self.episodes, episode_time) + '\t' +
                 ' * Avg Reward : {:.5f}'.format(avg_reward) + ', Reward/time : {:.5f}'.format(avg_reward_over_time))
        self._log_stats_to_dashboards(self.total_steps, "Valid", {**avg_actors_reward, "Avg_reward": avg_reward, "Reward_over_time": avg_reward_over_time}, **num_stats)

        # restore values
        self.episode, self.steps, self.total_metrics, self.episode_start_time = self.holding_episode_info

    def _log_stats_to_dashboards(self, step, prefix, stats):
        for name, value in stats.items():
            namet = prefix.capitalize() + "/" + prefix.lower() + "_" + name
            namew = prefix.capitalize() + "/" + prefix.lower() + "_" + name
            if self.tensorboard is not None:
                self.tensorboard.add_scalar(namet, value, step)
            if self.wandb:
                wandb.log({namew: value}, step)


class Writer(object):
    """Create an object with a write method that writes to a
    specific place on the screen, defined at instantiation.

    This is the glue between blessings and progressbar.
    """

    def __init__(self, t, location):
        """
        Input: location - tuple of ints (x, y), the position
                        of the bar in the terminal
        """
        self.location = location
        self.t = t

    def write(self, string):
        print(string)

    def flush(self):
        return

