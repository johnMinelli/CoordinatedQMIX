from argparse import Namespace
from typing import Callable
import gym
import numpy as np
import torch

from core.base_agent import BaseAgent
from core.ma_gym.baselines.ppo_agent import PPOAgent
from core.ma_gym.comix.comix_agent import CoordQMixGymAgent
from core.ma_gym.baselines.covdn_agent import CoordVDNGymAgent
from core.ma_gym.baselines.idqn import IDQNGym
from core.ma_gym.baselines.comaddpg_agent import CoordMADDPGGymAgent
from core.ma_gym.baselines.qmix import QMixGym
from envs.wrappers import MaGymEnvWrap
from utils.logger import Logger

global _device

def play_online_loop(opt: Namespace, env_fn: Callable[[], MaGymEnvWrap], agent_fn: Callable[[Namespace, MaGymEnvWrap], object], logger: Logger):
    # Initialize elements
    env = env_fn()
    agent = agent_fn(opt, env)

    agents_ids = env.agents_ids

    for episode in range(agent.start_epoch, opt.episodes):
        logger.episode_start(episode)
        step = 0
        total_rewards = np.zeros(env.n_agents)

        hxs = agent.init_hidden()
        current_state_obs, _, dones, _ = env.reset()
        current_state_obs = torch.Tensor(np.stack([current_state_obs[id] for id in agents_ids])).to(_device)
        dones = torch.Tensor(np.expand_dims(np.stack([dones[id] for id in agents_ids]), -1)).to(_device)
        agent.reset_buffer()
        agent.fill_buffer(state=current_state_obs, step=0)

        # cycle until end of episode
        while not dones.all():
            # fill rollout and update for n times during an episode depending on the siwwze of rollout storage
            for step in range(opt.rollout_size):
                with torch.no_grad():
                    actions, log_prob, state_value, hxs = agent.take_action(current_state_obs, hxs, dones)
                actions[dones.squeeze().bool()] = env.no_op

                # env step
                next_state_obs, rewards_dict, dones, info = env.step(
                    list(map(lambda a: None if dones[a[0]] else a[1][0].cpu().numpy(), enumerate(actions))))
                if opt.render_mode == "human": env.render()

                rewards = np.array(list(rewards_dict.values()))
                total_rewards += rewards

                # preprocess new state
                rewards = torch.Tensor(np.expand_dims(rewards, -1)).to(_device)
                next_state_obs = torch.Tensor(np.stack([next_state_obs[id] for id in agents_ids])).to(_device)
                dones = torch.Tensor(np.expand_dims(np.stack([dones[id] for id in agents_ids]), -1)).to(_device)

                agent.fill_buffer(current_state_obs, hxs, actions, log_prob, state_value, rewards, dones)

                # preprocess new state
                current_state_obs = next_state_obs

                if dones.all():
                    break

            # make the update with data collected (ignore if it is just last step)
            if step > 1:
                # add last state value estimate needed for advantage computing and **UPDATE**
                with torch.no_grad():
                    state_value = agent.get_value(current_state_obs, hxs, dones)
                agent.fill_buffer(state_value=state_value, step=step+1)
                # update with batch of data collected
                losses = agent.step(valid_steps=step)
                # n_log_steps = n_updates if opt.log_per_update else 1
                logger.train_step(step+1, {**rewards_dict, **losses}, agent.schedulers[0].get_last_lr()[0])
            else:  # be careful with that, you are throwing away last step, but also the reward of done
                agent.buffer.drain(steps=step)

        # optionally save models
        if episode != 0 and (episode % opt.agent_save_interval == 0):
            print(f'Saving agent at episode: {episode}.')
            agent.save_model("policy", episode)

        logger.episode_stop({"rewards": {k: v for k, v in zip(env.agents_ids, total_rewards)}},
                            {"success": env.get_success_metric()})
        agent.update_learning_rate()


        if episode != 0 and (episode % opt.agent_valid_interval == 0):
            test_env = env_fn()
            agent.switch_mode('eval')
            total_rewards = np.zeros(test_env.n_agents)
            success = 0

            logger.valid_start()
            for episode in range(opt.val_episodes):

                hxs = agent.init_hidden()
                current_state_obs, _, dones, _ = test_env.reset(options={"test": True})
                current_state_obs = torch.Tensor(np.stack([current_state_obs[id] for id in agents_ids])).to(_device)
                dones = torch.Tensor(np.expand_dims(np.stack([dones[id] for id in agents_ids]), -1)).to(_device)
                while not dones.all():

                    actions, _, _, hxs = agent.take_action(current_state_obs, hxs, dones)
                    next_state_obs, rewards_dict, dones, info = test_env.step(
                        list(map(lambda a: None if dones[a[0]] else a[1][0].cpu().numpy(), enumerate(actions))))
                    if opt.render_mode == "human" or opt.render_mode == "human_val": test_env.render()
                    total_rewards += np.array(list(rewards_dict.values()))
                    next_state_obs = torch.Tensor(np.stack([next_state_obs[id] for id in agents_ids])).to(_device)
                    dones = torch.Tensor(np.expand_dims(np.stack([dones[id] for id in agents_ids]), -1)).to(_device)

                    current_state_obs = next_state_obs
                logger.valid_step(1, {})
                success += test_env.get_success_metric()
            test_env.close()
            logger.valid_stop({"rewards": {k: v for k, v in zip(test_env.agents_ids, total_rewards)}},
                              {"success": success / opt.val_episodes})
            agent.switch_mode('train')
    env.close()


def play_loop(opt: Namespace, env_fn: Callable[[], MaGymEnvWrap], agent_fn: Callable[[Namespace, MaGymEnvWrap], BaseAgent], logger: Logger):
    # Initialize elements
    env = env_fn()
    agent = agent_fn(opt, env)

    agents_ids = env.agents_ids

    for episode in range(agent.start_epoch, opt.episodes):
        warmup_ended = agent.learning
        if warmup_ended: logger.episode_start(episode)
        total_rewards = np.zeros(env.n_agents)

        agent.decay_exploration(episode)
        hxs = agent.init_hidden()
        current_state_obs, _, dones, _ = env.reset()
        current_state_obs = torch.Tensor(np.stack([current_state_obs[id] for id in agents_ids])).to(_device)
        dones = torch.Tensor(np.expand_dims(np.stack([dones[id] for id in agents_ids]), -1)).to(_device)
        # cycle until end of episode
        while not dones.all():
            # select agent actions from observations
            actions, hxs, add_in = agent.take_action(current_state_obs, hxs, dones)

            # env step
            next_state_obs, rewards_dict, dones, info = env.step(list(map(lambda a: None if dones[a[0]] else a[1][0].cpu().numpy(), enumerate(actions))))
            if opt.render_mode == "human": env.render()

            rewards = np.array(list(rewards_dict.values()))
            total_rewards += rewards

            # preprocess new state
            rewards = torch.Tensor(np.expand_dims(rewards, -1)).to(_device)
            next_state_obs = torch.Tensor(np.stack([next_state_obs[id] for id in agents_ids])).to(_device)
            dones = torch.Tensor(np.expand_dims(np.stack([dones[id] for id in agents_ids]), -1)).to(_device)

            # let agent store it and take a learning take step (if `warmup_ended`); skip if ended by max_step
            n_updates, losses = agent.step(current_state_obs, add_in, actions, rewards, next_state_obs, dones)
            if warmup_ended:
                n_log_steps = n_updates if opt.log_per_update else 1
                logger.train_step(n_log_steps, {**rewards_dict, **losses}, agent.schedulers[0].get_last_lr()[0])
            # update for next iteration
            current_state_obs = next_state_obs

        if warmup_ended:
            # optionally save models
            if episode != 0 and (episode % opt.agent_save_interval == 0):
                print(f'Saving agent at episode: {episode}.')
                agent.save_model("policy", episode)

            logger.episode_stop({"rewards": {k: v for k,v in zip(env.agents_ids, total_rewards)}}, {"success": env.get_success_metric()})
            agent.update_learning_rate()

            # hard update (if implemented)
            if (episode%opt.update_target_interval)==0:
                agent.update_target_net()

            if episode != 0 and (episode % opt.agent_valid_interval == 0):
                test_env = env_fn()
                agent.switch_mode('eval')
                total_rewards = np.zeros(test_env.n_agents)
                success = 0

                logger.valid_start()
                for episode in range(opt.val_episodes):

                    hxs = agent.init_hidden()
                    current_state_obs, _, dones, _ = test_env.reset(options={"test": True})
                    current_state_obs = torch.Tensor(np.stack([current_state_obs[id] for id in agents_ids])).to(_device)
                    dones = torch.Tensor(np.expand_dims(np.stack([dones[id] for id in agents_ids]), -1)).to(_device)
                    while not dones.all():

                        actions, hxs, _ = agent.take_action(current_state_obs, hxs, dones)
                        next_state_obs, rewards_dict, dones, info = test_env.step(list(map(lambda a: None if dones[a[0]] else a[1][0].cpu().numpy(), enumerate(actions))))
                        if opt.render_mode == "human" or opt.render_mode == "human_val": test_env.render()
                        total_rewards += np.array(list(rewards_dict.values()))
                        next_state_obs = torch.Tensor(np.stack([next_state_obs[id] for id in agents_ids])).to(_device)
                        dones = torch.Tensor(np.expand_dims(np.stack([dones[id] for id in agents_ids]), -1)).to(_device)

                        current_state_obs = next_state_obs
                    logger.valid_step(1, {})
                    success += test_env.get_success_metric()
                test_env.close()
                logger.valid_stop({"rewards": {k: v/opt.val_episodes for k,v in zip(test_env.agents_ids, total_rewards)}}, {"success": success/opt.val_episodes})
                agent.switch_mode('train')
    env.close()

    print(f'Saving agent at episode: {opt.episodes}.')
    models = agent.save_model("policy", opt.episodes)
    for name, path in models.items():
        logger.log_artifact(path, name, f"checkpoint_{opt.episodes}")


def gym_loop(args: Namespace, device: torch.device, logger: Logger):
    global _device
    _device = device

    def create_env_fn():
        # Create the game environment; the env name is filtered by parser
        if "switch" in args.env:
            gym.envs.register(id='CustomSwitch4-v0', entry_point='envs.switch:Switch', kwargs=
            {'n_agents': 4, 'full_observable': False, 'step_cost': 0.0, 'max_steps': 500})
            env = MaGymEnvWrap(gym.make("CustomSwitch4-v0"))
        elif "predator_prey" in args.env:
            xy, n = (14, 8) if "predator_prey_8" in  args.env else (16, 16) if "predator_prey_16" in args.env else (12, 4)
            gym.envs.register(id='CustomPredatorPrey-v0', entry_point='envs.predator_prey:PredatorPrey', kwargs=
            {'grid_shape': (xy, xy), 'n_agents': n, 'n_preys': 16, 'prey_move_probs': (0.2, 0.2, 0.2, 0.2, 0.2), 'full_observable': False, 'penalty': 0, 'step_cost': 0, 'max_steps': 500, 'agent_view_range': (5, 5)})
            env = MaGymEnvWrap(gym.make("CustomPredatorPrey-v0"))
        elif "transport" in args.env:
            n, h = (4, 2) if "transport_2" in args.env else (8, 4) if "transport_4" in args.env else (2, 1)
            gym.envs.register(id='CustomTransport-v0', entry_point='envs.transport:Transport', kwargs=
            {'grid_size': (16,10), 'n_agents': n, 'n_loads': h, 'full_observable': False, 'step_cost': 0, 'max_steps': 500, 'agent_view_range': (5, 5)})
            env = MaGymEnvWrap(gym.make("CustomTransport-v0"))
        else:
            env = gym.make(args.env)
        return env

    def create_agent_fn(opt: Namespace, env):
        # Choose the agent implementation
        if args.agent == 'ippo':
            args.lr = 0.0005
            args.adv_gamma = 0.99
            args.adv_lambda = 0.95
            args.K_epochs = 1
            args.rollout_size = 2048
            agent = PPOAgent(opt, env, _device)
        elif args.agent == 'idqn':
            agent = IDQNGym(opt, env, _device)
        elif args.agent == 'vdn':
            agent = CoordVDNGymAgent(opt, env, _device)
        elif args.agent == 'qmix':
            agent = QMixGym(opt, env, _device)
        elif args.agent == 'maddpg':
            args.gumbel_max_temp = 10
            args.gumbel_min_temp = 0.1
            agent = CoordMADDPGGymAgent(opt, env, _device)
        else:
            # my implementation
            agent = CoordQMixGymAgent(opt, env, _device)

        return agent

    # run loop
    # play_online_loop(args, create_env_fn, create_agent_fn, logger) 
    play_loop(args, create_env_fn, create_agent_fn, logger)
