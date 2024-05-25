import copy
import logging

import gym
import math
import numpy as np
from PIL import ImageColor
from gym import spaces
from gym.utils import seeding

from envs.utils.action_space import MultiAgentActionSpace
from envs.utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text
from envs.utils.observation_space import MultiAgentObservationSpace

logger = logging.getLogger(__name__)


class Transport(gym.Env):
    """
    Pairs of two agents are need to transport a laad to the other side of a map in the docking area.
    The agents are attached to the load and cannot move away from it. The movement is allowed only if both select the
    same action ∈ {Left, Right, Up, Down, Rotate} at each time step resulting in the movement of the load as well.
    Presence of obstacles in the map make the task harder, since neither the agents nor the load can occupy the same
    position as one obstacle. Each pair is rewarded when accomplish its task, default value is 5. An auxiliary reward
    for decreasing the smallest distance between load-dock position is delivered to the agents, default 0.5.

    Each agent’s observation includes:
    - agent absolute coordinate position normalized to 1. e.g. [1.0,1.0] for br corner
    - boolean array for the cells in the observation area describing the presence of obstacle.
    e.g. [0]x(5,5) with an observation are af (5,5)

    The terminating condition of this task is when all agents take their load to the docking area of after n steps.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, grid_size=(16, 16), n_agents=2, n_loads=4, full_observable=False, step_cost=-0.01, dock_reward=5,
                 aux_reward=0.5, max_steps=500, agent_view_range=(5, 5)):
        assert len(grid_size) == 2, 'expected a tuple of size 2 for grid_shape, but found {}'.format(grid_size)
        assert len(agent_view_range) == 2, 'expected a tuple of size 2 for agent view mask,' \
                                           ' but found {}'.format(agent_view_range)
        assert grid_size[0] > 0 and grid_size[1] > 0, 'grid shape should be > 0'
        assert 1 <= n_loads <= 4, 'The environment support maximum 4 loads'
        assert n_agents == n_loads * 2, 'The environment should be initialized with a number of agents double respect the loads'

        self.n_agents = n_agents
        self.n_agents_dummy = n_agents  # + add dummy here for tests
        self.n_loads = n_loads
        self.full_observable = full_observable
        self._grid_size = grid_size
        self._max_steps = max_steps
        self._step_count = None
        self._steps_beyond_done = None
        self._step_cost = step_cost
        self._dock_reward = dock_reward
        self._aux_reward = aux_reward
        self._agent_view_range = agent_view_range
        self.viewer = None

        # Env constants
        mask_size = np.prod(self._agent_view_range)
        self._obs_high = np.array([1., 1.] + [1., 1.] + ([1.] * mask_size) + [1.] ,  dtype=np.float32)
        self._obs_low = np.array([0., 0.] + [0., 0.] + ([0.] * mask_size) + [0.] , dtype=np.float32)
        if self.full_observable:
            self._obs_high = np.tile(self._obs_high, self.n_agents)
            self._obs_low = np.tile(self._obs_low, self.n_agents)
        self.observation_space = MultiAgentObservationSpace(
            [spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])

        self.action_space = MultiAgentActionSpace([spaces.Discrete(5) for _ in range(self.n_agents)])
        self.agent_ids = [str(a) for a in range(self.n_agents)]

        self._init_load_pos = []
        self._init_agent_pos = []
        self._grid_shape = grid_size
        self._init_load_pos.append(
            [1, (math.ceil(self._grid_size[1] / 2) - 1) if self.n_loads <= 2 else self._grid_size[0] - 2])
        self._init_agent_pos.append([self._init_load_pos[-1][0], self._init_load_pos[-1][1] - 1])
        self._init_agent_pos.append([self._init_load_pos[-1][0], self._init_load_pos[-1][1] + 1])
        self._nav_area = [[3, self._grid_shape[0]], [0, self._grid_shape[1]]]
        if self.n_loads >= 2:
            self._grid_shape = [(grid_size[0] * 2) - 3, grid_size[1]]
            self._init_load_pos.append([self._grid_shape[0] - 2,
                                        (math.ceil(self._grid_size[1] / 2) - 1) if self.n_loads <= 2 else
                                        self._grid_size[0] - 2])
            self._init_agent_pos.append([self._init_load_pos[-1][0], self._init_load_pos[-1][1] - 1])
            self._init_agent_pos.append([self._init_load_pos[-1][0], self._init_load_pos[-1][1] + 1])
            self._nav_area = [[3, self._grid_shape[0]-3], [0, self._grid_shape[1]]]
        if self.n_loads >= 3:
            self._grid_shape = [(grid_size[0] * 2) - 3, grid_size[0]]
            self._init_load_pos.append([self._grid_size[0] - math.ceil(self._grid_size[1] / 2), 1])
            self._init_agent_pos.append([self._init_load_pos[-1][0] - 1, self._init_load_pos[-1][1]])
            self._init_agent_pos.append([self._init_load_pos[-1][0] + 1, self._init_load_pos[-1][1]])
            self._nav_area = [[3, self._grid_shape[0]-3], [3, self._grid_shape[1]]]
        if self.n_loads >= 4:
            self._grid_shape = [(grid_size[0] * 2) - 3, (grid_size[0] * 2) - 3]
            self._init_load_pos.append(
                [self._grid_size[0] - math.ceil(self._grid_size[1] / 2), self._grid_shape[1] - 2])
            self._init_agent_pos.append([self._init_load_pos[-1][0] - 1, self._init_load_pos[-1][1]])
            self._init_agent_pos.append([self._init_load_pos[-1][0] + 1, self._init_load_pos[-1][1]])
            self._nav_area = [[3, self._grid_shape[0]-3], [3, self._grid_shape[1]-3]]

        self._dock_pos = [grid_size[0] - 2,
                          (math.ceil(self._grid_size[1] / 2) - 1) if self.n_loads <= 2 else self._grid_size[0] - 2]
        self._max_dist = l1_distance(self._init_load_pos[0], self._dock_pos)

        # Episode variables
        self.agent_pos = {}
        self.load_pos = {}
        self._agent_dones = [False for _ in range(self.n_agents)]
        self._best_dock_dist = [99 for _ in range(self.n_loads)]
        self._loads_transported = 0
        self._total_episode_reward = None
        self._full_obs = self.__create_grid()

        self.seed()

    @property
    def step_count(self):
        return self._step_count

    @property
    def max_steps(self):
        return self._max_steps

    @property
    def success(self):
        tot = 0
        for i in range(self.n_loads):
            tot += 100 - (self._best_dock_dist[i] * 100 / l1_distance(self._init_load_pos[i], self._dock_pos))
        tot /= self.n_loads
        return int(tot)

    def get_action_meanings(self, agent_i=None):
        if agent_i is not None:
            assert agent_i <= self.n_agents
            return [ACTION_MEANING[i] for i in range(self.action_space[agent_i].n)]
        else:
            return [[ACTION_MEANING[i] for i in range(ac.n)] for ac in self.action_space]

    def action_space_sample(self):
        return [agent_action_space.sample() for agent_action_space in self.action_space]

    def __draw_base_img(self):
        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')
        for col in range(self._grid_shape[0]):
            for row in range(self._grid_shape[1]):
                if self._wall_exists((col, row)):
                    fill_cell(self._base_img, (col, row), cell_size=CELL_SIZE, fill=WALL_COLOR)

        fill_cell(self._base_img, self._dock_pos, cell_size=CELL_SIZE, fill=DOCK_COLOR)

    def __create_grid(self, test=False):
        x, y = self._grid_shape
        _grid = np.zeros((3, x, y))
        # add walls
        n = int((self._nav_area[0][1]-self._nav_area[0][0]) * (self._nav_area[1][1]-self._nav_area[1][0]) * 0.1)
        for i in range(n):
            x = np.random.randint(self._nav_area[0][0], self._nav_area[0][1])
            y = np.random.randint(self._nav_area[1][0], self._nav_area[1][1])
            if not self._dock_area((x, y)):
                _grid[WALL][x][y] = 1
        return _grid

    def __init_world(self, test=False):
        self._full_obs = self.__create_grid(test)
        # agents
        for i in range(self.n_agents):
            self.agent_pos[i] = self._init_agent_pos[i]
            self.__update_agent_view(i)

        for i in range(self.n_loads):
            self.load_pos[i] = self._init_load_pos[i]
            self.__update_load_view(i)
            self._update_docking_distance(i, reset=True)

        self.__draw_base_img()

    def get_agent_obs(self):
        _obs = []
        for agent_i in range(self.n_agents):
            pos = self.agent_pos[agent_i]
            # absolute coordinates
            _agent_i_obs = [round(pos[0] / (self._grid_shape[0] - 1), 2), round(pos[1] / (self._grid_shape[1] - 1), 2)]
            # dock coordinates
            _agent_i_obs.append(1-round(abs(pos[0] - self._dock_pos[0]) / (self._grid_size[0] - 1), 2))
            _agent_i_obs.append(1-round(abs(pos[1] - self._dock_pos[1]) / (self._grid_size[0] - 1), 2))
            # adding walls pos in observable area
            x_m = math.floor(self._agent_view_range[0] / 2)
            y_m = math.floor(self._agent_view_range[1] / 2)
            _obstacles_pos = np.zeros(self._agent_view_range)  # walls
            for i, col in enumerate(range(pos[0] - x_m, pos[0] + x_m + 1)):
                for j, row in enumerate(range(pos[1] - y_m, pos[1] + y_m + 1)):
                        if not self.is_valid((col, row)) or self._wall_exists((col, row)):
                            _obstacles_pos[i, j] = 1
            _agent_i_obs += _obstacles_pos.flatten().tolist()
            _agent_i_obs += [self._step_count / self._max_steps]  # adding time
            _obs.append(_agent_i_obs)

        if self.full_observable:
            _obs = np.array(_obs).flatten().tolist()
            _obs = [_obs for _ in range(self.n_agents)]
        return _obs

    def reset(self, seed=None, options=None):
        if options is None: options = {}
        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        self.agent_pos = {}
        self.load_pos = {}

        self.__init_world(options.get("test", False))
        self._step_count = 0
        self._steps_beyond_done = None
        self._agent_dones = [False for _ in range(self.n_agents)]
        self._loads_transported = 0

        return self.get_agent_obs(), {}

    def _update_docking_distance(self, load_i, reset=False):
        dist = l1_distance(self.load_pos[load_i], self._dock_pos)
        if dist < self._best_dock_dist[load_i] or reset:
            self._best_dock_dist[load_i] = dist
            return True
        else:
            return False

    def _dock_area(self, pos, area=1):
        return (self._dock_pos[0] - area <= pos[0] <= self._dock_pos[0] + area and self._dock_pos[1] - area <= pos[1] <=
                self._dock_pos[1] + area)

    def _wall_exists(self, pos):
        return self._full_obs[WALL][pos[0]][pos[1]] == 1

    def is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    def _is_cell_loadless(self, pos):
        return self._full_obs[LOAD][pos[0]][pos[1]] == 0

    def _is_cell_agentless(self, pos):
        return self._full_obs[AGENT][pos[0]][pos[1]] == 0

    def __update_agent_view(self, agent_i):
        self._full_obs[AGENT][self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]] += 1

    def __update_load_view(self, prey_i):
        self._full_obs[LOAD][self.load_pos[prey_i][0]][self.load_pos[prey_i][1]] += 1

    def __next_agent_pos(self, curr_pos, center, move):
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # rotate
            translated_point = [curr_pos[0] - center[0], curr_pos[1] - center[1]]
            next_pos = [center[0] + translated_point[1], center[1] - translated_point[0]]
        elif move == 5:  # no-op
            next_pos = curr_pos
        else:
            raise Exception('Action Not found!')
        return next_pos

    def __next_load_pos(self, curr_pos, move):
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # rotate
            next_pos = curr_pos
        elif move == 5:  # no-op
            next_pos = curr_pos
        else:
            raise Exception('Action Not found!')
        return next_pos

    def step(self, agents_action):
        assert (self._step_count is not None), "Call reset before using step method."

        # list to dict
        agents_action = {agent_i: action for agent_i, action in zip(range(self.n_agents), agents_action)}

        self._step_count += 1
        rewards = [self._step_cost for _ in range(self.n_agents)]

        for load_i in range(self.n_loads):
            agent_1 = load_i*2
            agent_2 = (load_i*2)+1
            if not self._agent_dones[agent_1] and agents_action[agent_1] == agents_action[agent_2]:
                cur_load_pos = copy.copy(self.load_pos[load_i])
                next_load_pos = self.__next_load_pos(cur_load_pos, agents_action[agent_2])
                agent_1_pos = copy.copy(self.agent_pos[agent_1])
                next_agent_1_pos = self.__next_agent_pos(agent_1_pos, cur_load_pos, agents_action[agent_1])
                agent_2_pos = copy.copy(self.agent_pos[agent_2])
                next_agent_2_pos = self.__next_agent_pos(agent_2_pos, cur_load_pos, agents_action[agent_2])

                if (agent_1_pos != next_agent_1_pos and self.is_valid(next_agent_1_pos) and not self._wall_exists(next_agent_1_pos) and (self._is_cell_loadless(next_agent_1_pos) or next_agent_1_pos == cur_load_pos) and self._is_cell_agentless(next_agent_1_pos) and
                        agent_2_pos != next_agent_2_pos and self.is_valid(next_agent_2_pos) and not self._wall_exists(next_agent_2_pos) and (self._is_cell_loadless(next_agent_2_pos) or next_agent_2_pos == cur_load_pos) and self._is_cell_agentless(next_agent_2_pos) and
                        (self._is_cell_loadless(next_load_pos) or cur_load_pos==next_load_pos) and (self._is_cell_agentless(next_load_pos) or next_load_pos==agent_1_pos or next_load_pos==agent_2_pos) and not self._wall_exists(next_load_pos)):
                    # Move and reward
                    self._full_obs[AGENT][agent_1_pos[0]][agent_1_pos[1]] -= 1
                    self._full_obs[AGENT][agent_2_pos[0]][agent_2_pos[1]] -= 1
                    self._full_obs[LOAD][cur_load_pos[0]][cur_load_pos[1]] -= 1
                    self.agent_pos[agent_1] = next_agent_1_pos
                    self.agent_pos[agent_2] = next_agent_2_pos
                    self.load_pos[load_i] = next_load_pos
                    if next_load_pos == self._dock_pos:
                        self._agent_dones[agent_1] = True
                        self._agent_dones[agent_2] = True
                        self._loads_transported += 1
                        self._update_docking_distance(load_i)
                        rew = self._dock_reward
                    else:
                        self.__update_agent_view(agent_1)
                        self.__update_agent_view(agent_2)
                        self.__update_load_view(load_i)
                        rew = self._aux_reward if self._update_docking_distance(load_i) else 0

                    rewards[agent_1] += rew
                    rewards[agent_2] += rew

        if self._step_count >= self._max_steps:
            for i in range(self.n_agents):
                self._agent_dones[i] = True

        for i in range(self.n_agents):
            self._total_episode_reward[i] += rewards[i]

        # Check for episode overflow
        if all(self._agent_dones):
            if self._steps_beyond_done is None:
                self._steps_beyond_done = 0
            else:
                if self._steps_beyond_done == 0:
                    logger.warn("You are calling 'step()' even though this "
                                "environment has already returned all(done) = True. You "
                                "should always call 'reset()' once you receive "
                                "'all(done) = True' -- any further steps are undefined "
                                "behavior.")
                self._steps_beyond_done += 1

        return self.get_agent_obs(), rewards, self._agent_dones, {}

    def render(self, mode='human'):
        assert (self._step_count is not None), "Call reset before using render method."

        img = copy.copy(self._base_img)
        for pos in self.agent_pos.values():
            for col in range(max(pos[0] - math.floor(self._agent_view_range[0] / 2), 0),
                             min(math.floor(self._agent_view_range[0] / 2) + pos[0] + 1, self._grid_shape[0])):
                for row in range(max(pos[1] - math.floor(self._agent_view_range[1] / 2), 0),
                                 min(math.floor(self._agent_view_range[1] / 2) + pos[1] + 1, self._grid_shape[1])):
                    if not self._wall_exists((col, row)) and [col, row] != self._dock_pos:
                        fill_cell(img, (col, row), cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)
            fill_cell(img, pos, cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)

        for agent_i, agent_pos in self.agent_pos.items():
            if not self._agent_dones[agent_i]:
                draw_circle(img, agent_pos, cell_size=CELL_SIZE, fill=AGENT_COLOR)
                write_cell_text(img, text=str(agent_i + 1), pos=agent_pos, cell_size=CELL_SIZE, fill='white', margin=0.4)

        for load_i, human_pos in self.load_pos.items():
            if self.load_pos[load_i] != self._dock_pos:
                draw_circle(img, human_pos, cell_size=CELL_SIZE, fill=HUMAN_COLOR)
                write_cell_text(img, text=str(load_i + 1), pos=human_pos, cell_size=CELL_SIZE, fill='white', margin=0.4)

        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def seed(self, n=None):
        self.np_random, seed = seeding.np_random(n)
        return [seed]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


AGENT_COLOR = ImageColor.getcolor('blue', mode='RGB')
AGENT_NEIGHBORHOOD_COLOR = (186, 238, 247)
HUMAN_COLOR = 'red'

CELL_SIZE = 35

WALL_COLOR = 'black'
DOCK_COLOR = 'green'

WALL = -1
LOAD = 0
AGENT = 1

ACTION_MEANING = {0: "RIGHT", 1: "UP", 2: "LEFT", 3: "DOWN", 4: "ROTATE", 5: "NOOP", }


def l1_distance(pos1, pos2):
    return round(np.abs(pos1[0] - pos2[0]) + np.abs(pos1[1] - pos2[1]), 2)
