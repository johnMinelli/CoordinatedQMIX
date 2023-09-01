import copy
import logging
import random
from netrc import netrc

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


class PredatorPrey(gym.Env):
    """
    Predator-prey involves a grid world, in which multiple predators attempt to capture randomly moving prey.
    Agents have a n × n view and select one of five actions ∈ {Left, Right, Up, Down, Stop} at each time step.
    Prey move according to selecting a random action (parametrized distribution) at each time step.

    We define the “capture” of a prey if this is surrounded in all its cardinal direction of predators or walls.
    Default reward for this is 5 for each agent. 
    A "tag" correspond to the presence of a predator and a prey in the same cell. A smaller reward is delivered to such
    agent in this case, with a default value of 0.1
    Each agent’s observation includes:
    - agent absolute coordinate position normalized to 1. e.g. [1.0,1.0] for br corner
    - boolean array for the cells in the observation area describing the presence of prey, agent, obstacle.
    e.g. [0]x(5,5,3) with an observation are af (5,5)

    The terminating condition of this task is when all preys are caught or after n steps. For every new episodes,
    preys are initialized into random locations.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, grid_shape=(5, 5), n_agents=2, n_preys=1, prey_move_probs=(0.175, 0.175, 0.175, 0.175, 0.3),
                 full_observable=False, penalty=-0.5, step_cost=-0.01, prey_capture_reward=5, prey_tag_reward=0.1,
                 max_steps=100, agent_view_range=(5, 5)):
        assert len(grid_shape) == 2, 'expected a tuple of size 2 for grid_shape, but found {}'.format(grid_shape)
        assert len(agent_view_range) == 2, 'expected a tuple of size 2 for agent view mask,' \
                                          ' but found {}'.format(agent_view_range)
        assert grid_shape[0] > 0 and grid_shape[1] > 0, 'grid shape should be > 0'
        assert 0 < agent_view_range[0] <= grid_shape[0], 'agent view mask has to be within (0,{}]'.format(grid_shape[0])
        assert 0 < agent_view_range[1] <= grid_shape[1], 'agent view mask has to be within (0,{}]'.format(grid_shape[1])

        self._grid_shape = grid_shape
        self.n_agents = n_agents
        self.n_agents_dummy = n_agents+16
        self.n_preys = n_preys
        self._max_steps = max_steps
        self._step_count = None
        self._steps_beyond_done = None
        self._penalty = penalty
        self._step_cost = step_cost
        self._prey_capture_reward = prey_capture_reward
        self._prey_tag_reward = prey_tag_reward
        self._agent_view_range = agent_view_range

        self.action_space = MultiAgentActionSpace([spaces.Discrete(5) for _ in range(self.n_agents)])
        self.agent_ids = [str(a) for a in range(self.n_agents)]
        self.agent_pos = {}
        self.prey_pos = {}
        self._prey_alive = None

        self._full_obs = self.__create_grid()
        self._agent_dones = [False for _ in range(self.n_agents)]
        self._prey_move_probs = prey_move_probs
        self.viewer = None
        self.full_observable = full_observable

        mask_size = np.prod(self._agent_view_range)
        self._obs_high = np.array([1., 1.] + [1.] * mask_size * 3, dtype=np.float32)
        self._obs_low = np.array([0., 0.] + [0.] * mask_size * 3, dtype=np.float32)
        if self.full_observable:
            self._obs_high = np.tile(self._obs_high, self.n_agents)
            self._obs_low = np.tile(self._obs_low, self.n_agents)
        self.observation_space = MultiAgentObservationSpace(
            [spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])

        self._total_episode_reward = None
        self.seed()

    @property
    def step_count(self):
        return self._step_count

    @property
    def max_steps(self):
        return self._max_steps

    @property
    def success(self):
        return self.n_preys-sum(self._prey_alive)

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

    def __create_grid(self):
        x, y = self._grid_shape
        _grid = np.zeros((3, x, y))
        # add center walls
        if x >= 8 and y >= 8:
            for col in range(3, x-3):
                for row in range(3, y-3):
                    _grid[WALL][col][row] = 1
        return _grid

    def __init_full_obs(self):
        self._full_obs = self.__create_grid()

        while True:
            pos = [np.random.randint(0, self._grid_shape[0] - 1),
                   np.random.randint(0, self._grid_shape[1] - 1)]
            if self._is_cell_preyless(pos) and not self._wall_exists(pos):
                break
        for agent_i in range(self.n_agents):
            self.agent_pos[agent_i] = pos
            self.__update_agent_view(agent_i)

        for prey_i in range(self.n_preys):
            while True:
                pos = [np.random.randint(0, self._grid_shape[0] - 1),
                       np.random.randint(0, self._grid_shape[1] - 1)]
                if self._is_cell_agentless(pos) and not self._wall_exists(pos):
                    self.prey_pos[prey_i] = pos
                    break
            self.__update_prey_view(prey_i)

        self.__draw_base_img()

    def get_agent_obs(self):
        _obs = []
        for agent_i in range(self.n_agents):
            pos = self.agent_pos[agent_i]
            _agent_i_obs = [round(pos[0] / (self._grid_shape[0] - 1), 2), round(pos[1] / (self._grid_shape[1] - 1), 2)]  # coordinates

            # check if prey is in the view area
            x_m = math.floor(self._agent_view_range[0]/2)
            y_m = math.floor(self._agent_view_range[1]/2)
            _prey_pos = np.zeros(self._agent_view_range)  # prey location in neighbour
            _agent_pos = np.zeros(self._agent_view_range)  # other agents location in neighbour
            _obstacle_pos = np.zeros(self._agent_view_range)  # other agents location in neighbour

            for i, col in enumerate(range(pos[0] - x_m, pos[0] + x_m + 1)):
                for j, row in enumerate(range(pos[1] - y_m, pos[1] + y_m + 1)):
                    if not self.is_valid((col, row)) or self._wall_exists((col, row)):
                        _obstacle_pos[i, j] = 1
                        continue
                    if self._full_obs[PREY][col][row] > 0:
                        _prey_pos[i, j] = min(1, self._full_obs[PREY][col][row])  # get relative position for the prey loc.
                    if self._full_obs[AGENT][col][row] > (1 if pos == [col, row] else 0):
                        _agent_pos[i, j] = min(1, self._full_obs[AGENT][col][row])

            _agent_i_obs += _prey_pos.flatten().tolist()  # adding prey pos in observable area
            _agent_i_obs += _agent_pos.flatten().tolist()  # adding prey pos in observable area
            _agent_i_obs += _obstacle_pos.flatten().tolist()  # adding prey pos in observable area
            _obs.append(_agent_i_obs)

        if self.full_observable:
            _obs = np.array(_obs).flatten().tolist()
            _obs = [_obs for _ in range(self.n_agents)]
        return _obs

    def reset(self, seed=None, options=None):
        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        self.agent_pos = {}
        self.prey_pos = {}

        self.__init_full_obs()
        self._step_count = 0
        self._steps_beyond_done = None
        self._agent_dones = [False for _ in range(self.n_agents)]
        self._prey_alive = [True for _ in range(self.n_preys)]

        return self.get_agent_obs(), {}

    def _wall_exists(self, pos):
        return self._full_obs[WALL][pos[0]][pos[1]] == 1

    def is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    def _is_cell_preyless(self, pos):
        return self.is_valid(pos) and (self._full_obs[PREY][pos[0]][pos[1]] == 0)

    def _is_cell_agentless(self, pos):
        return self.is_valid(pos) and (self._full_obs[AGENT][pos[0]][pos[1]] == 0)

    def __next_pos(self, curr_pos, move):
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            next_pos = curr_pos
        else:
            raise Exception('Action Not found!')
        return next_pos

    def __update_agent_pos(self, agent_i, move):

        curr_pos = copy.copy(self.agent_pos[agent_i])
        next_pos = self.__next_pos(curr_pos, move)

        if next_pos != curr_pos and self.is_valid(next_pos) and not self._wall_exists(next_pos):
            self.agent_pos[agent_i] = next_pos
            self._full_obs[AGENT][curr_pos[0]][curr_pos[1]] -= 1
            self.__update_agent_view(agent_i)

    def __update_prey_pos(self, prey_i, move):
        curr_pos = copy.copy(self.prey_pos[prey_i])
        if self._prey_alive[prey_i]:
            next_pos = self.__next_pos(curr_pos, move)

            if next_pos != curr_pos and self.is_valid(next_pos) and not self._wall_exists(next_pos):
                self.prey_pos[prey_i] = next_pos
                self._full_obs[PREY][curr_pos[0]][curr_pos[1]] -= 1
                self.__update_prey_view(prey_i)
        else:
            self._full_obs[PREY][curr_pos[0]][curr_pos[1]] -= 1

    def __update_agent_view(self, agent_i):
        self._full_obs[AGENT][self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]] += 1

    def __update_prey_view(self, prey_i):
        self._full_obs[PREY][self.prey_pos[prey_i][0]][self.prey_pos[prey_i][1]] += 1

    def step(self, agents_action):
        assert (self._step_count is not None), \
            "Call reset before using step method."

        def is_surrounding(pos_1, pos_2):
            """
            Check if pos_1 is surrounding pos_2, i.e. is in a coordinate with distance 1 from pos_2.
            """
            return np.abs(pos_1[0] - pos_2[0]) + np.abs(pos_1[1] - pos_2[1]) == 1

        self._step_count += 1
        rewards = [self._step_cost for _ in range(self.n_agents)]

        for agent_i, action in enumerate(agents_action):
            if not (self._agent_dones[agent_i]):
                self.__update_agent_pos(agent_i, action)

        for prey_i in range(self.n_preys):
            if self._prey_alive[prey_i]:
                surrounding_cells = [[disp_x + self.prey_pos[prey_i][0], disp_y + self.prey_pos[prey_i][1]] for disp_x, disp_y in [[0,-1],[0,1],[-1,0],[1,0]]
                                     if self.is_valid((disp_x + self.prey_pos[prey_i][0], disp_y + self.prey_pos[prey_i][1]))
                                     and not self._wall_exists((disp_x + self.prey_pos[prey_i][0], disp_y + self.prey_pos[prey_i][1]))]
                surrounding_predators = [(i, pos) for i, pos in self.agent_pos.items() if pos in surrounding_cells]
                tagging_predators = [i for i, pos in self.agent_pos.items() if pos == self.prey_pos[prey_i]]

                if len(set([tuple(i[1]) for i in surrounding_predators])) == len(surrounding_cells):
                    self._prey_alive[prey_i] = False
                    for agent in surrounding_predators:
                        rewards[agent[0]] += self._prey_capture_reward
                elif len(tagging_predators) > 0:
                    for agent_i in tagging_predators:
                        rewards[agent_i] += self._prey_tag_reward

                prey_move = None
                if self._prey_alive[prey_i]:
                    prey_move = np.random.choice(len(self._prey_move_probs), 1, p=self._prey_move_probs)[0]
                self.__update_prey_pos(prey_i, prey_move)

        if (self._step_count >= self._max_steps) or (True not in self._prey_alive):
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
                    logger.warn(
                        "You are calling 'step()' even though this "
                        "environment has already returned all(done) = True. You "
                        "should always call 'reset()' once you receive "
                        "'all(done) = True' -- any further steps are undefined "
                        "behavior."
                    )
                self._steps_beyond_done += 1

        return self.get_agent_obs(), rewards, self._agent_dones, {'prey_alive': self._prey_alive}

    def __get_neighbour_coordinates(self, pos):
        neighbours = []
        if self.is_valid([pos[0] + 1, pos[1]]):
            neighbours.append([pos[0] + 1, pos[1]])
        if self.is_valid([pos[0] - 1, pos[1]]):
            neighbours.append([pos[0] - 1, pos[1]])
        if self.is_valid([pos[0], pos[1] + 1]):
            neighbours.append([pos[0], pos[1] + 1])
        if self.is_valid([pos[0], pos[1] - 1]):
            neighbours.append([pos[0], pos[1] - 1])
        return neighbours

    def render(self, mode='human'):
        assert (self._step_count is not None), \
            "Call reset before using render method."

        def __get_neighbour_coordinates( pos, area):
            x, y = pos
            neighbours = []
            for i in range(-area, area + 1):
                for j in range(-area, area + 1):
                    if (i == 0 and j == 0) or not self.is_valid((x + i, y + j)):
                        continue
                    neighbours.append((x + i, y + j))
            return neighbours

        img = copy.copy(self._base_img)
        for agent_i in range(self.n_agents):
            for neighbour in __get_neighbour_coordinates(self.agent_pos[agent_i], 2):
                if self._full_obs[WALL][neighbour[0]][neighbour[1]] == 0:
                    fill_cell(img, neighbour, cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)
            fill_cell(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)

        for agent_i in range(self.n_agents):
            draw_circle(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_COLOR)
            write_cell_text(img, text=str(agent_i + 1), pos=self.agent_pos[agent_i], cell_size=CELL_SIZE,
                            fill='white', margin=0.4)

        for prey_i in range(self.n_preys):
            if self._prey_alive[prey_i]:
                draw_circle(img, self.prey_pos[prey_i], cell_size=CELL_SIZE, fill=PREY_COLOR)
                write_cell_text(img, text=str(prey_i + 1), pos=self.prey_pos[prey_i], cell_size=CELL_SIZE,
                                fill='white', margin=0.4)

        img = np.asarray(img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def seed(self, n=None):
        np.random.seed(n)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


AGENT_COLOR = ImageColor.getcolor('blue', mode='RGB')
AGENT_NEIGHBORHOOD_COLOR = (186, 238, 247)
PREY_COLOR = 'red'

CELL_SIZE = 35

WALL_COLOR = 'black'

WALL = -1
PREY = 0
AGENT = 1

ACTION_MEANING = {
    0: "RIGHT",
    1: "UP",
    2: "LEFT",
    3: "DOWN",
    4: "NOOP",
}

PRE_IDS = {
    'agent': 'A',
    'prey': 'P',
    'wall': 'W',
    'empty': '0'
}
import copy
import logging
import random
from netrc import netrc

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


class PredatorPrey(gym.Env):
    """
    Predator-prey involves a grid world, in which multiple predators attempt to capture randomly moving prey.
    Agents have a n × n view and select one of five actions ∈ {Left, Right, Up, Down, Stop} at each time step.
    Prey move according to selecting a random action (parametrized distribution) at each time step.

    We define the “capture” of a prey if this is surrounded in all its cardinal direction of predators or walls.
    Default reward for this is 5 for each agent. 
    A "tag" correspond to the presence of a predator and a prey in the same cell. A smaller reward is delivered to such
    agent in this case, with a default value of 0.1
    Each agent’s observation includes:
    - agent absolute coordinate position normalized to 1. e.g. [1.0,1.0] for br corner
    - boolean array for the cells in the observation area describing the presence of prey, agent, obstacle.
    e.g. [0]x(5,5,3) with an observation are af (5,5)

    The terminating condition of this task is when all preys are caught or after n steps. For every new episodes,
    preys are initialized into random locations.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, grid_shape=(5, 5), n_agents=2, n_preys=1, prey_move_probs=(0.175, 0.175, 0.175, 0.175, 0.3),
                 full_observable=False, penalty=-0.5, step_cost=-0.01, prey_capture_reward=5, prey_tag_reward=0.1,
                 max_steps=100, agent_view_range=(5, 5)):
        assert len(grid_shape) == 2, 'expected a tuple of size 2 for grid_shape, but found {}'.format(grid_shape)
        assert len(agent_view_range) == 2, 'expected a tuple of size 2 for agent view mask,' \
                                          ' but found {}'.format(agent_view_range)
        assert grid_shape[0] > 0 and grid_shape[1] > 0, 'grid shape should be > 0'
        assert 0 < agent_view_range[0] <= grid_shape[0], 'agent view mask has to be within (0,{}]'.format(grid_shape[0])
        assert 0 < agent_view_range[1] <= grid_shape[1], 'agent view mask has to be within (0,{}]'.format(grid_shape[1])

        self._grid_shape = grid_shape
        self.n_agents = n_agents
        self.n_agents_dummy = n_agents+16
        self.n_preys = n_preys
        self._max_steps = max_steps
        self._step_count = None
        self._steps_beyond_done = None
        self._penalty = penalty
        self._step_cost = step_cost
        self._prey_capture_reward = prey_capture_reward
        self._prey_tag_reward = prey_tag_reward
        self._agent_view_range = agent_view_range

        self.action_space = MultiAgentActionSpace([spaces.Discrete(5) for _ in range(self.n_agents)])
        self.agent_ids = [str(a) for a in range(self.n_agents)]
        self.agent_pos = {}
        self.prey_pos = {}
        self._prey_alive = None

        self._full_obs = self.__create_grid()
        self._agent_dones = [False for _ in range(self.n_agents)]
        self._prey_move_probs = prey_move_probs
        self.viewer = None
        self.full_observable = full_observable

        mask_size = np.prod(self._agent_view_range)
        self._obs_high = np.array([1., 1.] + [1.] * mask_size * 3, dtype=np.float32)
        self._obs_low = np.array([0., 0.] + [0.] * mask_size * 3, dtype=np.float32)
        if self.full_observable:
            self._obs_high = np.tile(self._obs_high, self.n_agents)
            self._obs_low = np.tile(self._obs_low, self.n_agents)
        self.observation_space = MultiAgentObservationSpace(
            [spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])

        self._total_episode_reward = None
        self.seed()

    @property
    def step_count(self):
        return self._step_count

    @property
    def max_steps(self):
        return self._max_steps

    @property
    def success(self):
        return self.n_preys-sum(self._prey_alive)

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

    def __create_grid(self):
        x, y = self._grid_shape
        _grid = np.zeros((3, x, y))
        # add center walls
        if x >= 8 and y >= 8:
            for col in range(3, x-3):
                for row in range(3, y-3):
                    _grid[WALL][col][row] = 1
        return _grid

    def __init_full_obs(self):
        self._full_obs = self.__create_grid()

        while True:
            pos = [np.random.randint(0, self._grid_shape[0] - 1),
                   np.random.randint(0, self._grid_shape[1] - 1)]
            if self._is_cell_preyless(pos) and not self._wall_exists(pos):
                break
        for agent_i in range(self.n_agents):
            self.agent_pos[agent_i] = pos
            self.__update_agent_view(agent_i)

        for prey_i in range(self.n_preys):
            while True:
                pos = [np.random.randint(0, self._grid_shape[0] - 1),
                       np.random.randint(0, self._grid_shape[1] - 1)]
                if self._is_cell_agentless(pos) and not self._wall_exists(pos):
                    self.prey_pos[prey_i] = pos
                    break
            self.__update_prey_view(prey_i)

        self.__draw_base_img()

    def get_agent_obs(self):
        _obs = []
        for agent_i in range(self.n_agents):
            pos = self.agent_pos[agent_i]
            _agent_i_obs = [round(pos[0] / (self._grid_shape[0] - 1), 2), round(pos[1] / (self._grid_shape[1] - 1), 2)]  # coordinates

            # check if prey is in the view area
            x_m = math.floor(self._agent_view_range[0]/2)
            y_m = math.floor(self._agent_view_range[1]/2)
            _prey_pos = np.zeros(self._agent_view_range)  # prey location in neighbour
            _agent_pos = np.zeros(self._agent_view_range)  # other agents location in neighbour
            _obstacle_pos = np.zeros(self._agent_view_range)  # other agents location in neighbour

            for i, col in enumerate(range(pos[0] - x_m, pos[0] + x_m + 1)):
                for j, row in enumerate(range(pos[1] - y_m, pos[1] + y_m + 1)):
                    if not self.is_valid((col, row)) or self._wall_exists((col, row)):
                        _obstacle_pos[i, j] = 1
                        continue
                    if self._full_obs[PREY][col][row] > 0:
                        _prey_pos[i, j] = min(1, self._full_obs[PREY][col][row])  # get relative position for the prey loc.
                    if self._full_obs[AGENT][col][row] > (1 if pos == [col, row] else 0):
                        _agent_pos[i, j] = min(1, self._full_obs[AGENT][col][row])

            _agent_i_obs += _prey_pos.flatten().tolist()  # adding prey pos in observable area
            _agent_i_obs += _agent_pos.flatten().tolist()  # adding prey pos in observable area
            _agent_i_obs += _obstacle_pos.flatten().tolist()  # adding prey pos in observable area
            _obs.append(_agent_i_obs)

        if self.full_observable:
            _obs = np.array(_obs).flatten().tolist()
            _obs = [_obs for _ in range(self.n_agents)]
        return _obs

    def reset(self, seed=None, options=None):
        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        self.agent_pos = {}
        self.prey_pos = {}

        self.__init_full_obs()
        self._step_count = 0
        self._steps_beyond_done = None
        self._agent_dones = [False for _ in range(self.n_agents)]
        self._prey_alive = [True for _ in range(self.n_preys)]

        return self.get_agent_obs(), {}

    def _wall_exists(self, pos):
        return self._full_obs[WALL][pos[0]][pos[1]] == 1

    def is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    def _is_cell_preyless(self, pos):
        return self.is_valid(pos) and (self._full_obs[PREY][pos[0]][pos[1]] == 0)

    def _is_cell_agentless(self, pos):
        return self.is_valid(pos) and (self._full_obs[AGENT][pos[0]][pos[1]] == 0)

    def __next_pos(self, curr_pos, move):
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            next_pos = curr_pos
        else:
            raise Exception('Action Not found!')
        return next_pos

    def __update_agent_pos(self, agent_i, move):

        curr_pos = copy.copy(self.agent_pos[agent_i])
        next_pos = self.__next_pos(curr_pos, move)

        if next_pos != curr_pos and self.is_valid(next_pos) and not self._wall_exists(next_pos):
            self.agent_pos[agent_i] = next_pos
            self._full_obs[AGENT][curr_pos[0]][curr_pos[1]] -= 1
            self.__update_agent_view(agent_i)

    def __update_prey_pos(self, prey_i, move):
        curr_pos = copy.copy(self.prey_pos[prey_i])
        if self._prey_alive[prey_i]:
            next_pos = self.__next_pos(curr_pos, move)

            if next_pos != curr_pos and self.is_valid(next_pos) and not self._wall_exists(next_pos):
                self.prey_pos[prey_i] = next_pos
                self._full_obs[PREY][curr_pos[0]][curr_pos[1]] -= 1
                self.__update_prey_view(prey_i)
        else:
            self._full_obs[PREY][curr_pos[0]][curr_pos[1]] -= 1

    def __update_agent_view(self, agent_i):
        self._full_obs[AGENT][self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]] += 1

    def __update_prey_view(self, prey_i):
        self._full_obs[PREY][self.prey_pos[prey_i][0]][self.prey_pos[prey_i][1]] += 1

    def step(self, agents_action):
        assert (self._step_count is not None), \
            "Call reset before using step method."

        def is_surrounding(pos_1, pos_2):
            """
            Check if pos_1 is surrounding pos_2, i.e. is in a coordinate with distance 1 from pos_2.
            """
            return np.abs(pos_1[0] - pos_2[0]) + np.abs(pos_1[1] - pos_2[1]) == 1

        self._step_count += 1
        rewards = [self._step_cost for _ in range(self.n_agents)]

        for agent_i, action in enumerate(agents_action):
            if not (self._agent_dones[agent_i]):
                self.__update_agent_pos(agent_i, action)

        for prey_i in range(self.n_preys):
            if self._prey_alive[prey_i]:
                surrounding_cells = [[disp_x + self.prey_pos[prey_i][0], disp_y + self.prey_pos[prey_i][1]] for disp_x, disp_y in [[0,-1],[0,1],[-1,0],[1,0]]
                                     if self.is_valid((disp_x + self.prey_pos[prey_i][0], disp_y + self.prey_pos[prey_i][1]))
                                     and not self._wall_exists((disp_x + self.prey_pos[prey_i][0], disp_y + self.prey_pos[prey_i][1]))]
                surrounding_predators = [(i, pos) for i, pos in self.agent_pos.items() if pos in surrounding_cells]
                tagging_predators = [i for i, pos in self.agent_pos.items() if pos == self.prey_pos[prey_i]]

                if len(set([tuple(i[1]) for i in surrounding_predators])) == len(surrounding_cells):
                    self._prey_alive[prey_i] = False
                    for agent in surrounding_predators:
                        rewards[agent[0]] += self._prey_capture_reward
                elif len(tagging_predators) > 0:
                    for agent_i in tagging_predators:
                        rewards[agent_i] += self._prey_tag_reward

                prey_move = None
                if self._prey_alive[prey_i]:
                    prey_move = np.random.choice(len(self._prey_move_probs), 1, p=self._prey_move_probs)[0]
                self.__update_prey_pos(prey_i, prey_move)

        if (self._step_count >= self._max_steps) or (True not in self._prey_alive):
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
                    logger.warn(
                        "You are calling 'step()' even though this "
                        "environment has already returned all(done) = True. You "
                        "should always call 'reset()' once you receive "
                        "'all(done) = True' -- any further steps are undefined "
                        "behavior."
                    )
                self._steps_beyond_done += 1

        return self.get_agent_obs(), rewards, self._agent_dones, {'prey_alive': self._prey_alive}

    def __get_neighbour_coordinates(self, pos):
        neighbours = []
        if self.is_valid([pos[0] + 1, pos[1]]):
            neighbours.append([pos[0] + 1, pos[1]])
        if self.is_valid([pos[0] - 1, pos[1]]):
            neighbours.append([pos[0] - 1, pos[1]])
        if self.is_valid([pos[0], pos[1] + 1]):
            neighbours.append([pos[0], pos[1] + 1])
        if self.is_valid([pos[0], pos[1] - 1]):
            neighbours.append([pos[0], pos[1] - 1])
        return neighbours

    def render(self, mode='human'):
        assert (self._step_count is not None), \
            "Call reset before using render method."

        def __get_neighbour_coordinates( pos, area):
            x, y = pos
            neighbours = []
            for i in range(-area, area + 1):
                for j in range(-area, area + 1):
                    if (i == 0 and j == 0) or not self.is_valid((x + i, y + j)):
                        continue
                    neighbours.append((x + i, y + j))
            return neighbours

        img = copy.copy(self._base_img)
        for agent_i in range(self.n_agents):
            for neighbour in __get_neighbour_coordinates(self.agent_pos[agent_i], 2):
                if self._full_obs[WALL][neighbour[0]][neighbour[1]] == 0:
                    fill_cell(img, neighbour, cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)
            fill_cell(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)

        for agent_i in range(self.n_agents):
            draw_circle(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_COLOR)
            write_cell_text(img, text=str(agent_i + 1), pos=self.agent_pos[agent_i], cell_size=CELL_SIZE,
                            fill='white', margin=0.4)

        for prey_i in range(self.n_preys):
            if self._prey_alive[prey_i]:
                draw_circle(img, self.prey_pos[prey_i], cell_size=CELL_SIZE, fill=PREY_COLOR)
                write_cell_text(img, text=str(prey_i + 1), pos=self.prey_pos[prey_i], cell_size=CELL_SIZE,
                                fill='white', margin=0.4)

        img = np.asarray(img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def seed(self, n=None):
        np.random.seed(n)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


AGENT_COLOR = ImageColor.getcolor('blue', mode='RGB')
AGENT_NEIGHBORHOOD_COLOR = (186, 238, 247)
PREY_COLOR = 'red'

CELL_SIZE = 35

WALL_COLOR = 'black'

WALL = -1
PREY = 0
AGENT = 1

ACTION_MEANING = {
    0: "RIGHT",
    1: "UP",
    2: "LEFT",
    3: "DOWN",
    4: "NOOP",
}

PRE_IDS = {
    'agent': 'A',
    'prey': 'P',
    'wall': 'W',
    'empty': '0'
}
