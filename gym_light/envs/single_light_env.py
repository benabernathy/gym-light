from gym import spaces, logger
from gym.utils import seeding
import gym

import numpy as np


class SingleLightEnv(gym.Env):
    """
    Defines a simple 2D light environment. The agent must move closer to the
    single light source to gain points.

    Original Author:
        Benjamin Abernathy https://github.com/benabernathy

    Observation:
        Type: Box(8)
        Description: Light measurements in 8 directions
        Num Observation             Min             Max
        0   North                   0               Inf
        1   Northeast               0               Inf
        2   East                    0               Inf
        3   Southeast               0               Inf
        4   South                   0               Inf
        5   Southwest               0               Inf
        6   West                    0               Inf
        7   Northwest               0               Inf

    Actions:
        Type: Discrete(8)
        Description: Direction of movement
        Num Action
        0   Move agent to the north
        1   Move agent to the northeast
        2   Move agent to the east
        3   Move agent to the southeast
        4   Move agent to the south
        5   Move agent to the southwest
        6   Move agent to the west
        7   Move agent to the northwest

    Reward:
        Reward is 1 whenever the light intensity increases

    Starting State:
        All observations are assigned a uniform random variable between 0 and .5

    Episode Termination:
        - Agent stored energy reaches zero
        - Episode length is greater than 1,000
        - Agent distance from light > A units for 50 epochs
        - Solved requirements: Agent distance from light is < B units for 50 epochs
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.__version__ = "0.0.1"

        # Define the action space for the agent. It has eight directions of movement: N, NE E, SE, S, SW, W, NW
        self.action_space = spaces.Discrete(8)

        self.board_dims = (50, 50)
        self.light_source_pos = (25, 25)

        # It takes energy to live and we decrement the amount of energy each step by d_energy.
        # We also increment the energy by this amount if we're close enough to a light.
        self.d_energy = 1

        # The starting energy the agent has.
        self.starting_energy = 25

        # Set the current energy
        self.current_energy = self.starting_energy

        # Starting position in the space
        self.starting_position = (10, 10)

        self.current_position = self.starting_position

        # How to define the observation space?
        # For now try to use light level in the 8 directions
        high = np.array([np.finfo(np.float32).max,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max])
        low = np.array([0.,
                        0.,
                        0.,
                        0.,
                        0.,
                        0.,
                        0.,
                        0.])

        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state

        done = self.current_energy <= 0 \
               or self.current_position[0] < 0 \
               or self.current_position[0] > self.board_dims[0] \
               or self.current_position[1] < 0 \
               or self.current_position[2] > self.board_dims[1]

        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            #just ran out of energy
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np.random

