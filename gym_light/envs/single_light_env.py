import math

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
        - Reward is 1 whenever the light intensity increases

    Starting State:
        - All observations are assigned a uniform random variable between
          0 and .05

    Episode Termination:
        - Agent stored energy < 0 (self.min_energy)
        - Episode length is greater than 4,000 (self.max_episode_length)
        - Agent distance from light > 200 units for 50 epochs
          200 = self.max_light_distance
          50 = self.distance_episode_length
        - Solved requirements: Agent distance from light is < 10 units for 50
          10 = self.min_light_distance
          50 = self.distance_episode_length
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.__version__ = "0.0.1"

        # The location of the light source. It's in the middle.
        self.light_source_pos = (25, 25)

        # It takes energy to live and we decrement the amount of energy each
        # step by d_energy. We also increment the energy by this amount if
        # we're close enough to a light.
        self.d_energy = 1

        # The minimum energy allowed in the agent before the episode ends
        self.min_energy = 0

        # Maximum episode length before the episode ends
        self.max_episode_length = 4000

        # Maximum agent distance from the light before the episode ends
        self.max_light_distance = 200

        # Number of episodes that have to be outside of a distance range
        # before the episode ends (think hysteresis)
        self.distance_episode_length = 50

        self.min_light_distance = 10

        # The starting energy the agent has.
        self.starting_energy = 25

        # Set the current energy
        self.current_energy = self.starting_energy

        # Starting position in the space
        self.starting_position = (10, 10)

        # Set the agent's current position to its starting position
        self.current_position = self.starting_position

        # Observation space upper limits
        high = np.array([np.finfo(np.float32).max,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max])

        # Observation space lower limits
        low = np.array([0.,
                        0.,
                        0.,
                        0.,
                        0.,
                        0.,
                        0.,
                        0.])

        # Observation space definition
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Why is the agent's position not included in the observation space?
        # Because we do not want it to learn how to find the light, not
        # where the light is. Teach it how to fish.

        # Define the action space for the agent. It has eight directions of
        # movement: N, NE E, SE, S, SW, W, NW
        self.action_space = spaces.Discrete(8)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        # First make sure the action is a valid action.
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # This will be the current state
        state = self.state

        # We check to see if we've met the done conditions specified earlier (Episode Termination)
        done = self._is_episode_terminated()
        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # just ran out of energy
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has "
                    "already returned done = True. You should always call "
                    "'reset()' once you receive 'done = True' -- any further "
                    "steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.random.uniform(low=0.0, high=0.05, size=(8,))

    def render(self, mode='human'):
        # TODO fill this out
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    @staticmethod
    def distance(p1, p2):
        """
        Euclidean distance function
        :param p1: Tuple (x1, y1)
        :param p2: Tuple (x2, y2)
        :return: distance between the two points
        """
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def distance_from_light(self):
        """
        Computes the distance of the agent from the light
        :return: distance
        """
        return self.distance(self.current_position, self.light_source_pos)

    def _is_episode_terminated(self):
        """
        Returns a True if the episode is finished.
        :return: True if the termination criteria is met, false otherwise
        """
        # ignoring hysteresis for now
        dist = self.distance_from_light()
        done = self.current_energy <= self.min_energy \
            or dist < self.min_light_distance \
            or dist > self.max_light_distance
        return bool(done)
