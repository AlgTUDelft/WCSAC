import gym
from gym import spaces
import random
from gym.utils import seeding
import numpy as np


class SpyBimodal(gym.Env):

  def __init__(self):
    super(SpyBimodal, self).__init__()
    # Define action and observation space
    self.action_space = spaces.Box(low=np.array([0]), high=np.array([1]))
    self.observation_space = spaces.Discrete(100)

    self.state = 0
    self.alpha = 0.5
    self.steps = 100
    self.ep_ret = 0

    self.seed()
    self.reset()

  def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

  def step(self, action):
    # Execute one time step within the environment
    reward = random.uniform(-0.25 + action, 0.75 + action + self.alpha*action**2)
    cost   = random.uniform(0.5 * action, 1.5 * action)
    info = {'cost': cost[0]}
    self.state = self.state + 1

    self.ep_ret += reward

    done = self.state >= (self.steps - 1)
    if self.ep_ret <= 0.15 * self.state and self.state > 5:
      done=True

    return self.state, reward, done, info

  def reset(self):
    # Reset the state of the environment to an initial state
    self.state = 0
    self.ep_ret = 0
    return self.state
