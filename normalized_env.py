'''
Implemented by ghliu
https://github.com/ghliu/pytorch-ddpg/blob/master/normalized_env.py
'''

import gym

# https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """
    def __init__(self, env):
        super(NormalizedEnv, self).__init__(env=env)
        self.action_high = 1.
        self.action_low = -1.

    def _action(self, action):
        act_k = (self.action_high - self.action_low)/ 2.
        act_b = (self.action_high + self.action_low)/ 2.
        return act_k * action + act_b

    def _reverse_action(self, action):
        act_k_inv = 2./(self.action_high - self.action_low)
        act_b = (self.action_high + self.action_low)/ 2.
        return act_k_inv * (action - act_b)