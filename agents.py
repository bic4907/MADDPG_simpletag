import numpy as np
import random

from network import Actor, Critic
from utils import hard_update
from random_process import OrnsteinUhlenbeckProcess

class Predator:

    def __init__(self, s_dim, a_dim, num_agent=3, **kwargs):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.config = kwargs['config']
        self.num_agent = num_agent

        self.actor = Actor(s_dim, a_dim)
        self.actor_target = Actor(s_dim, a_dim)
        self.critic = Critic(s_dim, a_dim)
        self.critic_target = Critic(s_dim, a_dim)

        if self.config.use_cuda:
            self.actor.cuda()
            self.actor_target.cuda()
            self.critic.cuda()
            self.critic_target.cuda()

        hard_update(self.actor, self.actor_target)
        hard_update(self.critic, self.critic_target)

        self.random_process = OrnsteinUhlenbeckProcess(size=self.a_dim, theta=self.config.ou_theta, mu=self.config.ou_mu, sigma=self.config.ou_sigma)

    def get_batches(self):
        experiences = random.sample(self.replay_buffer, self.batch_size)

        state_batches = np.array([_[0] for _ in experiences])
        action_batches = np.array([_[1] for _ in experiences])
        reward_batches = np.array([_[2] for _ in experiences])
        next_state_batches = np.array([_[3] for _ in experiences])
        done_batches = np.array([_[4] for _ in experiences])

        return state_batches, action_batches, reward_batches, next_state_batches, done_batches

    def random_action(self):
        action = np.random.uniform(low=-1., high=1., size=(self.num_agent, self.a_dim))
        return action

    def reset(self):
        self.random_process.reset_states()

    def train(self):
        pass


class Predators:

    def __init__(self, s_dim, a_dim, num_agent=3, **kwargs):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.config = kwargs['config']
        self.num_agent = num_agent

        self.agents = [Predator(s_dim, a_dim, **kwargs) for _ in range(num_agent)]
        self.replay_buffer = list()

    def memory(self, s, a, r, s_, done):
        self.replay_buffer.append((s, a, r, s_, done))

        if len(self.replay_buffer) >= self.config.memory_length:
            self.replay_buffer.pop(0)

    def random_action(self):
        action = np.random.uniform(low=-1., high=1., size=(self.num_agent, self.a_dim))
        return action

    def reset(self):
        [agent.reset() for agent in self.agents]

    def get_batches(self):
        experiences = random.sample(self.replay_buffer, self.batch_size)

        state_batches = np.array([_[0] for _ in experiences])
        action_batches = np.array([_[1] for _ in experiences])
        reward_batches = np.array([_[2] for _ in experiences])
        next_state_batches = np.array([_[3] for _ in experiences])
        done_batches = np.array([_[4] for _ in experiences])

        return state_batches, action_batches, reward_batches, next_state_batches, done_batches

    def train(self):
        # TODO train 구현
        # https://github.com/xuehy/pytorch-maddpg/blob/master/MADDPG.py


class Preyer:

    def __init__(self, s_dim, a_dim, **kwargs):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.config = kwargs['config']

        self.actor = Actor(s_dim, a_dim)
        self.actor_target = Actor(s_dim, a_dim)
        self.critic = Critic(s_dim, a_dim)
        self.critic_target = Critic(s_dim, a_dim)

        if self.config.use_cuda:
            self.actor.cuda()
            self.actor_target.cuda()
            self.critic.cuda()
            self.critic_target.cuda()

        hard_update(self.actor, self.actor_target)
        hard_update(self.critic, self.critic_target)

        self.random_process = OrnsteinUhlenbeckProcess(size=self.a_dim, theta=self.config.ou_theta, mu=self.config.ou_mu, sigma=self.config.ou_sigma)
        self.replay_buffer = list()

    def memory(self, s, a, r, s_, done):
        self.replay_buffer.append((s, a, r, s_, done))

        if len(self.replay_buffer) >= self.config.memory_length:
            self.replay_buffer.pop(0)

    def get_batches(self):
        experiences = random.sample(self.replay_buffer, self.batch_size)

        state_batches = np.array([_[0] for _ in experiences])
        action_batches = np.array([_[1] for _ in experiences])
        reward_batches = np.array([_[2] for _ in experiences])
        next_state_batches = np.array([_[3] for _ in experiences])
        done_batches = np.array([_[4] for _ in experiences])

        return state_batches, action_batches, reward_batches, next_state_batches, done_batches

    def choose_action(self, noisy=True):
        pass

    def random_action(self):
        action = np.random.uniform(low=-1., high=1., size=(1, self.a_dim))
        return action

    def reset(self):
        self.random_process.reset_states()

    def train(self):
        pass





