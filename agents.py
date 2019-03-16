import numpy as np

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

    def random_action(self):
        action = np.random.uniform(low=-1., high=1., size=(self.num_agent, self.a_dim))
        return action

    def reset(self):
        [agent.reset() for agent in self.agents]

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

    def choose_action(self, noisy=True):
        pass

    def random_action(self):
        action = np.random.uniform(low=-1., high=1., size=(1, self.a_dim))
        return action

    def reset(self):
        self.random_process.reset_states()

    def train(self):
        pass





