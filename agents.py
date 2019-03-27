import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random

from network import Actor, Critic
from utils import soft_update, hard_update
from random_process import OrnsteinUhlenbeckProcess

class Predator:

    def __init__(self, s_dim, a_dim, num_agent, **kwargs):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.config = kwargs['config']
        self.num_agent = num_agent

        self.actor = Actor(s_dim, a_dim)
        self.actor_target = Actor(s_dim, a_dim)
        self.critic = Critic(s_dim, a_dim, num_agent)
        self.critic_target = Critic(s_dim, a_dim, num_agent)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.a_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.c_lr)
        self.a_loss = 0
        self.c_loss = 0

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


class Predators:

    def __init__(self, s_dim, a_dim, num_agent=3, **kwargs):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.config = kwargs['config']
        self.num_agent = num_agent
        self.device = 'cuda' if self.config.use_cuda else 'cpu'

        self.agents = [Predator(s_dim, a_dim, num_agent, **kwargs) for _ in range(num_agent)]
        self.replay_buffer = list()
        self.epsilon = 1.
        self.depsilon = self.epsilon / self.config.epsilon_decay

    def memory(self, s, a, r, s_, done):
        self.replay_buffer.append((s, a, r, s_, done))

        if len(self.replay_buffer) >= self.config.memory_length:
            self.replay_buffer.pop(0)

    def choose_action(self, s, noisy=True):
        if self.config.use_cuda:
            s = Variable(torch.cuda.FloatTensor(s))
        else:
            s = Variable(torch.FloatTensor(s))

        actions = np.zeros((self.num_agent, self.a_dim), dtype=np.float)
        for agent_idx, agent in enumerate(self.agents):
            actions[agent_idx] = agent.actor.forward(s)[agent_idx].cpu().detach().numpy()
            if noisy:
                actions[agent_idx] += self.epsilon * agent.random_process.sample()
                actions[agent_idx] = np.clip(actions[agent_idx], -1., 1.)

        self.epsilon -= self.depsilon
        self.epsilon = max(self.epsilon, 0)

        return actions

    def random_action(self):
        action = np.random.uniform(low=-1., high=1., size=(self.num_agent, self.a_dim))
        return action

    def reset(self):
        [agent.reset() for agent in self.agents]

    def get_batches(self):
        experiences = random.sample(self.replay_buffer, self.config.batch_size)

        state_batches = np.array([_[0] for _ in experiences])
        action_batches = np.array([_[1] for _ in experiences])
        reward_batches = np.array([_[2] for _ in experiences])
        next_state_batches = np.array([_[3] for _ in experiences])
        done_batches = np.array([_[4] for _ in experiences])

        return state_batches, action_batches, reward_batches, next_state_batches, done_batches

    def prep_train(self):
        for a in self.agents:
            a.actor.train()
            a.critic.train()
            a.actor_target.train()
            a.critic_target.train()

    def prep_eval(self):
        for a in self.agents:
            a.actor.eval()
            a.critic.eval()
            a.actor_target.eval()
            a.critic_target.eval()

    def train(self):
        for agent_idx in range(len(self.agents)):
            state_batches, action_batches, reward_batches, next_state_batches, done_batches = self.get_batches()

            state_batches = Variable(torch.Tensor(state_batches).to(self.device))
            action_batches = Variable(torch.Tensor(action_batches).to(self.device))
            reward_batches = Variable(torch.Tensor(reward_batches).reshape(self.config.batch_size, self.num_agent, 1).to(self.device))
            next_state_batches = Variable(torch.Tensor(next_state_batches).to(self.device))
            done_batches = Variable(torch.Tensor((done_batches == False) * 1).reshape(self.config.batch_size, self.num_agent, 1).to(self.device))


            target_next_actions = Variable(torch.Tensor(np.zeros(shape=(self.config.batch_size, self.num_agent, self.a_dim), dtype=np.float))).to(self.device)
            for agt in range(self.num_agent):
                target_next_actions[:, agt] = self.agents[agt].actor_target.forward(next_state_batches[:, agt])
            target_next_q = self.agents[agent_idx].critic_target.forward(next_state_batches, target_next_actions).detach()

            main_q = self.agents[agent_idx].critic(state_batches, action_batches)

            # Critic Loss
            self.agents[agent_idx].critic.zero_grad()
            baselines =  self.config.reward_coef * reward_batches[:, agent_idx] + done_batches[:, agent_idx] * self.config.gamma * target_next_q
            loss_critic = torch.nn.MSELoss()(main_q, baselines)
            loss_critic.backward()
            torch.nn.utils.clip_grad_norm_(self.agents[agent_idx].critic.parameters(), 0.5)
            self.agents[agent_idx].critic_optimizer.step()

            # Actor Loss
            self.agents[agent_idx].actor.zero_grad()
            agent_idx_pure_action = self.agents[agent_idx].actor.forward(state_batches[:, agent_idx])
            pure_action_batches = action_batches.clone()
            for agt in range(self.num_agent):
                if agt == agent_idx: continue
                pure_action_batches[:, agt] = self.agents[agt].actor_target.forward(state_batches[:, agt])
            pure_action_batches[:, agent_idx] = agent_idx_pure_action
            '''
            pure_action_batches = action_batches.clone()
            # memory의 action들은 기본적으로 noisy가 있기때문에 선택된 agent에 대해서만 pure한 액션으로 바꾸어 학습시킴
            
            pure_action_batches[:, agent_idx] = agent_idx_pure_action
            '''

            loss_actor = -self.agents[agent_idx].critic.forward(state_batches, pure_action_batches).mean()

            loss_actor += (agent_idx_pure_action ** 2).mean() * 1e-3
            loss_actor.backward()
            torch.nn.utils.clip_grad_norm_(self.agents[agent_idx].actor.parameters(), 0.5)
            self.agents[agent_idx].actor_optimizer.step()

            # This is for logging
            self.agents[agent_idx].c_loss = loss_critic.item()
            self.agents[agent_idx].a_loss = loss_actor.item()

        for agent_idx in range(len(self.agents)):
            soft_update(self.agents[agent_idx].actor, self.agents[agent_idx].actor_target, self.config.tau)
            soft_update(self.agents[agent_idx].critic, self.agents[agent_idx].critic_target, self.config.tau)

    def getLoss(self):
        return np.mean([agent.c_loss for agent in self.agents]).item(), np.mean([agent.a_loss for agent in self.agents]).item()


class Preyer:

    def __init__(self, s_dim, a_dim, **kwargs):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.config = kwargs['config']
        self.device = 'cuda' if self.config.use_cuda else 'cpu'

        self.actor = Actor(s_dim, a_dim)
        self.actor_target = Actor(s_dim, a_dim)
        self.critic = Critic(s_dim, a_dim, 1)
        self.critic_target = Critic(s_dim, a_dim, 1)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.a_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.c_lr)
        self.c_loss = 0
        self.a_loss = 0

        if self.config.use_cuda:
            self.actor.cuda()
            self.actor_target.cuda()
            self.critic.cuda()
            self.critic_target.cuda()

        hard_update(self.actor, self.actor_target)
        hard_update(self.critic, self.critic_target)

        self.random_process = OrnsteinUhlenbeckProcess(size=self.a_dim, theta=self.config.ou_theta, mu=self.config.ou_mu, sigma=self.config.ou_sigma)
        self.replay_buffer = list()
        self.epsilon = 1.
        self.depsilon = self.epsilon / self.config.epsilon_decay

    def memory(self, s, a, r, s_, done):
        self.replay_buffer.append((s, a, r, s_, done))

        if len(self.replay_buffer) >= self.config.memory_length:
            self.replay_buffer.pop(0)

    def get_batches(self):
        experiences = random.sample(self.replay_buffer, self.config.batch_size)

        state_batches = np.array([_[0] for _ in experiences])
        action_batches = np.array([_[1] for _ in experiences])
        reward_batches = np.array([_[2] for _ in experiences])
        next_state_batches = np.array([_[3] for _ in experiences])
        done_batches = np.array([_[4] for _ in experiences])

        return state_batches, action_batches, reward_batches, next_state_batches, done_batches

    def choose_action(self, s, noisy=True):
        if self.config.use_cuda:
            s = Variable(torch.cuda.FloatTensor(s))
        else:
            s = Variable(torch.FloatTensor(s))
        a = self.actor.forward(s).cpu().detach().numpy()

        if noisy:
            a += max(self.epsilon, 0.001) * self.random_process.sample()
            self.epsilon -= self.depsilon
        a = np.clip(a, -1., 1.)

        return np.array([a])

    def random_action(self):
        action = np.random.uniform(low=-1., high=1., size=(1, self.a_dim))
        return action

    def reset(self):
        self.random_process.reset_states()

    def train(self):
        state_batches, action_batches, reward_batches, next_state_batches, done_batches = self.get_batches()

        state_batches = Variable(torch.Tensor(state_batches).to(self.device))
        action_batches = Variable(torch.Tensor(action_batches).reshape(-1, 1).to(self.device))
        reward_batches = Variable(torch.Tensor(reward_batches).reshape(-1, 1).to(self.device))
        next_state_batches = Variable(torch.Tensor(next_state_batches).to(self.device))
        done_batches = Variable(torch.Tensor((done_batches == False) * 1).reshape(-1, 1).to(self.device))


        target_next_actions = self.actor_target.forward(next_state_batches).detach()
        target_next_q = self.critic_target.forward(next_state_batches, target_next_actions).detach()

        main_q = self.critic(state_batches, action_batches)

        # Critic Loss
        self.critic.zero_grad()
        baselines = reward_batches + done_batches * self.config.gamma * target_next_q
        loss_critic = torch.nn.MSELoss()(main_q, baselines)
        loss_critic.backward()
        self.critic_optimizer.step()

        # Actor Loss
        self.actor.zero_grad()
        clear_action_batches = self.actor.forward(state_batches)
        loss_actor = (-self.critic.forward(state_batches, clear_action_batches)).mean()
        loss_actor.backward()
        self.actor_optimizer.step()

        # This is for logging
        self.c_loss = loss_critic.item()
        self.a_loss = loss_actor.item()

        soft_update(self.actor, self.actor_target, self.config.tau)
        soft_update(self.critic, self.critic_target, self.config.tau)

    def getLoss(self):
        return self.c_loss, self.a_loss
