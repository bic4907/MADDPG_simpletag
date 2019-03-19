import torch
import torch.nn as nn
from utils import fanin_init


class Actor(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Actor, self).__init__()


        self.s_dim = s_dim
        self.a_dim = a_dim

        self.norm1 = nn.BatchNorm1d(self.s_dim)
        self.fc1 = nn.Linear(self.s_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.a_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.norm1.weight.data.fill_(1)
        self.norm1.bias.data.fill_(0)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-0.003, 0.003)

    def forward(self, x):
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.tanh(x)
        return x


class Critic(nn.Module):
    def __init__(self, s_dim, a_dim, num_agent):
        super(Critic, self).__init__()

        self.num_agent = num_agent
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.norm1 = nn.BatchNorm1d(s_dim * num_agent)
        self.fc1 = nn.Linear(s_dim * num_agent, 64)
        self.fc2 = nn.Linear(64 + self.a_dim * num_agent, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

        self.norm1.weight.data.fill_(1)
        self.norm1.bias.data.fill_(0)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-0.003, 0.003)

    def forward(self, x_n, a_n):
        x = x_n.reshape(-1, self.s_dim * self.num_agent)
        a = a_n.reshape(-1, self.a_dim * self.num_agent)
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = torch.cat([x, a], 1)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x