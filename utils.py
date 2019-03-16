import torch
import numpy as np


def to_torch(np_array):
    return torch.from_numpy(np_array)


def hard_update(source, target):
    target.load_state_dict(source.state_dict())


def soft_update(source, target, tau):
    for src_param, tgt_param in zip(source.parameters(), target.parameters()):
        tgt_param.data.copy_(
            tgt_param.data * (1.0 - tau) + src_param.data * tau
        )


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


def split_obs(obs, num_predator=3):
    return np.array(obs[:num_predator]), np.array(obs[num_predator:])[0]


def merge_action(a1, a2):
    return np.concatenate((a1, a2), axis=0)