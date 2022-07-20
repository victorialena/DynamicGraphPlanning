import pdb

import numpy as np
from numpy.random import rand, choice

import torch
import torch.nn as nn

import networkx as nx

#--------------------- Policies

from utils.policy_base import Policy

class argmaxDiscretePolicy(nn.Module, Policy):
    def __init__(self, qf, dim=1, preprocessing=lambda x: x):
        super().__init__()
        self.qf = qf
        self.dim = dim
        self.preprocessing = preprocessing

    def get_action(self, obs):
        q_values = self.qf(*self.preprocessing(obs))
        return q_values.argmax(self.dim), {}

class doNothingPolicy(nn.Module, Policy):
    def __init__(self, default):
        super().__init__()
        self.default = default

    def get_action(self, obs):
        return torch.Tensor([self.default]*len(obs.x)).to(torch.int64), {}
    
class epsilonGreedyPolicy(nn.Module, Policy):
    def __init__(self, qf, space, eps=0.1, dim=1, preprocessing=lambda x: x):
        super().__init__()
        self.qf = qf
        self.aspace = space
        
        self.eps = np.clip(eps, .0, 1.)
        self.dim = dim
        self.preprocessing = preprocessing

    def get_action(self, obs):
        if rand() < self.eps:
            return torch.Tensor(self.aspace.sample()).to(torch.long), {}

        q_values = self.qf(*self.preprocessing(obs))
        return q_values.argmax(self.dim), {}
    
