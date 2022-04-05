import numpy as np
import torch
import torch.nn as nn

from numpy.random import rand
from rlkit.policies.base import Policy
    
class argmaxDiscretePolicy(nn.Module, Policy):
    def __init__(self, qf, dim=1):
        super().__init__()
        self.qf = qf
        self.dim = dim

    def get_action(self, obs):
        q_values = self.qf(obs)
        return q_values.argmax(self.dim), {}
    
# redundant code - clean this up
class epsilonGreedyPolicy(nn.Module, Policy):
    def __init__(self, qf, space, eps=0.1, dim=1, sim_annealing_fac=1.0, minimum=0.05, device='cpu'):
        super().__init__()
        self.qf = qf
        self.aspace = space
        
        self.eps = np.clip(eps, .0, 1.)
        self.dim = dim
        self.saf = np.clip(sim_annealing_fac, .0, 1.)
        self.min = minimum
        self.device = device
        
    def simulated_annealing(self):
        self.eps = max(self.min, self.eps*self.saf)

    def get_action(self, obs):
        if rand() < self.eps:
            return torch.Tensor(self.aspace.sample()).to(torch.long).to(self.device), {}
        q_values = self.qf(obs)
        return q_values.argmax(self.dim), {}
   
        