import pdb

import numpy as np
from numpy.random import rand, choice

import torch
import torch_geometric
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, to_networkx
from torch_geometric.utils.random import erdos_renyi_graph

import networkx as nx
import gym
from gym.spaces import MultiDiscrete

from collections import namedtuple
from copy import copy, deepcopy
from typing import Optional
from enum import Enum, IntEnum

class status(IntEnum):
    good=0
    faulty=1
    dead=2
    
class load(IntEnum):
    idle=0
    loaded=1
    success=2
    
class action(IntEnum): 
    noop=0 
    reboot=1
    
"""
# baseline config params
# [https://github.com/JuliaPOMDP/MultiAgentSysAdmin.jl/blob/master/src/MultiAgentSysAdmin.jl]

p_fail_base::Float64 = 0.4
p_fail_bonus::Float64 = 0.2
p_dead_base::Float64 = 0.1
p_dead_bonus::Float64 = 0.5
# load
p_load::Float64 = 0.6
p_doneG::Float64 = 0.9
p_doneF::Float64 = 0.6

discount::Float64 = 0.9
reboot_penalty = -0.0
"""

sysconfig = namedtuple("sysconfig", 
                       ['p_fail_base', 'p_fail_bonus', 'p_dead_base', 'p_dead_bonus',
                        'p_load', 'p_doneG', 'p_doneF',
                        'discount', 'reboot_penalty', 'working', 'done'],
                       defaults=[0.4, .2, .1, .5, .6, .9, .6, 
                                 .9, -.0, .0, 1.])
#                        defaults=[.25,.2,.1,.5,
#                                  .6,.9,.6,
#                                  .9, -.5, .1, 1.])

format_input = lambda x: F.one_hot(x, num_classes=len(status)).reshape(-1,len(status)+len(load)).to(torch.float32)

format_data = lambda x: (format_input(x.x), x.edge_index)

format_data_input = lambda x: format_input(x.x)

def neighbors_status(x, edge_index, nodeid):
    "Assumes the graph g in bidirectional."
    return x[edge_index[1][edge_index[0]==nodeid]]

class sysAdminProbe(gym.Env):
    """
    ### Description
    
    ### State Space
    The state is defined as an arbitrary input graph where each node represents a machine in a
    network with 2 state variables: Status s = {good, faulty, dead} and Laod l = {idle, loaded, done}.
    
    ### Transition
    A dead machine increases the odds of its neighbors dying, same for a faulty machine. A faulty machine
    requires more time for its process to finish and a dead machine looses the allocated job entirely.
    
    ### Action Space
    Each agent must decided whether or not to reboot at the current time.
    
    ### Rewards
    The system gets a reward of 1 for each machine finishing a job successfully. 
    
    ### Starting State
    The network topology is initialized at random, but we also use ring, star and ring-of-ring topologies.
    
    ### Episode Termination
    A simulation terminates when either all jobs have been completed or a maximum nr of steps has been reached.
    
    ### Arguments
    No additional arguments are currently supported.
    """

    def __init__(self, nnodes: int, njobs: int):
        self.n_nodes = nnodes
        self.n_jobs = njobs
        self.config = sysconfig()
        self.topology = 'random'
        
        self.aspace = MultiDiscrete([len(action)] * self.n_nodes)
        self.sspace = MultiDiscrete([len(status)*len(load)] * self.n_nodes)
        self.state = None
        self.count = njobs
        
    def reward(self, a):
        r = torch.sum(self.state.x[:, 1] == load.loaded) * self.config.working \
            + torch.sum(a==action.reboot) * self.config.reboot_penalty \
            + torch.sum(self.state.x[:, 1] == load.success) * self.config.done
        return r
    
    def ssample():
        s = self.sspace.sample()
        return Data(x=torch.Tensor(np.stack([s//len(status), s%len(status)]).T).to(torch.int64),
                    edge_index=self.sample_edge_index())
    
    def asample():
        return self.aspace.sample()
        
    def step(self, a):
        err_msg = f"{a!r} ({type(a)}) is not a valid action."
        assert self.aspace.contains(a.numpy()), err_msg
        
        self.count = self.count - torch.sum(self.state.x[:, 1] == load.success)
        
        reward = self.reward(a)
        done = self.count <= 0
        x0 = deepcopy(self.state.x[:, 0])
        
        egdes = self.state.edge_index
        
        p_fail = torch.tensor([(neighbors_status(x0, egdes, i) == status.faulty).numpy().mean()
                               if any(egdes[0]==i) else 0. 
                               for i in range(self.n_nodes)])
        p_fail = self.config.p_fail_base + self.config.p_fail_bonus*p_fail
        
        p_dead = torch.tensor([(neighbors_status(x0, egdes, i) == status.dead).numpy().mean()
                               if any(egdes[0]==i) else 0. 
                               for i in range(self.n_nodes)])
        p_dead = self.config.p_dead_base + self.config.p_fail_bonus*p_fail
                
        self.state.x[:, 0] = torch.where((x0 == status.good) & (torch.rand(self.n_nodes) < p_fail),
                                   torch.Tensor([status.faulty]*self.n_nodes).to(torch.int64),
                                   self.state.x[:, 0])
        
        self.state.x[:, 0] = torch.where((x0 == status.faulty) & (torch.rand(self.n_nodes) < p_dead),
                                   torch.Tensor([status.dead]*self.n_nodes).to(torch.int64),
                                   self.state.x[:, 0])
        
        self.state.x[:, 0] = torch.where((x0 == status.dead) & (a==action.reboot),
                                   torch.Tensor([status.good]*self.n_nodes).to(torch.int64),
                                   self.state.x[:, 0])
        
        x1 = deepcopy(self.state.x[:, 1])
        
        p = torch.where(x0 == status.good, self.config.p_doneG, self.config.p_doneF)
        self.state.x[:, 1] = torch.where((x1 == load.idle) & (torch.rand(self.n_nodes) < self.config.p_load),
                                   torch.Tensor([load.loaded]*self.n_nodes).to(torch.int64),
                                   self.state.x[:, 1])
        
        self.state.x[:, 1] = torch.where((x1 == load.loaded) & (torch.rand(self.n_nodes) < p),
                                   torch.Tensor([load.success]*self.n_nodes).to(torch.int64),
                                   self.state.x[:, 1])      
        
        self.state.x[:, 1] = torch.where((x1 == load.success) | (self.state.x[:, 0] == status.dead),
                                   torch.Tensor([load.idle]*self.n_nodes).to(torch.int64),
                                   self.state.x[:, 1])
                        
        return deepcopy(self.state), deepcopy(reward.item()), deepcopy(done), {}
    
    def sample_edge_index(self, topology=None):
        if topology == None:
            topology = self.topology
            
        if topology == 'random':
            return erdos_renyi_graph(self.n_nodes, 0.75, directed=False)
        elif topology == 'star':
            arr = torch.arange(1, self.n_nodes)
            edge_index = torch.stack([arr, torch.zeros(self.n_nodes-1, dtype=torch.int64)])
            return torch.hstack([edge_index, edge_index.flip(0)])
        elif topology == 'ring':
            arr = torch.arange(self.n_nodes)
            edge_index = torch.vstack([arr, arr.roll(-1,0)])
            return torch.hstack([edge_index, edge_index.flip(0)])
            
        err_msg = f"Unknown topology. Choose among 'ring', 'star', or 'random'."
        assert False, err_msg

    def reset(self, seed: Optional[int] = None, topology: str = 'random'):
        if not seed == None:
            super().reset(seed=seed)
        self.topology = topology
        edge_index = self.sample_edge_index()
        
        x = torch.zeros((self.n_nodes, 2), dtype=torch.int64) #torch.randint(high=len(status), size=(3,2))
        self.state = Data(x=x, edge_index=edge_index)
        self.count = self.n_jobs
        
        return deepcopy(self.state)

    def render(self, s=None):
        if not s:
            s = self.state
        g = torch_geometric.utils.to_networkx(s, to_undirected=True)
        colors = np.array(['green', 'yellow', 'red'])
        color_map = colors[s.x.numpy()[:, 0]]
        labeldict = {i: 'L' if v==load.loaded else ('I' if v==load.idle else 'S')  for i, v in enumerate(s.x[:, 1])}
        nx.draw_circular(g, node_color=color_map, labels=labeldict)
        
    def seed(self, n: int):
        super().reset(seed=seed)


#---------------------------- Helpers

def avg_traj_r(paths):
    return np.mean([np.sum(p['rewards']) for p in paths])

def cumulative_discounted_sum(vec, gamma):
    n = len(vec)
    discounted_vec = (gamma**torch.arange(n))*torch.tensor(vec)
    return discounted_vec.sum().item()

def avg_cumulative_discounted_r(paths, gamma=.9):
    return np.mean([cumulative_discounted_sum(p['rewards'], gamma) for p in paths])

#---------------------------- Add ons

from utils.policy_base import Policy
import torch.nn as nn

class rebootWhenDead(nn.Module, Policy):
    def __init__(self):
        super().__init__()

    def get_action(self, obs):
        return torch.where(obs.x[:, 0]==status.dead, action.reboot, action.noop), {}

def print_sar(s, a, r):
    colors = np.array(['G', 'F', 'D'])
    print('STATE ', colors[s.numpy()[:, 0]].tolist())
    colors = np.array(['I', 'L', 'S'])
    print('LOAD  ',colors[s.numpy()[:, 1]].tolist())
    colors = np.array(['N', 'R'])
    print('ACTIO ', colors[np.array(a)].tolist())
    print('R ', r, '\n')
    
print_path = lambda p : [print_sar(s.x, a, r) for s, a, r in zip(p['observations'], p['actions'], p['rewards'])]