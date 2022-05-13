import gym
import networkx as nx
import numpy as np
import pdb
import torch
import torch_geometric

from collections import namedtuple
from copy import copy, deepcopy
from enum import Enum, IntEnum
from gym.spaces import Box, MultiDiscrete
from numpy import rad2deg, deg2rad, cos, sin
from numpy.random import choice, randint, rand, uniform
from typing import Optional

from itertools import combinations, combinations_with_replacement
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, to_networkx, to_undirected

# https://courses.cit.cornell.edu/mae5070/DynamicEquations.pdf

# ------------- DEF

sysconfig = namedtuple("sysconfig", ['maxXYZ', 'collision_penalty', 'collision_thresh', 'action_cost',
                                     'dt', 'max_v', 'min_v', 'dv'], 
                       defaults=[720000., -1., 1., -.01, # meters
                                 1., 150., 200., 2.]) # tick = 1s and v [m/s]

# ------------- HELPERS

# ------------- 1/

def wrap(alpha):
    """ wrap angle alpha to interval [-pi, pi)"""
    return (alpha + np.pi) % (2.0 * np.pi) - np.pi

def rotx(a):
    """ wrap angle alpha to interval [-pi, pi)"""
    return torch.tensor([[1, 0, 0], [0, cos(a), -sin(a)], [0, sin(a), cos(a)]])

def roty(a):
    """ wrap angle alpha to interval [-pi, pi)"""
    return torch.tensor([[cos(a), 0, sin(a)], [0, 1, 0], [-sin(a), 0, cos(a)]])

def rotz(a):
    """ wrap angle alpha to interval [-pi, pi)"""
    return torch.tensor([[cos(a), -sin(a), 0], [sin(a), cos(a), 0], [0, 0, 1]])

def rot_b2i(a, b, c):
    return torch.matmul(torch.matmul(rotz(c).T, roty(b).T), rotx(a).T)

# ------------- 2/

def propagate_s(s, a, dt):
    sp = deepcopy(s)
    sp[3:] += a
    sp[:3] += torch.matmul(rot_b2i(*sp[3:-1]), torch.tensor([sp[-1], 0, 0])) * dt
    return sp

# ------------- 3/

def fully_connect_graph(n_nodes):
    """ Connect the graph s.t. all drones are interconnected. """
    
    idx = torch.combinations(torch.arange(n_nodes), r=2)
    return to_undirected(idx.t(), num_nodes=n_nodes)

def sparse_connect_graph(x, thresh):
    """ Connect the graph s.t. drones are only connected to other drones within a threshhold distance. """
    
    n = len(x)
    dis = torch.cdist(x[:, :3], x[:, :3], p=2)
    connected = (dis < thresh).triu(1)
    
    edge_index = torch.vstack(torch.where((dis < thresh).triu(1)))
    return to_undirected(edge_index)

# ------------- ENV
 
class collisionAvoidance(gym.Env):
    """
    ### Description
    
    ### Action Space
    Each agent in the scene can move horizontaly (R or L), vertically (U or D) or not at all.
    
    ### State Space
    The state is defined as an arbitrary input array of positions of n drones, appended by the location
    of the goal region.
    
    ### Rewards
    The reward of +1 is given to the system for when all drones reached a valid goal region, and a penality for any two drones
    colliding. Since this describes a discrete environment, to agents are considered in collision iif they occupy the same cell.
    
    ### Starting State
    Randomly initilized input array.
    
    ### Episode Termination
    When all drones have reached the goal region.
    
    ### Arguments
    No additional arguments are currently supported.
    """

    def __init__(self, args, device='cpu'):
        self.config = sysconfig()
        self.nagents = args.ndrones
        
        self.aspace = Box(low=np.array([-deg2rad(1)]*3+[-self.config.dv]), 
                          high=np.array([deg2rad(1)]*3+[self.config.dv]), 
                          dtype=np.float32)
        self.sspace = Box(low=np.array([-self.config.maxXYZ//2]*3+[-np.pi]*3+[self.config.min_v]), 
                          high=np.array([self.config.maxXYZ//2]*3+[np.pi]*3+[self.config.max_v]), 
                          dtype=np.float32)
        
#         self._isspace = Box(low=np.array([0]*3+[-np.deg2rad(45), -np.deg2rad(30), -np.pi]+[self.config.min_v]), 
#                             high=np.array([self.config.maxXYZ]*3+[np.deg2rad(45), np.deg2rad(30), np.pi]+[self.config.max_v]),
#                             dtype=np.float32)
        
        self.state = None
        self._device = device
        
    def get_distances(self):
        return torch.cdist(self.state.x[:, :3], self.state.x[:, :3], p=2)
    
    def in_collision(self):
        dis = self.get_distances() < self.config.collision_thresh
        return torch.triu(dis, 1).sum(1) #.sum().item()
    
    def is_terminal(self):
        return False
    
    def get_size(self):
        return torch.tensor(np.vstack([self.sspace.low, self.sspace.high]))
    
    def get_channels(self):
        """ Returns in, out channel sizes per agent. """
        return self.sspace.shape[0], self.aspace.shape[0]
        
    def reward(self, a):
        action_cost = abs(a)
        action_cost[:,-1] /= 100.
        return self.config.collision_penalty * self.in_collision() + \
               self.config.action_cost * action_cost.sum(1)
                        
    def step(self, a):
        err_msg = f"{a!r} ({type(a)}) is not a valid action."
        if torch.is_tensor(a): # model output
            assert ((a < 1.0) * (a > 0.)).all().item(), err_msg
            a = (a-0.5)*2.*self.aspace.high
        else: # from aspace.sample()
            assert all([self.aspace.contains(_a) for _a in a]) , err_msg
        
        reward = self.reward(a)
        done = self.is_terminal()
        
        for i, (s, _a) in enumerate(zip(self.state.x, a)):
            self.state.x[i] = propagate_s(s, _a, self.config.dt)
        
        # wrap to pi and bounds
        self.state.x[:, :3] = ((self.state.x[:, :3] + (self.config.maxXYZ//2)) % self.config.maxXYZ) - (self.config.maxXYZ//2)
        self.state.x[:, 3:-1] = wrap(self.state.x[:, 3:-1])
        self.state.x[:, -1] = self.state.x[:, -1].clamp(min=self.config.min_v, max=self.config.max_v)
        
        return deepcopy(self.state), deepcopy(reward), deepcopy(done), {}

    def reset(self, seed: Optional[int] = None):
        if not seed == None:
            super().reset(seed=seed)
                
#         x = torch.stack([torch.tensor(self.sspace.sample()) for _ in range(self.nagents)])
#         x[:, 3:-2] = 0.05*x[:, 3:-2] # reset roll, pitch
        self.nagents = 4
        x = torch.tensor([[self.config.maxXYZ//8, 0, 0, 0, 0, -np.pi, self.config.max_v],
                          [0, self.config.maxXYZ//8, 0, 0, 0, np.pi/2, self.config.max_v],
                          [-self.config.maxXYZ//8, 0, 0, 0, 0, 0, self.config.max_v],
                          [0, -self.config.maxXYZ//8, 0, 0, 0, -np.pi/2, self.config.max_v]])
    
        self.state = Data(x=x, edge_index=fully_connect_graph(self.nagents)).to(self._device)
        
        return deepcopy(self.state)
    
    def seed(self, n: int):
        super().seed(n)
        self.aspace.seed(n)
        self.sspace.seed(n)
        
    def to(self, device):
        self._device = device
        if self.state:
            self.state = self.state.to(device)
            
            