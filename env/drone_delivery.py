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
from numpy.random import choice, randint, rand, uniform
from typing import Optional

from itertools import combinations, combinations_with_replacement
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, to_networkx, to_undirected

# ------------- DEF

sysconfig = namedtuple("sysconfig", 
                       ['maxX', 'maxY', 'goal_reward', 'collision_penalty', 'distance_penalty'], 
                       defaults=[4, 4, 1., -.1, -.01])

actions = namedtuple("actions", 
                    ['right', 'left', 'up', 'down', 'drop'],
                    defaults=[np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4)])
action = actions()

# ------------- HELPERS

# ------------- 1/

def check_capacity(indices, cap):
    idx, c = indices.unique(return_counts=True)
    return (cap[idx] == c).all()

# ------------- 2/

def randomList(m, n):
    """ Function to generate a list of m random non-negative integers whose sum is n. """ 
    arr = [0] * m
    for i in range(n):
        arr[randint(0, n) % m] += 1
    return arr

def to_pos(arr, x, n):
    rows = torch.div(arr, x, rounding_mode='trunc') # "out//y" deprecated
    return torch.vstack((rows, arr%x, -torch.ones(n), torch.ones(n))).T

def sample_unique(x: int, y: int, n: int):   
    assert n < x*y, "Impossible query."    
    out = torch.randperm(x*y)[:n]
    return to_pos(out, x, n)

def sample_near(x: int, y: int, nd: int, ng: int, nsteps: int):   
    goals = torch.randperm(x*y)[:ng]
    candidates = (goals + torch.tensor([sum(_) for _ in \
                                        combinations_with_replacement([1, -1, x, -x], nsteps)]).unsqueeze(1)).flatten().unique()
    
    filter1 = (candidates >= 0) * (candidates < x*y)
    filter2 = ~(candidates.repeat(len(goals), 1).T == goals).any(1)
    candidates = candidates[filter1*filter2]
    len(candidates)>=nd, "Cannot sample: Maze is too small for nr of drones."
    
    idx = torch.randperm(len(candidates))[:nd]
    out = torch.hstack((candidates[idx],goals))
    return to_pos(out, x, nd+ng)

def sample_state(x, y, nd, ng, nsteps=-1):
    """
    # Sample unique positions for each goal and drone (no initial collision) and store goal capacity in 
    # feature vector. Drones have capacity -1. +/- denotes the node type!      
    """
    if nsteps>0:
        try:
            state = sample_near(x, y, nd, ng, nsteps)
            state[nd:, -2] = torch.ones(ng)
            idx, counts = torch.cdist(state[:nd, :2], state[nd:, :2], p=1).min(1).indices.unique(return_counts=True)
            state[nd:, -1] = 0.
            state[nd+idx, -1] = counts.to(torch.float)
            return state
        except:
            pass
    state = sample_unique(x, y, nd+ng)
    state[nd:, -2] = torch.ones(ng)
    state[nd:, -1] = torch.tensor(randomList(ng, nd))
    
    return state

# ------------- 3/

def fully_connect_graph(n_nodes):
    """ Connect the graph s.t. all drones and goals are interconnected. """
    
    idx = torch.combinations(torch.arange(n_nodes), r=2)
    return to_undirected(idx.t(), num_nodes=n_nodes)

def connect_graph(n_drones, n_goal, reverse=False):
    """ Connect the graph s.t. all drones are interconnected and each goal connects to all drones. """
    
    idx = torch.combinations(torch.arange(n_drones), r=2)
    edge_index = to_undirected(idx.t(), num_nodes=n_drones)

    arr = torch.arange(n_drones, n_drones+n_goal)
    for i in range(n_drones):
        goal2drone = torch.stack([arr, i*torch.ones(n_goal, dtype=torch.int64)]) if not reverse else \
                     torch.stack([i*torch.ones(n_goal, dtype=torch.int64), arr])
        edge_index = torch.hstack((edge_index, goal2drone))
    return edge_index

def sparse_connect_graph(n_drones, n_goal, reverse=False, undirected=False):
    """ Connect the graph s.t. all drones are connected to each goal only. Requires a model with GCN depth ≥ 2 """
    
    arr = torch.arange(n_drones, n_drones+n_goal)
    edge_index = torch.hstack([torch.stack([arr, i*torch.ones(n_goal, dtype=torch.int64)]) \
                               for i in range(n_drones)])
    if undirected:
        return to_undirected(edge_index)
    if reverse:
        return edge_index.flip(0)
    return edge_index

# ------------- ENV
 
class droneDelivery(gym.Env):
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
        self.config = sysconfig(maxX=args.maze_size, maxY=args.maze_size)
        self.ndrones = args.ndrones
        self.ngoals = args.ngoals
        
        # L, R, up, down, drop payload
        self.aspace = MultiDiscrete([len(action)]*self.ndrones)
        self.a2vecmap = torch.Tensor([[1., 0.],
                                      [-1, 0.],
                                      [0., 1.],
                                      [0, -1.],
                                      [0., 0.]]).to(device)
        # x, y, d|g, payload|cap
        self.sspace = MultiDiscrete([self.config.maxX, self.config.maxY, 2, args.ndrones])
        self.state = None
        
        if args.graph_type == "full":
            self.generate_g = lambda : fully_connect_graph(self.ndrones+self.ngoals) 
        elif args.graph_type == "sparse":
            self.generate_g = lambda : sparse_connect_graph(self.ndrones, self.ngoals, undirected=True) 
        elif args.graph_type == "light":
            self.generate_g = lambda : connect_graph(self.ndrones, self.ngoals) 
        else:
            assert False, "Unspecified graph type."
        
        self._device = device
        self._max_sample_distance = args.max_sample_distance
    
    def drone_mask(self):
        return self.state.x[:, 2] < 0
    
    def goal_mask(self):
        return self.state.x[:, 2] > 0
        
    def get_distances(self):
        dis = torch.cdist(self.state.x[self.drone_mask(), :2], self.state.x[self.goal_mask(), :2], p=1)
        return dis.min(1)
    
    def in_collision(self):
        dis = torch.cdist(self.state.x[self.drone_mask(), :2], self.state.x[self.drone_mask(), :2], p=1)
        return (dis == 0).float().sum(1)-1.
    
    def is_terminal(self):
        return (self.state.x[self.drone_mask(), -1] == 0).all()
    
    def get_size(self):
        return torch.Tensor([self.config.maxX, self.config.maxY, 2, self.ndrones])
    
    def get_channels(self):
        """ Returns in, out channel sizes per agent. """
        return self.sspace.shape[0], self.aspace.nvec[0]
            
    def get_max_len(self):
        return self.config.maxX #+ self.config.maxY - 1
        
    def reward(self, a):
        # TODO: should distance penalty be only active for drones that carry a payload?
        done = self.is_terminal().float().item()
        
        dis = self.get_distances()
        dropping = ((a == 4) * (self.state.x[self.drone_mask(), -1]>0) * (dis.values == 0) * \
                   (self.state.x[self.goal_mask(), -1][dis.indices]>0)).float()
        
        return (self.config.goal_reward) * dropping + \
               (self.config.collision_penalty/2.) * self.in_collision() + \
               (self.config.distance_penalty/self.ndrones) * self.get_distances().values
                        
    def step(self, a):
        err_msg = f"{a!r} ({type(a)}) is not a valid action."
        assert self.aspace.contains(a.cpu().numpy() if torch.is_tensor(a) else a) , err_msg
        
        reward = self.reward(a)
        dropping = ((a == 4) * (self.state.x[self.drone_mask(), -1]>0))
        a = self.a2vecmap[a]
        done = self.is_terminal()
    
        self.state.x[self.drone_mask(), :2] = (self.state.x[self.drone_mask(), :2]+a).clamp(min=0, max=self.config.maxX)
        
        # cargo management
        gidx = torch.where(self.drone_mask())[0]
        self.state.x[gidx[dropping], -1] -= 1

        dis = self.get_distances()
        idx, load = dis.indices[(dis.values == 0) * dropping].unique(return_counts=True)
        didx = torch.where(self.goal_mask())[0]
        self.state.x[didx[idx], -1] -= load     
        self.state.x[:, -1] = self.state.x[:, -1].clamp(min=0.)
        
        return deepcopy(self.state), deepcopy(reward), deepcopy(done), {}

    def reset(self, seed: Optional[int] = None):
        if not seed == None:
            super().reset(seed=seed)
                
        x = sample_state(self.config.maxX, self.config.maxY, self.ndrones, self.ngoals, self._max_sample_distance)
        self.state = Data(x=x, edge_index=self.generate_g()).to(self._device)
        
        return deepcopy(self.state)

    def render(self, s = None):
        if not s:
            s = self.state
        g = torch_geometric.utils.to_networkx(s, to_undirected=False)
        colors = np.array(['orange' if _[-1]<0 else 'green' for _ in s.x])
        pos = {i: x[:2].numpy() for i, x in enumerate(s.x)}
        nx.draw(g, pos=pos, node_color=colors)
    
    def seed(self, n: int):
        super().seed(n)
        self.aspace.seed(n)
        self.sspace.seed(n)
        
    def to(self, device):
        self._device = device
        self.a2vecmap = self.a2vecmap.to(device)
        if self.state:
            self.state = self.state.to(device)
            
            