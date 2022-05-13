import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

from collections import OrderedDict
from actor.gatv2_conv import GATv2Conv
from rlkit.policies.base import Policy
from warnings import warn

import pdb

# ------------------ Helpers

def interleave_lists(list1, list2):
    out = list1 + list2
    out[::2] = list1
    out[1::2] = list2
    return out

#-------------- Heuristic policy (baseline)
from env.drone_delivery import action

def pick(x):
    probs = np.array([x[0]>0, x[0]<0, x[1]>0, x[1]<0, x[1]==x[0]==0], dtype=int)
    return np.random.choice(action, p=probs/sum(probs))

def move2closestGoal(obs):
    goal_mask = obs.x[:, -2] > 0
    drone_mask = obs.x[:, -2] < 0

    idx = torch.cdist(obs.x[drone_mask, :2], obs.x[goal_mask, :2], p=1).min(1).indices
    dis = (obs.x[goal_mask][idx, :2] - obs.x[drone_mask, :2])
    return torch.Tensor([pick(d) for d in dis]).to(torch.long)

class sysRolloutPolicy(nn.Module, Policy):
    def __init__(self, device='cpu'):
        super().__init__()            
        self.device = device

    def get_action(self, obs):
        goal_mask = obs.x[:, -2] > 0
        drone_mask = obs.x[:, -2] < 0
        
        idx = torch.cdist(obs.x[drone_mask, :2], obs.x[goal_mask, :2], p=1).min(1).indices
        dis = (obs.x[goal_mask][idx, :2] - obs.x[drone_mask, :2])
        return move2closestGoal(obs).to(self.device), {}

class sysMultiRolloutPolicy(nn.Module, Policy):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

    def get_action(self, obs):
        goal_mask = obs.x[:, -2] > 0
        drone_mask = obs.x[:, -2] < 0
        active_drone_mask = drone_mask * (obs.x[:, -1] > 0)
        active_given_drone_mask = obs.x[drone_mask][:, -1] > 0
        
        xg = obs.x[goal_mask]
        xg = torch.repeat_interleave(xg, xg[:, -1].to(torch.int64), dim=0)[:active_drone_mask.sum().item()]
        
        out = torch.Tensor([4]*sum(drone_mask)).to(torch.long).to(self.device)
        if len(xg)==0: #empty
            return out, {}
        
        xd = obs.x[active_drone_mask]
        dis = (xg[:, :2] - xd[:, :2])
        out[active_given_drone_mask] = torch.Tensor([pick(d) for d in dis]).to(torch.long) #.to(self.device)
        return out, {}
    
#-------------- GCN model   
        
class droneDeliveryModel(nn.Module):
    
    def __init__(self, c_in, c_out, c_hidden=[], n_linear=1, bounds=None, **kwargs):
        
        super().__init__()
            
        activation_fn = kwargs['activation'] if 'activation' in kwargs.keys() else nn.ReLU(inplace=True)

        assert len(c_hidden) > 0, "Hidden dimension can not be zero => no GCN layer."
        layer_size = [c_in]+c_hidden+[c_out]
        n_sage = len(layer_size)-n_linear-1
        n_heads = kwargs['heads'] if 'heads' in kwargs.keys() else [4]*len(layer_size)
        
        layers = [(GATv2Conv(layer_size[i],
                             layer_size[i+1]//(n_heads[i] if i<n_sage-1 else 1), 
                             heads=n_heads[i], 
                             concat=(i<n_sage-1), 
                             dropout=kwargs['dropout']), 'x, edge_index -> x')
                  if i < n_sage else
                  nn.Linear(layer_size[i], layer_size[i+1]) 
                  for i in range(len(c_hidden)+1)]
        layers = interleave_lists(layers, [activation_fn]*(len(layer_size)-2))
        
        self.model = gnn.Sequential('x, edge_index', layers)
              
        self._device = 'cpu'
        self._upper_bound = bounds

    def forward(self, x):
        y = x.x
        drone_mask = x.x[:, -2] < 0
        if self._upper_bound is not None:
            y = y.div(self._upper_bound-1)
        return self.model(y, x.edge_index)[drone_mask]    
    
    def to(self, device):
        super().to(device)
        self._device = device
        if self._upper_bound is not None:
            self._upper_bound = self._upper_bound.to(device)

#-------------- GCN model (repeated)

class droneDeliveryModel_rep3(nn.Module):
    
    def __init__(self, c_in, c_out, c_hidden, k, bounds=None, **kwargs):
        
        super().__init__()
        
        self.encoder = nn.Linear(c_in, c_hidden)
        self.sage = gnn.SAGEConv(c_hidden, c_hidden)
        self.gat = GATv2Conv(c_hidden, c_out, heads=4, concat=False, dropout=kwargs['dropout'])
        self._k = k
        
        self._device = 'cpu'
        self._upper_bound = bounds 

    def forward(self, x):
        y = x.x
        drone_mask = x.x[:, -2] < 0
        if self._upper_bound is not None:
            y = y.div(self._upper_bound-1)
            
        out = self.encoder(y).relu()
        for i in range(self._k):
            out = self.sage(out, x.edge_index).relu()
        return self.gat(out, x.edge_index)[drone_mask]
    
    def to(self, device):
        super().to(device)
        self._device = device
        if self._upper_bound is not None:
            self._upper_bound = self._upper_bound.to(device)

class droneDeliveryModel_rep1(nn.Module):
    
    def __init__(self, c_in, c_out, c_hidden, k, bounds=None, **kwargs):
        
        super().__init__()
        
        self.encoder = nn.Linear(c_in, c_hidden)
        self.layer = gnn.SAGEConv(c_hidden, c_hidden)
        self.decoder = nn.Linear(c_hidden, c_out)
        self._k = k
        
        self._device = 'cpu'
        self._upper_bound = bounds 

    def forward(self, x):
        y = x.x
        drone_mask = x.x[:, -2] < 0
        if self._upper_bound is not None:
            y = y.div(self._upper_bound-1)
            
        out = self.encoder(y).relu()
        for i in range(self._k):
            out = self.layer(out, x.edge_index).relu()
        return self.decoder(out)[drone_mask]    
    
    def to(self, device):
        super().to(device)
        self._device = device
        if self._upper_bound is not None:
            self._upper_bound = self._upper_bound.to(device)

class droneDeliveryModel_rep2(nn.Module):
    
    def __init__(self, c_in, c_out, c_hidden=[], bounds=None, k=[], **kwargs):
        
        super().__init__()
        assert len(c_hidden) > 0, "Hidden dimension can not be zero => no GCN layer."
        
        self.encoder = nn.Linear(c_in, c_hidden[0])
        self.layers = [gnn.SAGEConv(l_size, l_size) for l_size in c_hidden]
        self.decoder = nn.Linear(c_hidden[-1], c_out)
        self._k = k
        
        assert len(k) == len(c_hidden), "Number of GCN layers doesn't match iterations per layer."
        
        self._device = 'cpu'
        self._upper_bound = bounds
        
        if len(c_hidden)>1:
            self._k = interleave_lists(k, [1]*(len(k)-1))
            self.layers = interleave_lists(self.layers, [gnn.SAGEConv(c_hidden[i], c_hidden[i+1]) for i in range(len(k)-1)])

    def forward(self, x):
        y = x.x
        drone_mask = x.x[:, -2] < 0
        if self._upper_bound is not None:
            y = y.div(self._upper_bound-1)
            
        out = self.encoder(y).relu()
        for k, layer in zip(self._k, self.layers):
            for i in range(k):
                out = layer(out, x.edge_index).relu()
        return self.decoder(out)[drone_mask]    
    
    def to(self, device):
        super().to(device)
        self._device = device
        if self._upper_bound is not None:
            self._upper_bound = self._upper_bound.to(device)
            
#-------------- heterogenous GCN model   

class droneDeliveryModelHeterogenous(nn.Module):
    
    def __init__(self, c_in, c_out, c_hidden=[], bounds=None):
        
        super().__init__()
        
        self.lin1_d = nn.Linear(c_in, c_hidden[0])
        self.lin1_g = nn.Linear(c_in, c_hidden[0])
        self.gat1 = gnn.GATv2Conv(c_hidden[0], c_hidden[1], heads=8, concat=False)
        self.lin2_d = nn.Linear(c_hidden[1], c_hidden[2])
        self.lin2_g = nn.Linear(c_hidden[1], c_hidden[2])
        self.gat2 = gnn.GATv2Conv(c_hidden[2], c_hidden[3], heads=8, concat=False)
        self.lin1 = nn.Linear(c_hidden[3], c_out)
        
        self._upper_bound = bounds
        self._c_hidden = c_hidden
    
    def forward(self, x):
        y = x.x
        drone_mask = x.x[:, -2] < 0
        goal_mask = x.x[:, -2] > 0
        
        if self._upper_bound is not None:
            y = y.div(self._upper_bound-1)
        
        out1 = self.lin1_d(y)
        out1[goal_mask] = self.lin1_g(y[goal_mask])
        out1 = self.gat1(out1, x.edge_index).relu()
        
        out = self.lin2_d(out1)
        out[goal_mask] = self.lin2_g(out1[goal_mask])
        out = self.gat2(out, x.edge_index).relu()
        
        return self.lin1(out)[drone_mask]
    