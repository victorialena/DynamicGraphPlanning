import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

from collections import OrderedDict
from gatv2_conv import GATv2Conv
from rlkit.policies.base import Policy
from warnings import warn

import pdb

# ------------------ Helpers

def interleave_lists(list1, list2):
    out = list1 + list2
    out[::2] = list1
    out[1::2] = list2
    return out
    
#-------------- GCN model   
        
class collisionAvoidanceModel(nn.Module):
    
    def __init__(self, c_in, c_out, c_hidden, bounds, n_linear=1, **kwargs):
        
        super().__init__()
            
        activation_fn = kwargs['activation'] if 'activation' in kwargs.keys() else nn.ReLU(inplace=True)

        assert len(c_hidden) > 0, "Hidden dimension can not be zero => no GCN layer."
        layer_size = [c_in]+c_hidden+[c_out]
        n_sage = len(layer_size)-n_linear-1
        n_heads = kwargs['heads'] if 'heads' in kwargs.keys() else 4
        
        layers = [(GATv2Conv(layer_size[i],
                                 layer_size[i+1]//(n_heads if i<n_sage-1 else 1), 
                                 heads=n_heads, 
                                 concat=(i<n_sage-1), 
                                 dropout=kwargs['dropout']), 'x, edge_index -> x')
                  if i < n_sage else
                  nn.Linear(layer_size[i], layer_size[i+1]) 
                  for i in range(len(c_hidden)+1)]
        layers = interleave_lists(layers, [activation_fn]*(len(layer_size)-2))
        
        self.model = gnn.Sequential('x, edge_index', layers)
              
        self._device = 'cpu'
        self._upper_bound = bounds[1]
        self._lower_bound = bounds[0]

    def forward(self, x):
        y = x.x
        if self._upper_bound is not None:
            y = (y-self._lower_bound).div(self._upper_bound-self._lower_bound)
        return self.model(y, x.edge_index).sigmoid()
    
    
    def to(self, device):
        super().to(device)
        self._device = device
        if self._upper_bound is not None:
            self._upper_bound = self._upper_bound.to(device)
            self._lower_bound = self._lower_bound.to(device)
