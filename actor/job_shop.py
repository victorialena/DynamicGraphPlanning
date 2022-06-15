import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import itertools
# import scipy.sparse as sp

from copy import copy, deepcopy

# ----------------- Helpers

def construct_readout_graph(g, etype):
    """ 
    Returns graph of edges to be evalutated, i.e., draw edges from each worker to each job node.
    If g is a batched graph representation, return edges within each subgraph.
    etype = use .canonical_etypes() 
    """
    
    utype, _, vtype = etype
    nu, nv = g.num_nodes(utype), g.num_nodes(vtype)
    if not is_batched_graph(g):
        src, dst = g.nodes(utype).repeat(nv), g.nodes(vtype).repeat_interleave(nu)
        
        return dgl.heterograph({etype: (src, dst)},
                               num_nodes_dict={utype: nu, vtype: nv})
    # else:
    node_ids = g.ndata['id']
    src = torch.cat([g.nodes(utype)[node_ids[utype]==i].repeat(c)
                     for i, c in zip(*node_ids[vtype].unique(return_counts=True))])
    dst = torch.cat([g.nodes(vtype)[node_ids[vtype]==i].repeat_interleave(c)
                     for i, c in zip(*node_ids[utype].unique(return_counts=True))])
    
    out = dgl.heterograph({etype: (src, dst)}, num_nodes_dict={utype: nu, vtype: nv})
    out.nodes['job'].data['id'] = g.nodes['job'].data['id']
    out.nodes['worker'].data['id'] = g.nodes['worker'].data['id']
    return out

def is_batched_graph(g):
    return len(g.ndata['id'])!=0

def num_subgraphs(g):
    return 1 if not is_batched_graph(g) else len(g.ndata['id']['job'].unique())

def sg_nworkers(g):
    if is_batched_graph(g):
        idx, cnts = g.ndata['id']['worker'].unique(return_counts=True)
        if len(cnts.unique()) == 1:
            return cnts.unique().item()
        return cnts
    return g.num_nodes('worker')

def batch_graphs(batch):
    n = len(batch)
    
    njs = torch.tensor([g.num_nodes('job') for g in batch])
    nws = torch.tensor([g.num_nodes('worker') for g in batch])
    
    def concat_edges(etype, idx, ns):
        return torch.cat([batch[i].edges(etype=etype)[idx]+ns[:i].sum() for i in range(n)])
    
    batched_graph_data = {
        ('job', 'precede', 'job'): (concat_edges('precede', 0, njs), concat_edges('precede', 1, njs)), 
        ('job', 'next', 'job'): (concat_edges('next', 0, njs), concat_edges('next', 1, njs)),
        ('worker', 'processing', 'job'): (concat_edges('processing', 0, nws), concat_edges('processing', 1, njs)),
    }

    state = dgl.heterograph(batched_graph_data, 
                            num_nodes_dict={'worker': nws.sum().item(), 'job': njs.sum().item()})

    state.nodes['job'].data['hv'] = torch.cat([g.nodes['job'].data['hv'] for g in batch])
    state.nodes['worker'].data['he'] = torch.cat([g.nodes['worker'].data['he'] for g in batch])
    state.nodes['job'].data['id'] = torch.arange(n).repeat_interleave(njs)
    state.nodes['worker'].data['id'] = torch.arange(n).repeat_interleave(nws)
    
    return state

# ----------------- ReadOut Op

class dotProductPredictor(nn.Module):
    """ returns scores for each job (row) per worker (col)"""
    def forward(self, graph, hv, he, _etype):
        # hv contains the node representations computed from the GNN
        utype, etype, vtype = _etype
        nu = sg_nworkers(graph)
        assert type(nu)==int, "Graphs have different workers counts. Can not proceed this eval without errors."
        with graph.local_scope():
            graph.nodes[vtype].data['hv'] = hv
            graph.nodes[utype].data['he'] = he
            graph.apply_edges(fn.u_dot_v('he', 'hv', 'score'), etype=etype)
            return graph.edges[etype].data['score'].view(num_subgraphs(graph), -1, nu).squeeze()
        
# ----------------- Model

class hgnn(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.embedding = dglnn.HeteroLinear({'job': 7, 'worker':3}, 16)
        self.conv = dglnn.HeteroGraphConv({
            'precede' : dglnn.GraphConv(16, 16),
            'next' : dglnn.GraphConv(16, 16),
            'processing' : dglnn.SAGEConv((16, 16), 16, 'mean')},
            aggregate='sum')
        self.pred = dotProductPredictor()
        
    def forward(self, g):
        h0 = {**g.ndata['hv'], **g.ndata['he']}
        h1 = self.embedding(h0)
        hv = self.conv(g, h1)['job']
        
        rg = construct_readout_graph(g, ('worker', 'processing', 'job'))
        return self.pred(rg, hv, h1['worker'], ('worker', 'processing', 'job'))
    
# --------------- Agent
from rlkit.policies.base import Policy

class epsilonGreedyPolicy(nn.Module, Policy):
    def __init__(self, net, eps=0.1):
        super().__init__()
        self.net = net
        self.eps = np.clip(eps, .0, 1.)

    def __call__(self, g):
        return self.get_action(g)
    
    def get_action(self, g):
        idx = np.where(g.ndata['hv']['job'][:, 3] == 0)[0] # idx: list of unscheduled job ids
        assert len(idx) > 0, "Unecessary query. Empty action space."
        
        if np.random.rand() < self.eps:
            nw = g.num_nodes(ntype='worker')
            return (np.random.randint(nw), np.random.choice(idx))
        
        out = self.net(g)
        val, workers = out[idx].max(1)
        j = val.argmax().item()
        w = workers[j].item()
        return (w, idx[j])
    
    def get_actions(self, g):
        assert is_batched_graph(g), "This is a single graph, call get_action() instead."
        
        n = num_subgraphs(g)
        idx = (g.ndata['hv']['job'][:, 3] == 0).view(n, -1)
        assert sum(idx).item() > 0, "Empty action space!"
        
        scores = self.net(g)
        out = {}
        values, workers = scores.max(-1, keepdims=False)
        for i in range(n):
            j = torch.where(idx[i])[0][values[i, idx[i]].argmax().item()].item()
            out[i] = (workers[i, j].item(), j)
        return out