#!/usr/bin/env python
import pdb

import argparse

import numpy as np
from numpy.random import rand, choice

import torch
import torch.nn as nn
from torch.nn import Linear, ReLU

import torch.nn.functional as F
from torch.optim import Adam

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import Sequential, GCNConv, SAGEConv
from torch_geometric.utils import to_dense_adj, to_networkx
from torch_geometric.utils.random import erdos_renyi_graph

import networkx as nx
import gym
from gym.spaces import MultiDiscrete

from collections import namedtuple
from copy import copy, deepcopy
from typing import Optional
from enum import Enum, IntEnum

from env.sys_admin import *

from utils.path_collector import MdpPathCollector
from utils.std_policies import *
from utils.pyg_rollout_functions import rollout
from utils.replay_buffer import anyReplayBuffer

import matplotlib.pyplot as plt

def parse_args():
    parser=argparse.ArgumentParser(description="Sys-Admin default settings.")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--n_nodes", type=int, default=6, help="number of nodes/workers available for processing")
    parser.add_argument("--n_jobs", type=int, default=100, help="number of jobs to process")
    parser.add_argument("--eps", type=float, default=0.1, help="epislon greedy exploration parameter")
    parser.add_argument("--replay_buffer_cap", type=int, default=10000, help="replay buffer size, max nr of past samples")
    parser.add_argument("--n_samples", type=int, default=128, help="number of trajectories sampled at each epoch.")
    parser.add_argument("--prioritized_replay", type=bool, default=False, help="use prioritzed replay when sampling for HER")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1E-3, help="learning rate")
    parser.add_argument("-ep", "--n_epoch", type=int, default=100, help="number of epochs")
    parser.add_argument("-it", "--n_iter", type=int, default=32, help="number of iterations per epoch")
    parser.add_argument("-bs", "--batch_size", type=int, default=64, help="batch size per iteration")
    parser.add_argument("-γ", "--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--max_path_len", type=int, default=100, help="maximum trajectory length during sampling.")
    parser.add_argument("--layer_sz", type=int, default=16, help="number of heads per GATv2 layer.")
    parser.add_argument("--heuristic_pol_len", type=int, default=100)
    
    parser.add_argument("--load_from", type=str, default="", help="load a pretrained network's weights")
    parser.add_argument("--save", type=bool, default=False, help="save last model instance")
    parser.add_argument("--plot", type=bool, default=True, help="plot training performance curves")

    args=parser.parse_args()
    return args


class sysAdminModel(nn.Module):    
    def __init__(self, c_in, c_out, c_hidden=16, **kwargs):
        super().__init__()
        
        self._device = kwargs['device'] if 'device' in kwargs.keys() else 'cpu'
        self.model = Sequential('x, edge_index', [
            (SAGEConv(c_in, c_hidden), 'x, edge_index -> x'),
            ReLU(inplace=True),
            Linear(c_hidden, c_out),
        ])

    def forward(self, x, edge_index):
        return self.model(x.to(self._device), edge_index.to(self._device))
    
    def to(self, device):
        super().to(device)
        self._device = device
        
def main(args):
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = sysAdminProbe(nnodes=args.n_nodes, njobs=args.n_jobs)
    x = env.reset(topology='star')

    in_channels, out_channels = len(status)+len(load), len(action)

    qf = sysAdminModel(in_channels, out_channels, args.layer_sz)
    target_qf = sysAdminModel(in_channels, out_channels, args.layer_sz)
    
    if args.load_from:
        qf.load_state_dict(torch.load("chkpt/sys_admin/"+args.load_from))
        target_qf.load_state_dict(torch.load("chkpt/sys_admin/"+args.load_from))
    
    qf_criterion = nn.MSELoss()
    eval_policy = argmaxDiscretePolicy(qf, preprocessing=format_data)
    expl_policy = epsilonGreedyPolicy(qf, env.aspace, eps=args.eps, preprocessing=format_data)

    expl_path_collector = MdpPathCollector(env, expl_policy, rollout)
    eval_path_collector = MdpPathCollector(env, eval_policy, rollout)
    replay_buffer = anyReplayBuffer(args.replay_buffer_cap, prioritized=args.prioritized_replay)
    optimizer = Adam(qf.parameters(), lr=args.learning_rate)

    loss = []
    avg_r_train = []
    avg_r_test = []

    for i in range(args.n_epoch):
        qf.train(False)
        paths = eval_path_collector.collect_new_paths(args.n_samples//4, args.max_path_len, False)
        avg_r_test.append(np.mean([np.mean(p['rewards']) for p in paths]))

        paths = expl_path_collector.collect_new_paths(args.n_samples, args.max_path_len, False)
        avg_r_train.append(np.mean([np.mean(p['rewards']) for p in paths]))
        replay_buffer.add_paths(paths)

        qf.train(True)    
        for _ in range(args.n_iter):
            batch = replay_buffer.random_batch(args.batch_size)
            _, counts = batch.batch.unique(return_counts=True)
            rewards = batch.r.repeat_interleave(counts)
            terminals = batch.t.repeat_interleave(counts).to(torch.float)
            actions = batch.a

            obs = batch.x
            next_obs = batch.next_s

            out = target_qf(format_input(batch.next_s), batch.edge_index)

            target_q_values = out.max(-1).values
            y_target = rewards + (1. - terminals) * args.gamma * target_q_values
            out = qf(format_input(batch.x), batch.edge_index)

            actions_one_hot = F.one_hot(actions.to(torch.int64), num_classes=len(action))
            y_pred = torch.sum(out * actions_one_hot, dim=-1)
            qf_loss = qf_criterion(y_pred, y_target)

            loss.append(qf_loss.item())

            optimizer.zero_grad()
            qf_loss.backward()
            optimizer.step()

        target_qf.load_state_dict(deepcopy(qf.state_dict()))
        print("iter ", i+1, " -> loss: ", np.mean(loss[-args.n_iter:]),
              ", rewards: (train) ", avg_r_train[-1],
              ", (test) ", avg_r_test[-1])
        
    if args.save:
        torch.save(target_qf.state_dict(), "chkpt/sys_admin/sysadmin_qf_sz%d" % (args.n_nodes))
        
    if args.plot:
        example_policy = doNothingPolicy(action.noop)
        path_collector = MdpPathCollector(env, example_policy, rollout)
        paths = path_collector.collect_new_paths(args.n_samples//4, args.max_path_len, False)
        expected_default = np.mean([np.mean(p['rewards']) for p in paths])

        example_policy = sysRolloutPolicy(env.aspace, 0.)
        path_collector = MdpPathCollector(env, example_policy, rollout)
        paths = path_collector.collect_new_paths(args.n_samples//4, args.max_path_len, False)
        expected_heuristic = np.mean([np.mean(p['rewards']) for p in paths])
        
        n_iter, n_epoch = args.n_iter, args.n_epoch
        plt.plot(np.arange(n_epoch), [expected_default]*(n_epoch), label = "do nothing", color='lightgray')
        plt.plot(np.arange(n_epoch), [expected_heuristic]*(n_epoch), label = "reboot when dead",  color='darkgray')

        plt.plot(np.arange(n_epoch), avg_r_train, label = "avg R (train)")
        plt.plot(np.arange(n_epoch), avg_r_test, label = "avg R (test)")
        plt.legend()
        plt.savefig('figs/sys_admin/training_log_sz%d.png' % (args.n_nodes), dpi=300)

device = 'cpu' #torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    args=parse_args()
    main(args)
    
    