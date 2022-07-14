import sys
sys.path.append('/home/victorialena/rlkit')

import pdb

import dgl
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import copy, deepcopy
from torch.optim import Adam

from env.job_shop import jobShopScheduling
from actor.job_shop import *
from utils.job_shop import * # custom replay buffer and episode sampler
from utils.jsp_makespan import *

from path_collector import MdpPathCollector

class hgnn(nn.Module):
    def __init__(self, embedding_dim=16, k=2):
        super().__init__()
        
        self.embedding = dglnn.HeteroLinear({'job': 7, 'worker':3}, embedding_dim)
        self.conv = dglnn.HeteroGraphConv({
            'precede' : dglnn.GraphConv(embedding_dim, embedding_dim),
            'next' : dglnn.GraphConv(embedding_dim, embedding_dim),
            'processing' : dglnn.SAGEConv((embedding_dim, embedding_dim), embedding_dim, 'mean')},
            aggregate='sum')
               
        self.pred = dotProductPredictor()
        self.num_loops = k
        
    def forward(self, g):
        h0 = {**g.ndata['hv'], **g.ndata['he']}
        hv = self.embedding(h0)
        hw = hv['worker']
        for _ in range(self.num_loops):
            hv = {'job': self.conv(g, hv)['job'],
                  'worker': hw}
            
        rg = construct_readout_graph(g, ('worker', 'processing', 'job'))
        return self.pred(rg, hv['job'], hw, ('worker', 'processing', 'job'))    


# #### Helpers

scientific_notation =  lambda x:"{:.2e}".format(x)

def get_scores(g, scores):
    n = scores.shape[0]
    idx = (g.ndata['hv']['job'][:, 3] == 0).view(n, -1)
    
    values, workers = scores.max(-1, keepdims=False)
    return torch.stack([values[i][idx[i]].max() if sum(idx[i]).item()>0 else torch.tensor(0.) for i in range(n)])

def mean_reward(paths):
    return torch.tensor([p['rewards'] for p in paths]).sum(1).mean().item()

def mean_makespan(paths):
    "Returns the average makespan successful paths from given list. Returns *nan* if no path was successful."
    return torch.tensor([p['makespan'] for p in paths if p['success']]).mean().item()

def relative_makespan_error(paths):
    """ From initial conditions of each path, evaluate optimal makespan for each path and compare against
    sampled trajectory. """
    err = []
    for p in paths:
        if not p['success']:
            continue
        jdata = g2jobdata(p['observations'][0], p['actions'])
        makespan, status = get_makespan(jdata)
        if makespan > -1: # feasible
            relative_error = p['makespan']/makespan - 1
            err.append(relative_error)
        else: 
            print("This should not be possible, check for bugs!!")
            
    return torch.tensor(err).mean().item()


# #### Q Learning

seed = 42 #42 #0
torch.manual_seed(seed)
np.random.seed(seed)

njobs, nworkers = 30, 15 #25, 10 #50, 20 
env = jobShopScheduling(njobs, nworkers)
g0 = env.reset()

load_from = "jobshop_qf_j25-w15_4x4_multilayer-from_script"

layersz = 16 #32

qf = hgnn(layersz)
expl_policy = epsilonGreedyPolicy(qf, .1)

target_qf = hgnn(layersz)
eval_policy = epsilonGreedyPolicy(target_qf, 0.)

if load_from:
    qf.load_state_dict(torch.load("chkpt/job_shop/"+load_from))
    target_qf.load_state_dict(torch.load("chkpt/job_shop/"+load_from))

expl_path_collector = MdpPathCollector(env, expl_policy, rollout_fn=sample_episode, parallelize=False)
eval_path_collector = MdpPathCollector(env, eval_policy, rollout_fn=sample_episode, parallelize=False)

replay_buffer_cap = 5000
replay_buffer = replayBuffer(replay_buffer_cap, prioritized=True)

learning_rate = 8e-5 #1e-4

optimizer = Adam(qf.parameters(), lr=learning_rate, weight_decay=0.01)
qf_criterion = nn.MSELoss()

max_len = njobs+1
n_samples = 128 #64 
n_epoch = 300 #400
n_iter = 64
batch_size = 32
gamma = 1.0

loss = []
avg_r_train = []
avg_r_eval = []
success_rates = []
relative_errors = []

for i in range(n_epoch):
    qf.train(False)
    paths = expl_path_collector.collect_new_paths(n_samples, max_len, False)
    train_r = mean_reward(paths)
    avg_r_train.append(train_r)
    replay_buffer.add_paths(paths)
    
    paths = eval_path_collector.collect_new_paths(n_samples//4, max_len, False)
    eval_r = mean_reward(paths)
    avg_r_eval.append(eval_r)
    
    success_rate = np.mean([p['success'] for p in paths])
    success_rates.append(success_rate)
    
    avg_makespan = mean_makespan(paths)
    relative_err = relative_makespan_error(paths)
    relative_errors.append(relative_err)

    qf.train(True)
#     if i==200:
#         expl_policy.eps = 0.05
        
    for _ in range(n_iter):
        batch = replay_buffer.random_batch(batch_size)

        rewards = torch.tensor([b.r for b in batch])
        terminals = torch.tensor([b.d for b in batch]).float()
        actions = torch.tensor([b.a for b in batch])
        
        states = batch_graphs([b.s for b in batch])
        next_s = batch_graphs([b.sp for b in batch])        

        out = target_qf(next_s) # shape = (|G|, |J|, |W|)
        target_q_values = get_scores(next_s, out)
        y_target = rewards + (1. - terminals) * gamma * target_q_values 
        
        out = qf(states)
        y_pred = out[torch.arange(batch_size), actions.T[1], actions.T[0]]
        qf_loss = qf_criterion(y_pred, y_target).to(torch.float)

        loss.append(qf_loss.item())

        optimizer.zero_grad()
        qf_loss.backward()
        optimizer.step()

    target_qf.load_state_dict(deepcopy(qf.state_dict()))
    err = 3
    print("Epoch", i+1, #"| lr:", scientific_notation(optimizer.param_groups[0]["lr"]) ,
          " -> Loss:", round(np.mean(loss[-n_iter:]), err),
          "| Rewards: (train)", round(train_r, err), "(test)", round(eval_r, err),
          "| Success rate:", round(success_rate, err), 
          "| Makespan:", round(avg_makespan, err), 
          "| Rel. error:", round(relative_err, err), )


target_qf.eval()
torch.save(target_qf.state_dict(), "chkpt/job_shop/jobshop_qf_j%d-w%d_4x4_multilayer-from_script" % (njobs, nworkers))


# #### Plot

import matplotlib.pyplot as plt

losses = [np.mean(loss[i*n_iter:(i+1)*n_iter]) for i in range(n_epoch)]

x = np.arange(n_epoch)

plt.figure(figsize=(15, 10))

plt.subplot(221)
plt.plot(x, losses)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.subplot(222)
plt.plot(x, avg_r_train, label="test")
plt.plot(x, avg_r_eval, 'r--', label="eval")
plt.legend()
plt.ylabel('Train/Test Rewards [path, avg]')
plt.xlabel('Epoch')
plt.subplot(223)
plt.plot(x, success_rates)
plt.ylabel('Success Rate')
plt.xlabel('Epoch')
plt.subplot(224)
plt.plot(x, [0.0]*len(x), 'lightgray', linestyle='--')
plt.plot(x, relative_errors)
plt.ylabel('Relative Error')
plt.xlabel('Epoch')
plt.suptitle('Training Performance Summary', y=.95)
plt.savefig('figs/job_shop/j%d-w%d_4x4_multilayer-from_script' % (njobs, nworkers), dpi=300)

