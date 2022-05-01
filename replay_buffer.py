import numpy as np
import pdb
import torch

from collections import deque, OrderedDict

from torch import is_tensor, Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler, BatchSampler

from scipy import signal
from warnings import warn

grep = lambda q, x : list(map(q.__getitem__, x))
grepslice = lambda q, x1, x2 : list(itertools.islice(q, x1, x2))
to_batch = lambda q : torch.stack(list(q))
softmax = lambda x : np.exp(x)/sum(np.exp(x))

def discount_cumsum(x, discount):
    assert len(x.shape) == 1
    n = len(x)
    a = torch.arange(0, n)
    A = (discount**torch.stack([a.roll(i) for i in range(n)])).triu()
    return (x.repeat(n,1)*A).sum(-1)
    
def discount_cumsum_n(X, discount):
    pdb.set_trace()
    return torch.vstack([discount_cumsum(x, discount) for x in X.T]).T

def statistics_scalar(x, with_min_and_max=False):
    """ Get mean/std and optional min/max of scalar x  """
    if len(x.shape) == 1:
        x = x.unsqueeze(-1)
    if with_min_and_max:        
        return x.mean(0), x.std(0), x.max(0), x.min(0)
    return x.mean(0), x.std(0)

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, max_replay_buffer_size, gamma=0.99, lam=0.95):
        self.obs_buf = deque([], max_replay_buffer_size)  
        self.act_buf = deque([], max_replay_buffer_size)  
        self.adv_buf = deque([], max_replay_buffer_size)  
        self.rew_buf = deque([], max_replay_buffer_size)  
        self.ret_buf = deque([], max_replay_buffer_size)  
        self.val_buf = deque([], max_replay_buffer_size)  
        self.logp_buf = deque([], max_replay_buffer_size)        
        
        self.gamma, self.lam = gamma, lam
        self.path_start_idx, self.max_replay_buffer_size = np.array([],dtype=int), max_replay_buffer_size

    def add_sample(self, obs, act, rew, val, logp):
        if self.get_size() == self.max_replay_buffer_size:
            self.path_start_idx -= 1
            
        self.obs_buf.appendleft(obs)
        self.act_buf.appendleft(act)
        self.rew_buf.appendleft(rew)
        self.val_buf.appendleft(val)
        self.logp_buf.appendleft(logp)

    def finish_path(self, last_val=0):
        """ last_val should be 0 if the trajectory ended because the agent reached a terminal state, 
        and otherwise should be V(s_T), the value function estimated for the last state. """
        
        if torch.is_tensor(last_val):
            last_val = last_val.item()
#             last_val = torch.ones(self.rew_buf[-1].shape)

        path_slice = torch.arange(self.path_start_idx[-1], self.get_size())
#         pdb.set_trace()
        rews = torch.tensor(grep(self.rew_buf, path_slice)+[last_val])
        vals = torch.tensor(grep(self.val_buf, path_slice)+[last_val])
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf.extendleft(discount_cumsum(deltas, self.gamma * self.lam))
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf.extendleft(discount_cumsum(rews, self.gamma)[:-1])
        
        # clean out old trajectories
        self.path_start_idx = self.path_start_idx[self.path_start_idx>=0]
        assert self.is_valid(), "Invalid buffer"
         
    def start_path(self):
        self.path_start_idx = np.append(self.path_start_idx, self.get_size())

    def get(self):
        # the next two lines implement the advantage normalization trick
        adv_buf = to_batch(self.adv_buf)
        adv_mean, adv_std = statistics_scalar(adv_buf)
        adv_buf = (adv_buf - adv_mean) / adv_std
        
        assert self.is_valid(), "Invalid buffer"
        
        data = dict(obs=Batch.from_data_list(self.obs_buf), act=to_batch(self.act_buf), ret=to_batch(self.ret_buf),
                    adv=adv_buf, logp=to_batch(self.logp_buf))
        return data #{k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}
    
    def get_size(self):
        return len(self.rew_buf)
    
    def number_of_paths(self):
        return len(self.path_start_idx)
    
    def is_valid(self):
        return len(self.act_buf) == len(self.adv_buf)
        

class anyReplayBuffer():

    def __init__(self, max_replay_buffer_size, replace = True, prioritized=False):
        self._max_replay_buffer_size = max_replay_buffer_size        
        self._replace = replace
        self._prioritized = prioritized
        
        self._weights = deque([], max_replay_buffer_size)
        self._observations = deque([], max_replay_buffer_size)            

    def add_sample(self, observation, action, reward, next_observation, terminal, env_info=None, **kwargs):
        data = Data(x=observation.x,
                    edge_index=observation.edge_index,
                    a=action,
                    r=reward,
                    next_s=next_observation.x,
                    t=terminal)
        self._observations.appendleft(data)
        self._weights.appendleft(reward.sum().item() if is_tensor(reward) else reward)
    
    def add_paths(self, paths):
        for path in paths:
            self.add_path(path)
    
    def add_path(self, path):
        for obs, action, reward, next_obs, terminal in zip(path["observations"],
                                                           path["actions"],
                                                           path["rewards"],
                                                           path["next_observations"],
                                                           path["terminals"],     ):
            self.add_sample(observation=obs,
                            action=action,
                            reward=reward,
                            next_observation=next_obs,
                            terminal=terminal)
        self.terminate_episode()
        
    def terminate_episode(self):
        pass

    def random_batch(self, batch_size):
        prio = softmax(self._weights) if self._prioritized else None
        indices = np.random.choice(self.get_size(), 
                                   size=batch_size, p=prio, 
                                   replace=self._replace or self._size < batch_size)
        if not self._replace and self._size < batch_size:
            warn('Replace was set to false, but is temporarily set to true \
            because batch size is larger than current size of replay.')
        
        batch = grep(self._observations, indices)
        
        return Batch.from_data_list(batch)
    
    def get_size(self):
        return len(self._weights)
        
    def rebuild_env_info_dict(self, idx):
        return self.batch_env_info_dict(idx)

    def batch_env_info_dict(self, indices):
        return {}

    def num_steps_can_sample(self):
        return self.get_size()

    def get_diagnostics(self):
        return OrderedDict([
            ('size', self.get_size())
        ])