import numpy as np
import pdb

from collections import deque, OrderedDict

from torch import is_tensor, Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler, BatchSampler

from warnings import warn

grep = lambda q, x : list(map(q.__getitem__, x))
softmax = lambda x : np.exp(x)/sum(np.exp(x))

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
#                     node_type=observation.node_type,
#                     edge_type=observation.edge_type,
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