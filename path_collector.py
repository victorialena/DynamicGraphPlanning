from collections import deque, OrderedDict
from functools import partial

import numpy as np

from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.samplers.data_collector.base import PathCollector
# from rollout_functions import *
import multiprocessing as mp
from multiprocessing.dummy import Pool

import pdb

num_cores = 12 #mp.cpu_count()
max_parallel = 20

"""
Notes:
1) Ray paralellisation only works for non-geometric networks. 'gnn.Sequential' can not get pickle-Serialization.
2) Ray and mp.Pool only work on CPU
3) mp.Pool can handle up to 25 trajectories per batch.
4) mp.Pool doesn't like lambda functions (in env.drone_delivery def)
"""

class MdpPathCollector(PathCollector):
    def __init__(self, env, policy, rollout_fn, max_num_epoch_paths_saved:int=None, parallelize:bool=False):
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        
        self._rollout_fn = rollout_fn
        self._multithreading = parallelize

    def collect_new_paths(self, n_paths, max_path_length, discard_incomplete_paths=False, flatten=False):
        paths = []

        for _ in range(n_paths):
            path = self._rollout_fn(self._env, self._policy, max_path_length=max_path_length)
            # if flatten: paths.extend(path)
            paths.append(path)
        
        self._epoch_paths.extend(paths)
        return paths
    
    def collect_nsteps(self, n_steps, max_path_length, discard_incomplete_paths=False, flatten=False):
        paths = []
        count = 0
        
        while count < n_steps:
            path = self._rollout_fn(self._env, self._policy, max_path_length=max_path_length)
            count += len(path['terminals'])
            # if flatten: paths.extend(path)
            paths.append(path)
        
        self._epoch_paths.extend(paths)
        return paths
    
    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', sum(path_lens)),
            ('number of epoch paths', len(self._epoch_paths)),
        ])
        return stats
