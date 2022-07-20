import abc
import numpy as np

from collections import OrderedDict, deque
from numbers import Number
# from functools import partial

def get_average_returns(paths):
    returns = [sum(path["rewards"]) for path in paths]
    return np.mean(returns)


def create_stats_ordered_dict(
        name,
        data,
        stat_prefix=None,
        always_show_all_stats=True,
        exclude_max_min=False,
):
    if stat_prefix is not None:
        name = "{}{}".format(stat_prefix, name)
    if isinstance(data, Number):
        return OrderedDict({name: data})

    if len(data) == 0:
        return OrderedDict()

    if isinstance(data, tuple):
        ordered_dict = OrderedDict()
        for number, d in enumerate(data):
            sub_dict = create_stats_ordered_dict(
                "{0}_{1}".format(name, number),
                d,
            )
            ordered_dict.update(sub_dict)
        return ordered_dict

    if isinstance(data, list):
        try:
            iter(data[0])
        except TypeError:
            pass
        else:
            data = np.concatenate(data)

    if (isinstance(data, np.ndarray) and data.size == 1
            and not always_show_all_stats):
        return OrderedDict({name: float(data)})

    stats = OrderedDict([
        (name + ' Mean', np.mean(data)),
        (name + ' Std', np.std(data)),
    ])
    if not exclude_max_min:
        stats[name + ' Max'] = np.max(data)
        stats[name + ' Min'] = np.min(data)
    return stats

class DataCollector(object, metaclass=abc.ABCMeta):
    def end_epoch(self, epoch):
        pass

    def get_diagnostics(self):
        return {}

    def get_snapshot(self):
        return {}

    @abc.abstractmethod
    def get_epoch_paths(self):
        pass

class PathCollector(DataCollector, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        pass
    
#--------------------------- Class def
    
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
