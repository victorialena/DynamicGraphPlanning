import numpy as np

from collections import namedtuple
from collections import deque
from copy import copy, deepcopy

grep = lambda q, x : list(map(q.__getitem__, x))
grepslice = lambda q, x1, x2 : list(itertools.islice(q, x1, x2))
to_batch = lambda q : torch.stack(list(q))
softmax = lambda x : np.exp(x)/sum(np.exp(x))

Sample = namedtuple("Sample", "s a r sp d")

class replayBuffer():
    def __init__(self, max_replay_buffer_size, replace = True, prioritized=False):
        self._max_replay_buffer_size = max_replay_buffer_size        
        self._replace = replace
        self._prioritized = prioritized
        
        self._weights = deque([], max_replay_buffer_size)
        self._samples = deque([], max_replay_buffer_size)            
    
    def add_paths(self, paths):
        for path in paths:
            self.add_path_samples(path)
    
    def add_path_samples(self, path):
        for s, a, r, sp, d, w in zip(path["observations"],
                                     path["actions"],
                                     path["rewards"],
                                     path["next_observations"],
                                     path["terminals"],
                                     path["rewards"]):
            self._samples.appendleft(Sample(s, a, r, sp, d))
            self._weights.appendleft(w)
        
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
        
        return grep(self._samples, indices)
    
    def get_size(self):
        return len(self._weights)
        
    def num_steps_can_sample(self):
        return self._max_replay_buffer_size - self.get_size()

    def get_diagnostics(self):
        return OrderedDict([
            ('size', self.get_size())
        ])
    
    
def sample_episode(env, agent, max_path_length:int=100):
    obs = env.reset()
    
    observations=[]
    actions=[]
    rewards=[]
    next_observations=[]
    terminals=[]
    
    success=False
    makespan=0.
    
    for _ in range(max_path_length):
        a = agent.get_action(obs)
        nextobs, reward, done, info = env.step(a)
        
        observations.append(obs)
        actions.append(deepcopy(a))
        rewards.append(reward)
        next_observations.append(nextobs)
        terminals.append(done)
        
        obs = nextobs
        success = info['success']
        makespan = info['makespan']
        
        if done: 
            break
            
    return dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=terminals,
        success=success,
        makespan=makespan,
    )

