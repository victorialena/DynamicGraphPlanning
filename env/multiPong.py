import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np
import torch
import supersuit

from gym.spaces import MultiDiscrete
from numpy.random import rand
from pettingzoo.atari import warlords_v3
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

import pdb

# ------------- Helpers

def crop_observation(obs, _min=14, _max=183):
    return obs[_min:_max]

def feature_mask(obs):
    masked_obs = obs.copy()
    masked_obs[:42, :48] = 0
    masked_obs[:42, -48:] = 0
    masked_obs[-42:, :48] = 0
    masked_obs[-42:, -48:] = 0
    return masked_obs

def get_agent_pos(obs, summed_cvalues):
    """ Finds agent positions, call on cropped observation. """
    masked_obs = feature_mask(obs).sum(-1)
    
#     assert all([(masked_obs==c).any() for c in summed_cvalues]), "Did not find agents"
#     if not all([(masked_obs==c).any() for c in summed_cvalues]):
#         pdb.set_trace()
    
    return np.stack([np.stack(np.where(masked_obs == c)).mean(1) for c in summed_cvalues])

def get_all_colors(masked_obs):
    idx = np.stack(np.nonzero(masked_obs.sum(-1)))
    return np.unique(np.stack([masked_obs[i, j] for (i, j) in idx.T]), axis=0, return_counts=True)

def brick_filter(obs, summed_cvalues):
    _col, counts = get_all_colors(obs)
    brick_colors = _col[[(v not in summed_cvalues) and (c<10000) for (v, c) in zip(_col.sum(-1), counts)]]
    brick_filter = np.full(obs.shape[:2], False)
    for v in brick_colors.sum(-1):
        brick_filter |= (obs.sum(-1) == v)
    return brick_filter

def get_smallest_island(grid, default):
    """ 
    Returns the upper left corner of the ball, or default value is none was
    found (i.e. currenly in collision).
    """
    def dfs(i,j,size):
        # Stop searching when index out of range or cell equals 0
        if i < 0 or j<0 or i >= len(grid) or j >= len(grid[0])or grid[i][j] != 1:
            return size
        # Change cell to 0 so we don't visit again
        grid[i][j] = 0
        # Increment size count
        size += 1
        size = dfs(i,j+1,size)
        size = dfs(i,j-1,size)
        size = dfs(i+1,j,size)
        size = dfs(i-1,j,size)
        # Return the incremented size count
        return size

    # Find the maximum size count
    max_size = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                sz = dfs(i,j,0)
                if sz < 20: # a single brick is > 50 (meausred 8x8 for the "easy" brick)
                    return np.array([i,j])
    return default

def rgb2gray(rgb):
    return np.dot(rgb, [0.2989, 0.5870, 0.1140])

def downsample(x):
    """ Requires oriented feature as input x. """
    x[24:,:28] = 0 # filter non-brick 
    m, n, _ = x.shape 
    return np.stack([[x[i*8:(i+1)*8,j*8:(j+1)*8].mean((0,1)) for j in range(n//8)]for i in range(m//8)])

def get_features(obs, scale=1):
    """ Return feature space for all 4 agents. Rotate so all feature vectors are symmetric. """
    UL = downsample(np.flip(obs[2:42, :48], axis=0))
    UR = downsample(np.flip(np.flip(obs[2:42, -48:], axis=0), axis=1))
    LL = downsample(obs[-41:-1, :48])
    LR = downsample(np.flip(obs[-41:-1, -48:], axis=1))
    return np.stack([rgb2gray(UL).flatten(), rgb2gray(UR).flatten(), rgb2gray(LL).flatten(), rgb2gray(LR).flatten()])/scale

def get_corners():
    return np.array([[42, 48], [42, 112], [128, 48], [128, 112]])

def fully_connect_graph(n_nodes):
    """ Connect the graph s.t. all drones and goals are interconnected. """    
    idx = torch.combinations(torch.arange(n_nodes), r=2)
    return to_undirected(idx.t(), num_nodes=n_nodes)

def normalize(a, b):
    return (a-b)/b

# ----------------- ENV

class multiPong():
    """
    ### Description
    
    ### Action Space
    
    ### State Space
    
    ### Rewards
    
    ### Starting State
    
    ### Episode Termination
    
    ### Arguments
    No additional arguments are currently supported.
    """
    
    def __init__(self, args=None):
        self.env = warlords_v3.env(obs_type='rgb_image', full_action_space=False, max_cycles=100000, auto_rom_install_path=None)

        self.g = fully_connect_graph(4)
        self.n_warmup = 110
        self.translation = np.array([84.5, 80. ])
        
        # self.action_space = MultiDiscrete([6]*4)
        self.action_conv = np.array([2*(i//2)+((i+1)%2) for i in range(6)]*2+[i for i in range(6)]*2).reshape(4, -1)
        self.prev_b = None
        self.b_pos = None
        self.a_pos = None
        self.agents = ['first_0', 'second_0', 'third_0', 'fourth_0']

        self.env.reset()
        observation, _, _, _ = self.env.last()
        non_zero_rows = np.nonzero(observation.sum(-1).sum(1))[0]
        non_zero_col_1 = np.nonzero(observation[non_zero_rows[0]].sum(-1))[0]
        non_zero_col_2 = np.nonzero(observation[non_zero_rows[-1]].sum(-1))[0]

        colors = {abs(j)+2*abs(i):observation[non_zero_rows[i], non_zero_col_1[j]] for i in [0, -1] for j in [0, -1]}
        assert len(np.unique([sum(i) for i in colors.values()])) == 4, "Summed colo chanels is confounding agents."
        self.summed_cvalues = [sum(v) for v in colors.values()]
        
    def reset(self, seed=None):
        self.env.reset(seed)
        self.warmup()
        
        return self.get_s()
        
    def warmup(self):
        n = np.random.randint(self.n_warmup, 500)
        for i in range(n):
            if self.isterminal():
                return False
            for agent in self.agents:
                self.env.step(self.env.action_space(agent).sample())
            if i == n-2:
                self.prev_b = self.translation
                self.prev_b = self.find_ball()
        
        a_pos = self.find_agents()    
        self.pos_a = normalize(a_pos, self.translation)
        return True
    
    def find_ball(self):
        assert self.prev_b is not None, "Not initialized, please call reset(seed: opt)"
        
        observation, _, _, _ = self.env.last()
        obs = crop_observation(observation)

        grid = brick_filter(obs, self.summed_cvalues)
        b_pos = get_smallest_island(grid, self.prev_b)
        return b_pos
    
    def find_agents(self):
        observation, _, _, _ = self.env.last()
        obs = crop_observation(observation)
        return get_agent_pos(obs, self.summed_cvalues)
                
    def get_s(self):
        
        b_pos = self.find_ball()
        self.b_pos = b_pos.copy()
        a_pos = self.find_agents()

        s = torch.tensor(np.hstack([normalize(a_pos, self.translation), 
                                    normalize(get_corners(),self.translation), 
                                    # feat,
                                    np.tile(normalize(b_pos,self.translation), (4,1)),
                                    np.tile(normalize(self.prev_b,self.translation), (4,1))]), dtype=torch.float)        
        return s
    
    def step(self, a):
        self.prev_b = self.b_pos.copy()
        for i, agent in enumerate(self.agents):
            if self.env.dones[agent]:
                continue
            self.env.step(self.action_conv[i, a[i].item()])
        
        s = self.get_s()
        dead_agents = list(self.env.dones.values())
        if any(dead_agents):
            s[dead_agents, :2] = self.pos_a[dead_agents]
            
        self.pos_a = s[:, :2]
        r = self.reward(s, a)
        return s, r, self.isterminal(), {}
            
    def isterminal(self):
        return any(self.env.dones.values())
    
    # new - "ball in your court"
    def reward(self, s, a):
        b_pos = s[0, 4:6]
        r = torch.stack([(b_pos < s[0, 2:4]).all(),
                         b_pos[0] < s[1, 2] and b_pos[1] > s[1, 3],
                         b_pos[0] > s[2, 2] and b_pos[1] < s[2, 3],
                         (b_pos > s[3, 2:4]).all()]).to(torch.float)
        return -r - 0.1*(a>1)
        
    def render(self):
        observation, _, _, _ = self.env.last()
        plt.imshow(observation)
        