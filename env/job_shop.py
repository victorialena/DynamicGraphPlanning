import dgl
import gym
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

# import torch.nn as nn
# import torch.nn.functional as F
# import scipy.sparse as sp

from copy import copy, deepcopy

# ------------------ Helpers

def get_random_queues(n: int, beta: float = 0.8):
    free = [True]*n
    out = []
    for i in np.random.permutation(n):
        if not free[i]:
            continue
        free[i] = False
        
        while np.random.rand() < beta:
            try:
                j = np.random.choice(np.where(free)[0])
            except:
                return out
            free[j] = False
            out.append([i, j])
            i = j
    return out

def count_q_length(_from, _to, n):
    counts, prev_counts = torch.zeros(n), torch.zeros(n)
    counts[_from] = 1
    while not all(counts == prev_counts):
        prev_counts = deepcopy(counts)
        counts[_from] = counts[_to]+1
    return counts

# ---------------- Specific Helpers

def get_random_intial_state(nw, nj):
    """
    Generate an inital graph / Job shop problem statement, including random dependencies between jobs.
    """
    _from, _to = torch.tensor([[0, 0]]+get_random_queues(nj)).T

    graph_data = {
       ('job', 'precede', 'job'): (_from, _to), # A ---before---> B
       ('job', 'next', 'job'): (torch.tensor([0]), torch.tensor([0])), # jobshop queue
       ('worker', 'processing', 'job'): (torch.tensor([0]), torch.tensor([0])) # nothing is scheduled
    }

    state = dgl.heterograph(graph_data, num_nodes_dict={'worker': nw, 'job': nj})
    # hack: can not init null vector for edges
    state.remove_edges(0, 'processing')
    state.remove_edges(0, 'precede')
    state.remove_edges(0, 'next')

    times = 0.1*torch.randint(1, 10, (nj,1)) # torch.rand(nj,1)
    _from, _to = _from[1:], _to[1:]
    counts = count_q_length(_from, _to, nj).unsqueeze(-1)
    state.nodes['job'].data['hv'] = torch.cat((times, torch.zeros(nj, 1), counts, torch.zeros(nj, 3), times), 1)
    state.nodes['worker'].data['he'] = torch.cat((torch.zeros(nw,2), torch.ones(nw,1)), 1)

    return state

# ---------------- Environment

class jobShopScheduling(gym.Env):
    """
    ### Description
    Learning to schedule a successful sequence of “job” to multiple workers respecting given constraints. 
    
    ### Action Space
    By adding an edge from a worker to an unscheduled job, the job gets queued to that thread.
    The resulting sequence can not be chnaged in hindsight.
    
    ### State Space    
    A disjunctive heterogeneous graph g = (V, C U D). Each node represents a “job” or a “worker”. 
    Edges in C denote succession requirements for jobs, edges in D denotes which jobs were assigned to 
    which worker. 
    
    ### Rewards
    The system recieves a positive unit reward for each executed job. And a penalty per time step.
    
    ### Starting State
    A random set of n jobs, including time requirements and succession constraints, e.g., task i requires 
    completion of task j.
    
    ### Episode Termination
    The episode terminates when all jobs have been scheduled. Then the action space has schunken to size 0.
    The final reward tallies up the remaining rewards to be versed (w/o time discounting).
    
    ### Arguments
    No additional arguments are currently supported.
    
    ### Baselines
    1) C* -> optimal makespan [https://developers.google.com/optimization/scheduling/job_shop]
    2) Heuristics -> Use same as Prof. Park's paper
    3) Other: Simualted Annealing
    
    ### Comments
    https://medium.datadriveninvestor.com/job-shop-scheduling-problem-jssp-an-overview-cd99970a02f8
    https://link.springer.com/article/10.1007/s40092-017-0204-z (flexible open shop problem)
    """

    def __init__(self, njobs: int, nworkers: int):
        self._njobs = njobs
        self._nworkers = nworkers
        self._jfeat = 7
        self._wfeat = 3
        self._dt = 0.1
        self._time_penalty = -0.01
        
        self._state = None
        
    def reward(self, a):
        assert False, "Not implemented. Do not call."
    
    def terminal(self):
        # Terminal state is reached when all the jobs have been scheduled. |A| is zero.
        return all(self._state.nodes['job'].data['hv'][:, 3] == 1)
    
    def worker_features(self):
        return ('n queued', 'expected run time', 'efficiency rate')
    
    def job_features(self):
        return ('time req', 
                'completion%', #1
                'nr of child nodes', #2 
                'status (one hot: scheduled, processing, finished)', #3-4-5
                'remaining time') #6
    
    def valid_action(self, a):
        _, j = a
        return self._state.nodes['job'].data['hv'][j, 3] == 0
    
    def check_job_requirements(self, j):
        # Return True if no incoming edges from preceding job requirements.
        _, dst = self._state.edges(etype='precede')
        return all(dst != j)
    
    def rollout(self, verbose=False):
        # Return number of jobs complete if we just waited until all workers exit (done if gridlock)
        # Does not take into account discount factor!
        state_hv = deepcopy(self._state.nodes['job'].data['hv'])
        state_he = deepcopy(self._state.nodes['worker'].data['he'])
        
        jdone = state_hv[:, 5] == 1
        
        reward = torch.tensor([0.])
        src, dst = deepcopy(self._state.edges(etype='processing'))
        sreq, dreq = deepcopy(self._state.edges(etype='precede'))
        
        makespan = 0.
        
        while True:
            idx = [dst[src==w][0].item() for w in src.unique().tolist()]
            idx = [j for j in idx if all(jdone[sreq[dreq==j]])]
            if len(idx) == 0:
                break # gridlock
            
            # get smallest remaining time for idx. -(.dt)
            j = idx[state_hv[idx, 6].argmin().item()]
            if verbose:
                print("executing job", j, "on worker", src[dst==j].item())
            jdone[j] = True
            dt = state_hv[j, 6].div(self._dt, rounding_mode='trunc')
            makespan += dt
            reward += 1. + dt*self._time_penalty
            state_hv[idx, 6] -= state_hv[j, 6] # mark that job as done
            
            # remove job from queue
            src = src[dst!=j]
            dst = dst[dst!=j]
            
        # clean up graph
        idx = np.where(jdone)[0]
        
        src, dst, cnts = self._state.edges('all', etype='processing')
        jidx = [(j in idx) for j in dst]
        self._state.remove_edges(cnts[jidx].tolist(), 'processing')                
        
        for etype in ['next', 'precede']:
            src, dst, cnts = self._state.edges('all', etype=etype)
            jidx = [(j in idx) for j in src]
            self._state.remove_edges(cnts[jidx].tolist(), etype)
            
        self._state.nodes['job'].data['hv'][idx, 2] = 1.
        self._state.nodes['job'].data['hv'][idx, 4] = 0.
        self._state.nodes['job'].data['hv'][idx, 5] = 1. # mark terminal
        self._state.nodes['job'].data['hv'][idx, 6] = 0. # set remaining time to 0
        
        src, _ = self._state.edges(etype='processing')
        if len(src):
            w, cnts = src.unique(return_counts=True)
            self._state.nodes['worker'].data['he'][w, 0] = cnts.float()

        return reward.item(), all(jdone), makespan
    
    def get_node_status(self, label, by_index=True):
        mask = self._state.nodes['job'].data['hv'][:, label] == 1
        if by_index:
            return np.where(mask)[0].tolist()
        return mask
        
    def get_scheduled(self, by_index=True):
        return self.get_node_status(3, by_index)
    
    def get_unscheduled(self, by_index=True):
        mask = ~self.get_node_status(3, False)
        if by_index:
            return np.where(mask)[0].tolist()
        return mask
    
    def get_processing(self, by_index=True):
        return self.get_node_status(4, by_index)
    
    def get_terminated(self, by_index=True):
        return self.get_node_status(5, by_index)
    
    def is_gridlocked(self):
        if len(self.get_processing()):
            return False

        src, dst = self._state.edges(etype='processing')
        if len(src.unique()) != self._nworkers:
            return False

        _, req = self._state.edges(etype='precede')
        newidx = [dst[src==w][0].item() for w in src.unique().tolist()]
        newidx = [j for j in newidx if j not in req]
        return len(newidx)>0
    
    def step(self, a):
        assert self.valid_action(a), "Invalid action taken: (w:%d, j:%d)" % a
        
        src, dst, cnts = self._state.edges('all', etype='processing')
        
        """ 
        1) Schedule job j for worker w: 
            a) Find last job scheduled for worker w, add edge from end of queue to new job j. 
            b) Add edge from w to j. 
            c) Update worker info (queue length, run time estimate).
            d) Mark job as scheduled.
        """
        w, j = a
        if w in src:
            _i = dst[src==w][-1].item() # add to end of q -- last edge added
            self._state.add_edge(_i, j, etype='next')        
        self._state.add_edge(w, j, etype='processing')
        
        state_hv = deepcopy(self._state.nodes['job'].data['hv'])
        state_he = deepcopy(self._state.nodes['worker'].data['he'])
        
        state_he[w, 0] += 1. # add job to work queue length
        state_he[w, 1] += state_hv[j, 0] # update worker' run time estimate
        state_hv[j, 3] = 1. # mark as scheduled
                
        """ 2) Assure the first job in queue is being processed at this time step. """
        _, req = self._state.edges(etype='precede')
        src, dst, cnts = self._state.edges('all', etype='processing') # call again to update processing
        newidx = [dst[src==w][0].item() for w in src.unique().tolist()]
        newidx = [j for j in newidx if j not in req]
        state_hv[newidx, 4] = 1 # set to processing (but completion % remain 0)
        
        # write info incase of early exit
        self._state.nodes['job'].data['hv'] = state_hv
        self._state.nodes['worker'].data['he'] = state_he

        """ 
        3) Update feature vectors:
            a) Progress time for node features: remaining time, completion % for jobs and workers
            b) Update info around terminal jobs, and remove processing edge if job has terminated.
            c) Remove next and precede edges for terminated jobs. 
        """
        # a
        processing_mask = state_hv[:, 4] == 1        
        if processing_mask.sum() == 0:
            return deepcopy(self._state), self._time_penalty, self.terminal(), {'success':False, 'makespan':0.}
        
        state_hv[processing_mask, 6] = torch.maximum(state_hv[processing_mask, 6]-self._dt,
                                                     torch.zeros(processing_mask.sum())).round(decimals=2) # update remaining time
        state_hv[processing_mask, 1] = torch.clamp(1-torch.div(state_hv[processing_mask, 6],
                                                               state_hv[processing_mask, 0]), 
                                                   min=0, max=1) # update completion %
        
        state_he[:, 1] = torch.maximum(state_he[:, 1]-self._dt, torch.zeros(self._nworkers)) # update remaining time
        
        # b
        state_hv[processing_mask, 5] = (state_hv[processing_mask, 1] == 1).float() # mark terminal
        state_hv[processing_mask, 4] = 1-state_hv[processing_mask, 5] # if terminal, job no longer processing        
        idx = torch.where(processing_mask)[0][torch.where(state_hv[processing_mask, 5])[0]].tolist() # job ids just terminated
        if len(idx):
            widx = [(j in idx) for j in dst]
            state_he[src[widx], 0] -= 1 # remove job from job count
            self._state.remove_edges(cnts[widx].tolist(), 'processing') # delete those edges?
        
            # c
            src, dst, cnts = self._state.edges('all', etype='next')
            ptridx = torch.tensor([cnts[src == j].item() for j in idx if j in src]) # this works because it is a queue: unique next node
            if len(ptridx):
                self._state.remove_edges(ptridx.tolist(), 'next')

            src, dst, cnts = self._state.edges('all', etype='precede')
            jidx = [(j in idx) for j in src]
            if len(jidx):
                self._state.remove_edges(cnts[jidx].tolist(), 'precede') # delete those edges?
                
        """ 5) Update feature vectors. """
        self._state.nodes['job'].data['hv'] = state_hv
        self._state.nodes['worker'].data['he'] = state_he
                
        """ 6) Compute reward and terminal state. """
        done = self.terminal()
        success = False
        makespan = 0.
        n_terminal = len(idx)
        reward = self._time_penalty + n_terminal
        
        if done:
            r, success, makespan = self.rollout()
            makespan += (self._njobs-1) #*self._dt
            reward += r
        
        return deepcopy(self._state), deepcopy(reward), deepcopy(done), {'success': success, 'makespan': makespan}

    def reset(self, seed: int = None, topology: str = 'random'):
        if not seed == None:
            super().reset(seed=seed)
                
        self._state = get_random_intial_state(self._nworkers, self._njobs)
        return deepcopy(self._state)
    
    def dump_state_info(self):
        print('scheduled:', self.get_scheduled())
        print('processing:', self.get_processing())
        print('terminated:', self.get_terminated())
        print('job data:')
        print(self._state.nodes['job'].data['hv'])
        print('worker data:')
        print(self._state.nodes['worker'].data['he'])

    def render(self):
        """
        import networkx as nx
        import matplotlib.pyplot as plt
        """
        self.dump_state_info()
        
        G = dgl.to_homogeneous(self._state).to_networkx(edge_attrs=['_TYPE'])
        
        node_color = ['red']*self._njobs+['orange']*self._nworkers
        edge_color = ['red' if e[-1].item() % 2 else 'orange' for e in G.edges(data='_TYPE')]
        nx.draw(G, node_color=node_color, edge_color=edge_color, with_labels=True)
        
    def seed(self, n: int):
        super().reset(seed=seed)