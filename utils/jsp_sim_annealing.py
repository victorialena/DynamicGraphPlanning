# import collections
import numpy as np
import pdb
import torch
import dgl

from copy import copy, deepcopy
from env.job_shop import get_random_queues, count_q_length, get_random_intial_state
from utils.job_shop import softmax

def assign_random_initial_state(state):
    """
    Assign jobs to machine randomly, this will be the starting point for simulated annealing.
    Edits input state in place.
    """
    # 1) assign jobs randomly to machine
    assignment = torch.randint(low=0, high=state.num_nodes('worker'), size=(state.num_nodes('job'), ))
    state.add_edges(assignment, state.nodes('job'), etype='processing')
    
    # 2) assign sequence within each worker
    for i in state.nodes('worker').tolist():
        if sum(assignment==i) < 2:
            continue
        out = state.nodes('job')[assignment==i][torch.randperm(sum(assignment==i))]
        state.add_edges(out[:-1], out[1:], etype='next')
        
def scoring_fn(state, dt=0.1, time_penalty=-0.01, verbose=False):
    """
    [same as env.rollout()] -> TODO: Move to utils file
    Return number of jobs complete if we just waited until all workers exit (done if gridlock)
    Does not take into account discount factor!
    """
    state_hv = deepcopy(state.nodes['job'].data['hv'])
    state_he = deepcopy(state.nodes['worker'].data['he'])

    jdone = state_hv[:, 5] == 1

    reward = torch.tensor([0.])
    src, dst = deepcopy(state.edges(etype='processing'))
    sreq, dreq = deepcopy(state.edges(etype='precede'))

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
        delta_t = state_hv[j, 6].div(dt, rounding_mode='trunc')
        makespan += delta_t.item()
        reward += 1. + delta_t*time_penalty
        state_hv[idx, 6] -= state_hv[j, 6] # mark that job as done

        # remove job from queue
        src = src[dst!=j]
        dst = dst[dst!=j]
    
    """
    if makespan == np.inf:
        print("makespan is inf")
        pdb.set_trace() 
    """
    assert makespan != np.inf, "Makespan cannnot be inf"
    
    return reward.item() # reward.item(), all(jdone), makespan

def get_neighbors(state):
    """
    This function  implements option 2, for efficiency reasons.
    Options for `get_neighbors()`:
    - evaluating every neighboring action, i.e., move job i to position j at worker k (|A| = w*n^2) 
    - eval every speudo neigboring state, i.e., move job i to end of queue for worker j (|A| = (n-1)*w)
    """
    out = []
    for j in state.nodes('job').tolist():
        prev_w = state.edges(etype='processing')[0][j].item()
        neigh = deepcopy(state)
        for w in state.nodes('worker').tolist():
            if w == prev_w:
                continue
            
            # assign job to new worker
            neigh.edges(etype='processing')[0][j] = w 
            
            # redo 'next' edges for old queue
            next_edges_from, next_edges_to = neigh.edges(etype='next')
            assert sum(next_edges_from==j)<=1 and sum(next_edges_to==j)<=1, "It's a queue, there can only be one dependency!"

            from_i = torch.nonzero(next_edges_from==j).item() if any(next_edges_from==j) else -1
            to_i = torch.nonzero(next_edges_to==j).item() if any(next_edges_to==j) else -1

            neigh.remove_edges([i for i in [from_i, to_i] if i >= 0], etype='next')
            if to_i >= 0 and from_i >= 0:
                neigh.add_edge(next_edges_from[to_i], next_edges_to[from_i], etype='next')

            # add j to new queue
            other_jobs = state.edges(etype='processing')[1][state.edges(etype='processing')[0] == w]
            if len(other_jobs) > 0:
                mask = [i not in next_edges_from for i in other_jobs]
                assert sum(mask) == 1, "That's a bug!"
                end_of_q = other_jobs[[i not in next_edges_from for i in other_jobs]].item()
                neigh.add_edge(end_of_q, j, etype='next')
            out.append(neigh)
    return out

def get_annealing_schedule(p, n):
    "Returns a function which decrements t every n iterations by p%."
    return lambda t, i: t*((1-p)**(i//n))

def simulated_annealing(init_s, objective, n_iterations, temp, annealing_schedule=None):
    """
    This function implements simulated annealing given a job shop problem def *init_s*. The
    algorithm moves to the best performing neighbor with increasing certainty. In the randomized
    outlier case, a neighbor is picked proportionally to its fitness. The temperature is on a 
    user-specified annealing schedule or the metropolis schedule can be used as well. 
    """
    nj = init_s.num_nodes('job')
    # generate and evaluate an initial point
    curr_s = deepcopy(init_s)
    assign_random_initial_state(curr_s)
    curr_eval = objective(curr_s)
    
    # current working solution
    best_s, best_eval = curr_s, curr_eval
    
    # run the algorithm
    for i in range(n_iterations):
        if best_eval > nj*0.9:
            return best_s, best_eval, i
            
        # states.append(curr_s) # store path
        # take a step and evaluate candidate point
        N = get_neighbors(curr_s)
        scores = [objective(gi) for gi in N]
        
        """
        # use metropolis schedule
        diff = max(scores) - curr_eval
        t = temp / float(i + 1)
        m = exp(-diff / t)
        """
        m = annealing_schedule(temp, i)
        if np.random.rand() < m:
            # pick next state based in proportion to scores
            i = np.random.choice(len(scores), p=softmax(scores))
            curr_s, curr_eval = N[i], scores[i]
            
        else:
            s = max(scores)    
            g = N[np.argmax(scores)]
            
        if curr_eval > best_eval:
            best_s, best_eval = curr_s, curr_eval
   
    return best_s, best_eval, n_iterations

def local_search(score_fn, g):
    s = score_fn(g)
    while True:
        G = get_neighbors(g)
        scores = [score_fn(gi) for gi in G]        
        if s <= max(scores):
            break
        
        s = max(scores)    
        g = G[argmax(scores)]
    return g     