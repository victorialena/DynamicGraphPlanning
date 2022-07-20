import numpy as np
import copy
from torch_geometric.data import Data

def condensed_rollout(env,
                      agent,
                      max_path_length=np.inf,
                      render=False,
                      render_kwargs=None,
                      preprocess_func=None,
                      get_action_kwargs=None,
                      postprocess_func=None,
                      reset_callback=None):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    if preprocess_func is None:
        preprocess_func = lambda x: x
        
    data = []
    path_length = 0
    agent.reset()
    
    o = env.reset()
    if reset_callback:
        reset_callback(env, agent, o)
    if render:
        env.render(**render_kwargs)
    
    while path_length < max_path_length:
        o_for_agent = preprocess_func(o)
        action, agent_info = agent.get_action(o_for_agent, **get_action_kwargs)
        
        if postprocess_func:
            postprocess_func(env, agent, o)
            
        next_o, reward, done, env_info = env.step(copy.deepcopy(action))
        sample = Data(x=o.x,
                      edge_index=o.edge_index,
                      a=action,
                      r=reward,
                      next_s=next_o.x,
                      t=done)
        
        if render:
            env.render(**render_kwargs)
        
        data.append(sample)
        path_length += 1
        if done:
            break
        o = next_o        
        
    return data, len(data)

def rollout(env,
            agent,
            max_path_length=np.inf,
            render=False,
            render_kwargs=None,
            preprocess_func=None,
            get_action_kwargs=None,
            postprocess_func=None,
            reset_callback=None):
    
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    if preprocess_func is None:
        preprocess_func = lambda x: x
        
    observations = []
    actions = []
    rewards = []
    terminals = []
    next_observations = []
    
    path_length = 0
    agent.reset()
    o = env.reset()
    
    if reset_callback:
        reset_callback(env, agent, o)
    if render:
        env.render(**render_kwargs)
    
    while path_length < max_path_length:
        o_for_agent = preprocess_func(o)
        a, agent_info = agent.get_action(o_for_agent, **get_action_kwargs)
        
        if postprocess_func:
            postprocess_func(env, agent, o)
            
        next_o, r, done, env_info = env.step(copy.deepcopy(a))
        if render:
            env.render(**render_kwargs)
        
        observations.append(o)
        rewards.append(r)
        terminals.append(done)
        actions.append(a)
        next_observations.append(next_o)
        
        path_length += 1
        if done:
            break
        o = next_o
        
    return dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=terminals,
    )
