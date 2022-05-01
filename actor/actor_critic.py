import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    
    def __init__(self, net):
        super().__init__()
        self.pi_net = net

    def _distribution(self, obs):
        logits = self.pi_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)
    

class MLPGaussianActor(Actor):

    def __init__(self, net):
        super().__init__()
        # get output size
        *_, last = net.model.children()
        act_dim = last.out_features
        
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.pi_net = net

    def _distribution(self, obs):
        mu = self.pi_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.v_net = net

    def forward(self, obs):
        out = torch.squeeze(self.v_net(obs), -1)
        if hasattr(obs, 'ptr'):
                n_obs = len(obs.ptr)-1
                out = out.reshape(n_obs, -1)
        return out # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, actor, critic, continous=True):
        super().__init__()
        
        self.pi = MLPGaussianActor(actor) if continous else MLPCategoricalActor(actor)
        self.v  = MLPCritic(critic)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample().sigmoid() #TODO: should we be doing this?
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs).sum(1)
        return a, v, logp_a #a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
    
    def train(self, flag):
        # TODO: does not work
        super().train(flag)
        self.pi.pi_net.train(flag)
        self.v.v_net.train(flag)