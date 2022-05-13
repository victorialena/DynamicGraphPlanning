import numpy as np
import pdb
import torch

from torch.optim import Adam

def ppo(env, actor_critic, args, 
        lam=0.97, target_kl=0.01, clip_ratio=0.2, pi_lr=3e-4, vf_lr=1e-3):    
    """ Proximal Policy Optimization (by clipping), with early stopping based on approximate KL """

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

    buf = PPOBuffer(args.replay_buffer_cap, args.gamma, lam)

    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = actor_critic.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((actor_critic.v(obs) - ret)**2).mean()

    pi_optimizer = Adam(actor_critic.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(actor_critic.v.parameters(), lr=vf_lr)

    def update():
        # TODO: should this be in the the training loop?
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(args.n_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = pi_info['kl'] # mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                break # Early stopping at step %d due to reaching max kl
            loss_pi.backward()
            # mpi_avg_grads(actor_critic.pi)    # average grads across MPI processes
            pi_optimizer.step()

        # Value function learning
        for i in range(args.n_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            # mpi_avg_grads(actor_critic.v)    # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        # kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        return loss_pi, loss_v

    o, ep_ret, ep_len = env.reset(), 0, 0
    
    for epoch in range(args.epochs):        
        actor_critic.train(True)
        ret_train = []
        for t in range(args.n_samples*max_ep_len):
            a, v, logp = actor_critic.step(torch.as_tensor(o, dtype=torch.float32))

            next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o, a, r, v, logp)
            
            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                ret_train.append(ep_ret)
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = actor_critic.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)
                o, ep_ret, ep_len = env.reset(), 0, 0
                

        # Perform PPO update!
        loss_pi, loss_v = update()
        
        # Print progress
        err = 8
        print("iter ", epoch+1, " -> loss (π): ", round(loss_pi, err), "(v): ", round(loss_v, err),
              "| rewards: (train) ", round(np.mean(ret_train), err))
        