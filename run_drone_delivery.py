import sys
sys.path.append('/home/victorialena/rlkit')

import pdb
import matplotlib.pyplot as plt
import rlkit
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

from argparse import ArgumentParser
from matplotlib import animation
from matplotlib.animation import FuncAnimation, PillowWriter 
from torch.optim import Adam
from torch_geometric.data import Data

from actor.drone_discrete import *
from env.drone_delivery import *
from path_collector import MdpPathCollector
from policies import argmaxDiscretePolicy, epsilonGreedyPolicy
from replay_buffer import anyReplayBuffer

# ------------------ Helpers
scientific_notation =  lambda x:"{:.2e}".format(x)

def mean_reward_per_traj(paths):
    return np.mean([torch.vstack(p['rewards']).sum().item() for p in paths])

def mean_reward(paths):
    return np.hstack([torch.vstack(p['rewards']).sum(1).numpy() for p in paths]).mean()

def max_reward_per_traj(paths):
    return np.mean([torch.vstack(p['rewards']).sum(1).max().item() for p in paths])

def get_success_rate(paths):
    success = [(p['observations'][-1].x[:, -1]==0).all().item() for p in paths]
    return sum(success)/len(paths)

def printSettings(args):
    print(args, '\n')
    
def make_plot(avg_r_train, avg_r_test, n_iter, n_epoch, baselines, saveas=""):
    for b, col, name in zip(baselines, 
                            ['lightgray', 'darkgray', 'gray'], 
                            ['random', 'closest', 'pre-assigned']):
        plt.plot(np.arange(n_iter*n_epoch), [b]*(n_iter*n_epoch), label=name, color=col)

    plt.plot(np.arange(n_iter*n_epoch), avg_r_train, label = "avg R (train)")
    plt.plot(np.arange(n_iter, n_iter*n_epoch+1, step=n_iter), avg_r_test, label = "avg R (test)")
    plt.legend()
    
    if saveas:
        plt.savefig("figs/drone_delivery/"+saveas+".png", dpi=300)
    else:
        plt.show()

def simple_animation(env, obs, saveas='animation_1'):
    fig, ax = plt.subplots(figsize=(6,4))    
    
    def env_simple_update(num, obs, ax):
        ax.clear()
        env.render(obs[num])
        ax.set_title("Frame {}".format(num))
    
    ani = animation.FuncAnimation(fig, env_simple_update, frames=len(obs), fargs=(obs, ax))
    ani.save("figs/drone_delivery/"+saveas+'.gif', writer='imagemagick', savefig_kwargs={'facecolor':'white'}, fps=1)
    
# ------------------ Baselines

def getBaseline(pol, max_len, n=128, verbose=False):
    path_collector = MdpPathCollector(env, pol)
    paths = path_collector.collect_new_paths(n, max_len, False)
    expected_heuristic_pt = mean_reward_per_traj(paths)
    print("Expected reward (per traj):", expected_heuristic_pt)
    expected_heuristic = mean_reward(paths)
    print("Expected reward (per step):", expected_heuristic, '\n')
    
    if verbose:
        idx = np.random.randint(100)
        for s, a, r, t in zip(paths[idx]['observations'], paths[idx]['actions'], 
                              paths[idx]['rewards'], paths[idx]['terminals']):
            print('\n', s)
            print(a)
            print(r)
            print(t)
    return expected_heuristic_pt, expected_heuristic

def getNetworkBaseline(n, g, max_len, in_channels, out_channels): 
    qf = droneDeliveryModel(in_channels, out_channels, [16, 16], dropout=0.0)
    policy = argmaxDiscretePolicy(qf) 
    return getBaseline(policy, max_len)

def getHeuristicBaseline(max_len):
    policy = sysRolloutPolicy(device=device)
    return getBaseline(policy, max_len)

def getHeuristicBaseline2(max_len):
    policy = sysMultiRolloutPolicy(device=device)
    return getBaseline(policy, max_len)


# ------------------ Train

def dqtrain(env, args):
    printSettings(args)
    
    in_channels, out_channels = env.get_channels()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)
    
    qf = droneDeliveryModel(in_channels, out_channels, args.c_hidden, n_linear=args.n_linear, 
                            bounds=env.get_size(), dropout=args.dropout, heads=args.n_heads)
    print(qf)
    if args.load_from:
        qf.load_state_dict(torch.load("chkpt/"+args.load_from+".pt"))
    qf.to(device)

    target_qf = droneDeliveryModel(in_channels, out_channels, args.c_hidden, n_linear=args.n_linear, 
                                   bounds=env.get_size(), dropout=args.dropout, heads=args.n_heads)
    
    if args.load_from:
        target_qf.load_state_dict(torch.load("chkpt/"+args.load_from+".pt"))
    target_qf.to(device)

    qf_criterion = nn.MSELoss()
    eval_policy = argmaxDiscretePolicy(qf)
    expl_policy = epsilonGreedyPolicy(qf, env.aspace, eps=args.eps, sim_annealing_fac=1.0, device=device)
    expl_policy.set_heuristic(True)

    expl_path_collector = MdpPathCollector(env, expl_policy) 
    eval_path_collector = MdpPathCollector(env, eval_policy)
    replay_buffer = anyReplayBuffer(args.replay_buffer_cap, prioritized=args.prioritized_replay)
    
    optimizer = Adam(qf.parameters(), lr=args.learning_rate)
    lam_fn = lambda epoch: args.sim_annealing_fac ** (epoch//args.saf_stepsize)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lam_fn)

    max_len = env.get_max_len()
    n_samples = min(args.n_samples, args.replay_buffer_cap//max_len) 

    loss = []
    avg_r_train = []
    avg_r_test = []
    max_r_test = []
    success_rate = []
    
    save_animation = args.save_to if args.save_to else args.plot_name

    for i in range(args.n_epoch):
        qf.train(False)
        paths = eval_path_collector.collect_new_paths(128, max_len, False)
        avg_r_test.append(mean_reward_per_traj(paths))
        max_r_test.append(max_reward_per_traj(paths))
        success_rate.append(get_success_rate(paths))

        paths = expl_path_collector.collect_new_paths(n_samples, max_len, False)
        replay_buffer.add_paths(paths)

        qf.train(True)
                
        if i > args.heuristic_pol_len:
            expl_policy.set_heuristic(False)

        for _ in range(args.n_iter):
            batch = replay_buffer.random_batch(args.batch_size) #//args.ndrones)
            
            rewards = batch.r
            terminals = batch.t.to(torch.float).repeat_interleave(args.ndrones)
            actions = batch.a
                        
            obs = batch
            next_obs = deepcopy(batch)
            next_obs.x = next_obs.next_s
            
            
            out = target_qf(next_obs)
            target_q_values = out.max(-1, keepdims=False).values
            y_target = rewards + (1. - terminals) * args.gamma * target_q_values 
            out = qf(obs)
            actions_one_hot = F.one_hot(actions.to(torch.int64), len(action))
            y_pred = torch.sum(out * actions_one_hot, dim=-1, keepdim=False)
            qf_loss = qf_criterion(y_pred, y_target)

            loss.append(qf_loss.item())
            avg_r_train.append(rewards.mean().item())

            optimizer.zero_grad() 
            qf_loss.backward()
            optimizer.step()
        scheduler.step()

        target_qf.load_state_dict(deepcopy(qf.state_dict()))
        err = 8
        print("epoch", i+1, "| lr:", scientific_notation(optimizer.param_groups[0]["lr"]) ,
              " -> loss:", round(np.mean(loss[-args.n_iter:]), err),
              "| rewards: (train)", round(np.mean(avg_r_train[-args.n_iter:]), err),
              "| (test)", round(avg_r_test[-1], err), "| (max)", round(max_r_test[-1], err),
              "| success rate:", round(success_rate[-1], err))
    
    if args.save_to:
        torch.save(qf.state_dict(), "chkpt/"+args.save_to+".pt")
        
    return loss, avg_r_train, avg_r_test, max_r_test

parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--c_hidden", type=int, nargs="+", default=[32, 32, 32], help="hidden channels in GCN")
parser.add_argument("--n_linear", type=int, default=1, help="number of linear layers in model")
parser.add_argument("--eps", type=float, default=0.1, help="epislon greedy exploration parameter")
parser.add_argument("-saf", "--sim_annealing_fac", type=float, default=1.0, help="simulated anealing factor")
parser.add_argument("--saf_stepsize", type=int, default=25, help="interval at which to reduce learning rate")
parser.add_argument("--replay_buffer_cap", type=int, default=10000, help="replay buffer size, max nr of past samples")
parser.add_argument("--n_samples", type=int, default=64, help="number of trajectories sampled at each epoch.")
parser.add_argument("--prioritized_replay", type=bool, default=True, help="use prioritzed replay when sampling for HER")
parser.add_argument("-lr", "--learning_rate", type=float, default=5E-3, help="learning rate")
parser.add_argument("-ep", "--n_epoch", type=int, default=15, help="number of epochs")
parser.add_argument("-it", "--n_iter", type=int, default=128, help="number of iterations per epoch")
parser.add_argument("-bs", "--batch_size", type=int, default=256, help="batch size per iteration")
parser.add_argument("-γ", "--gamma", type=float, default=0.9, help="discount factor")
parser.add_argument("--max_sample_distance", type=int, default=-1, 
                    help="distance between drones and closest goals at init (for curriculum learning)")
parser.add_argument("--dropout", type=float, default=0.0, help="dropout after GAT")
parser.add_argument("--n_heads", type=int, nargs="+", default=[4, 4], help="number of heads per GATv2 layer.")
parser.add_argument("--heuristic_pol_len", type=int, default=100) 

parser.add_argument("--graph_type", type=str, default="full", 
                    help="how the nodes are connected, e.g., \"full\", \"sparse\", or \"light\"")
parser.add_argument("--load_from", type=str, default="", help="load a pretrained network's weights")
parser.add_argument("--save_to", type=str, default="", help="save trained network params at")
parser.add_argument("--plot", type=bool, default=True, help="plot training performance curves")
parser.add_argument("--plot_name", type=str, default="", help="save training progress plot under")

parser.add_argument("--maze_size", type=int, default=5, help="number of drones spawning in the environment")
parser.add_argument("--ndrones", type=int, default=1, help="number of drones spawning in the environment")
parser.add_argument("--ngoals", type=int, default=1, help="number of goal regions spawning in the environment")

args = parser.parse_args()
device = 'cpu' #torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

env = droneDelivery(args, device=device)
x = env.reset()

loss, avg_r_train, avg_r_test, max_r_test = dqtrain(env, args)

if args.plot:
    net_pt, _ = getNetworkBaseline(args.ndrones, args.ngoals, env.get_max_len(), *env.get_channels())
    heuristic_pt, _ = getHeuristicBaseline(env.get_max_len())
    heuristic_pt2, _ = getHeuristicBaseline2(env.get_max_len())
    make_plot(avg_r_train, avg_r_test, args.n_iter, args.n_epoch, [net_pt, heuristic_pt, heuristic_pt2],
              args.plot_name if args.plot_name else args.save_to)

