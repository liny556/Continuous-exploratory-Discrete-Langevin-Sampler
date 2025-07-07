import torch.nn as nn
import pickle
import mmd
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import product
import argparse
from torch.distributions import Binomial
from torch import tensor
from tqdm import tqdm
#from utils import get_sampler
from samplers import (
    cLangevinSamplerOrdinal,
    LangevinSamplerOrdinal,
    PerDimMetropolisSamplerOrd,
    PerDimGibbsSamplerOrd,
)
#import samplers

EPS = 1e-10


def get_modes(num_modes, space_between_modes=5):
    # what should the mode centers be?
    dim = (num_modes + 1) * space_between_modes
    mode_centers_x = np.linspace(0, dim, num_modes + 2)[1:-1]
    mode_centers = list(product(mode_centers_x, repeat=2))
    return dim, torch.Tensor(mode_centers)


class SimpleBin(nn.Module):
    def __init__(self, ss_dim, mean, device) -> None:
        super().__init__()
        self.ss_dim = ss_dim
        self.mean = mean
        self.x_prob = self.mean[0] / self.ss_dim
        self.y_prob = self.mean[1] / self.ss_dim
        self.x_rv = Binomial(
            total_count=tensor([self.ss_dim]).to(device),
            probs=tensor([self.x_prob]).td(device),
        )
        self.y_rv = Binomial(
            total_count=tensor([self.ss_dim]).to(device),
            probs=tensor([self.y_prob]).to(device),
        )

    def forward(self, x):
        return self.x_rv.log_prob(x[:, 0]) + self.y_rv.log_prob(x[:, 1])


class MM_Bin(nn.Module):
    def __init__(self, ss_dim, device, means) -> None:
        super().__init__()
        self.ss_dim = ss_dim
        self.means = means
        self.bvs = []
        self.device = device
        for i in range(self.means.shape[0]):
            bv = SimpleBin(self.ss_dim, self.means[i, :], device=self.device)
            self.bvs.append(bv)

    def forward(self, x):
        out = torch.zeros((x.shape[0])).to(self.device)
        for bv in self.bvs:
            res = bv(x).exp() * (1 / len(self.bvs))
            out += res
        return torch.log(out + EPS)


class MM_Heat(nn.Module):
    def __init__(self, ss_dim, means, var, device, weights=None) -> None:
        super().__init__()
        self.means = means.float()
        self.means
        self.ss_dim = ss_dim
        self.one_hot = False
        self.var = var
        self.device = device
        self.weights = weights
        self.L = 1

    def forward(self, x):
        x = x.float()
        out = torch.zeros((x.shape[0])).to(x.device)
        self.means = self.means.to(x.device)
        if self.one_hot:
            dim_v = tensor(
                [[i for i in range(self.ss_dim)], [i for i in range(self.ss_dim)]]
            ).to(x.device)
            # turning from 1 hot to coordinate vector (2 dim with ss_dim potenital values)
            x = (dim_v * x).sum(axis=2)
        for m in range(self.means.shape[0]):
            if self.weights:
                out += (
                    torch.exp(
                        (-torch.norm(x - self.means[m, :], dim=1))
                        * (1 / (self.var * self.means.shape[0]))
                    )
                    * self.weights[m]
                )
            else:
                out += torch.exp(
                    (-torch.norm(x - self.means[m, :], dim=1))
                    * (1 / (self.var * self.means.shape[0]))
                )
            # for i in range(x.shape[0]):
            #     out[i] += torch.exp(
            #         (-torch.norm(x[i, :] - self.means[m, :]))
            #         * 1
            #         / (self.var * self.means.shape[0])
            #     )
        return torch.log(out + EPS)


def calc_probs(ss_dim, energy_function, device):
    energy_function.one_hot = False
    samples = []
    for i in range(ss_dim):
        for j in range(ss_dim):
            samples.append((i, j))
    samples = tensor(samples).to(device)
    energies = energy_function(samples)
    z = energies.exp().sum()
    probs = energies.exp() / z
    return samples, probs

def run_sampler(
    sampler,
    energy_function,
    sampling_steps,
    batch_size,
    device,
    start_coord,
    is_c,
    dim,
    x_init=None,
    show_a_s=True,
    rand_restarts=10,
):
    energy_function.one_hot = False
    # x = torch.zeros((batch_size, 2)).to(device)
    if x_init is not None:
        x = x_init
    else:
        x = torch.Tensor(start_coord).repeat(batch_size, 1).to(device)
    samples = []
    trajectories = []
    chain_a_s = []
    pg = tqdm(range(sampling_steps))
    restart_every = sampling_steps // rand_restarts
    for i in pg:
        x = x.to(device)
        energy_function = energy_function.to(device)
        trajectories.append(x.long().detach().cpu().numpy())
        if is_c:
            x,x_mid = sampler.step(x.detach(), energy_function)
            x = x.detach()
            x_mid = x_mid.detach()
            trajectories.append(x_mid.detach().cpu().numpy())
        else:
            x = sampler.step(x.detach(), energy_function).detach()
        # storing the acceptance rate
        samples += list(x.long().detach().cpu().numpy())
        if show_a_s:
            chain_a_s.append(sampler.a_s)
            samples += list(x.long().detach().cpu().numpy())
            # resetting the acceptance rate
            sampler.a_s = []
            if i % 10 == 0:
                pg.set_description(
                    f"mean a_s: {np.mean(chain_a_s[-100:])}"
                )
            # min_lr_val = (args.min_lr * dim)** (4) 
            # sampler.step_size = max(min_lr_val, sampler.step_size * args.exponential_decay)
        # if i % restart_every == 0:
        #     x = torch.randint(0, dim, (1, 2)).repeat((batch_size, 1)).to(device)
    
    trajectories.append(x.long().detach().cpu().numpy())
    return chain_a_s, samples, np.array(trajectories)

def get_sampler(args, dim, device):
    if 'dmala' or 'cdmala' in args.sampler:
        use_mh = True
    else:
        use_mh = False
    if args.sampler == "cdmala":
        sampler = cLangevinSamplerOrdinal(
            dim=int(2),
            max_val=dim,
            n_steps=1,
            mh=use_mh,
            step_size=args.step_size,
            step_size_c=args.step_size_c,
            bal=0.5,
            device=device,  
        )
    elif args.sampler == "dula":
        sampler = LangevinSamplerOrdinal(
            dim=int(2),
            max_val=dim,
            n_steps=1,
            mh=use_mh,
            step_size=args.step_size,
            bal=0.5,
            device=device,
        )
    elif args.sampler == "cdula":
        sampler = cLangevinSamplerOrdinal(
            dim=int(2),
            max_val=dim,
            n_steps=1,
            mh=use_mh,
            step_size=args.step_size,
            step_size_c=args.step_size_c,
            bal=0.5,
            device=device,
        )
    elif args.sampler == "dmala":
        sampler = LangevinSamplerOrdinal(
            dim=int(2),
            max_val=dim,
            n_steps=1,
            mh=use_mh,
            step_size=args.step_size,
            bal=0.5,
            device=device,
        )
    elif args.sampler == "gibbs":
        sampler = PerDimGibbsSamplerOrd(dim=2, max_val=dim)
    elif args.sampler == "rw":
        sampler = PerDimMetropolisSamplerOrd(dim=2, max_val=dim, dist_to_test=100)
    return sampler

def plot_trajectory(traja, dim, save_dir,mode,sampler):
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.grid(True,linestyle='--',alpha = 0.3)
    plt.xticks(np.arange(0, dim+1, 5))
    plt.yticks(np.arange(0, dim+1, 5))
    plt.xlim(0, dim)
    plt.ylim(0, dim)
    np.save(f"{save_dir}/traj.npy", traja)

    stride = 200
    traj = traja[::stride,40,:]
    if sampler == 'cdmala' or sampler == 'cdula':
        traj_1 = traj[::2]
        traj_2 = traj[1::2]
        sc = plt.scatter(traj_1[:,0], traj_1[:,1], c=range(len(traj_1)),
                cmap='viridis', s=50, edgecolor='k', alpha=0.7)
        plt.scatter(traj_2[:,0], traj_2[:,1],color='orange',alpha=0.5)
    else:
        sc = plt.scatter(traj[:,0], traj[:,1], c=range(len(traj)), 
                cmap='viridis', s=40, edgecolor='k', alpha=0.7)
    
    #key_indices = [0,len(traj)//4,len(traj)//2,-1]
    plt.plot(traj[:,0], traj[:,1], '--', lw=1, alpha=0.4,color = 'gray')  # white trajectory
    
    mode_center = mode
    # start, end and mode points
    plt.scatter(mode_center[:,0], mode_center[:,1], marker='*', s=300, c='gold', label='Modes',edgecolors='k')
    plt.scatter(traj[0,0], traj[0,1], marker='o', s=200, c='red', label='Start')
    plt.scatter(traj[-1,0], traj[-1,1], marker='X', s=200, c='black', label='End')
    
    plt.colorbar(sc,label="Step")
    #plt.title(f"Sampling Trajectory (dim={dim}", fontsize=40)
    plt.legend(fontsize=30)
    plt.savefig(f"{save_dir}/trajectory.png", dpi=300, bbox_inches='tight')
    plt.clf()   


def main(args):
    device = torch.device(
        "cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu"
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dim, modes_to_use = get_modes(args.num_modes, args.space_between_modes)
    energy_function = MM_Heat(
        ss_dim=dim,
        means=modes_to_use,
        var=args.dist_var,
        device=device,
    )
    sampler = get_sampler(args, dim, device)
    cur_dir = f"{args.save_dir}/{args.dist_type}_{args.dist_var}/"
    cur_dir += f"{args.modality}_{args.starting_point}/"
    if "acs" in args.sampler:
        cur_dir += f"{sampler.get_name()}_{args.num_cycles}/"
    elif args.sampler == "dmala" or args.sampler == "dula":
        cur_dir += f"{args.sampler}_{args.step_size}/"
    elif args.sampler == 'cdmala' or args.sampler == 'cdula':
        cur_dir += f"{args.sampler}_{args.step_size}_{args.step_size_c}/"
    else:
        cur_dir += f"{args.sampler}"
    os.makedirs(cur_dir, exist_ok=True)
    # plotting the ground truth distribution
    samples, probs = calc_probs(dim, energy_function, device)
    dist_img = np.zeros((dim, dim))
    for i in range(len(samples)):
        coord = samples[i]
        dist_img[coord[0], coord[1]] = probs[i]

    pickle.dump(tensor(dist_img), open(f"{cur_dir}/gt.pickle", "wb"))
    plt.imshow(dist_img, cmap="Blues")

    plt.axis("off")
    plt.savefig(f"{cur_dir}/init_dist.pdf", bbox_inches="tight")
    plt.savefig(f"{cur_dir}/init_dist.png", bbox_inches="tight")
    plt.savefig(f"{cur_dir}/init_dist.svg", bbox_inches="tight")
    print(f"gt: {cur_dir}/init_dist.png")
    start_coord_y = np.random.randint(0, dim)
    start_coord_x = np.random.randint(0, dim)
    #start_coord_x = start_coord_y = dim // 2
    start_coord = (start_coord_x, start_coord_y)
    est_img = torch.zeros((dim, dim))

    x_init = None
    if args.sampler in ["gibbs", "asb", "dula", "cyc_dula", "rw",'cdula']:
        show_a_s = False
    else:
        show_a_s = True
    if args.sampler == "cdmala" or args.sampler == "cdula":
        is_c = True
    else:
        is_c = False
    chain_a_s, samples,trajectories = run_sampler(
        energy_function=energy_function,
        batch_size=args.batch_size,
        sampling_steps=args.sampling_steps,
        device=device,
        sampler=sampler,
        start_coord=start_coord,
        is_c= is_c,
        x_init=x_init,
        show_a_s=show_a_s,
        dim=dim,
    )
    plot_trajectory(trajectories, dim, cur_dir,modes_to_use,args.sampler)

    for i in range(len(samples)):
        coord = samples[i]
        est_img[coord[0], coord[1]] += 1
    #normalize
    est_img = est_img / est_img.sum()
    pickle.dump(chain_a_s, open(f"{cur_dir}/chain_a_s.pickle", "wb"))
    pickle.dump(est_img, open(f"{cur_dir}/actual_probs.pickle", "wb"))
    plt.imshow(est_img, cmap="Blues")
    plt.axis("off")
    plt.savefig(f"{cur_dir}/est_dist.png", bbox_inches="tight")
    plt.savefig(f"{cur_dir}/est_dist.pdf", bbox_inches="tight")
    plt.savefig(f"{cur_dir}/est_dist.svg", bbox_inches="tight")
    print(f"est: {cur_dir}/est_dist.png")

    # compute the KL divergence
    gt = tensor(dist_img).to(device)
    est = est_img.to(device)
    kl = (gt * (gt / (est + EPS)).log()).sum()
    print(f"kl: {kl}")
    pickle.dump(kl, open(f"{cur_dir}/kl.pickle", "wb"))

    # compute the TV distance
    tv = torch.abs(gt - est).sum()
    print(f"tv: {tv}")
    pickle.dump(tv, open(f"{cur_dir}/tv.pickle", "wb"))

    # compute the JS divergence
    js = (gt * (gt / (0.5 * (gt + est) + EPS)).log()).sum()
    print(f"js: {js}")
    pickle.dump(js, open(f"{cur_dir}/js.pickle", "wb"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist_type", type=str, default="heat")
    parser.add_argument("--save_dir", type=str, default="figs/multi_modal")
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)#128
    parser.add_argument("--sampling_steps", type=int, default=1000)
    parser.add_argument("--dist_var", type=float, default=.9)#.9
    parser.add_argument("--sampler", type=str, default="dmala")
    parser.add_argument("--modality", type=str, default="multimodal")
    parser.add_argument("--starting_point", type=str, default="center")
    parser.add_argument("--num_modes", type=int, default=2)
    parser.add_argument("--space_between_modes", type=int, default=100)
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument("--step_size", default=.013, type=float)
    parser.add_argument("--step_size_c", default=.0035,type=float)
    parser.add_argument("--min_lr", default=.011, type=float)
    parser.add_argument("--input_type", type=str, default='binary')
    #parser.add_argument("--exponential_anneal", action="store_true")
    parser.add_argument('--seed', type=int, default=1234567)
    args = parser.parse_args()
    args.burn_in = 0
    args.n_steps = 1
    main(args)
