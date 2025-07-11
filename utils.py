import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as tr
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pickle
import rbm
import samplers
from tqdm import tqdm
import os
import block_samplers
from vamp_utils import load_caltech101silhouettes, load_omniglot


def difference_function(x, model):
    d = torch.zeros_like(x)
    orig_out = model(x).squeeze()
    for i in range(x.size(1)):
        x_pert = x.clone()
        x_pert[:, i] = 1. - x[:, i]
        delta = model(x_pert).squeeze() - orig_out
        d[:, i] = delta
    return d

def approx_difference_function(x, model):
    x = x.requires_grad_()
    gx = torch.autograd.grad(model(x).sum(), x)[0]
    wx = gx * -(2. * x - 1)
    return wx.detach()

def gradient_function(x, model):
    x = x.requires_grad_()
    gx = torch.autograd.grad(model(x).sum(), x)[0]
    return gx.detach()

def difference_function_multi_dim(x, model):
    d = torch.zeros_like(x)
    orig_out = model(x).squeeze()
    for i in range(x.size(1)):
        for j in range(x.size(2)):
            x_pert = x.clone()
            x_pert[:, i] = 0.
            x_pert[:, i, j] = 1.
            delta = model(x_pert).squeeze() - orig_out
            d[:, i, j] = delta
    return d

def langevin_approx_difference_function_multi_dim(x, model):
    x = x.requires_grad_()
    gx = torch.autograd.grad(model(x).sum(), x)[0]
    return gx.detach()

def approx_difference_function_multi_dim(x, model):
    x = x.requires_grad_()
    gx = torch.autograd.grad(model(x).sum(), x)[0]
    gx_cur = (gx * x).sum(-1)[:, :, None]
    return gx - gx_cur


def short_run_mcmc(logp_net, x_init, k, sigma, step_size=None):
    x_k = torch.autograd.Variable(x_init, requires_grad=True)
    # sgld
    if step_size is None:
        step_size = (sigma ** 2.) / 2.
    for i in range(k):
        f_prime = torch.autograd.grad(logp_net(x_k).sum(), [x_k], retain_graph=True)[0]
        x_k.data += step_size * f_prime + sigma * torch.randn_like(x_k)

    return x_k


def get_data(args):
    potential_datasets = ["mnist", "fashion", "emnist", "caltech", "omniglot", "kmnist"]

    if args.data in potential_datasets:
        transform = tr.Compose(
            [
                tr.Resize(args.img_size),
                tr.ToTensor(),
                lambda x: (x > 0.5).float().view(-1),
            ]
        )

        if args.data in ["mnist", "fashion", "emnist", "kmnist", "omniglot"]:
            if args.data == "mnist":
                train_data = torchvision.datasets.MNIST(
                    root="../data", train=True, transform=transform, download=True
                )
                test_data = torchvision.datasets.MNIST(
                    root="../data", train=False, transform=transform, download=True
                )
            elif args.data == "kmnist":
                train_data = torchvision.datasets.KMNIST(
                    root="../data", train=True, transform=transform, download=True
                )
                test_data = torchvision.datasets.KMNIST(
                    root="../data", train=False, transform=transform, download=True
                )
            elif args.data == "emnist":
                train_data = torchvision.datasets.EMNIST(
                    root="../data",
                    train=True,
                    split="mnist",
                    transform=transform,
                    download=False,
                )
                test_data = torchvision.datasets.EMNIST(
                    root="../data",
                    train=False,
                    split="mnist",
                    transform=transform,
                    download=False,
                )
            elif args.data == "fashion":
                train_data = torchvision.datasets.FashionMNIST(
                    root="../data", train=True, transform=transform, download=True
                )
                test_data = torchvision.datasets.FashionMNIST(
                    root="../data", train=False, transform=transform, download=True
                )

            elif args.data == "omniglot":
                transform = tr.Compose(
                    [
                        tr.Resize(args.img_size),
                        tr.RandomInvert(p=1),
                        tr.ToTensor(),
                        lambda x: (x > 0.5).float().view(-1),
                    ]
                )
                train_data = torchvision.datasets.Omniglot(
                    root="../data", transform=transform, download=True
                )
                test_data = train_data
            train_loader = DataLoader(
                train_data, args.batch_size, shuffle=True, drop_last=True
            )
            test_loader = DataLoader(
                test_data, args.batch_size, shuffle=True, drop_last=True
            )
        else:
            if args.data == "caltech":
                (
                    train_loader,
                    _,
                    test_loader,
                    args,
                ) = load_caltech101silhouettes(args)

        sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
        plot = lambda p, x: torchvision.utils.save_image(
            x.view(x.size(0), 1, args.img_size, args.img_size),
            p,
            normalize=True,
            nrow=sqrt(x.size(0)),
        )
        encoder = None
        viz = None

    elif args.data_file is not None:
        with open(args.data_file, "rb") as f:
            x = pickle.load(f)
        x = torch.tensor(x).float()
        train_data = TensorDataset(x)
        train_loader = DataLoader(
            train_data, args.batch_size, shuffle=True, drop_last=True
        )
        test_loader = train_loader
        viz = None
        if args.model == "lattice_ising" or args.model == "lattice_ising_2d":
            plot = lambda p, x: torchvision.utils.save_image(
                x.view(x.size(0), 1, args.dim, args.dim),
                p,
                normalize=False,
                nrow=int(x.size(0) ** 0.5),
            )
        elif args.model == "lattice_potts":
            plot = lambda p, x: torchvision.utils.save_image(
                x.view(x.size(0), args.dim, args.dim, 3).transpose(3, 1),
                p,
                normalize=False,
                nrow=int(x.size(0) ** 0.5),
            )
        else:
            plot = lambda p, x: None
    else:
        raise ValueError

    return train_loader, test_loader, plot, viz


def generate_data(args):
    if args.data_model == "lattice_potts":
        model = rbm.LatticePottsModel(args.dim, args.n_state, args.sigma)
        sampler = samplers.PerDimMetropolisSampler(model.data_dim, args.n_out, rand=False)
    elif args.data_model == "lattice_ising":
        model = rbm.LatticeIsingModel(args.dim, args.sigma)
        sampler = samplers.PerDimGibbsSampler(model.data_dim, rand=False)
    elif args.data_model == "lattice_ising_3d":
        model = rbm.LatticeIsingModel(args.dim, args.sigma, lattice_dim=3)
        sampler = samplers.PerDimGibbsSampler(model.data_dim, rand=False)
        print(model.sigma)
        print(model.G)
        print(model.J)
    elif args.data_model == "er_ising":
        model = rbm.ERIsingModel(args.dim, args.degree, args.sigma)
        sampler = samplers.PerDimGibbsSampler(model.data_dim, rand=False)
        print(model.G)
        print(model.J)
    else:
        raise ValueError

    model = model.to(args.device)
    samples = model.init_sample(args.n_samples).to(args.device)
    print("Generating {} samples from:".format(args.n_samples))
    print(model)
    for _ in tqdm(range(args.gt_steps)):
        samples = sampler.step(samples, model).detach()

    return samples.detach().cpu(), model


def load_synthetic(mat_file, batch_size):
    import scipy.io
    mat = scipy.io.loadmat(mat_file)
    ground_truth_J = mat['eij']
    ground_truth_h = mat['hi']
    ground_truth_C = mat['C']
    q = mat['q']
    n_out = q[0, 0]

    x_int = mat['sample']
    n_samples, dim = x_int.shape

    x_int = torch.tensor(x_int).long() - 1
    x_oh = torch.nn.functional.one_hot(x_int, n_out)
    assert x_oh.size() == (n_samples, dim, n_out)

    x = torch.tensor(x_oh).float()
    train_data = TensorDataset(x)
    train_loader = DataLoader(train_data, batch_size, shuffle=True, drop_last=True)
    test_loader = train_loader

    J = torch.tensor(ground_truth_J)
    j = J
    jt = j.transpose(0, 1).transpose(2, 3)
    ground_truth_J = (j + jt) / 2
    return train_loader, test_loader, x, \
           torch.tensor(ground_truth_J), torch.tensor(ground_truth_h), torch.tensor(ground_truth_C)

def get_sampler(temp, dim, device,args):
    data_dim = dim
    if args.input_type == "binary":
        if args.sampler == "gibbs":
            sampler = samplers.PerDimGibbsSampler(data_dim, rand=False)
        elif args.sampler == "rand_gibbs":
            sampler = samplers.PerDimGibbsSampler(data_dim, rand=True)
        elif args.sampler.startswith("bg-"):
            block_size = int(args.sampler.split('-')[1])
            sampler = block_samplers.BlockGibbsSampler(data_dim, block_size)
        elif args.sampler.startswith("hb-"):
            block_size, hamming_dist = [int(v) for v in args.sampler.split('-')[1:]]
            sampler = block_samplers.HammingBallSampler(data_dim, block_size, hamming_dist)
        elif args.sampler == "gwg":
            sampler = samplers.DiffSampler(data_dim, 1,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2.)
        elif args.sampler.startswith("gwg-"):
            n_hops = int(args.sampler.split('-')[1])
            sampler = samplers.MultiDiffSampler(data_dim, 1, approx=True, temp=2., n_samples=n_hops)
        elif args.sampler == "dmala":
            sampler = samplers.LangevinSampler(data_dim, 1,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2., step_size=args.step_size, mh=True)
        elif args.sampler == "dula":
            sampler = samplers.LangevinSampler(data_dim, 1,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2., step_size=args.step_size, mh=False)
        elif args.sampler == "cdmala":
            sampler = samplers.cLangevinSampler(data_dim, 1,
                                             fixed_proposal=False, approx=True, multi_hop=False, temp=2., step_size1=args.step_size, step_size2=args.step_size_c,mh=True)
        elif args.sampler == "cdula":
            sampler = samplers.cLangevinSampler(data_dim, 1,
                                             fixed_proposal=False, approx=True, multi_hop=False, temp=2., step_size1=args.step_size, step_size2=args.step_size_c,mh=False)
        
        else:
            raise ValueError("Invalid sampler...")
    else:
        if args.sampler == "gibbs":
            sampler = samplers.PerDimMetropolisSampler(data_dim, int(args.n_out), rand=False)
        elif args.sampler == "rand_gibbs":
            sampler = samplers.PerDimMetropolisSampler(data_dim, int(args.n_out), rand=True)
        elif args.sampler == "gwg":
            sampler = samplers.DiffSamplerMultiDim(data_dim, 1, approx=True, temp=2.)
        else:
            raise ValueError("invalid sampler")
    return sampler