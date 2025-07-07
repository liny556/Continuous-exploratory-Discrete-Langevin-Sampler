import argparse
import rbm
import torch
import numpy as np
import samplers
import mmd
import matplotlib.pyplot as plt
import os
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
import utils
import tensorflow_probability as tfp
import block_samplers
import time
import pickle
from torchvision.utils import make_grid
import tqdm


def makedirs(dirname):
    """
    Make directory only if it's not already there.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_ess(chain, burn_in):
    c = chain
    l = c.shape[0]
    bi = int(burn_in * l)
    c = c[bi:]
    cv = tfp.mcmc.effective_sample_size(c).numpy()
    cv[np.isnan(cv)] = 1.
    return cv

def get_gb_trained_rbm_sd(data, train_iter, rbm_name, save_dir):
    fn = (
        f"{save_dir}/{data}/zeroinit_False/rbm_iter_{train_iter}/{rbm_name}.pt"
    )
    print(fn)

    if os.path.isfile(fn):
        return torch.load(fn)
    return None

def main(args):
    makedirs(args.save_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
    def get_grid_img(x): 
        return make_grid(
            x.view(x.size(0), 1, args.img_size, args.img_size),
            normalize=True,
            nrow=sqrt(x.size(0)),
        )
    

    args.test_batch_size = args.batch_size
    args.train_batch_size = args.batch_size
    
    assert args.n_visible ==784
    args.img_size = 28

    model = rbm.BernoulliRBM(args.n_visible, args.n_hidden)
    model.to(device) 
    train_loader, test_loader, plot, viz = utils.get_data(args)

    init_data = []
    for x, _ in train_loader:
        init_data.append(x)
    init_data = torch.cat(init_data, 0)
    init_mean = init_data.mean(0).clamp(.01, .99)

    model = rbm.BernoulliRBM(args.n_visible, args.n_hidden, data_mean=init_mean)
    rbm_name = f"rbm_lr_{str(args.rbm_lr)}_n_hidden_{str(args.n_hidden)}"
    sd = get_gb_trained_rbm_sd(args.data, args.rbm_train_iter, rbm_name, args.save_dir)
    if sd is not None:
        model.load_state_dict(sd)
        model.to(device)
        print("Loaded pre-trained RBM")
    else:
        print("Training RBM")
        save_dir = f"{args.save_dir}/{args.data}/zeroinit_False/rbm_iter_{args.rbm_train_iter}/"
        os.makedirs(save_dir, exist_ok=True)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.rbm_lr)

        # train!
        itr = 0
        with tqdm.tqdm(total=args.rbm_train_iter) as pbar:
            while itr < args.rbm_train_iter:
                for x, _ in train_loader:
                    x = x.to(device)
                    xhat = model.gibbs_sample(v=x, n_steps=args.cd)

                    d = model.logp_v_unnorm(x)
                    m = model.logp_v_unnorm(xhat)

                    obj = d - m
                    loss = -obj.mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    desc_str ="{} | log p(data) = {:.4f}, log p(model) = {:.4f}, diff = {:.4f}".format(itr,d.mean(), m.mean(),
                                                    (d - m).mean())
                    itr += 1
                    pbar.update(1)
                    if itr % args.print_every == 0:
                        pbar.set_description(desc_str,refresh=True)
            torch.save(model.state_dict(), save_dir + f"{rbm_name}.pt")

    gt_samples = model.gibbs_sample(n_steps=args.gt_steps, n_samples=args.n_samples + args.n_test_samples, plot=True)
    kmmd = mmd.MMD(mmd.exp_avg_hamming, False)
    gt_samples, gt_samples2 = gt_samples[:args.n_samples], gt_samples[args.n_samples:]
    if plot is not None:
        cur_dir_pre = f'{args.save_dir}/{args.data}/zeroinit_{False}/rbm_iter_{args.rbm_train_iter}'
        plot("{}/ground_truth.png".format(cur_dir_pre), gt_samples2)
    opt_stat = kmmd.compute_mmd(gt_samples2, gt_samples)
    # print("gt <--> gt log-mmd", opt_stat, opt_stat.log10())

    new_samples = model.gibbs_sample(n_steps=0, n_samples=args.n_test_samples)

    log_mmds = {}
    log_mmds['gibbs'] = []
    ars = {}
    grad_f = {}
    grad_r = {}
    hops = {}
    ess = {}
    times = {}
    chains = {}
    chain = []
    sample_var = {}

    x0 = model.init_dist.sample((args.n_test_samples,)).to(device)
    temps = ['gwg','dula','dmala','cdula','cdmala']
    for temp in temps:

        if temp == 'dim-gibbs':
            sampler = samplers.PerDimGibbsSampler(args.n_visible)
        elif temp == "rand-gibbs":
            sampler = samplers.PerDimGibbsSampler(args.n_visible, rand=True)
        elif "bg-" in temp:
            block_size = int(temp.split('-')[1])
            sampler = block_samplers.BlockGibbsSampler(args.n_visible, block_size)
        elif "hb-" in temp:
            block_size, hamming_dist = [int(v) for v in temp.split('-')[1:]]
            sampler = block_samplers.HammingBallSampler(args.n_visible, block_size, hamming_dist)
        elif temp == "gwg":
            sampler = samplers.DiffSampler(args.n_visible, 1,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2.)
        elif "gwg-" in temp:
            n_hops = int(temp.split('-')[1])
            sampler = samplers.MultiDiffSampler(args.n_visible, 1, approx=True, temp=2., n_samples=n_hops)
        elif temp == "dmala":
            sampler = samplers.LangevinSampler(args.n_visible, 1,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2., step_size=0.2, mh=True)
        elif temp == "dula":
            sampler = samplers.LangevinSampler(args.n_visible, 1,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2., step_size=0.1, mh=False)
        elif temp == "cdula":
            sampler = samplers.cLangevinSampler(args.n_visible, 1,
                                             fixed_proposal=False, approx=True, multi_hop=False, temp=2., step_size1=0.2,step_size2=0.04, mh=False)
        elif temp == "cdmala":
            sampler = samplers.cLangevinSampler(args.n_visible, 1,
                                             fixed_proposal=False, approx=True, multi_hop=False, temp=2., step_size1=0.2,step_size2=0.04, mh=True)
        elif 'cdula-' in temp:
            nc = int(temp.split('-')[1])
            sampler = samplers.MulticLangevinSampler(args.n_visible, 1,
                                             fixed_proposal=False, approx=True, multi_hop=False, temp=2., step_size1=0.2,step_size2=0.03,mh = False ,nc = nc,decay=True)
        elif 'cdmala-' in temp:
            nc = int(temp.split('-')[1])
            sampler = samplers.MulticLangevinSampler(args.n_visible, 1,
                                             fixed_proposal=False, approx=True, multi_hop=False, temp=2., step_size1=0.2,step_size2=0.03,mh=True,nc=nc,decay=True)
        else:
            raise ValueError("Invalid sampler...")


        x = x0.clone().detach()

        log_mmds[temp] = []
        ars[temp] = []
        grad_f[temp] = []
        grad_r[temp] = []
        hops[temp] = []
        times[temp] = []
        chain = []
        sample_var[temp] = []
        cur_time = 0.
        for i in tqdm.tqdm(range(args.n_steps),desc = f"{temp}"):
            # do sampling and time it
            st = time.time()
            xhat = sampler.step(x.detach(), model).detach()
            cur_time += time.time() - st

            # compute hamming dist
            cur_hops = (x != xhat).float().sum(-1).mean().item()

            # update trajectory
            x = xhat

            if i % args.subsample == 0:
                if args.ess_statistic == "dims":
                    chain.append(x.cpu().numpy()[0][None])
                else:
                    xc = x[0][None]
                    h = (xc != gt_samples).float().sum(-1)
                    chain.append(h.detach().cpu().numpy()[None])

            if i % args.viz_every == 0 and plot is not None:
                plot("{}/temp_{}_samples_{}.png".format(args.save_dir, temp, i), x)

            if i % args.print_every == 0:
                hard_samples = x
                cur_sample_var = torch.var(hard_samples,dim = 1)
                mean_var = torch.mean(cur_sample_var).detach().cpu().numpy()
                sample_var[temp].append(mean_var)
                stat = kmmd.compute_mmd(hard_samples, gt_samples)
                log_stat = stat.log().item()
                log_mmds[temp].append(log_stat)
                times[temp].append(cur_time)
                hops[temp].append(cur_hops)
                if temp in ['cdmala','cdmala-2']:
                    grad_f[temp].append(np.mean(sampler.gfterm))
                    grad_r[temp].append(np.mean(sampler.grterm))
                if 'm' in temp:
                    ars[temp].append(np.mean(sampler.a_s))
        chain = np.concatenate(chain, 0)
        ess[temp] = get_ess(chain, args.burn_in)
        chains[temp] = chain
        print("ess = {} +/- {}".format(ess[temp].mean(), ess[temp].std()))
        print("mean hops = {}".format(np.mean(hops[temp])))
        print("mean time = {}".format(np.mean(times[temp])))
        print("final logmmd = {}".format(log_mmds[temp][-1]))
        if temp in ['dmala','cdmala','cdmala-2']:
            print("final ar = {}".format(ars[temp][-1]))
        # np.save("{}/rbm_sample_times_{}.npy".format(args.save_dir,temp),times[temp])
        # np.save("{}/rbm_sample_logmmd_{}.npy".format(args.save_dir,temp),log_mmds[temp])

    plt.clf()
    for temp in temps:
        plt.plot(log_mmds[temp], label="{}".format(temp))
        sample_var[temp] = np.array(sample_var[temp])
        plt.fill_between(range(len(log_mmds[temp])), log_mmds[temp]- sample_var[temp], log_mmds[temp]+sample_var[temp], alpha=0.2,label="_nolegend_")
    #plt.axhline(opt_stat.log10().detach().cpu().numpy(), color='black', linestyle='--', label="ground truth")
    x_ticks = [0,100, 200, 300, 400, 500]  
    x_labels = ['0','1000', '2000', '3000', '4000', '5000']  
    plt.xticks(x_ticks, x_labels)
    plt.grid(True)
    plt.legend(["GWG","DULA","DMALA","cDULA","cDMALA"],fontsize=17)
    plt.savefig("{}/logmmd.pdf".format(args.save_dir))

    plt.clf()
    for temp in temps:
        if 'm' in temp:
             plt.plot(ars[temp], label="{}".format(temp))
    plt.legend()
    plt.savefig("{}/ars.png".format(args.save_dir))

    plt.clf()
    for temp in temps:
        plt.plot(hops[temp], label="{}".format(temp))
    plt.legend()
    plt.savefig("{}/hops.png".format(args.save_dir))

    #plt.clf()
    #ess_list = [ess[temp] for temp in temps]
    #plt.boxplot(ess_list, labels=temps, showfliers=False)
    #plt.ylabel("ess")
    #plt.savefig("{}/ess.png".format(args.save_dir))

    #plt.clf()
    #plt.boxplot([ess[temp] / times[temp][-1] / (1. - args.burn_in) for temp in temps], labels=temps, showfliers=False)
    #plt.ylabel("ess_per_sec")
    #plt.savefig("{}/ess_per_sec.png".format(args.save_dir))
    #plt.clf()

    plt.clf()
    for temp in temps:
        plt.plot(times[temp],log_mmds[temp],label="{}".format(temp))
    plt.legend()
    plt.savefig("{}/time_vs_logmmd.png".format(args.save_dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="./figs/rbm_sample")
    parser.add_argument('--data', type=str, default='mnist',
                        choices=['mnist', 'random','emnist','omniglot','kmnist','caltech','fashion'])
    parser.add_argument('--n_steps', type=int, default=5000+1)
    parser.add_argument('--n_samples', type=int, default=500)
    parser.add_argument('--n_test_samples', type=int, default=100)
    parser.add_argument('--gt_steps', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=1234567)
 
    # rbm def
    parser.add_argument('--rbm_train_iter', type=int, default=1)#1 or 1000 or 3000
    parser.add_argument('--n_hidden', type=int, default=500)
    parser.add_argument('--n_visible', type=int, default=784)
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--viz_every', type=int, default=1000)
    # for rbm training
    parser.add_argument('--rbm_lr', type=float, default=.001)
    parser.add_argument('--cd', type=int, default=100)
    parser.add_argument('--img_size', type=int, default=28)
    parser.add_argument('--batch_size', type=int, default=100)
    # for ess 
    parser.add_argument('--subsample', type=int, default=1)
    parser.add_argument('--burn_in', type=float, default=.1)
    parser.add_argument('--ess_statistic', type=str, default="dims", choices=["hamming", "dims"])
    args = parser.parse_args()

    main(args)
