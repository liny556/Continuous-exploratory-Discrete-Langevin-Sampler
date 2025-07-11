import torch
import torch.nn as nn
import torch.distributions as dists
import GWG_release.utils as utils
import numpy as np

class ccLangevinSampler(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, multi_hop=False, fixed_proposal=False, temp=2., 
                 step_size1=0.2,step_size2 = 0.2, mh=True, decay=False,nc = 1):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.fixed_proposal = fixed_proposal
        self.multi_hop = multi_hop
        self.temp = temp
        self.step_size1 = step_size1
        self.step_size2 = step_size2
        #self.continuous = continuous 
        self.decay = decay
        self.nc = nc
        self.grad = lambda x, m: utils.gradient_function(x, m) / self.temp
        #self.test = test

        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp

        self.mh = mh
        self.a_s = []
        self.hops = []

    def ContinuousLangevin(self, x, model):
        x_cur = x
        nsteps = self.nc
        M = 1
        start = 0.0; end = self.step_size2
        if self.decay:
            alpha_schedule = np.ones(nsteps) * self.step_size2
            for rcounter in range(nsteps):
                cos_inner = np.pi*(rcounter%(nsteps//M))
                cos_inner /= nsteps//M
                cos_out = np.cos(cos_inner)+1
                alpha_schedule[rcounter] = start + (end-start)*cos_out/2
            alpha_schedule = torch.tensor(alpha_schedule)
            
        else:
            alpha_schedule = torch.ones(nsteps) * self.step_size2

        aa = []
        for i in range(nsteps):
            forward_delta = self.grad(x_cur, model)
            noise = torch.randn_like(x_cur)
            x_cur1 = x_cur + alpha_schedule[i] * forward_delta  + np.sqrt(alpha_schedule[i]) * noise
            aa.append(x_cur1)

        return aa

    def step(self, x, model):

        x_cur1 = x
        # x_cur2 = x_cur1
        
        m_terms = []
        prob_terms = []
        prob_terms2 = []
        la_terms = []  
        
        EPS = 1e-10
            
        x_cur2 = self.ContinuousLangevin(x_cur1, model)

        for i in x_cur2:
            forward_delta = self.grad(i, model)
            term2 = 1./(2*self.step_size1) * (1 - x_cur1 - i)**2
            
            term22= 1./(2*self.step_size1) * (x_cur1 - i)**2

            flip_prob = torch.exp(forward_delta * (1 - x_cur1- i) - term2)/(torch.exp(forward_delta * (1- x_cur1- i)
                                                                                                    -term2) + torch.exp(forward_delta * (x_cur1- i) -term22))
            flip_prob = flip_prob 

            #print(flip_prob)
            prob_terms.append(flip_prob)
        
        flip_prob = torch.stack(prob_terms).mean(0)
        rr = torch.rand_like(x_cur1)
        ind = (rr<flip_prob)*1
        x_delta = (1. - x_cur1) * ind + x_cur1 * (1. - ind)

        if self.mh:
                probs = flip_prob * ind + (1 - flip_prob) * (1. - ind)
                lp_for = torch.sum(torch.log(probs + EPS),dim=-1)## q(theta'|theta1)

                reverse_delta = self.grad(x_delta, model)
                forward_delta2 = self.grad(x_cur1, model)
                mean_forward = x_cur1  + self.step_size2 * forward_delta2
                mean_reverse = x_delta + self.step_size2 * reverse_delta
                
                for j in x_cur2:
                    reverse_delta = self.grad(j, model)
                    term2 = 1./(2*self.step_size1) * (1 - x_delta - j)**2
                    term22= 1./(2*self.step_size1) * (x_delta - j)**2
                    flip_prob = torch.exp(reverse_delta * (1 - x_delta- j) - term2)/(torch.exp(reverse_delta * (1 - x_delta - j)
                                                                                                    -term2) + torch.exp(reverse_delta * (x_delta - j)
                                                                                                    -term22))
                    probs = flip_prob*ind + (1 - flip_prob) * (1. - ind)
                    lp_rev = torch.sum(torch.log(probs + EPS),dim=-1)# q(theta|theta1)
                    lp_reverse = (-1./(2 * self.step_size2) * (j - mean_reverse)**2).sum(-1) + lp_rev
                    lp_forward = (-1./(2 * self.step_size2) * (j - mean_forward)**2).sum(-1) + lp_for
                    # q(theta|theta1)q(theta1|theta')
                    # q(theta'|theta1)q(theta1|theta)
                    m_term = (model(x_delta).squeeze() - model(x_cur1).squeeze())
                    la = m_term + lp_reverse - lp_forward
                    la_terms.append(la.exp())
                

                la = torch.stack(la_terms).mean(0)
                a = (la > torch.rand_like(la)).float()
                self.a_s.append(a.mean().item())
                x_cur = x_delta * a[:, None] + x_cur1 * (1. - a[:, None])
        else:
                x_cur = x_delta 

        return x_cur


class cLangevinSampler(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, multi_hop=False, fixed_proposal=False, temp=2., 
                 step_size1=0.2,step_size2 = 0.2, mh=True, nc = 1, test = False):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.fixed_proposal = fixed_proposal
        self.multi_hop = multi_hop
        self.temp = temp
        self.step_size1 = step_size1
        self.step_size2 = step_size2
        self.nc = nc
        self.grad = lambda x, m: utils.gradient_function(x, m) / self.temp
        self.test = test

        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp

        self.mh = mh
        self.a_s = []
        self.hops = []

    def ContinuousLangevin(self, x, model):
        x_cur = x
        nsteps = self.nc
        alpha_schedule = torch.ones(nsteps) * self.step_size2

        aa = []
        for i in range(nsteps):
            forward_delta = self.grad(x_cur, model)
            noise = torch.randn_like(x_cur)
            x_cur1 = x_cur + alpha_schedule[i] * forward_delta + np.sqrt(alpha_schedule[i]) * noise
            aa.append(x_cur1)

        return aa

    def step(self, x, model):

        x_cur1 = x
        
        m_terms = []
        prob_terms = []
        prob_terms2 = []
        
        EPS = 1e-10
            
        x_cur2 = self.ContinuousLangevin(x_cur1, model)

        for i in x_cur2:
            forward_delta = self.grad(i, model)
                #print(forward_delta)
            term2 = 1./(2*self.step_size1) * (1 - x_cur1 - i)**2
            term22= 1./(2*self.step_size1) * (x_cur1 - i)**2
            if self.test:
                flip_prob = torch.exp(forward_delta * (1 - x_cur1- i) - term2)/(torch.exp(forward_delta * (1- x_cur1- i)
                                                                                                    -term2) + torch.exp(forward_delta * (x_cur1- i)
                                                                                                    -term22)+1)
            else:
                flip_prob = torch.exp(forward_delta * (1 - x_cur1- i) - term2)/(torch.exp(forward_delta * (1- x_cur1- i)
                                                                                                    -term2) + torch.exp(forward_delta * (x_cur1- i)
                                                                                                    -term22))
            prob_terms.append(flip_prob)
        
        rr = torch.rand_like(i)
        flip_prob = torch.stack(prob_terms).mean(0)

        ind = (rr<flip_prob)*1
        x_delta = (1. - x_cur1)*ind + x_cur1 * (1. - ind)

        if self.mh:
                probs = flip_prob * ind + (1 - flip_prob) * (1. - ind)
                lp_forward = torch.sum(torch.log(probs+EPS),dim=-1)
                x_cur2 = self.ContinuousLangevin(x_delta, model)

                for j in x_cur2:
                    reverse_delta = self.grad(j, model)
                    term2 = 1./(2*self.step_size1) * (1 - x_delta - j)**2
                    term22= 1./(2*self.step_size1) * (x_delta - j)**2
                    if self.test:
                       flip_prob = torch.exp(reverse_delta * (1 - x_delta- j) - term2)/(torch.exp(reverse_delta * (1 - x_delta - j)
                                                                                                    -term2) + torch.exp(reverse_delta * (x_delta - j)
                                                                                                    -term22)+1)
                    else:
                        flip_prob = torch.exp(reverse_delta * (1 - x_delta- j) - term2)/(torch.exp(reverse_delta * (1 - x_delta - j)
                                                                                                    -term2) + torch.exp(reverse_delta * (x_delta - j)
                                                                                                    -term22))
                    prob_terms2.append(flip_prob)

                flip_prob = torch.stack(prob_terms2).mean(0)

                probs = flip_prob * ind + (1 - flip_prob) * (1. - ind)
                lp_reverse = torch.sum(torch.log(probs+EPS),dim=-1)    
                m_term = (model(x_delta).squeeze() - model(x_cur1).squeeze())
                la = m_term + lp_reverse - lp_forward 
                #print(m_term.mean(), lp_reverse.mean(), lp_forward.mean())  
                a = (la.exp() > torch.rand_like(la)).float()
                self.a_s.append(a.mean().item())
                x_cur = x_delta * a[:, None] + x_cur1 * (1. - a[:, None])
        else:
                x_cur = x_delta 

        return x_cur

class LangevinSampler(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, multi_hop=False, fixed_proposal=False, temp=2., step_size=0.2, mh=True, test = True):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.fixed_proposal = fixed_proposal
        self.multi_hop = multi_hop
        self.temp = temp
        self.step_size = step_size  
        self.test = test

        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp

        self.mh = mh
        self.a_s = []
        self.hops = []

    def step(self, x, model):

        x_cur = x
        
        m_terms = []
        prop_terms = []
        
        EPS = 1e-10
        for i in range(self.n_steps):
            forward_delta = self.diff_fn(x_cur, model)
            term2 = 1./(2*self.step_size) # for binary {0,1}, the L2 norm is always 1        
            flip_prob = torch.exp(forward_delta-term2)/(torch.exp(forward_delta-term2)+1)

            rr = torch.rand_like(x_cur)
            ind = (rr<flip_prob)*1
            x_delta = (1. - x_cur)*ind + x_cur * (1. - ind)

            if self.mh:
                probs = flip_prob*ind + (1 - flip_prob) * (1. - ind)
                lp_forward = torch.sum(torch.log(probs+EPS),dim=-1)

                reverse_delta = self.diff_fn(x_delta, model)
                flip_prob = torch.exp(reverse_delta-term2)/(torch.exp(reverse_delta-term2)+1)

                probs = flip_prob*ind + (1 - flip_prob) * (1. - ind)
                lp_reverse = torch.sum(torch.log(probs+EPS),dim=-1)

                m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
                la = m_term + lp_reverse - lp_forward
                
                a = (la.exp() > torch.rand_like(la)).float()
                self.a_s.append(a.mean().item())
                x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
            else:
                x_cur = x_delta

        return x_cur


# Gibbs-With-Gradients for binary data
class DiffSampler(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, multi_hop=False, fixed_proposal=False, temp=2., step_size=1.0):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.fixed_proposal = fixed_proposal
        self.multi_hop = multi_hop
        self.temp = temp
        self.step_size = step_size
        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp


    def step(self, x, model):

        x_cur = x
        a_s = []
        m_terms = []
        prop_terms = []

        if self.multi_hop:
            if self.fixed_proposal:
                delta = self.diff_fn(x, model)
                cd = dists.Bernoulli(probs=delta.sigmoid() * self.step_size)
                for i in range(self.n_steps):
                    changes = cd.sample()
                    x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)
                    la = (model(x_delta).squeeze() - model(x_cur).squeeze())
                    a = (la.exp() > torch.rand_like(la)).float()
                    x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
                    a_s.append(a.mean().item())
                self._ar = np.mean(a_s)
            else:
                for i in range(self.n_steps):
                    forward_delta = self.diff_fn(x_cur, model)
                    cd_forward = dists.Bernoulli(logits=(forward_delta * 2 / self.temp))
                    changes = cd_forward.sample()

                    lp_forward = cd_forward.log_prob(changes).sum(-1)

                    x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)


                    reverse_delta = self.diff_fn(x_delta, model)
                    cd_reverse = dists.Bernoulli(logits=(reverse_delta * 2 / self.temp))

                    lp_reverse = cd_reverse.log_prob(changes).sum(-1)

                    m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
                    la = m_term + lp_reverse - lp_forward
                    a = (la.exp() > torch.rand_like(la)).float()
                    x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
                    a_s.append(a.mean().item())
                    m_terms.append(m_term.mean().item())
                    prop_terms.append((lp_reverse - lp_forward).mean().item())
                self._ar = np.mean(a_s)
                self._mt = np.mean(m_terms)
                self._pt = np.mean(prop_terms)
        else:
            if self.fixed_proposal:
                delta = self.diff_fn(x, model)
                cd = dists.OneHotCategorical(logits=delta)
                for i in range(self.n_steps):
                    changes = cd.sample()

                    x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)
                    la = (model(x_delta).squeeze() - model(x_cur).squeeze())
                    a = (la.exp() > torch.rand_like(la)).float()
                    x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
                    a_s.append(a.mean().item())
                self._ar = np.mean(a_s)
            else:
                for i in range(self.n_steps):
                    forward_delta = self.diff_fn(x_cur, model)
                    cd_forward = dists.OneHotCategorical(logits=forward_delta)
                    changes = cd_forward.sample()

                    lp_forward = cd_forward.log_prob(changes)

                    x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)

                    reverse_delta = self.diff_fn(x_delta, model)
                    cd_reverse = dists.OneHotCategorical(logits=reverse_delta)

                    lp_reverse = cd_reverse.log_prob(changes)

                    m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
                    la = m_term + lp_reverse - lp_forward
                    a = (la.exp() > torch.rand_like(la)).float()
                    x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])

        return x_cur


# Gibbs-With-Gradients variant which proposes multiple flips per step
class MultiDiffSampler(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, temp=1., n_samples=1):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.temp = temp
        self.n_samples = n_samples
        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp
        self.a_s = []
        self.hops = []

    def step(self, x, model):

        x_cur = x
        a_s = []
        m_terms = []
        prop_terms = []

        for i in range(self.n_steps):
            forward_delta = self.diff_fn(x_cur, model)
            cd_forward = dists.OneHotCategorical(logits=forward_delta)
            changes_all = cd_forward.sample((self.n_samples,))

            lp_forward = cd_forward.log_prob(changes_all).sum(0)

            changes = (changes_all.sum(0) > 0.).float()

            x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)
            # self._phops = (x_delta != x).float().sum(-1).mean().item()
            cur_hops = (x_cur[0] != x_delta[0]).float().sum(-1).item()
            self.hops.append(cur_hops)

            reverse_delta = self.diff_fn(x_delta, model)
            cd_reverse = dists.OneHotCategorical(logits=reverse_delta)

            lp_reverse = cd_reverse.log_prob(changes_all).sum(0)

            m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
            la = m_term + lp_reverse - lp_forward
            a = (la.exp() > torch.rand_like(la)).float()
            x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
            self.a_s.append(a.mean().item())
            m_terms.append(m_term.mean().item())
            prop_terms.append((lp_reverse - lp_forward).mean().item())
        self._ar = np.mean(a_s)
        self._mt = np.mean(m_terms)
        self._pt = np.mean(prop_terms)
        # print(self._ar)
        self._hops = (x != x_cur).float().sum(-1).mean().item()
        return x_cur


class PerDimGibbsSampler(nn.Module):
    def __init__(self, dim, rand=False):
        super().__init__()
        self.dim = dim
        self.changes = torch.zeros((dim,))
        self.change_rate = 0.
        self.p = nn.Parameter(torch.zeros((dim,)))
        self._i = 0
        self._ar = 0.
        self._hops = 0.
        self._phops = 1.
        self.rand = rand

    def step(self, x, model):
        sample = x.clone()
        lp_keep = model(sample).squeeze()
        if self.rand:
            changes = dists.OneHotCategorical(logits=torch.zeros((self.dim,))).sample((x.size(0),)).to(x.device)
        else:
            changes = torch.zeros((x.size(0), self.dim)).to(x.device)
            changes[:, self._i] = 1.

        sample_change = (1. - changes) * sample + changes * (1. - sample)

        lp_change = model(sample_change).squeeze()

        lp_update = lp_change - lp_keep
        update_dist = dists.Bernoulli(logits=lp_update)
        updates = update_dist.sample()
        sample = sample_change * updates[:, None] + sample * (1. - updates[:, None])
        self.changes[self._i] = updates.mean()
        self._i = (self._i + 1) % self.dim
        self._hops = (x != sample).float().sum(-1).mean().item()
        self._ar = self._hops
        return sample

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0.

class PerDimMetropolisSampler(nn.Module):
    def __init__(self, dim, n_out, rand=False):
        super().__init__()
        self.dim = dim
        self.n_out = n_out
        self.changes = torch.zeros((dim,))
        self.change_rate = 0.
        self.p = nn.Parameter(torch.zeros((dim,)))
        self._i = 0
        self._j = 0
        self._ar = 0.
        self._hops = 0.
        self._phops = 0.
        self.rand = rand

    def step(self, x, model):
        if self.rand:
            i = np.random.randint(0, self.dim)
        else:
            i = self._i

        logits = []
        ndim = x.size(-1)

        for k in range(ndim):
            sample = x.clone()
            sample_i = torch.zeros((ndim,))
            sample_i[k] = 1.
            sample[:, i, :] = sample_i
            lp_k = model(sample).squeeze()
            logits.append(lp_k[:, None])
        logits = torch.cat(logits, 1)
        dist = dists.OneHotCategorical(logits=logits)
        updates = dist.sample()
        sample = x.clone()
        sample[:, i, :] = updates
        self._i = (self._i + 1) % self.dim
        self._hops = ((x != sample).float().sum(-1) / 2.).sum(-1).mean().item()
        self._ar = self._hops
        return sample

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0.

class PerDimLB(nn.Module):
    def __init__(self, dim, rand=False):
        super().__init__()
        self.dim = dim
        self.changes = torch.zeros((dim,))
        self.change_rate = 0.
        self.p = nn.Parameter(torch.zeros((dim,)))
        self._i = 0
        self._j = 0
        self._ar = 0.
        self._hops = 0.
        self._phops = 0.
        self.rand = rand

    def step(self, x, model):
        logits = []
        ndim = x.size(-1)
        fx = model(x).squeeze()
        for k in range(ndim):
            sample = x.clone()
            sample[:, k] = 1-sample[:, k] 
            lp_k = (model(sample).squeeze()-fx)/2.
            logits.append(lp_k[:, None])
        logits = torch.cat(logits, 1)
        Z_forward = torch.sum(torch.exp(logits),dim=-1)
        dist = dists.OneHotCategorical(logits=logits)
        changes = dist.sample()
        x_delta = (1. - x) * changes + x * (1. - changes)
        fx_delta = model(x_delta)
        logits = []
        for k in range(ndim):
            sample = x_delta.clone()
            sample[:, k] = 1-sample[:, k] 
            lp_k = (model(sample).squeeze()-fx_delta)/2.
            logits.append(lp_k[:, None])
        logits = torch.cat(logits, 1)
        Z_reverse = torch.sum(torch.exp(logits),dim=-1)
        la =  Z_forward/Z_reverse
        a = (la > torch.rand_like(la)).float()
        x = x_delta * a[:, None] + x * (1. - a[:, None])
        # a_s.append(a.mean().item())
        # self._ar = np.mean(a_s)
        return x

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0.


# Gibbs-With-Gradients for categorical data
class DiffSamplerMultiDim(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, temp=1.):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.temp = temp
        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function_multi_dim(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function_multi_dim(x, m) / self.temp

    def step(self, x, model):

        x_cur = x
        a_s = []
        m_terms = []
        prop_terms = []


        for i in range(self.n_steps):
            constant = 1.
            forward_delta = self.diff_fn(x_cur, model)
            
            # make sure we dont choose to stay where we are!
            forward_logits = forward_delta - constant * x_cur
            #print(forward_logits)
            cd_forward = dists.OneHotCategorical(logits=forward_logits.view(x_cur.size(0), -1))
            changes = cd_forward.sample()
            # print(x_cur.shape,forward_delta.shape,changes.shape)
            # exit()
            # compute probability of sampling this change
            lp_forward = cd_forward.log_prob(changes)
            # reshape to (bs, dim, nout)
            changes_r = changes.view(x_cur.size())
            # get binary indicator (bs, dim) indicating which dim was changed
            changed_ind = changes_r.sum(-1)
            # mask out cuanged dim and add in the change
            x_delta = x_cur.clone() * (1. - changed_ind[:, :, None]) + changes_r

            reverse_delta = self.diff_fn(x_delta, model)
            reverse_logits = reverse_delta - constant * x_delta
            cd_reverse = dists.OneHotCategorical(logits=reverse_logits.view(x_delta.size(0), -1))
            reverse_changes = x_cur * changed_ind[:, :, None]

            lp_reverse = cd_reverse.log_prob(reverse_changes.view(x_delta.size(0), -1))

            m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
            la = m_term + lp_reverse - lp_forward
            a = (la.exp() > torch.rand_like(la)).float()
            x_cur = x_delta * a[:, None, None] + x_cur * (1. - a[:, None, None])
            a_s.append(a.mean().item())
            m_terms.append(m_term.mean().item())
            prop_terms.append((lp_reverse - lp_forward).mean().item())
        self._ar = np.mean(a_s)
        self._mt = np.mean(m_terms)
        self._pt = np.mean(prop_terms)

        self._hops = (x != x_cur).float().sum(-1).sum(-1).mean().item()
        return x_cur


class GibbsSampler(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.changes = torch.zeros((dim,))
        self.change_rate = 0.
        self.p = nn.Parameter(torch.zeros((dim,)))

    def step(self, x, model):
        sample = x.clone()
        for i in range(self.dim):
            lp_keep = model(sample).squeeze()

            xi_keep = sample[:, i]
            xi_change = 1. - xi_keep
            sample_change = sample.clone()
            sample_change[:, i] = xi_change

            lp_change = model(sample_change).squeeze()

            lp_update = lp_change - lp_keep
            update_dist = dists.Bernoulli(logits=lp_update)
            updates = update_dist.sample()
            sample = sample_change * updates[:, None] + sample * (1. - updates[:, None])
            self.changes[i] = updates.mean()
        return sample

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0.
