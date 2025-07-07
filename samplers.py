import torch
import torch.nn as nn
import torch.distributions as dists
import utils 
import numpy as np
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

class MulticLangevinSampler(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, multi_hop=False, fixed_proposal=False, temp=2., 
                 step_size1=0.2,step_size2 = 0.02, mh=False, nc = 1,decay = True):
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
        self.decay = decay

        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp

        self.mh = mh
        self.a_s = []
        self.hops = []
        self.mterm = []
        self.gfterm = []
        self.grterm = []

    def ContinuousLangevin(self, x, model):
        x_cur = x
        nsteps = self.nc
        M = 1
        start = 0.0; end = self.step_size2
        if self.decay:
            alpha_schedule = np.ones(nsteps) * end
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
            x_cur = x_cur + alpha_schedule[i] * forward_delta
            x_cur = torch.clamp(x_cur, 0, 1)

        return x_cur

    def step(self, x, model):

        x_cur1 = x
        
        m_terms = []
        prob_terms = []
        prob_terms2 = []
        
        EPS = 1e-10
            
        x_cur2= self.ContinuousLangevin(x_cur1, model)
        for _ in range(self.n_steps):
            forward_delta = self.grad(x_cur2, model)
            term2 = 1./(2*self.step_size1) * (1 - x_cur1 - x_cur2)**2
            term22= 1./(2*self.step_size1) * (x_cur1 - x_cur2)**2
            flip_prob = torch.exp(forward_delta * (1 - x_cur1- x_cur2) - term2)/(torch.exp(forward_delta * (1- x_cur1- x_cur2)
                                                                                                    -term2) + torch.exp(forward_delta * (x_cur1- x_cur2)
                                                                                                    -term22))

            rr = torch.rand_like(x_cur1)
            ind = (rr < flip_prob)*1
            x_delta = (1. - x_cur1) * ind + x_cur1 * (1. - ind)

        return x_delta
      
class cLangevinSampler(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, multi_hop=False, fixed_proposal=False, temp=2., 
                 step_size1=0.2,step_size2 = 0.2, mh=True):
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
        self.grad = lambda x, m: utils.gradient_function(x, m) / self.temp

        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp

        self.mh = mh
        self.a_s = []
        self.hops = []
        self.mterm = []
        self.gfterm = []
        self.grterm = []

    def ContinuousLangevin(self, x, model):
        x_cur = x
        forward = self.grad(x_cur, model)  
        x_cur1 = x_cur + self.step_size2 * forward
        x_cur1 = torch.clamp(x_cur1, 0, 1)

        return x_cur1

    def step(self, x, model):

        x_cur1 = x
        
        m_terms = []
        prob_terms = []
        prob_terms2 = []
        
        EPS = 1e-10
            
        x_cur2 = self.ContinuousLangevin(x_cur1, model)
        for _ in range(self.n_steps):
            forward_delta = self.grad(x_cur2, model)
            term2 = 1./(2*self.step_size1) * (1 - x_cur1 - x_cur2)**2
            term22= 1./(2*self.step_size1) * (x_cur1 - x_cur2)**2
            flip_prob = torch.exp(forward_delta * (1 - x_cur1- x_cur2) - term2)/(torch.exp(forward_delta * (1- x_cur1- x_cur2)
                                                                                                    -term2) + torch.exp(forward_delta * (x_cur1- x_cur2)
                                                                                                    -term22))

            #diff = forward_delta - grad
            #self.gfterm.append(diff.mean().item())
            rr = torch.rand_like(x_cur1)
            ind = (rr < flip_prob)*1
            x_delta = (1. - x_cur1) * ind + x_cur1 * (1. - ind)

            if self.mh:
                probs = flip_prob * ind + (1 - flip_prob) * (1. - ind)
                lp_forward = torch.sum(torch.log(probs+EPS),dim=-1)
                x_cur2 = self.ContinuousLangevin(x_delta, model)
                reverse_delta = self.grad(x_cur2, model)
                term2 = 1./(2*self.step_size1) * (1 - x_delta - x_cur2)**2
                term22= 1./(2*self.step_size1) * (x_delta - x_cur2)**2
                flip_prob = torch.exp(reverse_delta * (1 - x_delta- x_cur2) - term2)/(torch.exp(reverse_delta * (1 - x_delta - x_cur2)
                                                                                                    -term2) + torch.exp(reverse_delta * (x_delta - x_cur2)
                                                                                                    -term22))
                #diff = reverse_delta - gradf
                #self.grterm.append(diff.mean().item())
                probs = flip_prob * ind + (1 - flip_prob) * (1. - ind)
                lp_reverse = torch.sum(torch.log(probs+EPS),dim=-1)    
                m_term = (model(x_delta).squeeze() - model(x_cur1).squeeze())
                la = m_term + lp_reverse - lp_forward 
                a = (la.exp() > torch.rand_like(la)).float()
                self.a_s.append(a.mean().item())
                x_cur = x_delta * a[:, None] + x_cur1 * (1. - a[:, None])
            else:
                x_cur = x_delta 
        return x_cur    
    
class LangevinSampler(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, multi_hop=False, fixed_proposal=False, temp=2., step_size=0.2, mh=True):
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

class LangevinSamplerOrdinal(nn.Module):
    def __init__(
        self,
        dim,
        bal,
        max_val=3,
        n_steps=10,
        multi_hop=False,
        temp=1.0,
        step_size=0.2,
        mh=True,
        device=None,
    ):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.0
        self._mt = 0.0
        self._pt = 0.0
        self._hops = 0.0
        self._phops = 0.0
        self.multi_hop = multi_hop
        self.temp = temp
        # rbm sampling: accpt prob is about 0.5 with lr = 0.2, update 16 dims per step (total 784 dims). ising sampling: accept prob 0.5 with lr=0.2
        # ising learning: accept prob=0.7 with lr=0.2
        # ebm: statistic mnist: accept prob=0.45 with lr=0.2
        self.a_s = []
        self.bal = bal
        self.mh = mh
        self.max_val = max_val  ### number of classes in each dimension
        self.step_size = (step_size * self.max_val) ** (self.dim**2)
        # self.step_size = step_size

    def get_grad(self, x, model):
        x = x.requires_grad_()
        out = model(x)
        gx = torch.autograd.grad(out.sum(), x)[0]
        return gx.detach()

    def _calc_logits(self, x_cur, grad):
        # creating the tensor of discrete values to compute the probabilities for
        batch_size = x_cur.shape[0]
        disc_values = torch.tensor([i for i in range(self.max_val)])[None, None, :]
        disc_values = disc_values.repeat((batch_size, self.dim, 1)).to(x_cur.device)
        term1 = torch.zeros((batch_size, self.dim, self.max_val))
        term2 = torch.zeros((batch_size, self.dim, self.max_val))
        x_expanded = x_cur[:, :, None].repeat((1, 1, self.max_val)).to(x_cur.device)
        grad_expanded = grad[:, :, None].repeat((1, 1, self.max_val)).to(x_cur.device)
        term1 = self.bal * grad_expanded * (disc_values - x_expanded)
        term2 = (disc_values - x_expanded) ** 2 * (1 / (2 * self.step_size))
        return term1 - term2

    def step(self, x, model, use_dula=False):
        """
        input x : bs * dim, every dim contains a integer of 0 to (num_cls-1)
        """
        x_cur = x
        m_terms = []
        prop_terms = []

        EPS = 1e-10
        for i in range(self.n_steps):
            # batch size X dim
            grad = self.get_grad(x_cur.float(), model)
            logits = self._calc_logits(x_cur, grad)
            cat_dist = torch.distributions.categorical.Categorical(logits=logits)
            x_delta = cat_dist.sample()

            if self.mh:
                lp_forward = torch.sum(cat_dist.log_prob(x_delta), dim=1)
                grad_delta = self.get_grad(x_delta.float(), model) / self.temp

                logits_reverse = self._calc_logits(x_delta, grad_delta)

                cat_dist_delta = torch.distributions.categorical.Categorical(
                    logits=logits_reverse
                )
                lp_reverse = torch.sum(cat_dist_delta.log_prob(x_cur), dim=1)

                m_term = model(x_delta).squeeze() - model(x_cur).squeeze()
                la = m_term + lp_reverse - lp_forward
                a = (la.exp() > torch.rand_like(la)).float()
                self.a_s.append(a.mean().item())
                if use_dula:
                    x_cur = x_delta
                else:
                    x_cur = x_delta * a[:, None] + x_cur * (1.0 - a[:, None])
            else:
                x_cur = x_delta
            x_cur = x_cur.long()
        return x_cur

class cLangevinSamplerOrdinal(nn.Module):
    def __init__(
        self,
        dim,
        bal,
        max_val=3,
        n_steps=10,
        multi_hop=False,
        temp=1.0,
        step_size=0.2,
        step_size_c = 0.02,
        mh=True,
        device=None,
    ):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.0
        self._mt = 0.0
        self._pt = 0.0
        self._hops = 0.0
        self._phops = 0.0
        self.multi_hop = multi_hop
        self.temp = temp
        self.a_s = []
        self.bal = bal
        self.mh = mh
        self.max_val = max_val  ### number of classes in each dimension
        self.step_size = (step_size * self.max_val) ** (self.dim**2)
        self.step_size_c = (step_size_c * self.max_val) ** (self.dim**2)
        # self.step_size = step_size

    def get_grad(self, x, model):
        x = x.requires_grad_()
        out = model(x)
        gx = torch.autograd.grad(out.sum(), x)[0]
        return gx.detach()

    def _calc_logits(self, x_cur, grad):
        # creating the tensor of discrete values to compute the probabilities for
        batch_size = x_cur.shape[0]
        disc_values = torch.tensor([i for i in range(self.max_val)])[None, None, :]
        disc_values = disc_values.repeat((batch_size, self.dim, 1)).to(x_cur.device)
        term1 = torch.zeros((batch_size, self.dim, self.max_val))
        term2 = torch.zeros((batch_size, self.dim, self.max_val))
        x_expanded = x_cur[:, :, None].repeat((1, 1, self.max_val)).to(x_cur.device)
        grad_expanded = grad[:, :, None].repeat((1, 1, self.max_val)).to(x_cur.device)
        term1 = self.bal * grad_expanded * (disc_values - x_expanded)
        term2 = (disc_values - x_expanded) ** 2 * (1 / (2 * self.step_size))
        return term1 - term2

    def step(self, x, model, use_dula=False):
        """
        input x : bs * dim, every dim contains a integer of 0 to (num_cls-1)
        """
        x_cur = x
        m_terms = []
        prop_terms = []

        EPS = 1e-10
        for i in range(self.n_steps):
            # batch size X dim
            grad = self.get_grad(x_cur.float(), model)
            x_1 = x_cur + self.step_size_c*.5*grad
            x_1 = torch.clamp(x_1,0,self.max_val-1)
            grad_1 = self.get_grad(x_1.float(),model)
            logits = self._calc_logits(x_1, grad_1)
            cat_dist = torch.distributions.categorical.Categorical(logits=logits)
            x_delta = cat_dist.sample()

            if self.mh:
                lp_forward = torch.sum(cat_dist.log_prob(x_delta), dim=1)
                grad_delta = self.get_grad(x_delta.float(), model) / self.temp
                x_2 = x_delta + self.step_size_c*.5*grad_delta
                x_2 = torch.clamp(x_2,0,self.max_val-1)
                grad_2 = self.get_grad(x_2.float(),model)
                logits_reverse = self._calc_logits(x_2, grad_2)

                cat_dist_delta = torch.distributions.categorical.Categorical(
                    logits=logits_reverse
                )
                lp_reverse = torch.sum(cat_dist_delta.log_prob(x_cur), dim=1)

                m_term = model(x_delta).squeeze() - model(x_cur).squeeze()
                la = m_term + lp_reverse - lp_forward
                a = (la.exp() > torch.rand_like(la)).float()
                self.a_s.append(a.mean().item())
                if use_dula:
                    x_cur = x_delta
                else:
                    x_cur = x_delta * a[:, None] + x_cur * (1.0 - a[:, None])
            else:
                x_cur = x_delta
            x_cur = x_cur.long()
        return x_cur, x_1
      
class PerDimGibbsSamplerOrd(nn.Module):
    def __init__(self, dim, max_val, rand=False):
        super().__init__()
        self.dim = dim
        self.changes = torch.zeros((dim,))
        self.change_rate = 0.0
        self.p = nn.Parameter(torch.zeros((dim,)))
        self._i = 0
        self._ar = 0.0
        self._hops = 0.0
        self._phops = 1.0
        self.rand = rand
        self.max_val = max_val

    def step(self, x, model):
        sample = x.clone()
        lp_keep = model(sample).squeeze()
        if self.rand:
            changes = (
                dists.OneHotCategorical(logits=torch.zeros((self.dim,)))
                .sample((x.size(0),))
                .to(x.device)
            )
        else:
            changes = torch.zeros((x.size(0), self.dim)).to(x.device)
            changes[:, self._i] = 1.0
        # need to calculate the energies of all the possible values
        sample_expanded = torch.repeat_interleave(sample, self.max_val, dim=0)
        values_to_test = torch.Tensor([[i] for i in range(self.max_val)]).repeat(
            (sample.size(0), 1)
        )
        sample_expanded[:, self._i] = values_to_test[:, 0].to(sample.device)
        energies = model(sample_expanded).squeeze()
        cat_dist = dists.categorical.Categorical(
            energies.reshape((sample.size(0), self.max_val)).exp()
        )
        new_coords = cat_dist.sample()
        sample[:, self._i] = new_coords
        self._i = (self._i + 1) % self.dim
        self._hops = (x != sample).float().sum(-1).mean().item()
        self._ar = self._hops
        return sample

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0.0

class PerDimMetropolisSamplerOrd(nn.Module):
    def __init__(self, dim, dist_to_test, max_val, rand=False):
        super().__init__()
        self.dim = dim
        self.changes = torch.zeros((dim,))
        self.change_rate = 0.0
        self.p = nn.Parameter(torch.zeros((dim,)))
        self._i = 0
        self._j = 0
        self._ar = 0.0
        self._hops = 0.0
        self._phops = 0.0
        self.rand = rand
        self.max_val = max_val
        self.dist_to_test = dist_to_test

    def step(self, x, model):
        if self.rand:
            i = np.random.randint(0, self.dim)
        else:
            i = self._i

        logits = []
        # ndim = x.size(-1)
        logits = torch.zeros((x.size(0), self.max_val)).to(x.device)
        len_values_to_test = 2 * self.dist_to_test + 1
        values_to_test = (
            torch.arange(-self.dist_to_test, self.dist_to_test + 1, step=1)[:, None]
            .repeat((x.size(0), 1))
            .to(x.device)
        )
        x_expanded = torch.repeat_interleave(x, len_values_to_test, dim=0)
        x_expanded[:, i] = torch.clamp(
            x_expanded[:, i] - values_to_test[:, 0], min=0, max=self.max_val - 1
        )
        coordinates_tested = x_expanded[:, i].reshape((x.size(0), len_values_to_test))
        energies = model(x_expanded).squeeze().reshape((x.size(0), len_values_to_test))
        logits[torch.arange(logits.size(0)).unsqueeze(1), coordinates_tested] = energies
        #
        #
        #
        # for k in range(-self.dist_to_test, self.dist_to_test + 1):
        #     sample = x.clone()
        #     # sample_i = torch.zeros((ndim,))
        #     # sample_i[k] = 1.0
        #     sample[:, i] = sample[:, i] + k
        #     # make sure values fall inside sample space
        #     sample[sample < 0] = 0
        #     sample[sample >= self.max_val] = self.max_val - 1
        #
        #     lp_k = model(sample).squeeze()
        #     logits[:, sample[:, self._i]] = lp_k
        dist = dists.Categorical(logits=logits)
        updates = dist.sample()
        sample = x.clone()
        sample[:, i] = updates
        self._i = (self._i + 1) % self.dim
        self._hops = ((x != sample).float().sum(-1) / 2.0).sum(-1).mean().item()
        self._ar = self._hops
        return sample

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0.0
