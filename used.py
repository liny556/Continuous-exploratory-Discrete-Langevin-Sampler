class rLangevinSampler(nn.Module):## dl clip
    def __init__(self, dim, n_steps=10, approx=False, multi_hop=False, fixed_proposal=False, temp=2., step_size=0.2, mh=True, var = .01):
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
        self.var = var

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
            #term2 = 1./(2*self.step_size) # for binary {0,1}, the L2 norm is always 1  
            eps = torch.randn_like(x_cur) *  self.var
            term = np.sqrt(4*self.step_size)  * eps * (1 - 2 * x_cur)    
            flip_prob = torch.exp(2 * self.step_size * forward_delta- 1 + term)/(torch.exp(2 * self.step_size * forward_delta- 1+term)+1)

            rr = torch.rand_like(x_cur)
            ind = (rr<flip_prob)*1
            x_delta = (1. - x_cur)*ind + x_cur * (1. - ind)

            if self.mh:
                probs = flip_prob*ind + (1 - flip_prob) * (1. - ind)
                lp_forward = torch.sum(torch.log(probs+EPS),dim=-1)

                reverse_delta = self.diff_fn(x_delta, model)
                eps = torch.randn_like(x_cur) *  self.var #?
                term = np.sqrt(4*self.step_size)* eps * (1 - 2 * x_delta)
                flip_prob = torch.exp(2 * self.step_size * reverse_delta - 1 + term)/(torch.exp(2*self.step_size * reverse_delta - 1 + term)+1)

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


class nLangevinSampler(nn.Module):## based nn
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
                term2 = 1./(2.*self.step_size) # for binary {0,1}, the L2 norm is always 1        
                flip_prob = torch.exp(forward_delta-term2)/(torch.exp(forward_delta-term2)+1)
        
                rr = torch.rand_like(x_cur)
                ind = (rr<flip_prob)*1
                x_1 = (1. - x_cur)*ind + x_cur * (1. - ind)
            
                mid_delta = self.diff_fn(x_1, model)
                term22 = 1./(2*self.step_size) # for binary {0,1}, the L2 norm is always 1  
                flip_prob_1 = torch.exp(mid_delta-term22)/(torch.exp(mid_delta-term22)+1)
                
                rr_1 = torch.rand_like(x_cur)
                ind_1 = (rr_1<flip_prob_1)* 1
                x_delta = (1. - x_1)*ind_1 + x_1 * (1. - ind_1)

                if self.mh:
                    probs =  flip_prob_1*ind_1 + (1 - flip_prob_1) * (1. - ind_1) + flip_prob*ind + (1 - flip_prob) * (1. - ind) 
                    lp_forward = torch.sum(torch.log(probs+EPS),dim=-1)

                    reverse_delta = self.diff_fn(x_delta, model)
                    flip_prob = torch.exp(reverse_delta-term2)/(torch.exp(reverse_delta-term2)+1)
                    #flip_prob_1 = torch.exp(mid_delta-term22)/(torch.exp(mid_delta-term22)+1)

                    probs = flip_prob_1*ind + (1 - flip_prob_1) * (1. - ind) + flip_prob*ind_1 + (1 - flip_prob) * (1. - ind_1) 
                    lp_reverse = torch.sum(torch.log(probs+EPS),dim=-1)

                    m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
                    la = m_term + lp_reverse - lp_forward
                                
                    a = (la.exp() > torch.rand_like(la)).float()
                    self.a_s.append(a.mean().item())
                    x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
                else:
                    x_cur = x_delta   

            return x_cur


class aLangevinSampler(nn.Module):#Like AMOGLOD
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

    def step(self, x, model, rho=0.):

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

            probs = flip_prob*ind + (1 - flip_prob) * (1. - ind)
            lp_forward = torch.sum(torch.log(probs+EPS),dim=-1)

            reverse_delta = self.diff_fn(x_delta, model)
            flip_prob = torch.exp(reverse_delta-term2)/(torch.exp(reverse_delta-term2)+1)

            probs = flip_prob*ind + (1 - flip_prob) * (1. - ind)
            lp_reverse = torch.sum(torch.log(probs+EPS),dim=-1)
            rho = rho + lp_reverse - lp_forward

            if self.mh:
                m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
                la = m_term + rho
                a = (la.exp() > torch.rand_like(la)).float()
                self.a_s.append(a.mean().item())
                x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
            else:
                x_cur = x_delta

        return x_cur , rho


class testLangevinSampler(nn.Module):##like reflected Langevin, it shows that it is unnecessary to change dimesion as many as possile.
    def __init__(self, dim, n_steps=10, approx=False, multi_hop=False, fixed_proposal=False, temp=2., 
                 step_size1=0.2,step_size2 = 0.2, mh=True, nc = 1, var = .01):
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
        self.var = var
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
        nsteps = self.nc
        alpha_schedule = torch.ones(nsteps) * self.step_size2

        aa = []
        for i in range(nsteps):
            forward_delta = self.grad(x_cur, model)
            noise = torch.randn_like(x_cur) * self.var
            x_cur1 = x_cur + alpha_schedule[i] * forward_delta + np.sqrt(alpha_schedule[i]) * noise
            x_cur1 = torch.clamp(x_cur1, 0.01, 0.99)
            #x_cur1 = 2.*x_cur11 - x_cur1
            aa.append(x_cur1)

        return aa,forward_delta

    def step(self, x, model):

        x_cur1 = x
        
        m_terms = []
        prob_terms = []
        prob_terms2 = []
        
        EPS = 1e-10
            
        x_cur2,grad = self.ContinuousLangevin(x_cur1, model)
        for _ in range(self.n_steps):
            for i in x_cur2:
                forward_delta = self.grad(i, model)
                term2 = 1./(2*self.step_size1) * (1 - x_cur1 - i)**2
                term22= 1./(2*self.step_size1) * (x_cur1 - i)**2
                flip_prob = torch.exp(forward_delta * (1 - x_cur1- i) - term2)/(torch.exp(forward_delta * (1- x_cur1- i)
                                                                                                    -term2) + torch.exp(forward_delta * (x_cur1- i)
                                                                                                    -term22))
                diff = grad- forward_delta
                self.gfterm.append(diff.mean().item())
                prob_terms.append(flip_prob)

            flip_prob = torch.stack(prob_terms).mean(0)
            #self.mterm.append(flip_prob.item())
            rr = torch.rand_like(i)
            ind = (rr < flip_prob)*1
            #print(ind.sum().item())
            x_delta = (1. - x_cur1) * ind + x_cur1 * (1. - ind)

            if self.mh:
                probs = flip_prob * ind + (1 - flip_prob) * (1. - ind)
                lp_forward = torch.sum(torch.log(probs+EPS),dim=-1)
                x_cur2,grad2 = self.ContinuousLangevin(x_delta, model)

                for j in x_cur2:
                    reverse_delta = self.grad(j, model)
                    term2 = 1./(2*self.step_size1) * (1 - x_delta - j)**2
                    term22= 1./(2*self.step_size1) * (x_delta - j)**2
                    flip_prob = torch.exp(reverse_delta * (1 - x_delta- j) - term2)/(torch.exp(reverse_delta * (1 - x_delta - j)
                                                                                                    -term2) + torch.exp(reverse_delta * (x_delta - j)
                                                                                                    -term22))
                    diff = grad2- reverse_delta
                    
                    self.grterm.append(diff.mean().item())
                    prob_terms2.append(flip_prob)

                flip_prob = torch.stack(prob_terms2).mean(0)

                probs = flip_prob * ind + (1 - flip_prob) * (1. - ind)
                lp_reverse = torch.sum(torch.log(probs+EPS),dim=-1)    
                m_term = (model(x_delta).squeeze() - model(x_cur1).squeeze())
                la = m_term + lp_reverse - lp_forward 
                #print(m_term.mean().item(), lp_reverse.mean().item(), lp_forward.mean().item())
                a = (la.exp() > torch.rand_like(la)).float()
                self.a_s.append(a.mean().item())
                x_cur = x_delta * a[:, None] + x_cur1 * (1. - a[:, None])
            else:
                x_cur = x_delta 
        return x_cur

class testLangevinSampler(nn.Module):##0.2 and 0.00004 no superior
    def __init__(self, dim, n_steps=10, approx=False, multi_hop=False, fixed_proposal=False, temp=2., 
                 step_size1=0.2,step_size2 = 0.2, mh=True, nc = 1, var = 1.):
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
        self.var = var
        self.grad = lambda x, m: utils.gradient_function(x, m) / self.temp

        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp

        self.mh = mh
        self.a_s = []
        self.hops = []
        self.mterm = []

    def ContinuousLangevin(self, x, model):
        x_cur = x
        nsteps = self.nc
        alpha_schedule = torch.ones(nsteps) * self.step_size2

        aa = []
        for i in range(nsteps):
            forward_delta = self.grad(x_cur, model)
            noise = torch.randn_like(x_cur) * self.var
            x_cur1 = x_cur + alpha_schedule[i] * forward_delta + np.sqrt(alpha_schedule[i]) * noise
            x_cur11 = torch.clamp(x_cur1, 0, 1)
            x_cur1 = 2.*x_cur11 - x_cur1
            aa.append(x_cur1)

        return aa

    def step(self, x, model):

        x_cur1 = x
        
        m_terms = []
        prob_terms = []
        prob_terms2 = []
        
        EPS = 1e-10
            
        x_cur2 = self.ContinuousLangevin(x_cur1, model)
        for _ in range(self.n_steps):
            for i in x_cur2:
                forward_delta = self.grad(i, model)
                term2 = 1./(2*self.step_size1) * (1 - x_cur1 - i)**2
                term22= 1./(2*self.step_size1) * (x_cur1 - i)**2
                flip_prob = torch.exp(forward_delta * (1 - x_cur1- i) - term2)/(torch.exp(forward_delta * (1- x_cur1- i)
                                                                                                    -term2) + torch.exp(forward_delta * (x_cur1- i)
                                                                                                    -term22))
                prob_terms.append(flip_prob)

            flip_prob = torch.stack(prob_terms).mean(0)
            rr = torch.rand_like(i)
            ind = (rr < flip_prob)*1
            #print(ind.sum().item())
            x_delta = (1. - x_cur1) * ind + x_cur1 * (1. - ind)

            if self.mh:
                probs = flip_prob * ind + (1 - flip_prob) * (1. - ind)
                lp_forward = torch.sum(torch.log(probs+EPS),dim=-1)
                x_cur2 = self.ContinuousLangevin(x_delta, model)

                for j in x_cur2:
                    reverse_delta = self.grad(j, model)
                    term2 = 1./(2*self.step_size1) * (1 - x_delta - j)**2
                    term22= 1./(2*self.step_size1) * (x_delta - j)**2
                    flip_prob = torch.exp(reverse_delta * (1 - x_delta- j) - term2)/(torch.exp(reverse_delta * (1 - x_delta - j)
                                                                                                    -term2) + torch.exp(reverse_delta * (x_delta - j)
                                                                                                    -term22))
                    prob_terms2.append(flip_prob)

                flip_prob = torch.stack(prob_terms2).mean(0)

                probs = flip_prob * ind + (1 - flip_prob) * (1. - ind)
                lp_reverse = torch.sum(torch.log(probs+EPS),dim=-1)    
                m_term = (model(x_delta).squeeze() - model(x_cur1).squeeze())
                la = m_term + lp_reverse - lp_forward 
                #print(m_term.mean().item(), lp_reverse.mean().item(), lp_forward.mean().item())
                a = (la.exp() > torch.rand_like(la)).float()
                self.a_s.append(a.mean().item())
                x_cur = x_delta * a[:, None] + x_cur1 * (1. - a[:, None])
            else:
                x_cur = x_delta 
        return x_cur

class bLangevinSampler(nn.Module):##cdmala several steps 
    def __init__(self, dim, n_steps=10, approx=False, multi_hop=False, fixed_proposal=False, temp=2., 
                 step_size1=0.2,step_size2 = 0.2, mh=True, nc = 1):
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
        term2 = 1./(2*self.step_size1)
        for _ in range(self.n_steps):
            for i in x_cur2:
                forward_delta = self.grad(i, model)
                flip_prob = torch.exp(forward_delta * (1 - x_cur1- i) - term2)/(torch.exp(forward_delta * (1- x_cur1- i)
                                                                                                    -term2) + torch.exp(forward_delta * (x_cur1- i)))
                prob_terms.append(flip_prob)

            flip_prob = torch.stack(prob_terms).mean(0)
            rr = torch.rand_like(i)
            ind = (rr < flip_prob)*1
            x_delta = (1. - x_cur1) * ind + x_cur1 * (1. - ind)

            if self.mh:
                probs = flip_prob * ind + (1 - flip_prob) * (1. - ind)
                lp_forward = torch.sum(torch.log(probs+EPS),dim=-1)
                x_cur2 = self.ContinuousLangevin(x_delta, model)

                for j in x_cur2:
                    reverse_delta = self.grad(j, model)
                    flip_prob = torch.exp(reverse_delta * (1 - x_delta- j) - term2)/(torch.exp(reverse_delta * (1 - x_delta - j)
                                                                                                    -term2) + torch.exp(reverse_delta * (x_delta - j)))
                    prob_terms2.append(flip_prob)

                flip_prob = torch.stack(prob_terms2).mean(0)

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


class hLangevinSampler(nn.Module):#it works but no guarantee
    def __init__(self, dim, n_steps=10, approx=False, multi_hop=False, fixed_proposal=False, temp=2., step_size=0.2, mh=True, beta = .7):
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
        self.beta = beta  

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
            flip_prob = torch.max(flip_prob, (flip_prob > self.beta).float())

            rr = torch.rand_like(x_cur)
            ind = (rr<flip_prob)*1
            x_delta = (1. - x_cur)*ind + x_cur * (1. - ind)

            if self.mh:
                probs = flip_prob*ind + (1 - flip_prob) * (1. - ind)
                lp_forward = torch.sum(torch.log(probs+EPS),dim=-1)

                reverse_delta = self.diff_fn(x_delta, model)
                flip_prob = torch.exp(reverse_delta-term2)/(torch.exp(reverse_delta-term2)+1)
                flip_prob = torch.max(flip_prob, (flip_prob > self.beta).float())

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


class sLangevinSampler(nn.Module): ##not stable althought it works
    def __init__(self, dim, n_steps=10, approx=False, multi_hop=False, fixed_proposal=False, temp=2., 
                 step_size1=0.2,step_size2 = 0.2, mh=True, nc = 1):
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
        #self.test = test

        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp

        self.mh = mh
        self.a_s = []
        self.hops = []
        self.mterm = []

    def ContinuousLangevin(self, x, model):
        x_cur = x
        nsteps = self.nc
        alpha_schedule = torch.ones(nsteps) * self.step_size2
        forward_delta = self.grad(x_cur, model)
        aa = []

        for i in range(nsteps):
            noise = torch.randn_like(x_cur)
            x_cur1 = x_cur  + np.sqrt(alpha_schedule[i]) * noise + alpha_schedule[i] * forward_delta
            x_cur1 = torch.clamp(x_cur1, 0, 1)## necessary
            aa.append(x_cur1)
 
        return aa, forward_delta.detach()
    
    def PreLangevin(self, x, forward_delta):
        term2 = 1./(2*self.step_size1)        
        flip_prob = torch.exp(forward_delta* -(2.* x-1)-term2)/(torch.exp(forward_delta * -(2.*x - 1)-term2)+1)
        rr = torch.rand_like(x)
        ind = (rr < flip_prob)*1 
        ind_1  = (flip_prob > 0.5)* 1
        #diff = (ind == ind_1).float()
        #print(diff.mean().item())
        return flip_prob, ind

    def step(self, x, model):

        x_cur1 = x
        
        m_terms = []
        prob_terms = []
        prob_terms2 = []
        
        EPS = 1e-10
            
        for _ in range(self.n_steps):
            x_cur2,forward_delta = self.ContinuousLangevin(x_cur1, model)
            flip_prob, ind = self.PreLangevin(x_cur1,forward_delta)
            #print(ind.sum().item())

            for i in x_cur2:
                forward_delta = self.grad(i, model)
                term_2 = 1./(2*self.step_size1) * (1 - x_cur1 - i)**2
                term22= 1./(2*self.step_size1) * (x_cur1 - i)**2
                flip_prob1 = torch.exp(forward_delta * (1 - x_cur1- i) - term_2)/(torch.exp(forward_delta * (1- x_cur1- i)
                                                                                                    -term_2) + torch.exp(forward_delta * (x_cur1- i)
                                                                                                    -term22))
                prob_terms.append(flip_prob1)

            flip_prob_c = torch.stack(prob_terms).mean(0)
            flip_prob = flip_prob_c * ind + flip_prob * (1. - ind)
            rr = torch.rand_like(i)
            ind = (rr < flip_prob)*1
            #print(ind.sum().item())
            x_delta = (1. - x_cur1) * ind + x_cur1 * (1. - ind)

            if self.mh:
                probs = flip_prob * ind + (1 - flip_prob) * (1. - ind)
                lp_forward = torch.sum(torch.log(probs+EPS),dim=-1)
                x_cur2,reverse_delta = self.ContinuousLangevin(x_delta, model)
                
                flip_prob, ind1 = self.PreLangevin(x_delta, reverse_delta)
                for j in x_cur2:
                    reverse_delta = self.grad(j, model)
                    term2 = 1./(2*self.step_size1) * (1 - x_delta - j)**2
                    term22= 1./(2*self.step_size1) * (x_delta - j)**2
                    flip_prob1 = torch.exp(reverse_delta * (1 - x_delta- j) - term2)/(torch.exp(reverse_delta * (1 - x_delta - j)
                                                                                                    -term2) + torch.exp(reverse_delta * (x_delta - j)
                                                                                                    -term22))
                    prob_terms2.append(flip_prob1)

                flip_prob_c = torch.stack(prob_terms2).mean(0)
                #######not` sure`######
                flip_prob = flip_prob_c * ind + flip_prob * (1. - ind)

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

# Proposal by Expectation but fail.
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
    

class nLangevinSampler(nn.Module):
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
            x_1 = (1. - x_cur)*ind + x_cur * (1. - ind)
            print(x_1)
            mid_delta = self.diff_fn(x_1, model)
            term22 = 1./(2.*self.step_size) # for binary {0,1}, the L2 norm is always 1        
            flip_prob_1 = torch.exp(mid_delta-term22)/(torch.exp(mid_delta-term22)+1)
            print(flip_prob_1)
            
            rr_1 = torch.rand_like(x_cur)
            ind_1 = (rr_1<flip_prob_1)* 1.
            x_delta = (1. - x_1)*ind_1 + x_1 * (1. - ind_1)

            if self.mh:
                probs = flip_prob*ind + (1 - flip_prob) * (1. - ind) + flip_prob_1*ind_1 + (1 - flip_prob_1) * (1. - ind_1)
                lp_forward = torch.sum(torch.log(probs+EPS),dim=-1)

                reverse_delta = self.diff_fn(x_delta, model)
                flip_prob = torch.exp(reverse_delta-term22)/(torch.exp(reverse_delta-term22)+1)
                flip_prob_1 = torch.exp(mid_delta-term2)/(torch.exp(mid_delta-term2)+1)

                probs = flip_prob*ind_1 + (1 - flip_prob) * (1. - ind_1) + flip_prob_1*ind + (1 - flip_prob_1) * (1. - ind)
                lp_reverse = torch.sum(torch.log(probs+EPS),dim=-1)

                m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
                la = m_term + lp_reverse - lp_forward
                print(m_term,lp_forward,lp_reverse)
                print(la.exp())                
                a = (la.exp() > torch.rand_like(la)).float()
                self.a_s.append(a.mean().item())
                x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
            else:
                x_cur = x_delta   

        return x_cur
    

class nLangevinSampler(nn.Module):
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
                term2 = 1./(2.*self.step_size) # for binary {0,1}, the L2 norm is always 1        
                flip_prob = torch.exp(forward_delta-term2)/(torch.exp(forward_delta-term2)+1)
        
                rr = torch.rand_like(x_cur)
                ind = (rr<flip_prob)*1
                x_1 = (1. - x_cur)*ind + x_cur * (1. - ind)
            
                mid_delta = self.diff_fn(x_1, model)
                term22 = 1./(2*self.step_size) # for binary {0,1}, the L2 norm is always 1  
                flip_prob_1 = torch.exp(mid_delta-term22)/(torch.exp(mid_delta-term22)+1)
                
                rr_1 = torch.rand_like(x_cur)
                ind_1 = (rr_1<flip_prob_1)* 1
                x_delta = (1. - x_1)*ind_1 + x_1 * (1. - ind_1)

                if self.mh:
                    probs = flip_prob*ind + (1 - flip_prob) * (1. - ind) + flip_prob_1*ind_1 + (1 - flip_prob_1) * (1. - ind_1)
                    lp_forward = torch.sum(torch.log(probs+EPS),dim=-1)

                    reverse_delta = self.diff_fn(x_delta, model)
                    flip_prob = torch.exp(reverse_delta-term2)/(torch.exp(reverse_delta-term2)+1)
                    flip_prob_1 = torch.exp(mid_delta-term22)/(torch.exp(mid_delta-term22)+1)

                    probs = flip_prob*ind_1 + (1 - flip_prob) * (1. - ind_1) + flip_prob_1*ind + (1 - flip_prob_1) * (1. - ind)
                    lp_reverse = torch.sum(torch.log(probs+EPS),dim=-1)

                    m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
                    la = m_term + lp_reverse - lp_forward
                                
                    a = (la.exp() > torch.rand_like(la)).float()
                    self.a_s.append(a.mean().item())
                    x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
                else:
                    x_cur = x_delta   

            return x_cur
        
class nLangevinSampler(nn.Module):
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
        prop_terms2 = []
        
        EPS = 1e-10
        for i in range(self.n_steps):
            forward_delta = self.diff_fn(x_cur, model)
            term2 = 1./(2.*self.step_size) # for binary {0,1}, the L2 norm is always 1     
            term22 = 1./(2.*self.step_size) # for binary {0,1}, the L2 norm is always 1     
            flip_prob = torch.exp(forward_delta-term2)/(torch.exp(forward_delta-term2)+1)
            for _ in range(2):
                rr = torch.rand_like(x_cur)
                ind = (rr<flip_prob)*1
                x_1 = (1. - x_cur)*ind + x_cur * (1. - ind)
                mid_delta = self.diff_fn(x_1, model)
                      
                flip_prob_1 = torch.exp(mid_delta-term22)/(torch.exp(mid_delta-term22)+1)
                prop_terms.append(flip_prob_1)
            flip_prob = torch.stack(prop_terms).mean(0)    
            rr_1 = torch.rand_like(x_cur)
            ind_1 = (rr_1<flip_prob)* 1.
            x_delta = (1. - x_cur)*ind_1 + x_cur * (1. - ind_1)

            if self.mh:
                probs = flip_prob*ind_1 + (1 - flip_prob) * (1. - ind_1)
                lp_forward = torch.sum(torch.log(probs+EPS),dim=-1)

                reverse_delta = self.diff_fn(x_delta, model)
                flip_prob = torch.exp(reverse_delta-term2)/(torch.exp(reverse_delta-term2)+1)
                for _ in range(2):
                    rr = torch.rand_like(x_cur)
                    ind = (rr<flip_prob)*1
                    x_1 = (1. - x_delta)*ind + x_delta * (1. - ind)
                    mid_delta = self.diff_fn(x_1, model)

                    flip_prob_1 = torch.exp(mid_delta-term22)/(torch.exp(mid_delta-term22)+1)
                    prop_terms2.append(flip_prob_1)
                flip_prob = torch.stack(prop_terms2).mean(0)

                probs = flip_prob*ind_1 + (1 - flip_prob) * (1. - ind_1)
                lp_reverse = torch.sum(torch.log(probs+EPS),dim=-1)

                m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
                la = m_term + lp_reverse - lp_forward             
                a = (la.exp() > torch.rand_like(la)).float()
                self.a_s.append(a.mean().item())
                x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
            else:
                x_cur = x_delta   

        return x_cur