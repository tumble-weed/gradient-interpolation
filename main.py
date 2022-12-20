
import sympy
from sympy import Matrix,symbols,Symbol
from IPython.display import display
from sympy.solvers import solve
import copy
from approximation import setup_approximations,get_approx_change_in_f
from solve import setup_change_in_f_linear_equation,get_coeffs_for_linear_equation
import torch
from utils import get_mask_th
from solve import estimate_abc
import numpy as np
# from inference import get_change_in_f_as_function
from approximation import get_approx_change_in_f_as_function
TODO = None
def setup(ndim,nthresh):
    """
    creates the symbols for s,t, fixed thresholds, observed_f
    """
    s = Symbol('s')
    g = Matrix(symbols( [[f'g{i}__t{t}' for t in range(1,(nthresh+1)+1)] for i in range(1,ndim+1)]))
    x = Matrix(symbols(f'x1:{ndim+1}'))
    t = Symbol('t')
    ti = symbols(f't1:{nthresh+1}')
    f_obs = Matrix(symbols( [f'fhat{t}' for t in range(1,nthresh+1+1)]
                       ))
    Mfixed = Matrix(symbols( [[f'mtilde{i}__t{t}' for t in range(1,nthresh+1)] for i in range(1,ndim+1)]))
    gfixed = Matrix(symbols( [[f'g{i}__t{t}' for t in range(1,nthresh+1)] for i in range(1,ndim+1)]))
    M = Matrix(symbols(f'm1:{ndim+1}'))
    return x,g,s,t,ti,f_obs,M,Mfixed,gfixed
#=============================
class GradientInterpolator():
    def __init__(self,ndim=3,nthresh=2):
        self.ndim = ndim
        self.nthresh = nthresh
        assert self.nthresh==2,'nthresh 2 is enough'
        x,g,s,t,ti,f_obs,M,Mfixed,gfixed = setup(ndim,nthresh)
        self.x,self.g,self.s,self.t,self.ti,self.f_obs,self.M,self.Mfixed,self.gfixed = x,g,s,t,ti,f_obs,M,Mfixed,gfixed
        nintervals = nthresh-1
        a,b,c = setup_approximations(nintervals,ndim)      
        self.a,self.b,self.c = a,b,c
        display(a,b,c)
        #TODO: does this need ti[1] and Mfixed[:,1]?
        assert a.shape == (ndim,1),'a shapre should be ndim,1'
        self.approx_change_in_f = get_approx_change_in_f(a[:,0],b[:,0],c[:,0],
                            x,s,
                            M,Mfixed[:,0],Mfixed[:,1],
                            t,ti[0],ti[1])
    

    def setup_linear_system(self):
        lin_eq1 = setup_change_in_f_linear_equation(
            self.approx_change_in_f,
            # to_replace,
            self.gfixed[:,0],self.gfixed[:,1],
            self.a[:,0],self.b[:,0],self.c[:,0],
            self.M,self.Mfixed[:,0],self.Mfixed[:,1])
        coeffs,constants = get_coeffs_for_linear_equation(lin_eq1,
                                self.a[:,0],
                                self.s,self.x,
                                self.Mfixed[:,:2],
                                self.ti[0],self.ti[1])        

        m_at_0,m_at_1 = self.Mfixed[:,0],self.Mfixed[:,1]
        g_at_0,g_at_1 = self.g[:,0],self.g[:,1]
        coeffs_for_lambdify = copy.deepcopy(coeffs)
        print('how does get_multiplier_of_a_ and lhs_constant_ not require access to s and t?')
        get_multiplier_of_a_ = sympy.lambdify([m_at_0[0],m_at_1[0]],coeffs_for_lambdify[0])
        get_lhs_constant_= sympy.lambdify([m_at_0[0],m_at_1[0],g_at_0[0],g_at_1[0]],constants[0])            
        self.get_multiplier_of_a_ = get_multiplier_of_a_
        self.get_lhs_constant_ = get_lhs_constant_
        return get_multiplier_of_a_,get_lhs_constant_

    def solve(self,g_at_0_,g_at_1_,f0_,f1_,m_at_0_,m_at_1_):
        get_multiplier_of_a_,get_lhs_constant_ = self.setup_linear_system()
        estimated_a,estimated_b,estimated_c = estimate_abc(m_at_0_,m_at_1_,g_at_0_,g_at_1_,f0_,f1_,get_multiplier_of_a_,get_lhs_constant_)
        return estimated_a,estimated_b,estimated_c

    def get_change_in_f_as_function(self):
        self.get_change_in_f_ = get_approx_change_in_f_as_function(
                        self.approx_change_in_f,
                          self.a,self.b,self.c,
                           self.x,self.s,
                           self.Mfixed[:,0],self.ti[0],self.ti[1])
    def get_approx_df_by_dM(self):
        from approximation import get_approx_df_by_dM
        approx_df_by_dM = get_approx_df_by_dM(self.M,self.a,self.b,self.c)
        get_approx_df_by_dM_ = sympy.lambdify([self.M[0],self.a[0],self.b[0],self.c[0]],approx_df_by_dM[0])
        return get_approx_df_by_dM_
    

#=========================================================================

ndim = 250*250
nthresh = 2
if True:
    f_obs_ = torch.randn(nthresh)
    g_ = torch.randn(ndim,nthresh)
    x_ = torch.randn(ndim)
else:
    f_obs_ = torch.zeros(nthresh).double()
    g_ = torch.zeros(ndim,nthresh).double()
    f_obs_[1]  = 1
    g_[:,-1] = 1
    x_ = torch.zeros(ndim).double()
    x_[1] = 1
Tfixed_ = torch.tensor([0.,1.])[None,:].double()
s_ = 1.
Mfixed_ = get_mask_th(x_.unsqueeze(-1),s_,Tfixed_)
assert Mfixed_.shape == (ndim,nthresh)
grad_interpolator = GradientInterpolator()
grad_interpolator.setup_linear_system()

# grad_interpolator.get_multiplier_of_a_(Mfixed_[:,0],Mfixed_[:,1])
# grad_interpolator.get_lhs_constant_(Mfixed_[:,0],Mfixed_[:,1],g_[:,0],g_[:,1])

estimated_a,estimated_b,estimated_c = grad_interpolator.solve(g_[:,0],g_[:,1],f_obs_[0],f_obs_[1],Mfixed_[:,0],Mfixed_[:,1])
assert isinstance(estimated_a,torch.Tensor)
assert isinstance(estimated_b,torch.Tensor)

grad_interpolator.get_change_in_f_as_function()
obs_change_in_f_ = f_obs_[-1] - f_obs_[0]
integrate_grad_wrt_x = estimated_change_in_f_ = grad_interpolator.get_change_in_f_(
        estimated_a,estimated_b,estimated_c,
        x_,s_,
        Mfixed_[:,0],
        0,1
    
)

print('difference between estimated and observed change in f',(estimated_change_in_f_.sum()-obs_change_in_f_).abs().sum())
# import IPython;IPython.embed()

get_approx_df_by_dM_ = grad_interpolator.get_approx_df_by_dM()
estimated_g_at_0_ = get_approx_df_by_dM_(Mfixed_[:,0]-Mfixed_[:,0],estimated_a,estimated_b,estimated_c)
estimated_g_at_1_ = get_approx_df_by_dM_(Mfixed_[:,1]-Mfixed_[:,0],estimated_a,estimated_b,estimated_c)
print( (estimated_g_at_0_-g_[:,0]).abs().sum())
print( (estimated_g_at_1_-g_[:,1]).abs().sum())
import IPython;IPython.embed()
'''
import pdb;pdb.set_trace()
found_a = estimate_a(m_at_0,m_at_1,
               g_at_0,g_at_1,
               f0,f1)
'''
# display(g)
# display(f_obs)
# display(x)
# display(ti)