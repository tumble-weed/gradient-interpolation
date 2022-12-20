
import sympy
from sympy import Matrix,symbols,Symbol
from IPython.display import display
from sympy.solvers import solve
import copy
from typing import Any,List
import torch
import numpy as np
from approximation import get_approx_df_by_dM
import logging
dprint = logging.getLogger('debug')
dprint.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
dprint.addHandler(handler)
def setup_change_in_f_linear_equation(change_in_f,
                                      g_at_0,g_at_1,
                                     a,b,c,
                                      M,m_at_0,m_at_1):
    #===========================================================================
    ndim = g_at_0.shape[0]
    to_replace = {'c':g_at_0}
    b_of_a = []
    approx_df_by_dM = get_approx_df_by_dM(M-m_at_0,a,b,c)
    for i in range(ndim):
        to_solve = approx_df_by_dM[i,-1].subs(c[i],g_at_0[i])
        to_solve = to_solve.subs(M[i],m_at_1[i])
        b_of_a.append(solve(to_solve - g_at_1[i],b[i]))
        # import pdb;pdb.set_trace()
    b_of_a = Matrix(b_of_a)
    to_replace['b'] = b_of_a    
    # import pdb;pdb.set_trace()
    #===========================================================================
    ndim = a.shape[0]
    substituted = Matrix(copy.deepcopy(change_in_f))
    for i in range(ndim):
        substituted[i,-1] = substituted[i,-1].subs(c[i],to_replace['c'][i])
        substituted[i,-1] = substituted[i,-1].subs(b[i],to_replace['b'][i])
        # substituted[i,-1] = substituted[i,-1].subs(M[i],Mfixed[i,-1])    
        substituted[i,-1] = substituted[i,-1].subs(M[i],m_at_1[i])    
    #===========================================================================
    return substituted

'''
def get_coeffs_for_linear_equation(approx_change_in_f,
                                   a,
                                   s,x,
                                #    m_at_0,m_at_1,
                                   Mfixed,
                                   t1,t2):
    ndim = approx_change_in_f.shape[0]
    nthresh = 2
    substituted = Matrix(copy.deepcopy(approx_change_in_f))
    # get coeffs of the change_in_f equations
    coeffs = []
    constants = []
    # y = []
    Tfixed = Matrix([t1,t2])
    # Mfixed = [m_at_0,m_at_1]
    for i in range(1,ndim):
        term = substituted[i]
        term = copy.deepcopy(term)
        constant_i = term.subs(a[i],0)

        coeff_i = term.subs(a[i],1) - constant_i
        # display(coeff_i)
        print('='*50)
        # for j in range(ndim):
        j=i
        for k in range(nthresh):
            coeff_i = coeff_i.subs(
                sympy.exp(-(s*(x[j]-Tfixed[k]))),
                -1 + 1./Mfixed[j,k]
            )
            constant_i = constant_i.subs(
                sympy.exp(-(s*(x[j]-Tfixed[k]))),
                -1 + 1./Mfixed[j,k]
            )
        import pdb;pdb.set_trace()
        coeff_i = coeff_i.simplify(rational=True)
        constant_i = constant_i.simplify(rational=True)
        
        coeffs.append(coeff_i)
        constants.append(constant_i)
        # import pdb;pdb.set_trace()
        # break
    # print('TODO:coeffs should not be referring to x?')
    display(coeffs)
    print('='*50)
    display(constants)

    return coeffs
'''
def get_coeffs_for_linear_equation(substituted,
                                   a,
                                   s,x,
                                #    m_at_0,m_at_1,
                                   Mfixed,
                                   t1,t2):
    substituted = Matrix(copy.deepcopy(substituted))
    ndim = substituted.shape[0]
    nthresh = 2    
    ##############################################
    # manually find coefficients
    ##############################################
    coeffs = []
    constants = []
    y = []
    Tfixed = Matrix([t1,t2])
    for i in range(ndim):
        term = substituted[i]
        term = copy.deepcopy(term)
        constant_i = term.subs(a[i],0)

        coeff_i = term.subs(a[i],1) - constant_i
        display(coeff_i)
        print('='*50)
        for j in range(ndim):
            for k in range(nthresh):
                coeff_i = coeff_i.subs(
                    sympy.exp(-(s*(x[j]-Tfixed[k]))),
                    -1 + 1./Mfixed[j,k]
                )
                constant_i = constant_i.subs(
                    sympy.exp(-(s*(x[j]-Tfixed[k]))),
                    -1 + 1./Mfixed[j,k]
                )
        coeff_i = coeff_i.simplify(rational=True)
        constant_i = constant_i.simplify(rational=True)
        coeffs.append(coeff_i)
        constants.append(constant_i)
        # break
    print('TODO:coeffs should not be referring to x?')
    # import pdb;pdb.set_trace()
    print('values of coeffs and constants verified against gradient_interpolation5.py')
    display(coeffs)
    print('='*50)
    display(constants)
    # import pdb;pdb.set_trace()
    return coeffs,constants

def estimate_a(
    # coeffs,constants,
    get_multiplier_of_a_,get_lhs_constant_,
    m_at_0,m_at_1,
    g_at_0,g_at_1,
    f0,f1):
    #=============================================================
    #=============================================================
    # coeffs_for_lambdify = copy.deepcopy(coeffs)
    # get_multiplier_of_a_ = sympy.lambdify([m_at_0,m_at_1],coeffs_for_lambdify)
    # get_lhs_constant_= sympy.lambdify([m_at_0,m_at_1,g_at_0,g_at_1],constants[0])    
    #=============================================================
    del_f = f1 - f0
    lhs_constants = get_lhs_constant_(m_at_0,m_at_1,g_at_0,g_at_1) 
    rhs = del_f - lhs_constants.sum()
    dprint.debug(f'm_at_0,m_at_1 {m_at_0.numpy()},{m_at_1.numpy()}')
    multipliers = get_multiplier_of_a_(m_at_0,m_at_1)
    dprint.debug(f'multipliers {multipliers.numpy()}')
    # import pdb;pdb.set_trace()
    assert multipliers.ndim == 1
    A = multipliers[None,:]
    At = A.T
    dprint.debug(f'A {A.numpy()}',)
    # import pdb;pdb.set_trace()
    AAt = torch.einsum('ij,jk->ik',A,At)
    AAt = torch.atleast_2d(AAt)
    dprint.debug(f'AAt {AAt.numpy()}')
    AAt_inv = torch.linalg.inv(AAt)
    dprint.debug(f'AAt_inv {AAt_inv.numpy()}')
    # AAt_inv = 1/AAt
    rhs = torch.tensor([[rhs]])
    found_a = torch.einsum('ij,jk,kl->il',At,AAt_inv,rhs)
    found_a = found_a.squeeze()
    return found_a

# def estimate_abc(m_at_0:Any[List[torch.Tensor,np.ndarray]],
#                  m_at_1:Any[List[torch.Tensor,np.ndarray]],
#                  g_at_0:Any[List[torch.Tensor,np.ndarray]],
#                  g_at_1:Any[List[torch.Tensor,np.ndarray]],
#                  f0:Any[List[torch.Tensor,np.ndarray]],
#                  f1:Any[List[torch.Tensor,np.ndarray]],
#                  get_multiplier_of_a_,get_lhs_constant_)->Any[List[torch.Tensor,np.ndarray]]:

def estimate_abc(m_at_0,
                 m_at_1,
                 g_at_0,
                 g_at_1,
                 f0,
                 f1,
                 get_multiplier_of_a_,get_lhs_constant_):
    
    # lin_eq1 = setup_change_in_f_linear_equation(approx_change_in_f,
    #                                         # to_replace,
    #                                         g_at_0,g_at_1,
    #                                     a[:,0],b[:,0],c[:,0],
    #                                     M,m_at_0,m_at_1)    
    # coeffs,constants = get_coeffs_for_linear_equation(lin_eq1,
    #                            a[:,0],
    #                            s,x,
    #                            Mfixed[:,:2],
    #                            ti[0],ti[1])
    del_f = (f1 -f0)
    c = g_at_0
    a = estimate_a(get_multiplier_of_a_,get_lhs_constant_,m_at_0,m_at_1,g_at_0,g_at_1,f0,f1)
    b = (
        (g_at_1 - c)/(m_at_1 - m_at_0)
        - 
        a * (m_at_1 - m_at_0)
    )
    return a,b,c
if False:
    def solve(a,s,x,Mfixed,ti,gfixed,f0,f1):
        # coeffs,constants = get_coeffs_for_linear_equation(lin_eq1,
        #                             a[:,0],
        #                             s,x,
        #                             Mfixed[:,:2],
        #                             ti[0],ti[1])

        # coeffs_for_lambdify = coeffs[0].copy()
        get_coeffs,get_constants = get_lin_eq_parameters(a,b,c,x,s,t,
                            M,Mfixed,
                            ti,gfixed)
        
        # get_multiplier_of_a_ = sympy.lambdify([Mfixed[0,0],Mfixed[0,1]],coeffs_for_lambdify)
        # get_lhs_constant_= sympy.lambdify([Mfixed[0,0],Mfixed[0,1],gfixed[0,0],gfixed[0,1]],constants[0])
        return estimate_abc(coeffs,constants,
                            Mfixed[:,0],Mfixed[:,1],
                            gfixed[:,0],gfixed[:,1],
                            f0,f1)