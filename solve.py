
import sympy
from sympy import Matrix,symbols,Symbol
from IPython.display import display
from sympy.solvers import solve
import copy
from approximation import get_approx_df_by_dM
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


def get_coeffs_for_linear_equation(approx_change_in_f,t0,t1):
    ndim = approx_change_in_f.shape[0]
    substituted = Matrix(copy.deepcopy(approx_change_in_f))
    # get coeffs of the change_in_f equations
    coeffs = []
    constants = []
    # y = []
    Tfixed = Matrix([t0,t1])
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
    # print('TODO:coeffs should not be referring to x?')
    display(coeffs)
    print('='*50)
    display(constants)

    return coeffs
