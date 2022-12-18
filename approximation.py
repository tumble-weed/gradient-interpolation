
import sympy
from sympy import Matrix,symbols,Symbol
from IPython.display import display
from sympy.solvers import solve
import copy

def setup_approximations(n_intervals,ndim):
    a = Matrix(symbols( [[f'a{i}__{t}{t+1}' for t in range(n_intervals)] for i in range(1,ndim+1)]))
    b = Matrix(symbols( [[f'b{i}__{t}{t+1}' for t in range(n_intervals)] for i in range(1,ndim+1)]))
    c = Matrix(symbols( [[f'c{i}__{t}{t+1}' for t in range(n_intervals)] for i in range(1,ndim+1)]))

    # a = Matrix(symbols(f'a1:{ndim+1}'))
    # b = Matrix(symbols(f'b1:{ndim+1}'))
    # c = Matrix(symbols(f'c1:{ndim+1}'))
    return a,b,c
    pass
def get_approx_df_by_dM(M,a,b,c):
    return a.multiply_elementwise(M.applyfunc(lambda el:el**2)) + b.multiply_elementwise(M) + c

def get_approx_change_in_f(a,b,c,
                           x,s,
                           M,m_at_0,m_at_1,
                           t,t0,t1):
    ndim = x.shape[0]
    approx_df_by_dM = get_approx_df_by_dM(M-m_at_0,a,b,c)
    T = Matrix([t for _ in range(ndim)])
    m = get_mask(x,s,T)
    dm_by_dt = sympy.diff(m,t)
    integrand = approx_df_by_dM.subs(zip(M,m)).multiply_elementwise(dm_by_dt)
    approx_change_in_f = sympy.integrate(integrand,(t,t0,t1))
    return approx_change_in_f

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
        substituted[i,-1] = substituted[i,-1].subs(M[i],Mfixed[i,-1])    
    #===========================================================================
    return substituted

