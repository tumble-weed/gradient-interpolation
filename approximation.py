
import sympy
from sympy import Matrix,symbols,Symbol
from IPython.display import display
from sympy.solvers import solve
import copy
from utils import get_mask_th
from utils import get_mask
def setup_approximations(n_intervals,ndim):
    a = Matrix(symbols( [[f'a{i}__{t}{t+1}' for t in range(1,1+n_intervals)] for i in range(1,ndim+1)]))
    b = Matrix(symbols( [[f'b{i}__{t}{t+1}' for t in range(1,1+n_intervals)] for i in range(1,ndim+1)]))
    c = Matrix(symbols( [[f'c{i}__{t}{t+1}' for t in range(1,1+n_intervals)] for i in range(1,ndim+1)]))

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
                           t,t1,t2):
    ndim = x.shape[0]
    approx_df_by_dM = get_approx_df_by_dM(M-m_at_0,a,b,c)
    T = Matrix([t for _ in range(ndim)])
    m = get_mask(x,s,T)
    dm_by_dt = sympy.diff(m,t)
    integrand = approx_df_by_dM.subs(zip(M,m)).multiply_elementwise(dm_by_dt)
    approx_change_in_f = sympy.integrate(integrand,(t,t1,t2))
    return approx_change_in_f

def get_approx_change_in_f_as_function(
                        approx_change_in_f,
                          a,b,c,
                           x,s,
                           m_at_0,
                           t1,t2):
    approx_change_in_f_for_lambdify = approx_change_in_f.copy()
    approx_change_in_f_function = sympy.lambdify([
        a[0],b[0],c[0],
        x[0],s,
        m_at_0[0],
        t1,t2
        ],
        approx_change_in_f_for_lambdify[0])
    return approx_change_in_f_function
