
import sympy
from sympy import Matrix,symbols,Symbol
from IPython.display import display
from sympy.solvers import solve
import copy

ndim = 2
nthresh = 3
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
x,g,s,t,ti,f_obs,M,Mfixed,gfixed = setup(ndim,nthresh)
a,b,c = setup_approximations(2,2)
display(a,b,c)
display(a[:,1],b[:,1],c[:,1])

approx_change_in_f = get_approx_change_in_f(a[:,0],b[:,0],c[:,0],
                           x,s,
                           M,Mfixed[:,0],Mfixed[:,1],
                           t,ti[0],ti[1])
lin_eq1 = setup_change_in_f_linear_equation(approx_change_in_f,
                                            # to_replace,
                                            gfixed[:,0],gfixed[:,1],
                                        a[:,0],b[:,0],c[:,0],
                                        M,Mfixed[:,0],Mfixed[:,1])
# display(g)
# display(f_obs)
# display(x)
# display(ti)