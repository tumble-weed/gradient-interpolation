#%%
v = 1
#%%
v = 1
import itertools
import numpy as np
from numpy import polyval
import torch
def get_poly_coeff_from_roots(roots):
    n_roots = len(roots)
    order = n_roots
    n_coeffs = 1 + n_roots
    coeffs = []
    for n_constants in range(n_coeffs):
        # n_variable = (n_roots - n_constants)
        # assert n_variable>= 0 
        coeff_i = ((-1)**n_constants)*sum(
            [np.prod(C) if len(C) >0  else 1 for C in itertools.combinations(roots,n_constants)]
            )
        coeffs.append(coeff_i)
    coeffs = coeffs[::-1]
    return coeffs
if 'initial test':
    roots = [1,2,3]
    coeffs_d = get_poly_coeff_from_roots(roots)
    coeffs_poly =  [ c/(pow_x+1) for c,pow_x in zip(coeffs_d,range(len(coeffs_d)))] 
    coeffs_poly.insert(0,np.pi)
    print('coefficients are in increasing power of x')
    print(coeffs_d)
    print(coeffs_poly)
    # print('coefficients in decreasing power of x')
    coeffs_poly = coeffs_poly[::-1]
print(polyval(coeffs_poly,0))
#=========================================
if 'multiple variables':
    n_variables = 4
    n_roots = 3
    polys = []
    for i in range(n_variables):
        roots = np.random.uniform(size = (n_roots,))
        coeffs_d = get_poly_coeff_from_roots(roots)
        coeffs_poly =  [ c/(pow_x+1) for c,pow_x in zip(coeffs_d,range(len(coeffs_d)))]         
        coeffs_poly.insert(0,1.)
        coeffs_poly = coeffs_poly[::-1]
        polys.append(coeffs_poly)
x = np.random.randn(n_variables)
y = []
for xi,pi in zip(x,polys):
    yi = polyval(pi,xi)
    y.append(yi)
print(y)
# print(polys)
#=========================================
# polyval in torch
def polyval_t(x,coeffs):
    device = x.device
    powers = torch.arange(len(coeffs)-1,-1,-1).to(device)
    x_pow = x**powers
    out = (x_pow * coeffs).sum()
    return out
print(polyval_t(torch.tensor(x[0]),torch.tensor(polys[0])))
print(
    [polyval_t(torch.tensor(xx),torch.tensor(p)) for p,xx in zip(polys,x)]
      )

