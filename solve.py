
import sympy
from sympy import Matrix,symbols,Symbol
from IPython.display import display
from sympy.solvers import solve
import copy

def get_coeffs_for_linear_equation(approx_change_in_f,t0,t1):
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
get_coeffs_for_linear_equation(approx_change_in_f,t0,t1)