import sympy
# from sympy import Matrix,symbols,Symbol
# from IPython.display import display
# from sympy.solvers import solve
import copy
import numpy as np
import torch
def sigmoid(x):
    return 1/(1+sympy.exp(-x))
def sigmoid_np(x):
    return 1/(1+np.exp(-x))
def get_mask(x,s,T):
    return (s*(x-T)).applyfunc(sigmoid)
def get_mask_np(x,s,T):
    return sigmoid_np(s*(x-T))    