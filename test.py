import numpy as np
import torch
from utils import get_mask_th
import numpy as np
if False:
  def f1(x, t):
    return 1 / (1 + torch.exp(-x + t))

  def f2(x):
    return 1 - (0.5 - torch.mean(x,dim=-1))**2

  import matplotlib.pyplot as plt

  # Define the range of values to evaluate f2 at
  t = torch.linspace(-5, 5, 100)
  ndim = 3
  nthresh = 4
  x = torch.randn(ndim)
  # Evaluate f2 at each value in the range
  y = f2(f1(x[None,:], t[:,None]))

  # Plot f2 as a function of t
  plt.plot(t, y)
  plt.show()

  def observe(masks):
      assert masks.ndim == 2,'batch_size,ndim'
      masks = masks.detach().requires_grad_(True)
      y = f2(masks)
      grads = y.sum().backward()

      return y,grads

  ndim = 3
  nthresh = 4
  x = torch.randn(ndim)
  # Evaluate f2 at each value in the range
  masks = f1(x[None,:], t[:,None])
  # masks = torch.rand(nthresh,ndim)
  y,grads = observe(masks)
  # Plot f2 as a function of t
  plt.plot(y.detach().numpy())
  plt.show()

if True:
  from main import *
  from solve import *
  def test_estimate_a(m_at_0,m_at_1,
                g_at_0,g_at_1,
                f0,f1,
                ref_estimate):
      new_estimate = estimate_a(m_at_0,m_at_1,
                g_at_0,g_at_1,
                f0,f1)
      '''
      if np.allclose(ref_estimate,new_estimate):
          print('close enough')
      else:
          print('NOT CLOSE!')
          print(ref_estimate)
          print(new_estimate)
      '''
      new_estimate

  f_obs_ = torch.randn(nthresh)
  g_ = torch.randn(ndim,nthresh)
  '''
  Mfixed_ = np.random.random(size=(ndim,nthresh))
  Mfixed_[:,0] = Mfixed_[:,0]*0.5/Mfixed_[:,0].sum()
  Mfixed_ = Mfixed_.cumsum(axis=-1)
  Mfixed_ = np.clip(Mfixed_,0,1)
  '''
  x_ = torch.randn(ndim,1)
  t_ = torch.tensor([0.,1.])[None,:]
  s_ = 1.
  Mfixed_ = get_mask_th(x_,s_,t_)

  test_estimate_a(
      Mfixed_[:,0],Mfixed_[:,1],
      g_[:,0],g_[:,1],
      f_obs_[0],f_obs_[1],
      # found_a_copy_
      None
      )  
  
  def test_estimate_coeffs(ref_a,ref_b,ref_c,
                           coeffs,constants,
                          m_at_0,m_at_1,g_at_0,g_at_1,f0,f1):
      new_a,new_b,new_c = estimate_abc(
          coeffs,constants,
          m_at_0,m_at_1,g_at_0,g_at_1,f0,f1    
          )
      
      '''
      for name,r,n in zip(['a','b','c'],[ref_a,ref_b,ref_c],[new_a,new_b,new_c]):
          if np.allclose(r,n):
              print(name,' close enough')
          else:
              print(name,' NOT CLOSE')
              print(r)
              print(n)
      '''
      # from approximation import get_approx_change_in_f_as_function
      # approx_change_in_f_as_function = get_approx_change_in_f_as_function(
      #                   approx_change_in_f,
      #                     a,b,c,
      #                      x,s,
      #                      m_at_0,t)
      # approx_change_in_f_as_function()
  coeffs,constants = 
  test_estimate_coeffs(
      # found_a_copy_,found_b_copy_,found_c_copy_,
      None,None,None,
      coeffs,constants,
      Mfixed_[:,0],Mfixed_[:,1],
          g_[:,0],g_[:,1],
          f_obs_[0],f_obs_[1]
          )  
  
  def test_solve(ref_a,ref_b,ref_c,
                m_at_0,m_at_1,g_at_0,g_at_1,f0,f1):
      '''
      TODO: this seems to require the lin_eq, but elsewhere 
      estimate_coeffs worked without it, how?
      '''
      new_a,new_b,new_c = solve(lin_eq1,a,s,x,Mfixed,ti,gfixed,f0,f1)
      # new_a,new_b,new_c = estimate_coeffs(
      #     m_at_0,m_at_1,g_at_0,g_at_1,f0,f1    
      #     )
      
      '''
      for name,r,n in zip(['a','b','c'],[ref_a,ref_b,ref_c],[new_a,new_b,new_c]):
          if np.allclose(r,n):
              print(name,' close enough')
          else:
              print(name,' NOT CLOSE')
              print(r)
              print(n)
      '''
      from approximation import get_approx_change_in_f_as_function
      approx_change_in_f_as_function = get_approx_change_in_f_as_function(
                        approx_change_in_f,
                          a,b,c,
                           x,s,
                           m_at_0,t)
      approx_change_in_f_as_function()
