from dataclasses import dataclass

import equinox as eqx 
import eqx.nn as nn
import jax 

class Head(eqx.Module):
  key: eqx.Module 
  query: eqx.Module
  value: eqx.Module
  head_size: int

  def __init__(self, head_size): 
    self.key = nn.Linear(n_embd, head_size, use_bias=False)
    self.query = nn.Linear(n_embd, head_size, use_bias=False)
    self.value = nn.Linear(n_embd, head_size, use_bias=False)

  def __call__(self, x):
    B, T, C = x.shape



  