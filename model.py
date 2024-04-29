from dataclasses import dataclass

import equinox as eqx 
import jax 

@dataclass 
class GPTConfig: 
  block_size: int = 1024
  vocab_size: int = 50304 
  n_layer: int = 12
  n_head: int = 12
  n_embd: int = 768
  dropout: float = 0.0
  bias: bool = True

class GPT(eqx.Module): 
  weight: jax.Array
  bias: jax.Array

  def __init__(self): 
    pass 

  def __call__(self, x): 
    return x

  