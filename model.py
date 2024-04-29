import equinox as eqx 
import jax 

class Transformer(eqx.Module): 
  weight: jax.Array
  bias: jax.Array

  def __init__(self): 
    pass 

  def __call__(self, x): 
    return x

@jax.jit
@jax.grad
def loss_fn(model, x, y): 
  pred_y = jax.vmap(model)(x)
  return jax.numpy.mean((y - pred_y) ** 2);
  