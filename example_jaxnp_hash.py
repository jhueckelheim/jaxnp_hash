import jax.numpy as jnp
import jax
import jaxnp_hash as jnph

def myfunc(x, mode, tolerance = 0):
    with jnph.hash_mode(mode, tolerance):
        tensor = jnph.HashTensor(x)
        return jnph.max(jnph.abs(tensor)).value

myfunc_g = jax.value_and_grad(myfunc)

print(myfunc_g(jnp.array([1.0, 2.0, -3.0]), "record", 0.1))

while True:
    print(myfunc_g(jnp.array([1.0, 17.0, 5.0]), "replay"))
    if not jnph.next_hash():
        break
