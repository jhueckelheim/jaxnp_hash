from contextlib import contextmanager
import jax.numpy as jnp

is_recording = False
path_hash = []

@contextmanager
def hash_mode(mode = None):
    global is_recording
    global path_hash
    if(mode == "record"):
        is_recording = True
        path_hash = []
    elif(mode == "consume"):
        is_recording = False
    elif(mode == "replay"):
        is_recording = False
        old_hash = path_hash.copy()
    else:
        raise Exception(f"Unexpected hash recording mode {mode}.")
    yield
    is_recording = False
    if(mode == "replay"):
        path_hash = old_hash

def hash_append(name, val):
    path_hash.append((name, val))

def hash_popf(name):
    namep, val = path_hash.pop(0)
    if(namep != name):
        raise Exception(f"Unexpected operation in Hash, expected {name} but got {namep}.")
    return val

# Wrapper class for JAX tensors
class HashTensor:
    def __init__(self, value):
        self.value = value 

    def __repr__(self):
        return f'HashTensor({self.value})'

    def __add__(self, other):
        return HashTensor(self.value + other.value)

    def __sub__(self, other):
        return HashTensor(self.value - other.value)

    def __mul__(self, other):
        return HashTensor(self.value * other.value)

def max(inval):
    if(is_recording):
        loc = jnp.argmax(inval.value)
        hash_append("argmax", loc)
    else:
        loc = hash_popf("argmax")
    return HashTensor(inval.value[loc])

def maximum(one, two):
    if(is_recording):
        loc = one.value >= two.value
        hash_append("maximum", loc)
    else:
        loc = hash_popf("maximum")
    return HashTensor(jnp.where(loc, one.value, two.value))

def min(inval):
    if(is_recording):
        loc = jnp.argmax(inval.value)
        hash_append("argmin", loc)
    else:
        loc = hash_popf("argmin")
    return HashTensor(inval.value[loc])

def minimum(one, two):
    if(is_recording):
        loc = one.value <= two.value
        hash_append("minimum", loc)
    else:
        loc = hash_popf("minimum")
    return HashTensor(jnp.where(loc, one.value, two.value))

def sum(inval):
    return HashTensor(jnp.sum(inval.value))

def abs(inval):
    if(is_recording):
        loc = inval.value >= 0
        hash_append("abs", loc)
    else:
        loc = hash_popf("abs")
    return HashTensor(jnp.where(loc, inval.value, -inval.value))
