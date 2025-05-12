from contextlib import contextmanager
import jax.numpy as jnp

is_recording = False
path_hash = []
tolerance = 0
path_hash_pos = 0

@contextmanager
def hash_mode(mode = None, tol = 0):
    global is_recording
    global path_hash
    global path_hash_pos
    global tolerance
    if(mode == "record"):
        is_recording = True
        path_hash = []
        path_hash_pos = 0
        tolerance = tol
    elif(mode == "replay"):
        is_recording = False
        path_hash_pos = 0
    else:
        raise Exception(f"Unexpected hash recording mode {mode}.")
    yield
    is_recording = False

def hash_append(name, choices):
    global path_hash
    global path_hash_pos
    path_hash.append(_TraceNode(name, choices))
    path_hash_pos += 1

def hash_popf(name):
    global path_hash
    global path_hash_pos
    node = path_hash[path_hash_pos]
    path_hash_pos += 1
    if(node.name != name):
        raise Exception(f"Unexpected operation in Hash, expected {name} but got {node.name}.")
    return node.currentChoice()

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

class _TraceNode:
    def __init__(self, name, choices):
        self.num = len(choices)
        self.pos = 0
        self.choices = choices
        self.name = name

    def currentChoice(self):
        return self.choices[self.pos]

    def incrementChoice(self):
        if(self.pos + 1 >= self.num):
            self.pos = 0
            return False
        else:
            self.pos += 1
            return True

def next_hash():
    for i in reversed(range(len(path_hash))):
        if(path_hash[i].incrementChoice()):
            return True
    return False

def max(inval):
    if(is_recording):
        loc = jnp.argmax(inval.value)
        val = inval.value[loc]
        nearby_locs, = jnp.where(inval.value >= val - tolerance)
        hash_append("max", nearby_locs)
    else:
        loc = hash_popf("max")
        val = inval.value[loc]
    return HashTensor(val)

def maximum(one, two):
    if(is_recording):
        loc = one.value >= two.value
        hash_append("maximum", [loc,])
    else:
        loc = hash_popf("maximum")
    return HashTensor(jnp.where(loc, one.value, two.value))

#def quantile(vals, n):

def min(inval):
    if(is_recording):
        loc = jnp.argmax(inval.value)
        hash_append("min", [loc,])
    else:
        loc = hash_popf("min")
    return HashTensor(inval.value[loc])

def min(inval):
    if(is_recording):
        loc = jnp.argmin(inval.value)
        val = inval.value[loc]
        nearby_locs, = jnp.where(inval.value <= val + tolerance)
        hash_append("min", nearby_locs)
    else:
        loc = hash_popf("min")
        val = inval.value[loc]
    return HashTensor(val)

def minimum(one, two):
    if(is_recording):
        loc = one.value <= two.value
        hash_append("minimum", [loc,])
    else:
        loc = hash_popf("minimum")
    return HashTensor(jnp.where(loc, one.value, two.value))

def sum(inval):
    return HashTensor(jnp.sum(inval.value))

def abs(inval):
    if(is_recording):
        loc = inval.value >= 0
        hash_append("abs", [loc,])
    else:
        loc = hash_popf("abs")
    return HashTensor(jnp.where(loc, inval.value, -inval.value))
