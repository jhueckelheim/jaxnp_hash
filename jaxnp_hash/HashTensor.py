from contextlib import contextmanager
import jax.numpy as jnp
import logging

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

is_recording = False
path_hash = []
tolerance = 0
path_hash_pos = 0

def get_path_hash():
    global path_hash
    return path_hash

@contextmanager
def hash_mode(mode = None, tol = 0):
    global is_recording
    global path_hash
    global path_hash_pos
    global tolerance
    logger.debug(f"Entering hash_mode: mode={mode}, tolerance={tol}")
    if(mode == "record"):
        is_recording = True
        path_hash = []
        path_hash_pos = 0
        tolerance = tol
        logger.debug("Starting recording mode")
    elif(mode == "replay"):
        is_recording = False
        path_hash_pos = 0
        logger.debug("Starting replay mode")
    else:
        raise Exception(f"Unexpected hash recording mode {mode}.")
    yield
    is_recording = False
    logger.debug("Exiting hash_mode")

def hash_append(name, choices):
    global path_hash
    global path_hash_pos
    logger.debug(f"hash_append: name={name}, num_choices={len(choices)}")
    path_hash.append(_TraceNode(name, choices))
    path_hash_pos += 1

def hash_popf(name):
    global path_hash
    global path_hash_pos
    logger.debug(f"hash_popf: name={name}, pos={path_hash_pos}")
    node = path_hash[path_hash_pos]
    path_hash_pos += 1
    if(node.name != name):
        raise Exception(f"Unexpected operation in Hash, expected {name} but got {node.name}.")
    choice = node.currentChoice()
    logger.debug(f"hash_popf: returning choice={choice}")
    return choice

# Wrapper class for JAX tensors
class HashTensor:
    def __init__(self, value):
        logger.debug(f"HashTensor.__init__: value={value}")
        self.value = value 

    def __repr__(self):
        return f'HashTensor({self.value})'
    
    def __str__(self):
        return f'HashTensor({self.value})'

    def __add__(self, other):
        logger.debug(f"HashTensor.__add__: self={self.value}, other={other.value}")
        return HashTensor(self.value + other.value)

    def __sub__(self, other):
        logger.debug(f"HashTensor.__sub__: self={self.value}, other={other.value}")
        return HashTensor(self.value - other.value)

    def __mul__(self, other):
        logger.debug(f"HashTensor.__mul__: self={self.value}, other={other.value}")
        return HashTensor(self.value * other.value)

class _TraceNode:
    def __init__(self, name, choices):
        logger.debug(f"_TraceNode.__init__: name={name}, num_choices={len(choices)}")
        self.num = len(choices)
        self.pos = 0
        self.choices = choices
        self.name = name

    def __repr__(self):
        return f'_TraceNode(name="{self.name}", pos={self.pos}/{self.num}, current_choice={self.currentChoice()})'
    
    def __str__(self):
        return f'_TraceNode(name="{self.name}", pos={self.pos}/{self.num}, current_choice={self.currentChoice()})'

    def currentChoice(self):
        choice = self.choices[self.pos]
        logger.debug(f"_TraceNode.currentChoice: name={self.name}, pos={self.pos}, choice={choice}")
        return choice

    def incrementChoice(self):
        logger.debug(f"_TraceNode.incrementChoice: name={self.name}, pos={self.pos}, num={self.num}")
        if(self.pos + 1 >= self.num):
            self.pos = 0
            return False
        else:
            self.pos += 1
            return True

def next_hash():
    global path_hash
    global path_hash_pos
    logger.debug("next_hash: attempting to increment choices")
    path_hash_pos = 0
    for i in reversed(range(len(path_hash))):
        if(path_hash[i].incrementChoice()):
            logger.debug(f"next_hash: successfully incremented choice at position {i}")
            return True
    logger.debug("next_hash: no more choices available")
    return False

def max(inval):
    logger.debug(f"max: input={inval.value}")
    if(is_recording):
        loc = jnp.argmax(inval.value)
        val = inval.value[loc]
        nearby_locs, = jnp.where(inval.value >= val - tolerance)
        logger.debug(f"max: recording - loc={loc}, val={val}, nearby_locs={nearby_locs}")
        hash_append("max", nearby_locs)
    else:
        loc = hash_popf("max")
        val = inval.value[loc]
        logger.debug(f"max: replaying - loc={loc}, val={val}")
    return HashTensor(val)

def maximum(one, two):
    logger.debug(f"maximum: one={one.value}, two={two.value}")
    if(is_recording):
        # Find the maximum value between the two tensors
        max_val = jnp.maximum(one.value, two.value)
        # Find all indices where either value is within tolerance of the maximum
        nearby_indices = jnp.where(
            jnp.abs(one.value - two.value) <= tolerance
        )[0]
        
        logger.debug(f"maximum: recording - max_val={max_val}, nearby_indices={nearby_indices}")
        
        # Generate all possible combinations of choices for these indices
        choices = []
        n_choices = 2 ** len(nearby_indices)  # Number of possible combinations
        
        for i in range(n_choices):
            # Convert the number to binary and pad with False
            choice = jnp.zeros_like(one.value, dtype=bool)
            # Set the bits for the nearby indices
            for j, idx in enumerate(nearby_indices):
                if (i >> j) & 1:  # Check if j-th bit is set
                    choice = choice.at[idx].set(True)
            choices.append(choice)
            
        logger.debug(f"maximum: recording - generated {len(choices)} choices")
        hash_append("maximum", choices)
    else:
        choice = hash_popf("maximum")
        # First compute standard maximum
        standard_max = jnp.maximum(one.value, two.value)
        # For indices where choice is True, flip the selection
        result = HashTensor(jnp.where(choice, jnp.where(one.value > two.value, two.value, one.value), standard_max))
        logger.debug(f"maximum: replaying - choice={choice}, result={result.value}")
        return result
    return HashTensor(jnp.maximum(one.value, two.value))

#def quantile(vals, n):

def min(inval):
    logger.debug(f"min: input={inval.value}")
    if(is_recording):
        loc = jnp.argmin(inval.value)
        val = inval.value[loc]
        nearby_locs, = jnp.where(inval.value <= val + tolerance)
        logger.debug(f"min: recording - loc={loc}, val={val}, nearby_locs={nearby_locs}")
        hash_append("min", nearby_locs)
    else:
        loc = hash_popf("min")
        val = inval.value[loc]
        logger.debug(f"min: replaying - loc={loc}, val={val}")
    return HashTensor(val)

def minimum(one, two):
    logger.debug(f"minimum: one={one.value}, two={two.value}")
    if(is_recording):
        # Find the minimum value between the two tensors
        min_val = jnp.minimum(one.value, two.value)
        # Find all indices where either value is within tolerance of the minimum
        nearby_indices = jnp.where(
            jnp.abs(one.value - two.value) <= tolerance
        )[0]
        
        logger.debug(f"minimum: recording - min_val={min_val}, nearby_indices={nearby_indices}")
        
        # Generate all possible combinations of choices for these indices
        choices = []
        n_choices = 2 ** len(nearby_indices)  # Number of possible combinations
        
        for i in range(n_choices):
            # Convert the number to binary and pad with False
            choice = jnp.zeros_like(one.value, dtype=bool)
            # Set the bits for the nearby indices
            for j, idx in enumerate(nearby_indices):
                if (i >> j) & 1:  # Check if j-th bit is set
                    choice = choice.at[idx].set(True)
            choices.append(choice)
            
        logger.debug(f"minimum: recording - generated {len(choices)} choices")
        hash_append("minimum", choices)
    else:
        choice = hash_popf("minimum")
        # First compute standard minimum
        standard_min = jnp.minimum(one.value, two.value)
        # For indices where choice is True, flip the selection
        result = HashTensor(jnp.where(choice, jnp.where(one.value < two.value, two.value, one.value), standard_min))
        logger.debug(f"minimum: replaying - choice={choice}, result={result.value}")
        return result
    return HashTensor(jnp.minimum(one.value, two.value))

def sum(inval):
    logger.debug(f"sum: input={inval.value}")
    result = HashTensor(jnp.sum(inval.value))
    logger.debug(f"sum: result={result.value}")
    return result

def abs(inval):
    logger.debug(f"abs: input={inval.value}")
    if(is_recording):
        # Find all indices where the value is within tolerance of 0
        nearby_indices = jnp.where(jnp.abs(inval.value) <= tolerance)[0]
        logger.debug(f"abs: recording - nearby_indices={nearby_indices}")
        
        # Generate all possible combinations of choices for these indices
        choices = []
        n_choices = 2 ** len(nearby_indices)  # Number of possible combinations
        
        for i in range(n_choices):
            # Convert the number to binary and pad with False
            choice = jnp.zeros_like(inval.value, dtype=bool)
            # Set the bits for the nearby indices
            for j, idx in enumerate(nearby_indices):
                if (i >> j) & 1:  # Check if j-th bit is set
                    choice = choice.at[idx].set(True)
            choices.append(choice)
            
        logger.debug(f"abs: recording - generated {len(choices)} choices")
        hash_append("abs", choices)
    else:
        choice = hash_popf("abs")
        result = HashTensor(jnp.where(choice, -jnp.abs(inval.value), jnp.abs(inval.value)))
        logger.debug(f"abs: replaying - choice={choice}, result={result.value}")
        return result
    return HashTensor(jnp.abs(inval.value))
