from contextlib import contextmanager
import jax.numpy as jnp
import logging

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

is_recording = False
path_hash = []
current_hash = []
tolerance = 0

class HashSet:
    """A set-like data structure that stores nearby path hashes in a compressed
       format, and allows the usual set operations (iterate over elements,
       check for contained elements, union, intersection, etc.)"""
    
    def __init__(self, trace):
        """Initialize with a trace (list of _TraceNode objects)."""
        self.trace = trace.copy()  # Keep original trace intact
        logger.debug(f"HashSet.__init__: trace with {len(trace)} nodes")
    
    def __iter__(self):
        """Iterate through all possible hash paths."""
        # Reset all nodes to initial position
        for node in self.trace:
            node.pos = 0
        
        # Yield first hash
        yield self._current_hash()
        
        # Generate subsequent hashes
        while self._increment_trace():
            yield self._current_hash()
    
    def _current_hash(self):
        """Get current hash as a list of _TraceNode objects with single choices."""
        return [_TraceNode(node.name, [node.currentChoice()]) for node in self.trace]
    
    def _increment_trace(self):
        """Increment the trace to the next combination."""
        for i in reversed(range(len(self.trace))):
            if self.trace[i].incrementChoice():
                return True
        return False
    
    def __len__(self):
        """Return the total number of possible hashes."""
        total = 1
        for node in self.trace:
            total *= node.num
        return total
    
    def __getitem__(self, index):
        """Get hash at specific index using random access."""
        if not isinstance(index, int):
            raise TypeError("Index must be an integer")
        
        total_len = len(self)
        if index < 0:
            index = total_len + index  # Handle negative indices
        
        if index < 0 or index >= total_len:
            raise IndexError(f"Index {index} is out of range for HashSet with {total_len} elements")
        
        # Convert index to combination of choices using mixed-radix representation
        # Create a copy of the trace to avoid modifying the original
        trace_copy = [_TraceNode(node.name, node.choices) for node in self.trace]
        
        # Reset all nodes to initial position
        for node in trace_copy:
            node.pos = 0
        
        # Calculate the position for each node based on the index
        remaining_index = index
        for i in reversed(range(len(trace_copy))):
            node = trace_copy[i]
            # Calculate how many combinations come after this node
            combinations_after = 1
            for j in range(i + 1, len(trace_copy)):
                combinations_after *= trace_copy[j].num
            
            # Set this node's position
            node.pos = remaining_index // combinations_after
            remaining_index = remaining_index % combinations_after
        
        # Return the hash for this combination (using the copy)
        return [_TraceNode(node.name, [node.currentChoice()]) for node in trace_copy]
    
    def __contains__(self, item):
        """Check if a specific hash exists in the set."""
        if not isinstance(item, list) or len(item) != len(self.trace):
            return False
        
        # Check if item matches any possible path
        for hash_item in self:
            if self._hashes_equal(hash_item, item):
                return True
        return False
    
    def _hashes_equal(self, hash1, hash2):
        """Check if two hashes are equal."""
        if len(hash1) != len(hash2):
            return False
        
        for node1, node2 in zip(hash1, hash2):
            if node1.name != node2.name:
                return False
            if len(node1.choices) != 1 or len(node2.choices) != 1:
                return False
            
            choice1 = node1.choices[0]
            choice2 = node2.choices[0]
            
            # Handle tuple choices (e.g., from maximum operation)
            if isinstance(choice1, tuple) and isinstance(choice2, tuple):
                if len(choice1) != len(choice2):
                    return False
                for c1, c2 in zip(choice1, choice2):
                    if hasattr(c1, 'shape') and hasattr(c2, 'shape'):
                        # Both are JAX arrays
                        if not jnp.array_equal(c1, c2):
                            return False
                    elif c1 != c2:
                        # Other values
                        return False
            # Handle JAX arrays
            elif hasattr(choice1, 'shape') and hasattr(choice2, 'shape'):
                if not jnp.array_equal(choice1, choice2):
                    return False
            # Handle other values
            elif choice1 != choice2:
                return False
        return True
    
    def __bool__(self):
        """Return True if the set is not empty."""
        return len(self.trace) > 0
    
    def __repr__(self):
        return f'HashSet(trace_length={len(self.trace)}, total_hashes={len(self)})'
    
    def __str__(self):
        return f'HashSet with {len(self)} possible hashes from {len(self.trace)} trace nodes'
    
    # Additional set-like methods
    def union(self, other):
        """Return union with another HashSet."""
        if not isinstance(other, HashSet):
            raise TypeError("Union requires another HashSet")
        
        # Create a new HashSet from the combined unique hashes
        all_hashes = set()
        
        # Add all hashes from self
        for hash_item in self:
            all_hashes.add(self._hash_to_tuple(hash_item))
        
        # Add all hashes from other
        for hash_item in other:
            all_hashes.add(self._hash_to_tuple(hash_item))
        
        # Convert back to trace format
        combined_trace = self._create_trace_from_hashes(list(all_hashes))
        return HashSet(combined_trace)
    
    def intersection(self, other):
        """Return intersection with another HashSet."""
        if not isinstance(other, HashSet):
            raise TypeError("Intersection requires another HashSet")
        
        # Find common hashes
        common_hashes = set()
        other_hashes = set()
        
        # Get all hashes from other
        for hash_item in other:
            other_hashes.add(self._hash_to_tuple(hash_item))
        
        # Find hashes in self that are also in other
        for hash_item in self:
            hash_tuple = self._hash_to_tuple(hash_item)
            if hash_tuple in other_hashes:
                common_hashes.add(hash_tuple)
        
        # Convert back to trace format
        common_trace = self._create_trace_from_hashes(list(common_hashes))
        return HashSet(common_trace)
    
    def difference(self, other):
        """Return difference with another HashSet (hashes in self but not in other)."""
        if not isinstance(other, HashSet):
            raise TypeError("Difference requires another HashSet")
        
        # Find hashes in self but not in other
        diff_hashes = set()
        other_hashes = set()
        
        # Get all hashes from other
        for hash_item in other:
            other_hashes.add(self._hash_to_tuple(hash_item))
        
        # Find hashes in self that are not in other
        for hash_item in self:
            hash_tuple = self._hash_to_tuple(hash_item)
            if hash_tuple not in other_hashes:
                diff_hashes.add(hash_tuple)
        
        # Convert back to trace format
        diff_trace = self._create_trace_from_hashes(list(diff_hashes))
        return HashSet(diff_trace)
    
    def _hash_to_tuple(self, hash_item):
        """Convert a hash (list of _TraceNode objects) to a hashable tuple."""
        elements = []
        for node in hash_item:
            choice = node.choices[0]
            
            # Convert JAX arrays to Python values for hashing
            if hasattr(choice, 'tolist'):
                # JAX array or numpy array
                try:
                    elements.append((node.name, tuple(choice.tolist()) if hasattr(choice.tolist(), '__iter__') else choice.tolist()))
                except:
                    elements.append((node.name, float(choice)))
            elif isinstance(choice, tuple):
                # Tuple (e.g., from maximum operation)
                try:
                    nearby_indices, choice_int = choice
                    nearby_tuple = tuple(nearby_indices.tolist()) if hasattr(nearby_indices, 'tolist') else tuple(nearby_indices)
                    elements.append((node.name, (nearby_tuple, int(choice_int))))
                except:
                    elements.append((node.name, choice))
            else:
                # Regular Python value
                elements.append((node.name, choice))
        
        return tuple(elements)
    
    def _create_trace_from_hashes(self, hash_tuples):
        """Create a trace from a list of hash tuples."""
        if not hash_tuples:
            return []
        
        # Group choices by node name/position
        node_choices = {}
        for hash_tuple in hash_tuples:
            for i, (node_name, choice) in enumerate(hash_tuple):
                if i not in node_choices:
                    node_choices[i] = {'name': node_name, 'choices': set()}
                node_choices[i]['choices'].add(choice)
        
        # Create trace nodes
        trace = []
        for i in sorted(node_choices.keys()):
            node_info = node_choices[i]
            choices_list = list(node_info['choices'])
            
            # Convert back from tuple format to original format
            converted_choices = []
            for choice in choices_list:
                if isinstance(choice, tuple) and len(choice) == 2 and isinstance(choice[0], tuple):
                    # This is a tuple from maximum operation
                    nearby_indices, choice_int = choice
                    nearby_array = jnp.array(nearby_indices)
                    converted_choices.append((nearby_array, choice_int))
                elif isinstance(choice, tuple):
                    # This is a regular tuple/array
                    try:
                        converted_choices.append(jnp.array(choice))
                    except:
                        converted_choices.append(choice)
                else:
                    # Regular value
                    converted_choices.append(choice)
            
            trace.append(_TraceNode(node_info['name'], converted_choices))
        
        return trace
    
    def format_hash_choice(self, hash_choice):
        """Format a hash choice (list of _TraceNode objects) for display."""
        if not hash_choice:
            return "No decision points"
        
        lines = []
        for step, node in enumerate(hash_choice):
            choice_val = node.choices[0]
            if isinstance(choice_val, tuple) and len(choice_val) == 2:
                # This is from maximum/minimum operations
                nearby_indices, choice_int = choice_val
                try:
                    if len(nearby_indices) == 0:
                        lines.append(f"  Step {step+1} ({node.name}): standard choice (no nearby values)")
                    else:
                        # Decode which indices are flipped based on the binary pattern
                        flipped_indices = []
                        for j, idx in enumerate(nearby_indices):
                            if (choice_int >> j) & 1:  # Check if j-th bit is set
                                flipped_indices.append(int(idx))  # Convert JAX array to Python int
                        
                        # Convert nearby_indices to readable format
                        nearby_list = [int(idx) for idx in nearby_indices]
                        
                        if len(flipped_indices) == 0:
                            lines.append(f"  Step {step+1} ({node.name}): standard choice (no flips)")
                        else:
                            lines.append(f"  Step {step+1} ({node.name}): flip indices {flipped_indices} (from nearby {nearby_list})")
                except:
                    lines.append(f"  Step {step+1} ({node.name}): indices {nearby_indices}, pattern {choice_int:b}")
            elif hasattr(choice_val, 'shape') and choice_val.shape == ():
                # JAX scalar
                lines.append(f"  Step {step+1} ({node.name}): scalar choice = {choice_val}")
            elif hasattr(choice_val, 'dtype') and choice_val.dtype == bool:
                # Boolean array from abs operation
                try:
                    true_indices = [j for j, val in enumerate(choice_val) if val]
                    if len(true_indices) == 0:
                        lines.append(f"  Step {step+1} ({node.name}): standard absolute value")
                    else:
                        lines.append(f"  Step {step+1} ({node.name}): negate at indices {true_indices}")
                except:
                    lines.append(f"  Step {step+1} ({node.name}): boolean array choice")
            elif hasattr(choice_val, '__len__'):
                try:
                    if len(choice_val) > 0:
                        # Index selection from max/min
                        lines.append(f"  Step {step+1} ({node.name}): selected index {choice_val}")
                    else:
                        lines.append(f"  Step {step+1} ({node.name}): empty choice")
                except:
                    # JAX array that doesn't support len()
                    lines.append(f"  Step {step+1} ({node.name}): array choice = {choice_val}")
            else:
                lines.append(f"  Step {step+1} ({node.name}): choice = {choice_val}")
        return "\n".join(lines)

class HashModeResult:
    """A result object that becomes a HashSet after recording is complete."""
    
    def __init__(self):
        self._trace = []
        self._hash_set = None
        self._is_recording = True
    
    def _finalize(self, trace):
        """Convert the trace to a HashSet."""
        self._trace = trace
        self._hash_set = HashSet(trace)
        self._is_recording = False
    
    def __len__(self):
        """Return the length of the HashSet."""
        if self._is_recording:
            # During recording, calculate based on current trace
            if not self._trace:
                return 0
            total = 1
            for node in self._trace:
                total *= node.num
            return total
        return len(self._hash_set)
    
    def __getitem__(self, index):
        """Get hash at specific index using random access."""
        if self._is_recording:
            raise RuntimeError("Cannot perform random access during recording")
        return self._hash_set[index]
    
    def _hashes_equal(self, hash1, hash2):
        """Check if two hashes are equal."""
        if self._is_recording:
            raise RuntimeError("Cannot compare hashes during recording")
        return self._hash_set._hashes_equal(hash1, hash2)
    
    def __contains__(self, item):
        """Check if item is in the HashSet."""
        if self._is_recording:
            return False
        return item in self._hash_set
    
    def __bool__(self):
        """Return True if HashSet is not empty."""
        if self._is_recording:
            return len(self._trace) > 0
        return bool(self._hash_set)
    
    def __repr__(self):
        if self._is_recording:
            return f"HashModeResult(recording... {len(self._trace)} nodes)"
        return repr(self._hash_set)
    
    def __str__(self):
        if self._is_recording:
            return f"HashModeResult(recording... {len(self._trace)} nodes)"
        return str(self._hash_set)
    
    def format_hash_choice(self, hash_choice):
        """Format a hash choice for display."""
        if self._is_recording:
            return "Cannot format hash choice during recording"
        return self._hash_set.format_hash_choice(hash_choice)
    
    def union(self, other):
        """Return union with another HashSet or HashModeResult."""
        if self._is_recording:
            raise RuntimeError("Cannot perform union operation during recording")
        if isinstance(other, HashModeResult):
            if other._is_recording:
                raise RuntimeError("Cannot perform union with HashModeResult that is still recording")
            other = other._hash_set
        return self._hash_set.union(other)
    
    def intersection(self, other):
        """Return intersection with another HashSet or HashModeResult."""
        if self._is_recording:
            raise RuntimeError("Cannot perform intersection operation during recording")
        if isinstance(other, HashModeResult):
            if other._is_recording:
                raise RuntimeError("Cannot perform intersection with HashModeResult that is still recording")
            other = other._hash_set
        return self._hash_set.intersection(other)
    
    def difference(self, other):
        """Return difference with another HashSet or HashModeResult."""
        if self._is_recording:
            raise RuntimeError("Cannot perform difference operation during recording")
        if isinstance(other, HashModeResult):
            if other._is_recording:
                raise RuntimeError("Cannot perform difference with HashModeResult that is still recording")
            other = other._hash_set
        return self._hash_set.difference(other)
    
    def _update_trace(self, trace):
        """Update the trace during recording."""
        self._trace = trace
    
    def __iter__(self):
        """Iterate through the HashSet."""
        if self._is_recording:
            # During recording, return empty iterator
            return iter([])
        return iter(self._hash_set)

@contextmanager 
def hash_mode(mode=None, tol=0, replay_hash=None):
    global is_recording
    global path_hash
    global current_hash
    global tolerance
    logger.debug(f"Entering hash_mode: mode={mode}, tolerance={tol}, replay_hash={replay_hash}")
    
    if mode == "record":
        is_recording = True
        path_hash = []
        tolerance = tol
        logger.debug("Starting recording mode")
        
        result = HashModeResult()
        
        # Make the result object aware of the trace being built
        result._trace = path_hash
        
        try:
            yield result
        finally:
            # After recording is complete, create the HashSet
            result._finalize(path_hash)
            logger.debug(f"Created HashSet with {len(result)} possible hashes")

    elif mode == "replay":
        is_recording = False
        if replay_hash is None:
            raise ValueError("replay_hash must be provided in replay mode")
        
        # Handle the case where replay_hash is a HashModeResult
        if isinstance(replay_hash, HashModeResult):
            if len(replay_hash) == 1:
                # Extract the single hash choice
                current_hash = next(iter(replay_hash))
            else:
                raise ValueError("HashModeResult with multiple hashes cannot be used directly as replay_hash. Iterate over it first.")
        else:
            current_hash = replay_hash
            
        logger.debug("Starting replay mode")
        yield
    else:
        raise Exception(f"Unexpected hash recording mode {mode}.")
    
    is_recording = False
    current_hash = None
    logger.debug("Exiting hash_mode")

def hash_append(name, choices):
    global path_hash
    logger.debug(f"hash_append: name={name}, num_choices={len(choices)}")
    path_hash.append(_TraceNode(name, choices))

def hash_popf(name):
    global path_hash
    if current_hash is None:
        raise ValueError("No hash provided for replay mode")
    if not current_hash:
        raise ValueError("Hash exhausted")
    node = current_hash.pop(0)
    if node.name != name:
        raise ValueError(f"Expected hash node {name}, got {node.name}")
    return node.choices[0]  # Return the first choice from the choices list

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

    def __truediv__(self, other):
        logger.debug(f"HashTensor.__truediv__: self={self.value}, other={other.value}")
        return HashTensor(self.value / other.value)

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
        # Find all indices where the values are within tolerance
        nearby_indices = jnp.where(jnp.abs(one.value - two.value) <= tolerance)[0]
        logger.debug(f"maximum: recording - nearby_indices={nearby_indices}")
        n_choices = 2 ** len(nearby_indices)
        # Store the indices, not the full boolean vectors
        choices = []
        for i in range(n_choices):
            choices.append((nearby_indices, i))
        hash_append("maximum", choices)
    else:
        nearby_indices, choice_int = hash_popf("maximum")
        # Reconstruct the boolean mask from the indices and the integer
        choice = jnp.zeros_like(one.value, dtype=bool)
        for j, idx in enumerate(nearby_indices):
            if (choice_int >> j) & 1:
                choice = choice.at[idx].set(True)
        standard_max = jnp.maximum(one.value, two.value)
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
        # Find all indices where the values are within tolerance
        nearby_indices = jnp.where(jnp.abs(one.value - two.value) <= tolerance)[0]
        logger.debug(f"minimum: recording - nearby_indices={nearby_indices}")
        n_choices = 2 ** len(nearby_indices)
        # Store the indices, not the full boolean vectors
        choices = []
        for i in range(n_choices):
            choices.append((nearby_indices, i))
        hash_append("minimum", choices)
    else:
        nearby_indices, choice_int = hash_popf("minimum")
        # Reconstruct the boolean mask from the indices and the integer
        choice = jnp.zeros_like(one.value, dtype=bool)
        for j, idx in enumerate(nearby_indices):
            if (choice_int >> j) & 1:
                choice = choice.at[idx].set(True)
        standard_min = jnp.minimum(one.value, two.value)
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
