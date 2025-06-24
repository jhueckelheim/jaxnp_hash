#!/usr/bin/env python3
"""
Example demonstrating the HashSet interface for jaxnp_hash.

This example shows how to:
1. Record operations with different tolerances 
2. Use the HashSet as a set-like data structure
3. Compare branch paths taken during execution, not just final results
4. Explore how different hash choices represent different execution paths
"""

import jax
import jax.numpy as jnp
import jaxnp_hash as jnph

def branching_function(x, y):
    """An example function that has nearby branches."""
    # Convert to HashTensors
    hx = jnph.HashTensor(x)
    hy = jnph.HashTensor(y)
    
    result1 = jnph.maximum(hx, hy)  # Choice point 1: indices [0, 2] within tolerance
    
    result2 = jnph.abs(result1)     # Choice point 2: element [1] near zero
    
    result3 = jnph.max(result2)      # Choice point 3: scaled values close to each other
    
    return result3.value

print("=== Simple HashSet Example ===\n")

# Create test data
x = jnp.array([1.0, 0.05, 0.94])
y = jnp.array([1.05, -1.5, 1.01])

print(f"Input x: {x}")
print(f"Input y: {y}")
print()

# Example 1: Record with tight tolerance
print("1. Recording with tight tolerance (0.01):")
with jnph.hash_mode("record", tol=0.01) as tight_hashes:
    result = branching_function(x, y)

print(f"   Result: {result}")
print(f"   Number of execution paths: {len(tight_hashes)}")
print(f"   HashSet: {tight_hashes}")
print()

# Example 2: Record with loose tolerance  
print("2. Recording with loose tolerance (0.1):")
with jnph.hash_mode("record", tol=0.1) as loose_hashes:
    result = branching_function(x, y)

print(f"   Result: {result}")
print(f"   Number of execution paths: {len(loose_hashes)}")
print(f"   HashSet: {loose_hashes}")
print()

# Example 3: Iterate through some paths
print("3. Exploring different execution paths:")
with jnph.hash_mode("record", tol=0.1) as hashes:
    branching_function(x, y)

print(f"   Total paths available: {len(hashes)}")
print("   Showing all paths:")

for i, hash_choice in enumerate(hashes):
    with jnph.hash_mode("replay", replay_hash=hash_choice):
        path_result = branching_function(x, y)
    
    print(f"     Path {i+1}: Result = {path_result}")
    print(f"       Branches taken:")
    
    # Create a formatted trace showing the decision points
    if hasattr(hashes, '_hash_set') and hashes._hash_set and hashes._hash_set.trace:
        from jaxnp_hash.HashTensor import _TraceNode
        manual_choice = []
        for node in hashes._hash_set.trace:
            manual_choice.append(_TraceNode(node.name, [node.currentChoice()]))
        
        formatted_trace = hashes.format_hash_choice(manual_choice)
        print(formatted_trace)
    else:
        print("       No trace information available")

print("=== HashSet Demo Complete! ===")
