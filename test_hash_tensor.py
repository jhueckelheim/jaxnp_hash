import jax.numpy as jnp
import jax
import jaxnp_hash as jnph
from itertools import product

def test_maximum():
    # Test case 1: No values within tolerance
    x1 = jnp.array([1.0, 2.0, 2.5])
    x2 = jnp.array([0.5, 1.5, 3.0])
    
    # Record mode
    with jnph.hash_mode("record", 0.1):
        result = jnph.maximum(jnph.HashTensor(x1), jnph.HashTensor(x2))
        assert jnp.allclose(result.value, jnp.array([1.0, 2.0, 3.0])), f"Expected [1.0, 2.0, 3.0], got {result.value}"

    # Replay mode
    with jnph.hash_mode("replay"):
        result = jnph.maximum(jnph.HashTensor(x1), jnph.HashTensor(x2))
        assert jnp.allclose(result.value, jnp.array([1.0, 2.0, 3.0])), f"Expected [1.0, 2.0, 3.0], got {result.value}"

    # Test case 2: One value within tolerance
    x1 = jnp.array([1.0, 2.0, 2.5])
    x2 = jnp.array([0.95, 1.5, 3.0])  # 0.95 is within 0.1 of 1.0
    
    # Record mode
    with jnph.hash_mode("record", 0.1):
        result = jnph.maximum(jnph.HashTensor(x1), jnph.HashTensor(x2))
        assert jnp.allclose(result.value[1:], jnp.array([2.0, 3.0])), f"Expected [2.0, 3.0] for indices 1:3, got {result.value[1:]}"
        assert result.value[0] in [1.0, 0.95], f"Expected first value to be 1.0 or 0.95, got {result.value[0]}"

    # Replay mode
    expected_results = set()
    for first_val in [1.0, 0.95]:
        result = jnp.array([first_val, 2.0, 3.0])
        expected_results.add(tuple(result.tolist()))
    
    actual_results = set()
    with jnph.hash_mode("replay"):
        while True:
            result = jnph.maximum(jnph.HashTensor(x1), jnph.HashTensor(x2))
            actual_results.add(tuple(result.value.tolist()))
            if not jnph.next_hash():
                break
    
    assert actual_results == expected_results, f"Expected results {expected_results}, got {actual_results}"

    # Test case 3: Multiple values within tolerance
    x1 = jnp.array([1.0, 2.0, 2.5])
    x2 = jnp.array([0.95, 1.95, 2.45])  # All values within 0.1
    
    # Record mode
    with jnph.hash_mode("record", 0.1):
        result = jnph.maximum(jnph.HashTensor(x1), jnph.HashTensor(x2))
        for i in range(3):
            assert result.value[i] in [x1[i], x2[i]], f"Expected value at index {i} to be {x1[i]} or {x2[i]}, got {result.value[i]}"

    # Replay mode
    expected_results = set()
    for vals in product([x1[0], x2[0]], [x1[1], x2[1]], [x1[2], x2[2]]):
        result = jnp.array(vals)
        expected_results.add(tuple(result.tolist()))
    
    actual_results = set()
    with jnph.hash_mode("replay"):
        while True:
            result = jnph.maximum(jnph.HashTensor(x1), jnph.HashTensor(x2))
            actual_results.add(tuple(result.value.tolist()))
            if not jnph.next_hash():
                break
    
    assert actual_results == expected_results, f"Expected results {expected_results}, got {actual_results}"

def test_minimum():
    # Test case 1: No values within tolerance
    x1 = jnp.array([1.0, 2.0, 2.5])
    x2 = jnp.array([0.5, 1.5, 3.0])
    
    # Record mode
    with jnph.hash_mode("record", 0.1):
        result = jnph.minimum(jnph.HashTensor(x1), jnph.HashTensor(x2))
        assert jnp.allclose(result.value, jnp.array([0.5, 1.5, 2.5])), f"Expected [0.5, 1.5, 2.5], got {result.value}"
    
    # Replay mode
    with jnph.hash_mode("replay"):
        result = jnph.minimum(jnph.HashTensor(x1), jnph.HashTensor(x2))
        assert jnp.allclose(result.value, jnp.array([0.5, 1.5, 2.5])), f"Expected [0.5, 1.5, 2.5], got {result.value}"

    # Test case 2: One value within tolerance
    x1 = jnp.array([1.0, 2.0, 2.5])
    x2 = jnp.array([1.05, 1.5, 3.0])  # 1.05 is within 0.1 of 1.0
    
    # Record mode
    with jnph.hash_mode("record", 0.1):
        result = jnph.minimum(jnph.HashTensor(x1), jnph.HashTensor(x2))
        assert jnp.allclose(result.value[1:], jnp.array([1.5, 2.5])), f"Expected [1.5, 2.5] for indices 1:3, got {result.value[1:]}"
        assert result.value[0] in [1.0, 1.05], f"Expected first value to be 1.0 or 1.05, got {result.value[0]}"
    
    # Replay mode
    expected_results = set()
    for first_val in [1.0, 1.05]:
        result = jnp.array([first_val, 1.5, 2.5])
        expected_results.add(tuple(result.tolist()))
    
    actual_results = set()
    with jnph.hash_mode("replay"):
        while True:
            result = jnph.minimum(jnph.HashTensor(x1), jnph.HashTensor(x2))
            actual_results.add(tuple(result.value.tolist()))
            if not jnph.next_hash():
                break
    
    assert actual_results == expected_results, f"Expected results {expected_results}, got {actual_results}"

    # Test case 3: Multiple values within tolerance
    x1 = jnp.array([1.0, 2.0, 2.5])
    x2 = jnp.array([1.05, 2.05, 2.55])  # All values within 0.1
    
    # Record mode
    with jnph.hash_mode("record", 0.1):
        result = jnph.minimum(jnph.HashTensor(x1), jnph.HashTensor(x2))
        for i in range(3):
            assert result.value[i] in [x1[i], x2[i]], f"Expected value at index {i} to be {x1[i]} or {x2[i]}, got {result.value[i]}"
    
    # Replay mode
    expected_results = set()
    for vals in product([x1[0], x2[0]], [x1[1], x2[1]], [x1[2], x2[2]]):
        result = jnp.array(vals)
        expected_results.add(tuple(result.tolist()))
    
    actual_results = set()
    with jnph.hash_mode("replay"):
        while True:
            result = jnph.minimum(jnph.HashTensor(x1), jnph.HashTensor(x2))
            actual_results.add(tuple(result.value.tolist()))
            if not jnph.next_hash():
                break
    
    assert actual_results == expected_results, f"Expected results {expected_results}, got {actual_results}"

def test_abs():
    # Test case 1: No values within tolerance of 0
    x = jnp.array([-1.0, -2.0, 2.5])
    
    # Record mode
    with jnph.hash_mode("record", 0.1):
        result = jnph.abs(jnph.HashTensor(x))
        assert jnp.allclose(result.value, jnp.array([1.0, 2.0, 2.5])), f"Expected [1.0, 2.0, 2.5], got {result.value}"
    
    # Replay mode
    with jnph.hash_mode("replay"):
        result = jnph.abs(jnph.HashTensor(x))
        assert jnp.allclose(result.value, jnp.array([1.0, 2.0, 2.5])), f"Expected [1.0, 2.0, 2.5], got {result.value}"

    # Test case 2: One value within tolerance of 0
    x = jnp.array([0.05, -2.0, 2.5])  # 0.05 is within 0.1 of 0
    
    # Record mode
    with jnph.hash_mode("record", 0.1):
        result = jnph.abs(jnph.HashTensor(x))
        assert result.value[0] == 0.05, f"Expected first value to be 0.05, got {result.value[0]}"
        assert jnp.allclose(result.value[1:], jnp.array([2.0, 2.5])), f"Expected [2.0, 2.5] for indices 1:3, got {result.value[1:]}"
    
    # Replay mode
    expected_results = set()
    for first_val in [0.05, -0.05]:
        result = jnp.array([first_val, 2.0, 2.5])
        expected_results.add(tuple(result.tolist()))
    
    actual_results = set()
    with jnph.hash_mode("replay"):
        while True:
            result = jnph.abs(jnph.HashTensor(x))
            actual_results.add(tuple(result.value.tolist()))
            if not jnph.next_hash():
                break
    
    assert actual_results == expected_results, f"Expected results {expected_results}, got {actual_results}"

    # Test case 3: Multiple values within tolerance of 0
    x = jnp.array([0.05, -0.05, 0.03])  # All values within 0.1 of 0
    
    # Record mode
    with jnph.hash_mode("record", 0.1):
        result = jnph.abs(jnph.HashTensor(x))
        for i in range(3):
            assert result.value[i] in [abs(x[i]), -abs(x[i])], f"Expected value at index {i} to be +-{abs(x[i])}, got {result.value[i]}"
    
    # Replay mode
    expected_results = set()
    for vals in product([0.05, -0.05], [0.05, -0.05], [0.03, -0.03]):
        result = jnp.array(vals)
        expected_results.add(tuple(result.tolist()))
    
    actual_results = set()
    with jnph.hash_mode("replay"):
        while True:
            result = jnph.abs(jnph.HashTensor(x))
            actual_results.add(tuple(result.value.tolist()))
            if not jnph.next_hash():
                break
    
    assert actual_results == expected_results, f"Expected results {expected_results}, got {actual_results}"

if __name__ == "__main__":
    test_maximum()
    test_minimum()
    test_abs()
    print("All tests passed!")
