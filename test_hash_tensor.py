import jax.numpy as jnp
import jax
import jaxnp_hash as jnph
from itertools import product

def test_maximum():
    # Test case 1: No values within tolerance
    x1 = jnp.array([1.0, 2.0, 2.5])
    x2 = jnp.array([0.5, 1.5, 3.0])
    
    # Record mode
    with jnph.hash_mode("record", 0.1) as hashes:
        result = jnph.maximum(jnph.HashTensor(x1), jnph.HashTensor(x2))
        assert jnp.allclose(result.value, jnp.array([1.0, 2.0, 3.0])), f"Expected [1.0, 2.0, 3.0], got {result.value}"
        assert len(hashes) == 1  # One hash for original execution path

    # Replay mode
    with jnph.hash_mode("replay", replay_hash=hashes):
        result = jnph.maximum(jnph.HashTensor(x1), jnph.HashTensor(x2))
        assert jnp.allclose(result.value, jnp.array([1.0, 2.0, 3.0])), f"Expected [1.0, 2.0, 3.0], got {result.value}"

    # Test case 2: One value within tolerance
    x1 = jnp.array([1.0, 2.0, 2.5])
    x2 = jnp.array([0.95, 1.5, 3.0])  # 0.95 is within 0.1 of 1.0
    
    # Record mode
    with jnph.hash_mode("record", 0.1) as hashes:
        result = jnph.maximum(jnph.HashTensor(x1), jnph.HashTensor(x2))
        assert jnp.allclose(result.value[1:], jnp.array([2.0, 3.0])), f"Expected [2.0, 3.0] for indices 1:3, got {result.value[1:]}"
        assert result.value[0] in [1.0, 0.95], f"Expected first value to be 1.0 or 0.95, got {result.value[0]}"

        assert len(hashes) == 2, f"Expected 2 hashes, got {len(hashes)} hashes"  # One hash for original execution path, one for the value within tolerance

    # Replay mode - test both possible choices
    expected_results = set()
    for first_val in [1.0, 0.95]:
        result = jnp.array([first_val, 2.0, 3.0])
        expected_results.add(tuple(result.tolist()))
    
    actual_results = set()
    for hash_choice in hashes:
        with jnph.hash_mode("replay", replay_hash=hash_choice):
            result = jnph.maximum(jnph.HashTensor(x1), jnph.HashTensor(x2))
            actual_results.add(tuple(result.value.tolist()))
    
    assert actual_results == expected_results, f"Expected results {expected_results}, got {actual_results}"

    # Test case 3: Multiple values within tolerance
    x1 = jnp.array([1.0, 2.0, 2.5])
    x2 = jnp.array([0.95, 1.95, 2.45])  # All values within 0.1
    
    # Record mode
    with jnph.hash_mode("record", 0.1) as hashes:
        result = jnph.maximum(jnph.HashTensor(x1), jnph.HashTensor(x2))
        for i in range(3):
            assert result.value[i] in [x1[i], x2[i]], f"Expected value at index {i} to be {x1[i]} or {x2[i]}, got {result.value[i]}"

        assert len(hashes) == 8, f"Test case 3 - Expected 8 hashes (2^3 combinations), got {len(hashes)}"  # All combinations of choices

    # Replay mode
    expected_results = set()
    for vals in product([x1[0], x2[0]], [x1[1], x2[1]], [x1[2], x2[2]]):
        result = jnp.array(vals)
        expected_results.add(tuple(result.tolist()))
    
    actual_results = set()
    for hash_choice in hashes:
        with jnph.hash_mode("replay", replay_hash=hash_choice):
            result = jnph.maximum(jnph.HashTensor(x1), jnph.HashTensor(x2))
            actual_results.add(tuple(result.value.tolist()))
    
    assert actual_results == expected_results, f"Expected results {expected_results}, got {actual_results}"

def test_minimum():
    # Test case 1: No values within tolerance
    x1 = jnp.array([1.0, 2.0, 2.5])
    x2 = jnp.array([0.5, 1.5, 3.0])
    
    # Record mode
    with jnph.hash_mode("record", 0.1) as hashes:
        result = jnph.minimum(jnph.HashTensor(x1), jnph.HashTensor(x2))
        assert jnp.allclose(result.value, jnp.array([0.5, 1.5, 2.5])), f"Expected [0.5, 1.5, 2.5], got {result.value}"
        assert len(hashes) == 1  # One hash for original execution path
    
    # Replay mode
    with jnph.hash_mode("replay", replay_hash=hashes):
        result = jnph.minimum(jnph.HashTensor(x1), jnph.HashTensor(x2))
        assert jnp.allclose(result.value, jnp.array([0.5, 1.5, 2.5])), f"Expected [0.5, 1.5, 2.5], got {result.value}"

    # Test case 2: One value within tolerance
    x1 = jnp.array([1.0, 2.0, 2.5])
    x2 = jnp.array([1.05, 1.5, 3.0])  # 1.05 is within 0.1 of 1.0
    
    # Record mode
    with jnph.hash_mode("record", 0.1) as hashes:
        result = jnph.minimum(jnph.HashTensor(x1), jnph.HashTensor(x2))
        assert jnp.allclose(result.value[1:], jnp.array([1.5, 2.5])), f"Expected [1.5, 2.5] for indices 1:3, got {result.value[1:]}"
        assert result.value[0] in [1.0, 1.05], f"Expected first value to be 1.0 or 1.05, got {result.value[0]}"
        assert len(hashes) == 2, f"Expected 2 hashes (2^1 combinations), got {len(hashes)}"  # One index within tolerance
    
    # Replay mode
    expected_results = set()
    for first_val in [1.0, 1.05]:
        result = jnp.array([first_val, 1.5, 2.5])
        expected_results.add(tuple(result.tolist()))
    
    actual_results = set()
    for hash_choice in hashes:
        with jnph.hash_mode("replay", replay_hash=hash_choice):
            result = jnph.minimum(jnph.HashTensor(x1), jnph.HashTensor(x2))
            actual_results.add(tuple(result.value.tolist()))
    
    assert actual_results == expected_results, f"Expected results {expected_results}, got {actual_results}"

    # Test case 3: Multiple values within tolerance
    x1 = jnp.array([1.0, 2.0, 2.5])
    x2 = jnp.array([1.05, 2.05, 2.55])  # All values within 0.1
    
    # Record mode
    with jnph.hash_mode("record", 0.1) as hashes:
        result = jnph.minimum(jnph.HashTensor(x1), jnph.HashTensor(x2))
        for i in range(3):
            assert result.value[i] in [x1[i], x2[i]], f"Expected value at index {i} to be {x1[i]} or {x2[i]}, got {result.value[i]}"
        assert len(hashes) == 8, f"Expected 8 hashes (2^3 combinations), got {len(hashes)}"  # All combinations of choices
    
    # Replay mode
    expected_results = set()
    for vals in product([x1[0], x2[0]], [x1[1], x2[1]], [x1[2], x2[2]]):
        result = jnp.array(vals)
        expected_results.add(tuple(result.tolist()))
    
    actual_results = set()
    for hash_choice in hashes:
        with jnph.hash_mode("replay", replay_hash=hash_choice):
            result = jnph.minimum(jnph.HashTensor(x1), jnph.HashTensor(x2))
            actual_results.add(tuple(result.value.tolist()))
    
    assert actual_results == expected_results, f"Expected results {expected_results}, got {actual_results}"

def test_abs():
    # Test case 1: No values within tolerance of 0
    x = jnp.array([-1.0, -2.0, 2.5])
    
    # Record mode
    with jnph.hash_mode("record", 0.1) as hashes:
        result = jnph.abs(jnph.HashTensor(x))
        assert jnp.allclose(result.value, jnp.array([1.0, 2.0, 2.5])), f"Expected [1.0, 2.0, 2.5], got {result.value}"
        assert len(hashes) == 1  # One hash for original execution path
    
    # Replay mode
    with jnph.hash_mode("replay", replay_hash=hashes):
        result = jnph.abs(jnph.HashTensor(x))
        assert jnp.allclose(result.value, jnp.array([1.0, 2.0, 2.5])), f"Expected [1.0, 2.0, 2.5], got {result.value}"

    # Test case 2: One value within tolerance of 0
    x = jnp.array([0.05, -2.0, 2.5])  # 0.05 is within 0.1 of 0
    
    # Record mode
    with jnph.hash_mode("record", 0.1) as hashes:
        result = jnph.abs(jnph.HashTensor(x))
        assert result.value[0] == 0.05, f"Expected first value to be 0.05, got {result.value[0]}"
        assert jnp.allclose(result.value[1:], jnp.array([2.0, 2.5])), f"Expected [2.0, 2.5] for indices 1:3, got {result.value[1:]}"
        assert len(hashes) == 2, f"Expected 2 hashes (2^1 combinations), got {len(hashes)}"  # One index within tolerance
    
    # Replay mode
    expected_results = set()
    for first_val in [0.05, -0.05]:
        result = jnp.array([first_val, 2.0, 2.5])
        expected_results.add(tuple(result.tolist()))
    
    actual_results = set()
    for hash_choice in hashes:
        with jnph.hash_mode("replay", replay_hash=hash_choice):
            result = jnph.abs(jnph.HashTensor(x))
            actual_results.add(tuple(result.value.tolist()))
    
    assert actual_results == expected_results, f"Expected results {expected_results}, got {actual_results}"

    # Test case 3: Multiple values within tolerance of 0
    x = jnp.array([0.05, -0.05, 0.03])  # All values within 0.1 of 0
    
    # Record mode
    with jnph.hash_mode("record", 0.1) as hashes:
        result = jnph.abs(jnph.HashTensor(x))
        for i in range(3):
            assert result.value[i] in [abs(x[i]), -abs(x[i])], f"Expected value at index {i} to be +-{abs(x[i])}, got {result.value[i]}"
        assert len(hashes) == 8, f"Expected 8 hashes (2^3 combinations), got {len(hashes)}"  # All combinations of choices
    
    # Replay mode
    expected_results = set()
    for vals in product([0.05, -0.05], [0.05, -0.05], [0.03, -0.03]):
        result = jnp.array(vals)
        expected_results.add(tuple(result.tolist()))
    
    actual_results = set()
    for hash_choice in hashes:
        with jnph.hash_mode("replay", replay_hash=hash_choice):
            result = jnph.abs(jnph.HashTensor(x))
            actual_results.add(tuple(result.value.tolist()))
    
    assert actual_results == expected_results, f"Expected results {expected_results}, got {actual_results}"

def test_hash_set_operations():
    """Test union, intersection, and difference operations."""
    import jax.numpy as jnp
    from jaxnp_hash import hash_mode, HashTensor, max, abs
    
    # Create two different HashSets with different tolerances
    arr1 = jnp.array([1.01, 2.02, 3.03])
    arr2 = jnp.array([0.02, -0.01, 0.03])
    
    # Record with tolerance 0.01 (tight)
    with hash_mode("record", tol=0.01) as tight_hashes:
        result1 = max(HashTensor(arr1))
        result2 = abs(HashTensor(arr2))
    
    # Record with tolerance 0.1 (loose) 
    with hash_mode("record", tol=0.1) as loose_hashes:
        result3 = max(HashTensor(arr1)) 
        result4 = abs(HashTensor(arr2))
    
    print(f"Tight tolerance: {len(tight_hashes)} hashes")
    print(f"Loose tolerance: {len(loose_hashes)} hashes")
    
    # Test union
    union_set = tight_hashes.union(loose_hashes)
    print(f"Union: {len(union_set)} hashes")
    assert len(union_set) >= len(tight_hashes) and len(union_set) >= len(loose_hashes)
    
    # Test intersection
    intersection_set = tight_hashes.intersection(loose_hashes)
    print(f"Intersection: {len(intersection_set)} hashes")
    assert len(intersection_set) <= len(tight_hashes) and len(intersection_set) <= len(loose_hashes)
    
    # Test difference
    diff_set = loose_hashes.difference(tight_hashes)
    print(f"Difference (loose - tight): {len(diff_set)} hashes")
    
    # Verify set properties
    # Union should contain all elements from both sets
    for hash_item in tight_hashes:
        assert hash_item in union_set or any(tight_hashes._hashes_equal(hash_item, u) for u in union_set)
    
    for hash_item in loose_hashes:
        assert hash_item in union_set or any(loose_hashes._hashes_equal(hash_item, u) for u in union_set)
    
    # Intersection should only contain common elements
    for hash_item in intersection_set:
        # Check that this hash exists in both original sets
        in_tight = hash_item in tight_hashes or any(tight_hashes._hashes_equal(hash_item, t) for t in tight_hashes)
        in_loose = hash_item in loose_hashes or any(loose_hashes._hashes_equal(hash_item, l) for l in loose_hashes)
        assert in_tight and in_loose, "Intersection element should be in both sets"
    
    print("All set operations tests passed!")

def test_hash_set_with_no_tolerance():
    """Test HashSet behavior when no values are within tolerance."""
    print("Testing HashSet with no tolerance effects...")
    
    # Use values that are far apart
    x1 = jnp.array([1.0, 5.0, 10.0])
    x2 = jnp.array([2.0, 6.0, 11.0])
    
    with jnph.hash_mode("record", 0.1) as hashes:  # Tolerance too small to matter
        result = jnph.maximum(jnph.HashTensor(x1), jnph.HashTensor(x2))
    
    # Should only have 1 hash since no values are within tolerance
    assert len(hashes) == 1, f"Expected 1 hash when no values within tolerance, got {len(hashes)}"
    
    # Test the single hash
    hash_list = list(hashes)
    assert len(hash_list) == 1, "Should be able to iterate and get exactly 1 hash"
    
    # Test replay with the single hash
    with jnph.hash_mode("replay", replay_hash=hash_list[0]):
        replay_result = jnph.maximum(jnph.HashTensor(x1), jnph.HashTensor(x2))
        expected = jnp.array([2.0, 6.0, 11.0])  # Standard maximum
        assert jnp.allclose(replay_result.value, expected), \
            f"Expected {expected}, got {replay_result.value}"
    
    print("✓ No tolerance test passed!")

def test_hash_set_membership():
    """Test membership operations on HashSet."""
    print("Testing HashSet membership...")
    
    x1 = jnp.array([1.0, 2.0])
    x2 = jnp.array([1.05, 1.95])  # Both within 0.1 tolerance
    
    with jnph.hash_mode("record", 0.1) as hashes:
        result = jnph.maximum(jnph.HashTensor(x1), jnph.HashTensor(x2))
    
    # Should have 4 hashes (2^2)
    assert len(hashes) == 4, f"Expected 4 hashes, got {len(hashes)}"
    
    # Test iteration consistency - each iteration should produce the same number of hashes
    iteration1 = list(hashes)
    iteration2 = list(hashes)
    iteration3 = list(hashes)
    
    assert len(iteration1) == len(iteration2) == len(iteration3) == 4, \
        "Each iteration should produce the same number of hashes"
    
    # Test that all hashes produce valid results when replayed
    valid_results = set()
    for i, hash_choice in enumerate(iteration1):
        # Get fresh copy for replay
        with jnph.hash_mode("record", 0.1) as fresh_hashes:
            result = jnph.maximum(jnph.HashTensor(x1), jnph.HashTensor(x2))
        fresh_list = list(fresh_hashes)
        
        # Use the i-th hash from the fresh list (should be equivalent)
        with jnph.hash_mode("replay", replay_hash=fresh_list[i]):
            replay_result = jnph.maximum(jnph.HashTensor(x1), jnph.HashTensor(x2))
            result_tuple = tuple(replay_result.value.tolist())
            valid_results.add(result_tuple)
            
            # Verify each value is from either x1 or x2
            for j, val in enumerate(replay_result.value):
                assert val in [x1[j], x2[j]], \
                    f"Result value {val} at index {j} should be from x1 or x2"
    
    # Should have exactly 4 unique results (2^2 combinations)
    assert len(valid_results) == 4, \
        f"Expected 4 unique results from 4 hashes, got {len(valid_results)}"
    
    print("✓ Membership test passed!")

def test_hash_set_random_access():
    """Test random access to HashSet elements using [] operator."""
    print("Testing HashSet random access...")
    
    x1 = jnp.array([1.0, 2.0])
    x2 = jnp.array([1.05, 1.95])  # Both within 0.1 tolerance
    
    with jnph.hash_mode("record", 0.1) as hashes:
        result = jnph.maximum(jnph.HashTensor(x1), jnph.HashTensor(x2))
    
    # Should have 4 hashes (2^2)
    assert len(hashes) == 4, f"Expected 4 hashes, got {len(hashes)}"
    
    # Test that we can access all elements by index
    accessed_hashes = []
    for i in range(len(hashes)):
        hash_at_i = hashes[i]
        accessed_hashes.append(hash_at_i)
        print(f"Hash at index {i}: {hash_at_i}")
    
    # Test that random access gives the same results as iteration
    iteration_hashes = list(hashes)
    assert len(accessed_hashes) == len(iteration_hashes), \
        "Random access should give same number of hashes as iteration"
    
    # Compare each hash (they should be in the same order)
    for i, (access_hash, iter_hash) in enumerate(zip(accessed_hashes, iteration_hashes)):
        assert hashes._hashes_equal(access_hash, iter_hash), \
            f"Hash at index {i} should be the same from random access and iteration"
    
    # Test negative indices
    assert hashes._hashes_equal(hashes[-1], hashes[3]), "Negative index -1 should equal index 3"
    assert hashes._hashes_equal(hashes[-2], hashes[2]), "Negative index -2 should equal index 2"
    
    # Test out-of-bounds access
    try:
        _ = hashes[4]  # Should raise IndexError
        assert False, "Should have raised IndexError for index 4"
    except IndexError:
        pass  # Expected
    
    try:
        _ = hashes[-5]  # Should raise IndexError
        assert False, "Should have raised IndexError for index -5"
    except IndexError:
        pass  # Expected
    
    # Test type error for non-integer index
    try:
        _ = hashes["invalid"]  # Should raise TypeError
        assert False, "Should have raised TypeError for string index"
    except TypeError:
        pass  # Expected
    
    # Test consistency - accessing the same index multiple times should give the same result
    first_access = hashes[1]
    second_access = hashes[1]
    assert hashes._hashes_equal(first_access, second_access), \
        "Multiple accesses to same index should give same result"
    
    print("✓ Random access test passed!")


if __name__ == "__main__":
    test_maximum()
    test_minimum()
    test_abs()
    test_hash_set_operations()
    test_hash_set_with_no_tolerance()
    test_hash_set_membership()
    test_hash_set_random_access()
    print("All tests passed!")
