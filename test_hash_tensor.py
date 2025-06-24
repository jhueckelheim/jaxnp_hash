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
    """Test set-like operations on recorded hashes with different tolerances."""
    print("Testing HashSet operations...")
    
    # Test data with values that are close together
    x1 = jnp.array([1.0, 2.0, 3.0])
    x2 = jnp.array([1.02, 1.98, 3.03])  # Values within 0.05 tolerance
    
    # Record with tight tolerance (0.01) - should only get 1 hash (no values within tolerance)
    with jnph.hash_mode("record", 0.01) as hashes_tight:
        result_tight = jnph.maximum(jnph.HashTensor(x1), jnph.HashTensor(x2))
    
    # Record with loose tolerance (0.1) - should get 8 hashes (2^3 combinations)
    with jnph.hash_mode("record", 0.1) as hashes_loose:
        result_loose = jnph.maximum(jnph.HashTensor(x1), jnph.HashTensor(x2))
    
    print(f"Tight tolerance (0.01): {len(hashes_tight)} hashes")
    print(f"Loose tolerance (0.1): {len(hashes_loose)} hashes")
    
    # Test basic set properties
    assert len(hashes_tight) == 1, f"Expected 1 hash with tight tolerance, got {len(hashes_tight)}"
    assert len(hashes_loose) == 8, f"Expected 8 hashes with loose tolerance, got {len(hashes_loose)}"
    assert bool(hashes_tight) == True, "Tight tolerance set should not be empty"
    assert bool(hashes_loose) == True, "Loose tolerance set should not be empty"
    
    # Test iteration - collect all hashes from both sets
    tight_hashes = list(hashes_tight)
    loose_hashes = list(hashes_loose)
    
    print(f"Collected {len(tight_hashes)} hashes from tight tolerance")
    print(f"Collected {len(loose_hashes)} hashes from loose tolerance")
    
    # Test that there are more hashes with loose tolerance than tight tolerance
    assert len(loose_hashes) > len(tight_hashes), \
        "Loose tolerance should have more hashes than tight tolerance"
    
    # Test specific hash replay to verify they work
    print("Testing hash replay...")
    test_count = 0
    for hash_choice in loose_hashes[:3]:  # Test first 3 hashes
        with jnph.hash_mode("replay", replay_hash=hash_choice):
            replay_result = jnph.maximum(jnph.HashTensor(x1), jnph.HashTensor(x2))
            # Verify result is reasonable (should be elementwise max of some combination)
            for i in range(len(replay_result.value)):
                assert replay_result.value[i] in [x1[i], x2[i]], \
                    f"Replay result at index {i} should be from x1 or x2"
            test_count += 1
    
    print(f"Successfully tested replay for {test_count} hash choices")
    
    # Test that the sets produce different results when replayed
    # Note: We need fresh hashes for each test since hash_popf consumes the hash
    
    # Get fresh hashes for tight tolerance  
    with jnph.hash_mode("record", 0.01) as fresh_tight:
        result_tight = jnph.maximum(jnph.HashTensor(x1), jnph.HashTensor(x2))
    fresh_tight_hashes = list(fresh_tight)
    
    replay_results_tight = set()
    for hash_choice in fresh_tight_hashes:
        with jnph.hash_mode("replay", replay_hash=hash_choice):
            result = jnph.maximum(jnph.HashTensor(x1), jnph.HashTensor(x2))
            replay_results_tight.add(tuple(result.value.tolist()))
    
    # Get fresh hashes for loose tolerance
    with jnph.hash_mode("record", 0.1) as fresh_loose:
        result_loose = jnph.maximum(jnph.HashTensor(x1), jnph.HashTensor(x2))
    fresh_loose_hashes = list(fresh_loose)
    
    replay_results_loose = set()
    for hash_choice in fresh_loose_hashes:
        with jnph.hash_mode("replay", replay_hash=hash_choice):
            result = jnph.maximum(jnph.HashTensor(x1), jnph.HashTensor(x2))
            replay_results_loose.add(tuple(result.value.tolist()))
    
    print(f"Tight tolerance produced {len(replay_results_tight)} unique results")
    print(f"Loose tolerance produced {len(replay_results_loose)} unique results")
    
    # Verify that loose tolerance produces more unique results
    assert len(replay_results_loose) > len(replay_results_tight), \
        "Loose tolerance should produce more unique results"
    
    # Verify that tight tolerance results are subset of loose tolerance results
    assert replay_results_tight.issubset(replay_results_loose), \
        "Tight tolerance results should be subset of loose tolerance results"
    
    print("✓ All HashSet operations tests passed!")

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



if __name__ == "__main__":
    test_maximum()
    test_minimum()
    test_abs()
    test_hash_set_operations()
    test_hash_set_with_no_tolerance()
    test_hash_set_membership()
    print("All tests passed!")
