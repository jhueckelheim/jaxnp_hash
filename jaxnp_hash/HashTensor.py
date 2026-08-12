import logging
from contextlib import contextmanager
from contextvars import ContextVar

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

_is_recording: ContextVar[bool] = ContextVar('_is_recording', default=False)
_recorded_trace: ContextVar[list] = ContextVar('_recorded_trace', default=[])
_replay_path: ContextVar[list] = ContextVar('_replay_path', default=None)
_replay_pos: ContextVar[int] = ContextVar('_replay_pos', default=0)
_tolerance: ContextVar[float] = ContextVar('_tolerance', default=0)
_is_vmap_replay: ContextVar[bool] = ContextVar('_is_vmap_replay', default=False)
_vmap_trace: ContextVar[list] = ContextVar('_vmap_trace', default=None)
_vmap_selectors: ContextVar[tuple] = ContextVar('_vmap_selectors', default=None)


class _TraceNode:
    def __init__(self, name, choices):
        logger.debug("_TraceNode.__init__: name=%s, num_choices=%s", name, len(choices))
        self.name = name
        self.choices = list(choices)
        self.num = len(self.choices)
        self.pos = 0

    def __repr__(self):
        return f'_TraceNode(name="{self.name}", pos={self.pos}/{self.num}, current_choice={self.currentChoice()})'

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if not isinstance(other, _TraceNode):
            return NotImplemented
        return self.name == other.name and self.choices == other.choices

    def currentChoice(self):
        choice = self.choices[self.pos]
        logger.debug("_TraceNode.currentChoice: name=%s, pos=%s, choice=%s", self.name, self.pos, choice)
        return choice

    def incrementChoice(self):
        logger.debug("_TraceNode.incrementChoice: name=%s, pos=%s, num=%s", self.name, self.pos, self.num)
        if self.pos + 1 >= self.num:
            self.pos = 0
            return False
        else:
            self.pos += 1
            return True


class PathSet:
    def __init__(self, trace, _empty=False):
        self.trace = trace.copy()
        self._empty = _empty
        logger.debug("PathSet.__init__: trace with %s nodes", len(trace))

    def __iter__(self):
        if self._empty:
            return
        if not self.trace:
            yield []
            return

        for node in self.trace:
            node.pos = 0

        yield self._current_path()

        while self._increment_trace():
            yield self._current_path()

    def _current_path(self):
        return [_TraceNode(node.name, [node.currentChoice()]) for node in self.trace]

    def _increment_trace(self):
        for i in reversed(range(len(self.trace))):
            if self.trace[i].incrementChoice():
                return True
        return False

    def _iter_positions(self):
        # Mirrors __iter__/_increment_trace exactly, but yields the raw per-node
        # `pos` tuple instead of materializing _TraceNode copies -- used to build
        # vmap batch-selector arrays in the SAME path order/identity that __iter__
        # already guarantees (in particular, which path is index 0/"default").
        if self._empty:
            return
        if not self.trace:
            yield ()
            return

        for node in self.trace:
            node.pos = 0

        yield tuple(node.pos for node in self.trace)

        while self._increment_trace():
            yield tuple(node.pos for node in self.trace)

    def __len__(self):
        if self._empty:
            return 0
        total = 1
        for node in self.trace:
            total *= node.num
        return total

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise TypeError("Index must be an integer")

        total_len = len(self)
        if index < 0:
            index = total_len + index

        if index < 0 or index >= total_len:
            raise IndexError(f"Index {index} is out of range for PathSet with {total_len} elements")

        if not self.trace:
            return []

        trace_copy = [_TraceNode(node.name, node.choices) for node in self.trace]

        remaining_index = index
        for i in range(len(trace_copy)):
            node = trace_copy[i]
            combinations_after = 1
            for j in range(i + 1, len(trace_copy)):
                combinations_after *= trace_copy[j].num
            node.pos = remaining_index // combinations_after
            remaining_index = remaining_index % combinations_after

        return [_TraceNode(node.name, [node.currentChoice()]) for node in trace_copy]

    def __contains__(self, item):
        if isinstance(item, list):
            nodes = item
        else:
            return False
        if len(nodes) != len(self.trace):
            return False
        if not nodes and not self.trace:
            return True
        for item_node, trace_node in zip(nodes, self.trace):
            if item_node.name != trace_node.name:
                return False
            if len(item_node.choices) != 1:
                return False
            if item_node.choices[0] not in trace_node.choices:
                return False
        return True

    @staticmethod
    def _paths_equal(path1, path2):
        nodes1 = path1
        nodes2 = path2
        if len(nodes1) != len(nodes2):
            return False
        for node1, node2 in zip(nodes1, nodes2):
            if node1.name != node2.name:
                return False
            if len(node1.choices) != 1 or len(node2.choices) != 1:
                return False
            if node1.choices[0] != node2.choices[0]:
                return False
        return True

    def __bool__(self):
        return len(self) > 0

    def __repr__(self):
        return f'PathSet(trace_length={len(self.trace)}, total_paths={len(self)})'

    def __str__(self):
        return f'PathSet with {len(self)} possible paths from {len(self.trace)} trace nodes'

    def _path_to_tuple(self, path):
        return tuple((node.name, node.choices[0]) for node in path)

    def _create_trace_from_paths(self, path_tuples):
        if not path_tuples:
            return None

        node_choices = {}
        for path_tuple in path_tuples:
            for i, (node_name, choice) in enumerate(path_tuple):
                if i not in node_choices:
                    node_choices[i] = {'name': node_name, 'choices': []}
                if choice not in node_choices[i]['choices']:
                    node_choices[i]['choices'].append(choice)

        trace = []
        for i in sorted(node_choices.keys()):
            node_info = node_choices[i]
            trace.append(_TraceNode(node_info['name'], node_info['choices']))
        return trace

    def _build_from_tuples(self, path_tuples):
        trace = self._create_trace_from_paths(path_tuples)
        if trace is None:
            return PathSet([], _empty=True)
        return PathSet(trace)

    def union(self, other):
        if not isinstance(other, PathSet):
            raise TypeError("Union requires another PathSet")
        all_paths = set()
        for path in self:
            all_paths.add(self._path_to_tuple(path))
        for path in other:
            all_paths.add(self._path_to_tuple(path))
        return self._build_from_tuples(list(all_paths))

    def intersection(self, other):
        if not isinstance(other, PathSet):
            raise TypeError("Intersection requires another PathSet")
        other_paths = set()
        for path in other:
            other_paths.add(self._path_to_tuple(path))
        common_paths = []
        for path in self:
            pt = self._path_to_tuple(path)
            if pt in other_paths:
                common_paths.append(pt)
        return self._build_from_tuples(common_paths)

    def difference(self, other):
        if not isinstance(other, PathSet):
            raise TypeError("Difference requires another PathSet")
        other_paths = set()
        for path in other:
            other_paths.add(self._path_to_tuple(path))
        diff_paths = []
        for path in self:
            pt = self._path_to_tuple(path)
            if pt not in other_paths:
                diff_paths.append(pt)
        return self._build_from_tuples(diff_paths)

    def format_path(self, path):
        nodes = path
        if not nodes:
            return "No decision points"

        lines = []
        for step, node in enumerate(nodes):
            choice_val = node.choices[0]
            if isinstance(choice_val, tuple) and len(choice_val) == 2 and isinstance(choice_val[0], tuple) and isinstance(choice_val[1], tuple):
                # abs()'s collapsed choice: (nearby_indices, base_negate) -- there's
                # only ever one choice now, so there's nothing to report beyond which
                # indices were ambiguous (zero-gradient) at record time.
                nearby_indices, _base_negate = choice_val
                if len(nearby_indices) == 0:
                    lines.append(f"  Step {step+1} ({node.name}): standard absolute value (no ambiguous indices)")
                else:
                    lines.append(f"  Step {step+1} ({node.name}): ambiguous (zero-grad) indices {list(nearby_indices)}")
            elif isinstance(choice_val, tuple) and len(choice_val) >= 2 and isinstance(choice_val[0], tuple):
                nearby_indices, choice_int = choice_val[0], choice_val[1]
                if isinstance(nearby_indices, tuple):
                    if len(nearby_indices) == 0:
                        lines.append(f"  Step {step+1} ({node.name}): standard choice (no nearby values)")
                    else:
                        flipped_indices = [
                            nearby_indices[j] for j in range(len(nearby_indices))
                            if (choice_int >> j) & 1
                        ]
                        if len(flipped_indices) == 0:
                            lines.append(f"  Step {step+1} ({node.name}): standard choice (no flips)")
                        else:
                            lines.append(f"  Step {step+1} ({node.name}): flip indices {flipped_indices} (from nearby {list(nearby_indices)})")
                else:
                    lines.append(f"  Step {step+1} ({node.name}): indices {nearby_indices}, pattern {choice_int:b}")
            elif isinstance(choice_val, tuple):
                true_indices = [j for j, val in enumerate(choice_val) if val]
                if len(true_indices) == 0:
                    lines.append(f"  Step {step+1} ({node.name}): standard absolute value")
                else:
                    lines.append(f"  Step {step+1} ({node.name}): negate at indices {true_indices}")
            else:
                lines.append(f"  Step {step+1} ({node.name}): scalar choice = {choice_val}")
        return "\n".join(lines)


@contextmanager
def _branch_mode(mode, tol=0, replay_path=None, trace=None, selectors=None):
    logger.debug("Entering _branch_mode: mode=%s, tolerance=%s", mode, tol)

    rec_token = _is_recording.set(False)
    trace_token = _recorded_trace.set([])
    path_token = _replay_path.set(None)
    pos_token = _replay_pos.set(0)
    tol_token = _tolerance.set(0)
    vmap_flag_token = _is_vmap_replay.set(False)
    vmap_trace_token = _vmap_trace.set(None)
    vmap_sel_token = _vmap_selectors.set(None)

    try:
        if mode == "record":
            _is_recording.set(True)
            _tolerance.set(tol)
            logger.debug("Starting recording mode")

            trace = _recorded_trace.get()

            try:
                yield trace
            finally:
                pass

        elif mode == "replay":
            if replay_path is None:
                raise ValueError("replay_path must be provided in replay mode")

            if isinstance(replay_path, list):
                _replay_path.set(list(replay_path))
            elif isinstance(replay_path, PathSet):
                if len(replay_path) == 1:
                    path = next(iter(replay_path))
                    _replay_path.set(list(path))
                else:
                    raise ValueError("PathSet with multiple paths cannot be used directly as replay_path. Iterate over it first.")
            else:
                raise TypeError(f"Unexpected replay_path type: {type(replay_path)}")

            _replay_pos.set(0)
            logger.debug("Starting replay mode")
            yield

        elif mode == "vmap_replay":
            if trace is None or selectors is None:
                raise ValueError("trace and selectors must be provided in vmap_replay mode")

            _is_vmap_replay.set(True)
            _vmap_trace.set(trace)
            _vmap_selectors.set(tuple(selectors))
            _replay_pos.set(0)
            logger.debug("Starting vmap_replay mode")
            yield
        else:
            raise Exception(f"Unexpected branch recording mode {mode}.")
    finally:
        _is_recording.reset(rec_token)
        _recorded_trace.reset(trace_token)
        _replay_path.reset(path_token)
        _replay_pos.reset(pos_token)
        _tolerance.reset(tol_token)
        _is_vmap_replay.reset(vmap_flag_token)
        _vmap_trace.reset(vmap_trace_token)
        _vmap_selectors.reset(vmap_sel_token)
        logger.debug("Exiting _branch_mode")


def _trace_append(name, choices):
    trace = _recorded_trace.get()
    logger.debug("_trace_append: name=%s, num_choices=%s", name, len(choices))
    trace.append(_TraceNode(name, choices))


def _trace_popf(name):
    replay = _replay_path.get()
    pos = _replay_pos.get()
    if replay is None:
        raise ValueError("No path provided for replay mode")
    if pos >= len(replay):
        raise ValueError("Path exhausted")
    node = replay[pos]
    _replay_pos.set(pos + 1)
    if node.name != name:
        raise ValueError(f"Expected trace node {name}, got {node.name}")
    return node.choices[0]


def _trace_popf_vmap(name):
    trace = _vmap_trace.get()
    pos = _replay_pos.get()
    selectors = _vmap_selectors.get()
    if trace is None:
        raise ValueError("No trace provided for vmap_replay mode")
    if pos >= len(trace):
        raise ValueError("Path exhausted")
    node = trace[pos]
    selector = selectors[pos]
    _replay_pos.set(pos + 1)
    if node.name != name:
        raise ValueError(f"Expected trace node {name}, got {node.name}")
    return node, selector


class HashTensor:
    def __init__(self, value):
        logger.debug("HashTensor.__init__: value=%s", value)
        self.value = value

    def __repr__(self):
        return f'HashTensor({self.value})'

    def __str__(self):
        return f'HashTensor({self.value})'

    @staticmethod
    def _unwrap(other):
        if isinstance(other, HashTensor):
            return other.value
        return other

    def __add__(self, other):
        return HashTensor(self.value + self._unwrap(other))

    def __radd__(self, other):
        return HashTensor(self._unwrap(other) + self.value)

    def __sub__(self, other):
        return HashTensor(self.value - self._unwrap(other))

    def __rsub__(self, other):
        return HashTensor(self._unwrap(other) - self.value)

    def __mul__(self, other):
        return HashTensor(self.value * self._unwrap(other))

    def __rmul__(self, other):
        return HashTensor(self._unwrap(other) * self.value)

    def __truediv__(self, other):
        return HashTensor(self.value / self._unwrap(other))

    def __rtruediv__(self, other):
        return HashTensor(self._unwrap(other) / self.value)


def max(inval):
    logger.debug("max: input=%s", inval.value)
    if _is_recording.get():
        loc = jnp.argmax(inval.value)
        val = inval.value[loc]
        nearby_locs, = jnp.where(inval.value >= val - _tolerance.get())
        nearby_locs = tuple(int(x) for x in nearby_locs.tolist())
        logger.debug("max: recording - loc=%s, val=%s, nearby_locs=%s", loc, val, nearby_locs)
        _trace_append("max", nearby_locs)
    elif _is_vmap_replay.get():
        node, selector = _trace_popf_vmap("max")
        # node.choices is the recorded tuple of tied indices; selector (a traced,
        # batched scalar under vmap) indexes into it -- one gather, no Python branch.
        nearby_locs_arr = jnp.asarray(node.choices)
        loc = nearby_locs_arr[selector]
        val = inval.value[loc]
        logger.debug("max: vmap replaying - loc=%s, val=%s", loc, val)
    else:
        loc = _trace_popf("max")
        val = inval.value[loc]
        logger.debug("max: replaying - loc=%s, val=%s", loc, val)
    return HashTensor(val)


def min(inval):
    logger.debug("min: input=%s", inval.value)
    if _is_recording.get():
        loc = jnp.argmin(inval.value)
        val = inval.value[loc]
        nearby_locs, = jnp.where(inval.value <= val + _tolerance.get())
        nearby_locs = tuple(int(x) for x in nearby_locs.tolist())
        logger.debug("min: recording - loc=%s, val=%s, nearby_locs=%s", loc, val, nearby_locs)
        _trace_append("min", nearby_locs)
    elif _is_vmap_replay.get():
        node, selector = _trace_popf_vmap("min")
        # node.choices is the recorded tuple of tied indices; selector (a traced,
        # batched scalar under vmap) indexes into it -- one gather, no Python branch.
        nearby_locs_arr = jnp.asarray(node.choices)
        loc = nearby_locs_arr[selector]
        val = inval.value[loc]
        logger.debug("min: vmap replaying - loc=%s, val=%s", loc, val)
    else:
        loc = _trace_popf("min")
        val = inval.value[loc]
        logger.debug("min: replaying - loc=%s, val=%s", loc, val)
    return HashTensor(val)


def _elementwise_minmax(one, two, name, jnp_op, prefer_first):
    logger.debug("%s: one=%s, two=%s", name, one.value, two.value)
    if _is_recording.get():
        nearby_indices = jnp.where(jnp.abs(one.value - two.value) <= _tolerance.get())[0]
        nearby_indices = tuple(int(x) for x in nearby_indices.tolist())
        logger.debug("%s: recording - nearby_indices=%s", name, nearby_indices)
        # Fixed per-index operand identity at record time (True => "two" is the
        # jnp_op winner). Indices outside nearby_indices are unambiguous, so replay
        # must keep this identity rather than recomparing at the (different) replay
        # point -- otherwise replaying this branch at another z silently jumps to
        # whichever operand wins there, instead of extending the recorded manifold.
        base_pick_two = tuple(bool(x) for x in (jnp_op(one.value, two.value) == two.value).tolist())
        n_choices = 2 ** len(nearby_indices)
        choices = [(nearby_indices, i, base_pick_two) for i in range(n_choices)]
        _trace_append(name, choices)
    elif _is_vmap_replay.get():
        node, selector = _trace_popf_vmap(name)
        nearby_indices, _, base_pick_two = node.choices[0]
        # nearby_indices/base_pick_two are the SAME across every choice of this node
        # (only the middle "choice_int" element differs) -- fixed closure constants.
        # selector (a traced, batched scalar under vmap) plays the role choice_int
        # played in the plain-replay branch below, but the bit-unpacking has to be
        # pure jnp ops (one vectorized scatter) instead of a Python loop, since a
        # traced value can't drive Python-level control flow or list mutation.
        m = len(base_pick_two)
        # dtype=np.intp: np.array(()) defaults to float64, which .at[].set() rejects
        # as an indexer -- an empty nearby_indices (no ties) must stay integer-typed.
        nearby_arr = np.array(nearby_indices, dtype=np.intp)
        base_arr = np.array(base_pick_two)
        bits = (selector >> jnp.arange(len(nearby_indices))) & 1
        flip = jnp.zeros(m, dtype=bool).at[nearby_arr].set(bits.astype(bool))
        pick_two = jnp.logical_xor(base_arr, flip)
        result = HashTensor(jnp.where(pick_two, two.value, one.value))
        logger.debug("%s: vmap replaying - pick_two=%s, result=%s", name, pick_two, result.value)
        return result
    else:
        nearby_indices, choice_int, base_pick_two = _trace_popf(name)
        # nearby_indices/choice_int/base_pick_two are plain Python values, so the flip
        # is done as a Python list op -- looping with per-index jnp .at[].set() calls
        # here was dispatching one uncompiled JAX op per tied index per replay, which
        # dominated replay cost when several indices tie at once.
        pick_two = list(base_pick_two)
        for j, idx in enumerate(nearby_indices):
            if (choice_int >> j) & 1:
                pick_two[idx] = not pick_two[idx]
        # Plain numpy here (not jnp.array): pick_two is a constant boolean mask, never
        # differentiated, and jnp.array would dispatch an explicit host->device put; a
        # numpy array is broadcast into jnp.where without that extra dispatch.
        pick_two = np.array(pick_two)
        result = HashTensor(jnp.where(pick_two, two.value, one.value))
        logger.debug("%s: replaying - pick_two=%s, result=%s", name, pick_two, result.value)
        return result
    return HashTensor(jnp_op(one.value, two.value))


def maximum(one, two):
    return _elementwise_minmax(one, two, "maximum", jnp.maximum, prefer_first=True)


def minimum(one, two):
    return _elementwise_minmax(one, two, "minimum", jnp.minimum, prefer_first=False)


def sum(inval):
    logger.debug("sum: input=%s", inval.value)
    result = HashTensor(jnp.sum(inval.value))
    logger.debug("sum: result=%s", result.value)
    return result


def _abs_from_choice(value, nearby_indices, base_negate):
    # Unambiguous components (outside nearby_indices) extend the recorded linear piece
    # with their fixed recorded sign -- gradient is the constant recorded sign, and
    # replaying at a different z intentionally does NOT recompute the sign there (see
    # note below). Ambiguous (near-zero-at-record-time) components instead collapse to
    # a single zero-gradient choice, matching the hand-coded h_one_norm's "0" hash
    # (general_nonsmooth_h_funs.py) rather than enumerating all 2**len(nearby_indices)
    # sign combinations -- that enumeration is exact but intractable once more than a
    # handful of components tie, and (per that hand-coded implementation's own
    # comment) the exactness doesn't matter for the convex-hull criticality
    # calculation manifold sampling actually needs.
    negate = np.array(base_negate)
    ambiguous = np.zeros(len(base_negate), dtype=bool)
    if len(nearby_indices) > 0:
        ambiguous[np.array(nearby_indices, dtype=np.intp)] = True

    linear_part = jnp.where(negate, -value, value)
    # stop_gradient keeps the forward value as the true current abs() (so ambiguous
    # components are always reported correctly), while forcing their gradient
    # contribution to exactly 0 -- jnp.where's gradient only flows through the branch
    # each element selected, so this leaves unambiguous components' ±1 gradient
    # untouched.
    zero_grad_part = jax.lax.stop_gradient(jnp.abs(value))
    return jnp.where(ambiguous, zero_grad_part, linear_part)


def abs(inval):
    logger.debug("abs: input=%s", inval.value)
    if _is_recording.get():
        nearby_indices = jnp.where(jnp.abs(inval.value) <= _tolerance.get())[0]
        nearby_indices = tuple(int(x) for x in nearby_indices.tolist())
        logger.debug("abs: recording - nearby_indices=%s", nearby_indices)
        # Fixed per-index sign at record time: False => extend with "+z_i", True =>
        # extend with "-z_i". Indices outside nearby_indices are unambiguous, so
        # replay must keep this raw linear extension rather than recomputing abs at
        # the (different) replay point's value -- otherwise replaying this branch at
        # another z silently reverts to the true abs there, instead of extending the
        # recorded manifold.
        base_negate = tuple(bool(x) for x in (inval.value < 0).tolist())
        # Exactly one choice regardless of len(nearby_indices) -- see
        # _abs_from_choice's docstring-comment for why we don't enumerate sign
        # combinations for ambiguous components.
        choices = [(nearby_indices, base_negate)]
        _trace_append("abs", choices)
        result = HashTensor(_abs_from_choice(inval.value, nearby_indices, base_negate))
        logger.debug("abs: recording - result=%s", result.value)
        return result
    elif _is_vmap_replay.get():
        node, _selector = _trace_popf_vmap("abs")
        nearby_indices, base_negate = node.choices[0]
        # No selector-driven branch selection needed anymore -- there's only ever one
        # choice per node now, so every batch element uses the same formula.
        result = HashTensor(_abs_from_choice(inval.value, nearby_indices, base_negate))
        logger.debug("abs: vmap replaying - result=%s", result.value)
        return result
    else:
        nearby_indices, base_negate = _trace_popf("abs")
        result = HashTensor(_abs_from_choice(inval.value, nearby_indices, base_negate))
        logger.debug("abs: replaying - result=%s", result.value)
        return result


def record(fun, tol=0.0):
    def recorded(*args, **kwargs):
        with _branch_mode("record", tol=tol) as trace:
            value = fun(*args, **kwargs)
        paths = PathSet(trace)
        return value, paths
    return recorded


def replay(fun, path):
    def replayed(*args, **kwargs):
        with _branch_mode("replay", replay_path=path):
            value = fun(*args, **kwargs)
        return value
    return replayed


def grad(fun, argnums=0, tol=0.0, has_aux=False):
    def grad_fn(*args, **kwargs):
        _, paths = record(fun, tol=tol)(*args, **kwargs)
        default_path = paths[0]

        with _branch_mode("replay", replay_path=default_path):
            jax_grad_fn = jax.grad(fun, argnums=argnums, has_aux=has_aux)
            grad_result = jax_grad_fn(*args, **kwargs)

        if has_aux:
            grads, aux = grad_result
            return (grads, aux), paths
        else:
            return grad_result, paths
    return grad_fn


def value_and_grad(fun, argnums=0, tol=0.0, has_aux=False):
    def val_grad_fn(*args, **kwargs):
        record_result, paths = record(fun, tol=tol)(*args, **kwargs)
        default_path = paths[0]

        if has_aux:
            record_value = record_result[0] if isinstance(record_result, tuple) else record_result
        else:
            record_value = record_result

        with _branch_mode("replay", replay_path=default_path):
            jax_grad_fn = jax.grad(fun, argnums=argnums, has_aux=has_aux)
            grad_result = jax_grad_fn(*args, **kwargs)

        if has_aux:
            grads, aux = grad_result
            return (record_value, grads, aux), paths
        else:
            return (record_value, grad_result), paths
    return val_grad_fn


def replay_grad(fun, path, argnums=0, has_aux=False):
    def replayed_grad(*args, **kwargs):
        with _branch_mode("replay", replay_path=path):
            jax_grad_fn = jax.grad(fun, argnums=argnums, has_aux=has_aux)
            grad_result = jax_grad_fn(*args, **kwargs)

        if has_aux:
            grads, aux = grad_result
            return grads, aux
        else:
            return grad_result
    return replayed_grad


def replay_value_and_grad(fun, path, argnums=0, has_aux=False, _jax_vg_fn=None):
    # _jax_vg_fn lets callers that replay many paths against the same fun (e.g.
    # all_value_and_grad's per-path loop, h_fun's H0 replay loop) build the
    # jax.value_and_grad transform once and reuse it, instead of once per path. This
    # doesn't change tracing/caching behavior -- jax.value_and_grad(fun, ...) itself
    # does no tracing until called, and each call below still runs eagerly under its
    # own _branch_mode, so this is exactly equivalent to rebuilding it every call.
    jax_vg_fn = _jax_vg_fn if _jax_vg_fn is not None else jax.value_and_grad(fun, argnums=argnums, has_aux=has_aux)

    def replayed_val_grad(*args, **kwargs):
        with _branch_mode("replay", replay_path=path):
            vg_result = jax_vg_fn(*args, **kwargs)

        if has_aux:
            (value, aux), grads = vg_result
            return value, grads, aux
        else:
            value, grads = vg_result
            return value, grads
    return replayed_val_grad


def all_value_and_grad(fun, argnums=0, tol=0.0, has_aux=False):
    def all_vg_fn(*args, **kwargs):
        defaultresult, paths = record(fun, tol=tol)(*args, **kwargs)

        if kwargs or not paths.trace:
            # Fall back to the original one-path-at-a-time replay loop when there's
            # nothing to batch over (no tie-break nodes recorded) or for a calling
            # convention (kwargs) the batched jax.vmap path below doesn't support.
            jax_vg_fn = jax.value_and_grad(fun, argnums=argnums, has_aux=has_aux)
            results = []
            for path in paths:
                vg_fn = replay_value_and_grad(fun, path, argnums=argnums, has_aux=has_aux, _jax_vg_fn=jax_vg_fn)
                results.append(vg_fn(*args, **kwargs))
            return defaultresult, results, paths

        # Batch every enumerated path into a SINGLE jax.vmap dispatch instead of one
        # eager jax.value_and_grad call per path. Each trace node's per-path "choice"
        # becomes a traced, batched integer array (selector) fed in as an actual
        # vmapped argument; the tie-break primitives (abs/max/min/maximum/minimum)
        # read it via _is_vmap_replay/_trace_popf_vmap instead of the plain-Python
        # ContextVar path used by "replay" mode. _iter_positions reuses PathSet's
        # existing enumeration order exactly, so path identity (in particular,
        # index 0 == the "default"/no-flip path) is unchanged from before.
        n_args = len(args)
        n_nodes = len(paths.trace)
        positions = list(paths._iter_positions())
        n_paths = len(positions)
        selector_arrays = [jnp.asarray([pos[i] for pos in positions], dtype=jnp.int32) for i in range(n_nodes)]

        def vmap_body(*inner_args):
            call_args = inner_args[:n_args]
            selectors = inner_args[n_args:]
            with _branch_mode("vmap_replay", trace=paths.trace, selectors=selectors):
                return fun(*call_args)

        jax_vg_fn = jax.value_and_grad(vmap_body, argnums=argnums, has_aux=has_aux)
        vmap_vg_fn = jax.vmap(jax_vg_fn, in_axes=(None,) * n_args + (0,) * n_nodes)
        vg_out = vmap_vg_fn(*args, *selector_arrays)

        results = []
        if has_aux:
            (values, aux), grads = vg_out
            for k in range(n_paths):
                aux_k = jax.tree_util.tree_map(lambda a, k=k: a[k], aux)
                results.append((values[k], grads[k], aux_k))
        else:
            values, grads = vg_out
            for k in range(n_paths):
                results.append((values[k], grads[k]))

        return defaultresult, results, paths
    return all_vg_fn


def h_fun(fun, argnums=0, tol=0.0, has_aux=False):

    def wrapped(z, H0=None):
        z_jax = jnp.asarray(z)

        if H0 is None:
            defaultresult, results, paths = all_value_and_grad(fun, argnums=argnums, tol=tol, has_aux=has_aux)(z_jax)

            grads = np.zeros((z_jax.shape[0], len(paths)), dtype=float)
            h_vals = np.zeros(len(paths), dtype=float)
            for k, (v, g) in enumerate(results):
                h_vals[k] = float(v)
                grads[:, k] = np.asarray(g)

            return defaultresult, grads, paths
        else:
            J = len(H0)
            h = np.zeros(J, dtype=float)
            grads = np.zeros((z_jax.shape[0], J), dtype=float)
            jax_vg_fn = jax.value_and_grad(fun, argnums=argnums, has_aux=has_aux)

            for k, path in enumerate(H0):
                v, g = replay_value_and_grad(fun, path, argnums=argnums, has_aux=has_aux, _jax_vg_fn=jax_vg_fn)(z_jax)
                h[k] = float(v)
                grads[:, k] = np.asarray(g)

            return h, grads

    return wrapped
