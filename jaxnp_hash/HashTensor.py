import logging
from contextlib import contextmanager
from contextvars import ContextVar

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)

_is_recording: ContextVar[bool] = ContextVar('_is_recording', default=False)
_recorded_trace: ContextVar[list] = ContextVar('_recorded_trace', default=[])
_replay_path: ContextVar[list] = ContextVar('_replay_path', default=None)
_replay_pos: ContextVar[int] = ContextVar('_replay_pos', default=0)
_tolerance: ContextVar[float] = ContextVar('_tolerance', default=0)


class _TraceNode:
    def __init__(self, name, choices):
        logger.debug(f"_TraceNode.__init__: name={name}, num_choices={len(choices)}")
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
        logger.debug(f"_TraceNode.currentChoice: name={self.name}, pos={self.pos}, choice={choice}")
        return choice

    def incrementChoice(self):
        logger.debug(f"_TraceNode.incrementChoice: name={self.name}, pos={self.pos}, num={self.num}")
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
        logger.debug(f"PathSet.__init__: trace with {len(trace)} nodes")

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
            if isinstance(choice_val, tuple) and len(choice_val) == 2:
                nearby_indices, choice_int = choice_val
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
def _branch_mode(mode, tol=0, replay_path=None):
    logger.debug(f"Entering _branch_mode: mode={mode}, tolerance={tol}")

    rec_token = _is_recording.set(False)
    trace_token = _recorded_trace.set([])
    path_token = _replay_path.set(None)
    pos_token = _replay_pos.set(0)
    tol_token = _tolerance.set(0)

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
        else:
            raise Exception(f"Unexpected branch recording mode {mode}.")
    finally:
        _is_recording.reset(rec_token)
        _recorded_trace.reset(trace_token)
        _replay_path.reset(path_token)
        _replay_pos.reset(pos_token)
        _tolerance.reset(tol_token)
        logger.debug("Exiting _branch_mode")


def _trace_append(name, choices):
    trace = _recorded_trace.get()
    logger.debug(f"_trace_append: name={name}, num_choices={len(choices)}")
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


class HashTensor:
    def __init__(self, value):
        logger.debug(f"HashTensor.__init__: value={value}")
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
    logger.debug(f"max: input={inval.value}")
    if _is_recording.get():
        loc = jnp.argmax(inval.value)
        val = inval.value[loc]
        nearby_locs, = jnp.where(inval.value >= val - _tolerance.get())
        nearby_locs = tuple(int(x) for x in nearby_locs.tolist())
        logger.debug(f"max: recording - loc={loc}, val={val}, nearby_locs={nearby_locs}")
        _trace_append("max", nearby_locs)
    else:
        loc = _trace_popf("max")
        val = inval.value[loc]
        logger.debug(f"max: replaying - loc={loc}, val={val}")
    return HashTensor(val)


def min(inval):
    logger.debug(f"min: input={inval.value}")
    if _is_recording.get():
        loc = jnp.argmin(inval.value)
        val = inval.value[loc]
        nearby_locs, = jnp.where(inval.value <= val + _tolerance.get())
        nearby_locs = tuple(int(x) for x in nearby_locs.tolist())
        logger.debug(f"min: recording - loc={loc}, val={val}, nearby_locs={nearby_locs}")
        _trace_append("min", nearby_locs)
    else:
        loc = _trace_popf("min")
        val = inval.value[loc]
        logger.debug(f"min: replaying - loc={loc}, val={val}")
    return HashTensor(val)


def _elementwise_minmax(one, two, name, jnp_op, prefer_first):
    logger.debug(f"{name}: one={one.value}, two={two.value}")
    if _is_recording.get():
        nearby_indices = jnp.where(jnp.abs(one.value - two.value) <= _tolerance.get())[0]
        nearby_indices = tuple(int(x) for x in nearby_indices.tolist())
        logger.debug(f"{name}: recording - nearby_indices={nearby_indices}")
        n_choices = 2 ** len(nearby_indices)
        choices = [(nearby_indices, i) for i in range(n_choices)]
        _trace_append(name, choices)
    else:
        nearby_indices, choice_int = _trace_popf(name)
        nearby_indices = jnp.array(nearby_indices)
        choice = jnp.zeros_like(one.value, dtype=bool)
        for j, idx in enumerate(nearby_indices):
            if (choice_int >> j) & 1:
                choice = choice.at[idx].set(True)
        standard = jnp_op(one.value, two.value)
        if prefer_first:
            flipped = jnp.where(one.value > two.value, two.value, one.value)
        else:
            flipped = jnp.where(one.value < two.value, two.value, one.value)
        result = HashTensor(jnp.where(choice, flipped, standard))
        logger.debug(f"{name}: replaying - choice={choice}, result={result.value}")
        return result
    return HashTensor(jnp_op(one.value, two.value))


def maximum(one, two):
    return _elementwise_minmax(one, two, "maximum", jnp.maximum, prefer_first=True)


def minimum(one, two):
    return _elementwise_minmax(one, two, "minimum", jnp.minimum, prefer_first=False)


def sum(inval):
    logger.debug(f"sum: input={inval.value}")
    result = HashTensor(jnp.sum(inval.value))
    logger.debug(f"sum: result={result.value}")
    return result


def abs(inval):
    logger.debug(f"abs: input={inval.value}")
    if _is_recording.get():
        nearby_indices = jnp.where(jnp.abs(inval.value) <= _tolerance.get())[0]
        nearby_indices = tuple(int(x) for x in nearby_indices.tolist())
        logger.debug(f"abs: recording - nearby_indices={nearby_indices}")
        n_choices = 2 ** len(nearby_indices)
        choices = [(nearby_indices, i) for i in range(n_choices)]
        logger.debug(f"abs: recording - generated {n_choices} choices")
        _trace_append("abs", choices)
    else:
        nearby_indices, choice_int = _trace_popf("abs")
        nearby_indices = jnp.array(nearby_indices)
        negate = jnp.zeros_like(inval.value, dtype=bool)
        for j, idx in enumerate(nearby_indices):
            if (choice_int >> j) & 1:
                negate = negate.at[idx].set(True)
        result = HashTensor(jnp.where(negate, -jnp.abs(inval.value), jnp.abs(inval.value)))
        logger.debug(f"abs: replaying - negate={negate}, result={result.value}")
        return result
    return HashTensor(jnp.abs(inval.value))


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


def replay_value_and_grad(fun, path, argnums=0, has_aux=False):
    def replayed_val_grad(*args, **kwargs):
        with _branch_mode("replay", replay_path=path):
            jax_vg_fn = jax.value_and_grad(fun, argnums=argnums, has_aux=has_aux)
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
        results = []
        for path in paths:
            vg_fn = replay_value_and_grad(fun, path, argnums=argnums, has_aux=has_aux)
            results.append(vg_fn(*args, **kwargs))
        return defaultresult, results, paths
    return all_vg_fn


def h_fun(fun, argnums=0, tol=0.0, has_aux=False):
    import numpy as np

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

            for k, path in enumerate(H0):
                v, g = replay_value_and_grad(fun, path, argnums=argnums, has_aux=has_aux)(z_jax)
                h[k] = float(v)
                grads[:, k] = np.asarray(g)

            return h, grads

    return wrapped
