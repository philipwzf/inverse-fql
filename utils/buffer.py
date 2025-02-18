from flax import struct
from chex import Scalar, Array, ArrayTree, PRNGKey
from typing import Any, Callable
from functools import partial
from flax.struct import dataclass
import numpy as np
import jax
import jax.numpy as jnp

@dataclass
class BufferState:
    data: ArrayTree   # Pytree with leading dimension = buffer capacity
    idx: Scalar       # Next insertion index
    size: Scalar      # Current number of items in buffer
    max_size: Scalar  # Maximum buffer capacity

def _init_buffer(sample_item: ArrayTree, max_size: Scalar) -> BufferState:
    "Initialise replay buffer with a given capacity."
    def allocate(x):
        return jnp.zeros((max_size,) + x.shape, dtype=x.dtype)
    data = jax.tree.map(allocate, sample_item)
    return BufferState(
        data=data,
        idx=jnp.array(0, dtype=jnp.int32),
        size=jnp.array(0, dtype=jnp.int32),
        max_size=jnp.array(max_size, dtype=jnp.int32)
    )

class RolloutBuffer:
    def __init__(self, sample_item: ArrayTree, max_size: int):
        """
        Initialize the replay buffer.
        
        Args:
            sample_item: A pytree representing a single example (e.g., dict with 'state' and 'action').
            max_size: Maximum number of items the buffer can store.
        """
        self.state = _init_buffer(sample_item, max_size)
    
    @staticmethod
    @partial(jax.jit, donate_argnums=(0,))
    def _push(state: BufferState, item: ArrayTree) -> BufferState:
        "Internal JIT-ed function to push a new item into the buffer."
        new_data = jax.tree.map(
            lambda data, item_elem: data.at[state.idx].set(item_elem),
            state.data,
            item
        )
        new_idx = (state.idx + 1) % state.max_size
        new_size = jnp.minimum(state.size + 1, state.max_size)
        return BufferState(
            data=new_data,
            idx=new_idx,
            size=new_size,
            max_size=state.max_size
        )
    
    def push(self, item: ArrayTree):
        """
        Push a new item into the buffer.
        
        Args:
            item: A pytree matching the structure of the sample item.
        """
        self.state = self._push(self.state, item)
    
    @staticmethod
    @partial(jax.jit, static_argnums=(2,))
    def _sample(state: BufferState, key: PRNGKey, batch_size: int) -> ArrayTree:
        "Internal JIT-ed function to sample a batch of items."
        indices = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=state.size)
        return jax.tree.map(lambda data: data[indices], state.data)
    
    def sample(self, key: PRNGKey, batch_size: int) -> ArrayTree:
        """
        Sample a batch of items uniformly from the buffer.
        
        Args:
            key: A PRNG key for random number generation.
            batch_size: Number of items to sample.
            
        Returns:
            A pytree with the same structure as the stored items.
        """
        return self._sample(self.state, key, batch_size)
    
    def get(self) -> ArrayTree:
        """
        Retrieve all valid data from the buffer in insertion order (oldest first).
        
        Returns:
            A pytree of arrays containing all the data in the buffer.
        """
        size = int(self.state.size)
        max_size = int(self.state.max_size)
        if size < max_size:
            # Buffer not full yet: return the first `size` elements.
            return jax.tree.map(lambda arr: arr[:size], self.state.data)
        else:
            # When full, the buffer is circular.
            idx = int(self.state.idx)
            return jax.tree.map(lambda arr: jnp.concatenate([arr[idx:], arr[:idx]], axis=0), self.state.data)

# Tested on 2/18/2025, buffer should be working correctly.
def test_buffer():
    SAMPLE_ITEM = {
        'state': jnp.zeros((4,)),  # 4-dimensional state vector
        'action': jnp.array(0)     # integer action
    }

    MAX_SIZE = 10

    def test_init_buffer():
        """
        Test that a newly initialized buffer returns no data.
        """
        buffer = RolloutBuffer(SAMPLE_ITEM, MAX_SIZE)
        data = buffer.get()
        # Since no item has been pushed yet, expect 0 rows.
        assert data['state'].shape[0] == 0
        assert data['action'].shape[0] == 0

    def test_push_and_get():
        """
        Test that pushing items updates the buffer correctly and that `get` returns items in the correct order.
        """
        max_size = 2
        buffer = RolloutBuffer(SAMPLE_ITEM, max_size)
        
        # Push a single item and verify it appears in get()
        first_item = {'state': jnp.array([1, 2, 3, 4]), 'action': jnp.array(1)}
        buffer.push(first_item)
        data = buffer.get()
        np.testing.assert_array_equal(np.array(data['state'][0]), np.array([1, 2, 3, 4]))
        np.testing.assert_array_equal(np.array(data['action'][0]), np.array(1))
        
        # Push additional items to trigger circular wrap-around.
        for i in range(1, 7):  # Total of 7 pushes in a buffer of capacity 5.
            item = {
                'state': jnp.array([i, i+1, i+2, i+3]),
                'action': jnp.array(i)
            }
            buffer.push(item)
        
        # After 7 pushes, the valid items (in oldest-to-newest order) should be:
        expected_states = jnp.array([
            [2, 3, 4, 5],  # push 3
            [3, 4, 5, 6],  # push 4
            [4, 5, 6, 7],  # push 5
            [5, 6, 7, 8],  # push 6 (wrap-around)
            [6, 7, 8, 9]   # push 7 (wrap-around)
        ])
        expected_actions = jnp.array([2, 3, 4, 5, 6])
        
        data = buffer.get()
        np.testing.assert_array_equal(np.array(data['state']), np.array(expected_states))
        np.testing.assert_array_equal(np.array(data['action']), np.array(expected_actions))

    def test_sample():
        """
        Test that sampling returns data with the correct shapes and valid indices.
        """
        max_size = 10
        buffer = RolloutBuffer(SAMPLE_ITEM, max_size)
        
        # Push 5 items with known values.
        for i in range(5):
            item = {
                'state': jnp.array([i, i+1, i+2, i+3]),
                'action': jnp.array(i)
            }
            buffer.push(item)
        
        batch_size = 3
        key = jax.random.PRNGKey(0)
        sampled = buffer.sample(key, batch_size)
        
        # Verify shapes.
        assert sampled['state'].shape == (batch_size, 4)
        assert sampled['action'].shape == (batch_size,)
        
        # Check that sampled actions are within the valid range (0 to 4).
        for a in np.array(sampled['action']):
            assert 0 <= a < 5
    test_init_buffer()
    test_push_and_get()
    test_sample()