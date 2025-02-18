import copy
from typing import Any, Sequence, Tuple

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, LogParam, Value, ppoCritic
from utils.buffer import RolloutBuffer
from jax.experimental import checkify


# def calculate_gae(dones, values, rewards, next_values, gamma, lambd):
#     """
#     Compute advantages and targets using the sample-code style but with direct arrays.
    
#     Args:
#       dones:   Array of done flags of shape (T, 1) (will be squeezed to (T,)).
#       values:  Array of value estimates of shape (T, 1) (will be squeezed to (T,)).
#       rewards: Array of rewards of shape (T, 1) (will be squeezed to (T,)).
#       next_values: Scalar value (or 0-D array) representing the value estimate for the timestep after the trajectory.
#       gamma: Discount factor.
#       lambd: GAE lambda parameter.
    
#     Returns:
#       advantages: Array of advantage estimates (shape (T,)).
#       targets:    Array of targets computed as advantages + values (shape (T,)).
#     """
#     # # Remove extra dimensions so each element is a scalar.
#     dones = jnp.squeeze(dones, axis=-1)    # shape becomes (T,)
#     rewards = jnp.squeeze(rewards, axis=-1)  # shape becomes (T,)
    
#     # Initialize the carry as a tuple: (gae, next_value).
#     # Here, gae is initialized to zero and next_value to last_val.
#     next_values = next_values[-1]
#     init_carry = (jnp.zeros_like(next_values), next_values)
#     # print(f"init_carry: {init_carry[0].shape, init_carry[1].shape}")
    
#     def scan_fn(carry, inputs):
#         gae, next_value = carry
#         d, v, r = inputs
#         delta = r + gamma * next_value * (1 - d) - v
#         new_gae = delta + gamma * lambd * (1 - d) * gae
#         # Pass the current state's value to the next iteration.
#         new_carry = (new_gae, v)
#         return new_carry, new_gae

#     # Perform a reverse scan over the trajectory.
#     # reverse=True makes the scan iterate from the last timestep to the first.
#     _, advantages_rev = jax.lax.scan(
#         scan_fn,
#         init_carry,
#         (dones, values, rewards),
#         reverse=True,
#         unroll=16,
#     )
#     # Reverse the advantages so that they align with the original order.
#     advantages = advantages_rev[::-1]
#     # Compute targets as advantages plus the original values.
#     targets = advantages + values

#         # Normalize advantages.
#     gae_mean = jnp.mean(advantages)
#     gae_std = jnp.std(advantages) + 1e-8
#     gaes_norm = (advantages - gae_mean) / gae_std
#     return targets, gaes_norm
def calculate_gae(dones, values, rewards, next_values, gamma, lambd):
    """
    Compute GAE using a reverse scan. Assumes that `next_values` is an array of 
    the next state values for each timestep, with shape (T,).
    """
    # Remove extra dimensions.
    dones = jnp.squeeze(dones, axis=-1)    # shape (T,)
    rewards = jnp.squeeze(rewards, axis=-1)  # shape (T,)

    # Define the scan function. The carry here is the accumulated GAE from the future.
    def scan_fn(carry, inputs):
        d, v, r, nv = inputs  # nv: next state's value for this timestep
        delta = r + gamma * nv * (1 - d) - v
        new_gae = delta + gamma * lambd * (1 - d) * carry
        return new_gae, new_gae

    # Prepare inputs for the scan. Note that next_values is now an array.
    inputs = (dones, values, rewards, next_values)
    
    # Run a reverse scan (from T-1 down to 0).
    initial_carry = 0.0  # starting with zero advantage from the future.
    _, gaes = jax.lax.scan(scan_fn, 
                               initial_carry, 
                               inputs, 
                               reverse=True,
                               unroll=16,)
    
    # Compute targets as advantages plus values.
    targets = gaes + values

    # Normalize advantages.
    gae_mean = jnp.mean(gaes)
    gae_std = jnp.std(gaes) + 1e-8
    gaes_norm = (gaes - gae_mean) / gae_std
    return targets, gaes_norm

class PPOAgent(flax.struct.PyTreeNode):
    """PPO agent in a similar style as the SAC agent."""
    rng: Any
    network: Any
    config: Any = nonpytree_field()
    # The rollout buffer instance is stored as a non-pytree field.
    buffer: Any = nonpytree_field()

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the combined PPO loss (actor + critic)."""
        info = {}
        rng = rng if rng is not None else self.rng

        # Unpack the batch.
        states      = batch['states']
        actions     = batch['actions']
        rewards     = batch['rewards']
        # costs       = batch['costs']
        log_pis_old = batch['log_pis']
        next_states = batch['next_states']
        dones = batch['dones']

        # Get critic predictions.
        values = self.network.select('critic')(states, params=grad_params)
        next_values = self.network.select('critic')(next_states, params=grad_params)

        # If values come from an ensemble, average over the ensemble dimension.
        # if values.ndim == 2:
        #     values = jnp.mean(values, axis=0).reshape(-1, 1)
        # if next_values.ndim == 2:
        #     next_values = jnp.mean(next_values, axis=0).reshape(-1, 1)

        # Compute targets and advantages using GAE.
        targets, gaes = calculate_gae(
                            dones=dones,
                            values=values,
                            rewards=rewards,
                            next_values=next_values,  # assuming next_values is a scalar or a 0-D array; if it's (batch_size,1), consider squeezing.
                            gamma=self.config['discount'],
                            lambd=self.config['lambd']
                        )

        # Critic loss (mean squared error).
        loss_critic = jnp.mean((values - targets) ** 2)
        info['critic/loss'] = loss_critic

        # Actor loss.
        dist = self.network.select('actor')(states, params=grad_params)
        new_log_pis = dist.log_prob(actions)

        entropy = -jnp.mean(new_log_pis)
        info['actor/entropy'] = entropy

        ratios = jnp.exp(new_log_pis - log_pis_old)
        clip_eps = self.config['clip_eps']

        loss_actor_1 = -ratios * gaes
        loss_actor_2 = -jnp.clip(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * gaes
        loss_actor = jnp.mean(jnp.maximum(loss_actor_1, loss_actor_2))
        info['actor/loss'] = loss_actor

        # Total loss: combine actor and critic losses,
        # with optional coefficients.
        total_loss = loss_actor + self.config['value_coef'] * loss_critic - self.config['coef_ent'] * entropy
        info['total_loss'] = total_loss
        return total_loss, info

    @jax.jit
    def update(self, batch):
        """
        Update the agent's parameters with a batch of transitions.
        Returns an updated agent and a dictionary of loss information.
        """
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        # Update network parameters using a functional loss update.
        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor network."""
        dist = self.network.select('actor')(observations, temperature=temperature)
        actions = dist.sample(seed=seed)
        log_pis = dist.log_prob(actions)
        # Clip actions if necessary.
        actions = jnp.clip(actions, -1, 1)
        return actions, log_pis
    
    @jax.jit
    def get_best_action(
        self,
        observations,
        temperature=1.0,
    ):
        """
        Return the action with max probability (i.e. the mode of the distribution).
        For a Gaussian policy, this is simply the mean.
        """
        # Forward pass through the actor to get the distribution.
        dist = self.network.select('actor')(observations, temperature=temperature)

        # For a diagonal Gaussian, the 'mode' is its mean (also called 'loc' in some distributions).
        # If your policy distribution object doesn't have a `.mode` method, use `.mean`.
        best_action = dist.mean()  # or dist.mode if available

        # (Optional) Clip actions if the environment action space is [-1, 1].
        best_action = jnp.clip(best_action, -1, 1)

        # You can also compute log_prob of this action if you want it for diagnostics.
        log_pis = dist.log_prob(best_action)

        return best_action, log_pis

    # --- Buffer methods using the new functional buffer ---
    def store_transition(self, state, action, reward, done, log_pi, next_state):
        """
        Push a new transition into the buffer.
        Note: The new buffer is updated in place (its internal state is managed by the buffer itself).
        """
        # Create a pytree for the transition. Ensure that rewards, dones, and log_pis are arrays of shape (1,)
        transition = {
            'states': state,
            'actions': action,
            'rewards': jnp.array([reward], dtype=jnp.float32),
            # 'costs': jnp.array([cost], dtype=jnp.float32),
            'dones': jnp.array([done], dtype=jnp.int32),
            'log_pis': jnp.array([log_pi], dtype=jnp.float32),
            'next_states': next_state,
        }
        self.buffer.push(transition)

    def get_batch(self):
        """Retrieve all transitions from the buffer in insertion order."""
        return self.buffer.get()

    def sample_batch(self, num_samples, key):
        """Randomly sample a batch of transitions from the buffer."""
        return self.buffer.sample(key, num_samples)

    @classmethod
    def create(
        cls,
        seed: int,
        ex_observations: jnp.ndarray,
        ex_actions: jnp.ndarray,
        config: ml_collections.ConfigDict,
        buffer: RolloutBuffer,
    ):
        """Create a new PPO agent.
        
        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        action_dim = ex_actions.shape[-1]

        # Define networks.
        # critic_def = Value(
        #     hidden_dims=config['value_hidden_dims'],
        #     layer_norm=config['layer_norm'],
        #     num_ensembles=2,
        # )
        critic_def = ppoCritic(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
        )
        actor_def = Actor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            tanh_squash=config['tanh_squash'],
            state_dependent_std=config['state_dependent_std'],
            const_std=False,
            final_fc_init_scale=config['actor_fc_scale'],
        )

        # Pack the networks into a dictionary.
        network_info = dict(
            actor=(actor_def, (ex_observations,)),
            critic=(critic_def, (ex_observations,)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        # Create a ModuleDict that holds our networks.
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config),buffer=buffer)

def get_config():
    config = ml_collections.ConfigDict(
        dict(
        agent_name='ppo',
        lr=3e-4,
        batch_size= 256,
        actor_hidden_dims= (64, 64),
        value_hidden_dims= (64, 64),
        layer_norm=True,  # Whether to use layer normalization.
        actor_layer_norm=False,  # Whether to use layer normalization for the actor.
        discount= 0.995,
        lambd= 0.97,
        clip_eps= 0.2,
        coef_ent= 0.1,
        value_coef= 1.0,
        tanh_squash=False,
        state_dependent_std=False,
        actor_fc_scale=0.01,
        buffer_max_size=2048,  # Maximum capacity for the buffer.
    ))
    return config
