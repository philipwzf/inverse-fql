from typing import Any, Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import orthogonal, constant

def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')

def orthogonal_init(scale=1.0):
    return orthogonal(scale)


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0, 'intermediates': 0},
        split_rngs={'params': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class Identity(nn.Module):
    """Identity layer."""

    def __call__(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False
    bias_init: callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
            if i == len(self.hidden_dims) - 2:
                self.sow('intermediates', 'feature', x)
        return x


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class Actor(nn.Module):
    """Gaussian actor network.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, 
                             activate_final=True, 
                             layer_norm=self.layer_norm, 
                             kernel_init=orthogonal(jnp.sqrt(2)),
                             bias_init=constant(0.0))
        self.mean_net = nn.Dense(self.action_dim, 
                                 kernel_init=orthogonal_init(self.final_fc_init_scale), 
                                 bias_init=constant(0.0))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, 
                                        kernel_init=orthogonal_init(self.final_fc_init_scale),
                                        bias_init=constant(0.0))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        temperature=1.0,
    ):
        """Return action distributions.

        Args:
            observations: Observations.
            temperature: Scaling factor for the standard deviation.
        """
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, 
                                                      scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution
    
    # def evaluate_log_pi(self, states, actions):
    #     """Compute the log-probability of given actions under the policy.

    #     This method re-computes the Gaussian parameters and then evaluates the log probability,
    #     taking into account the tanh squashing transformation.
    #     """
    #     # TODO: This follows the original implementation, need to check if we can make this more efficient.
    #     # Forward pass to obtain the Gaussian parameters.
    #     if self.encoder is not None:
    #         inputs = self.encoder(states)
    #     else:
    #         inputs = states
    #     outputs = self.actor_net(inputs)
    #     means = self.mean_net(outputs)
    #     if self.state_dependent_std:
    #         log_stds = self.log_std_net(outputs)
    #     else:
    #         if self.const_std:
    #             log_stds = jnp.zeros_like(means)
    #         else:
    #             log_stds = self.log_stds
    #     log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        
    #     # Compute the "pre-squash" log probability.
    #     # Transform actions back through atanh.
    #     noises = (jnp.arctanh(actions) - means) / (jnp.exp(log_stds) + 1e-8)
    #     gaussian_log_probs = (
    #         jnp.sum(-0.5 * (noises ** 2) - log_stds, axis=-1, keepdims=True)
    #         - 0.5 * jnp.log(2 * jnp.pi) * log_stds.shape[-1]
    #     )
    #     # Adjust for the tanh squashing transformation.
    #     log_det = jnp.sum(jnp.log(1 - actions**2 + 1e-6), axis=-1, keepdims=True)
    #     return gaussian_log_probs - log_det
class ppoCritic(nn.Module):
    """Critic network with orthogonal initialization.
    
    The first two dense layers use orthogonal initialization with scale √2,
    and the final layer uses orthogonal initialization with scale 1.0.

    Follows the blog post: 
    https://towardsdatascience.com/breaking-down-state-of-the-art-ppo-implementations-in-jax-6f102c06c149/
    """
    hidden_dims: Sequence[int]
    layer_norm: bool = True
    encoder: nn.Module = None

    def setup(self):
        # Instead of a generic MLP, we explicitly construct the network
        # to control each layer's initializer.
        self.dense1 = nn.Dense(
            64,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0)
        )
        self.dense2 = nn.Dense(
            64,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0)
        )
        # Final output layer with scale 1.0.
        self.out = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0)
        )

    def __call__(self, observations, actions=None):
        # Optionally process observations with an encoder.
        inputs = [self.encoder(observations)] if self.encoder is not None else [observations]
        if actions is not None:
            inputs.append(actions)
        x = jnp.concatenate(inputs, axis=-1)
        x = self.dense1(x)
        x = nn.tanh(x)
        x = self.dense2(x)
        x = nn.tanh(x)
        v = self.out(x)
        return jnp.squeeze(v, axis=-1)


class Value(nn.Module):
    """Value/critic network.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        value_net = mlp_class((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)

        self.value_net = value_net

    def __call__(self, observations, actions=None):
        """Return values or critic values.

        Args:
            observations: Observations.
            actions: Actions (optional).
        """
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs).squeeze(-1)

        return v


class ActorVectorField(nn.Module):
    """Actor vector field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None

    def setup(self) -> None:
        self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)

    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False):
        """Return the vectors at the given states, actions, and times (optional).

        Args:
            observations: Observations.
            actions: Actions.
            times: Times (optional).
            is_encoded: Whether the observations are already encoded.
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            inputs = jnp.concatenate([observations, actions, times], axis=-1)

        v = self.mlp(inputs)

        return v
