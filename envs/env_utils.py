import collections
import re
import time

import gymnasium
import numpy as np
import dsrl
from gymnasium.spaces import Box

from utils.datasets import Dataset

def make_env(env_name):
    """Make DSRL environment."""
    # TODO: adapt to more envs similar to gymnasium.
    env = gymnasium.make(env_name, render_mode='rgb_array')
    # env = gymnasium.make('GymV21Environment-v0', env_id=env_name)
    env = EpisodeMonitor(env)
    return env


def get_dataset(
    env,
    env_name,
    safe_only=False
):
    """Make DSRL dataset.

    Args:
        env: Environment instance.
        env_name: Name of the environment.
    """
    dataset = env.get_dataset()

    terminals = np.zeros_like(dataset['rewards'])  # Indicate the end of an episode.
    masks = np.zeros_like(dataset['rewards'])  # Indicate whether we should bootstrap from the next state.
    rewards = dataset['rewards'].copy().astype(np.float32)
    costs = dataset['costs'].copy().astype(np.float32)

    for i in range(len(terminals) - 1):
        if (
            np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6
            or dataset['terminals'][i] == 1.0
        ):
            terminals[i] = 1
        else:
            terminals[i] = 0
        masks[i] = 1 - dataset['terminals'][i]
    print(costs.sum())
    # If safe_only is True, filter out transitions where cost > 0.
    if safe_only:
        # Create a boolean mask where True indicates a safe transition.
        safe_idx = costs == 0

        observations = dataset['observations'][safe_idx].astype(np.float32)
        actions = dataset['actions'][safe_idx].astype(np.float32)
        next_observations = dataset['next_observations'][safe_idx].astype(np.float32)
        terminals = terminals[safe_idx].astype(np.float32)
        rewards = rewards[safe_idx]
        costs = costs[safe_idx]
        masks = masks[safe_idx]
    else:
        observations = dataset['observations'].astype(np.float32)
        actions = dataset['actions'].astype(np.float32)
        next_observations = dataset['next_observations'].astype(np.float32)
    masks[-1] = 1 - dataset['terminals'][-1]
    terminals[-1] = 1
    return Dataset.create(
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        terminals=terminals,
        rewards=rewards,
        costs=costs,
        masks=masks,
    )


class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics."""

    def __init__(self, env, filter_regexes=None):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0
        self.filter_regexes = filter_regexes if filter_regexes is not None else []

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.cost_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        # cost = info['cost'] if 'cost' in info else print('No cost in info')

        # Remove keys that are not needed for logging.
        for filter_regex in self.filter_regexes:
            for key in list(info.keys()):
                if re.match(filter_regex, key) is not None:
                    del info[key]

        self.reward_sum += reward
        # self.cost_sum += cost
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if terminated or truncated:
            info['episode'] = {}
            info['episode']['final_reward'] = reward
            info['episode']['return'] = self.reward_sum
            # info['episode']['cost'] = self.cost_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

            # TODO: add normalized return and cost.
            # if hasattr(self.unwrapped, 'get_normalized_score'):
            #     info['episode']['normalized_return'],info['episode']['normalized_cost'] = (
            #         self.unwrapped.get_normalized_score(info['episode']['return'],info['episode']['cost']) * 100.0
            #     )

        # return observation, reward, cost, terminated, truncated, info
        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class FrameStackWrapper(gymnasium.Wrapper):
    """Environment wrapper to stack observations."""

    def __init__(self, env, num_stack):
        super().__init__(env)

        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)

        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if 'goal' in info:
            info['goal'] = np.concatenate([info['goal']] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        # cost = info['cost'] if 'cost' in info else print('No cost in info')

        self.frames.append(ob)
        # return self.get_observation(), reward, cost, terminated, truncated, info
        return self.get_observation(), reward, terminated, truncated, info

def make_env_and_datasets(env_name, frame_stack=None, action_clip_eps=1e-5):
    """Make offline RL environment and datasets.

    Args:
        env_name: Name of the environment or dataset.
        frame_stack: Number of frames to stack.
        action_clip_eps: Epsilon for action clipping.

    Returns:
        A tuple of the environment, evaluation environment, training dataset, and validation dataset.
    """

    # if 'singletask' in env_name:
    #     # OGBench.
    #     env, train_dataset, val_dataset = ogbench.make_env_and_datasets(env_name)
    #     eval_env = ogbench.make_env_and_datasets(env_name, env_only=True)
    #     env = EpisodeMonitor(env, filter_regexes=['.*privileged.*', '.*proprio.*'])
    #     eval_env = EpisodeMonitor(eval_env, filter_regexes=['.*privileged.*', '.*proprio.*'])
    #     train_dataset = Dataset.create(**train_dataset)
    #     val_dataset = Dataset.create(**val_dataset)
    # elif 'antmaze' in env_name and ('diverse' in env_name or 'play' in env_name or 'umaze' in env_name):
    #     # D4RL AntMaze.
    #     from envs import d4rl_utils

    #     env = d4rl_utils.make_env(env_name)
    #     eval_env = d4rl_utils.make_env(env_name)
    #     dataset = d4rl_utils.get_dataset(env, env_name)
    #     train_dataset, val_dataset = dataset, None
    # elif 'pen' in env_name or 'hammer' in env_name or 'relocate' in env_name or 'door' in env_name:
    #     # D4RL Adroit.
    #     import d4rl.hand_manipulation_suite  # noqa
    #     from envs import d4rl_utils

    #     env = d4rl_utils.make_env(env_name)
    #     eval_env = d4rl_utils.make_env(env_name)
    #     dataset = d4rl_utils.get_dataset(env, env_name)
    #     train_dataset, val_dataset = dataset, None
    # if 'halfcheetah' in env_name:
    env = make_env(env_name)
    eval_env = make_env(env_name)
    dataset = get_dataset(env, env_name)
    train_dataset, val_dataset = dataset, None
    # else:
    #     raise ValueError(f'Unsupported environment: {env_name}')

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)
        eval_env = FrameStackWrapper(eval_env, frame_stack)

    env.reset()
    eval_env.reset()

    # Clip dataset actions.
    if action_clip_eps is not None:
        train_dataset = train_dataset.copy(
            add_or_replace=dict(actions=np.clip(train_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
        )
        if val_dataset is not None:
            val_dataset = val_dataset.copy(
                add_or_replace=dict(actions=np.clip(val_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
            )

    return env, eval_env, train_dataset, val_dataset