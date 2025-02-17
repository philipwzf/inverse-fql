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
    # TODO: adapt to more envs similar to d4rl.
    env = gymnasium.make(env_name, render_mode='rgb_array')
    print(env.env.env.env.render_mode)
    print(env.env.env.env.__dict__)
    # env = EpisodeMonitor(env)
    return env

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
        cost = info['cost'] if 'cost' in info else print('No cost in info')

        # Remove keys that are not needed for logging.
        for filter_regex in self.filter_regexes:
            for key in list(info.keys()):
                if re.match(filter_regex, key) is not None:
                    del info[key]

        self.reward_sum += reward
        self.cost_sum += cost
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if terminated or truncated:
            info['episode'] = {}
            info['episode']['final_reward'] = reward
            info['episode']['return'] = self.reward_sum
            info['episode']['cost'] = self.cost_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

            # TODO: add normalized return and cost.
            # if hasattr(self.unwrapped, 'get_normalized_score'):
            #     info['episode']['normalized_return'],info['episode']['normalized_cost'] = (
            #         self.unwrapped.get_normalized_score(info['episode']['return'],info['episode']['cost']) * 100.0
            #     )

        return observation, reward, cost, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)

env = make_env('OfflinePointButton2Gymnasium-v0')
env.reset()
print(env.render())