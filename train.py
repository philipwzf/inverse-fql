import os
import json
import random
import time

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_env
from utils.evaluation import evaluate, flatten
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb, get_wandb_video
from utils.buffer import RolloutBuffer
from utils.datasets import Dataset, ReplayBuffer

FLAGS = flags.FLAGS

# Basic flags.
flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'InvertedPendulum-v4', 'Environment name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

# Online RL settings.
flags.DEFINE_integer('online_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2048, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

# Evaluation settings.
flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

# Augmentation / frame stacking (if needed).
flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')
flags.DEFINE_integer('balanced_sampling', 0, 'Whether to use balanced sampling for online fine-tuning.')

# Load agent configuration.
config_flags.DEFINE_config_file('agent', 'agents/ppo.py', lock_config=False)

def main(_):
    # Set up experiment logging.
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project='ppo', group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Create environment and evaluation environment.
    
    config = FLAGS.agent
    env = make_env(FLAGS.env_name)
    eval_env = make_env(FLAGS.env_name)

    # Set random seeds.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Create an example batch for agent initialization.
    # Here, we perform a reset and use its observation as a dummy example.
    ob, _ = env.reset()
    # Assume the environment's action space provides a shape attribute.
    action_shape = env.action_space.shape  
    example_obs = jnp.expand_dims(jnp.array(ob), axis=0)
    example_actions = jnp.zeros((1,) + action_shape, dtype=jnp.float32)

    # Create the agent using the online RL configuration.
    agent_class = agents['ppo']
        # Initialize the new functional rollout buffer.
    # Create a sample item that defines the structure of a single transition.
    sample_item = {
        'states': jnp.zeros(example_obs.shape[1:], dtype=jnp.float32),
        'actions': jnp.zeros(example_actions.shape[1:], dtype=jnp.float32),
        'rewards': jnp.zeros((1,), dtype=jnp.float32),
        'dones': jnp.zeros((1,), dtype=jnp.float32),
        'log_pis': jnp.zeros((1,), dtype=jnp.float32),
        'next_states': jnp.zeros(example_obs.shape[1:], dtype=jnp.float32),
    }
    buffer = RolloutBuffer(sample_item, max_size=FLAGS.buffer_size)
    # replay_buffer = ReplayBuffer.create(sample_item, size=FLAGS.buffer_size)
    
    agent = agent_class.create(
        FLAGS.seed,
        example_obs,
        example_actions,
        config,
        buffer
    )
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)


    agent = agent.replace(buffer=buffer)

    # Prepare loggers.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    step = 0
    done = True
    expl_metrics = dict()
    online_rng = jax.random.PRNGKey(FLAGS.seed)

    # Main online RL training loop.
    for i in tqdm.tqdm(range(1, FLAGS.online_steps + 1), smoothing=0.1, dynamic_ncols=True):
        online_rng, key = jax.random.split(online_rng)
        if done:
            step = 0
            ob, _ = env.reset()
        # Sample an action from the agent's policy.
        action, log_pi = agent.sample_actions(observations=ob, temperature=1.0, seed=key)
        action = np.array(action)
        next_ob, reward, terminated, truncated, info = env.step(action.copy())
        done = terminated or truncated

        # Store the transition in the functional buffer.
        agent.store_transition(ob, action, reward, float(done), log_pi, next_ob)
        # replay_buffer.add_transition(
        #         dict(
        #             observations=ob,
        #             actions=action,
        #             rewards=reward,
        #             terminals=float(done),
        #             log_pis=log_pi,
        #             masks=1.0 - terminated,
        #             next_observations=next_ob,
        #         )
        #     )

        ob = next_ob
        step += 1

        # Sample a batch from the buffer for updating the agent.
        if i % FLAGS.buffer_size == 0:
            online_rng, sample_key = jax.random.split(online_rng)
            batch = agent.get_batch()
            # batch = replay_buffer.sample(2048, jnp.arange(2048))
            agent, update_info = agent.update(batch)

        # Logging.
        # if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': float(v) for k, v in update_info.items()}
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            train_metrics.update(expl_metrics)
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluation.
        if FLAGS.eval_interval != 0 and (i == 1 or i % FLAGS.eval_interval == 0):
            renders = []
            eval_metrics = {}
            eval_info, trajs, cur_renders = evaluate(
                agent=agent,
                env=eval_env,
                config=config,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            renders.extend(cur_renders)
            for k, v in eval_info.items():
                eval_metrics[f'evaluation/{k}'] = float(v)
            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders)
                eval_metrics['video'] = video
            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent periodically.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
