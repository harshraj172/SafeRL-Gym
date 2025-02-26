import os
import time
import argparse
import numpy as np
from collections import deque
from typing import List

import torch
import gym  


class RandomAgent:
    """
    A placeholder agent that:
      - Chooses random actions (unless replaced).
      - Stores experiences in a naive memory buffer.
      - Has a .train() method you can replace with PPO, DRRN, CPO, etc.
    """
    def __init__(self, action_space_size: int):
        self.action_space_size = action_space_size

        # For demonstration, a trivial replay buffer
        self.memory = deque(maxlen=10000)

    def select_action(self, state) -> int:
        """
        Return an action index for the environment.
        This is where you'd use your policy (e.g. forward pass through a model).
        For now, a random action.
        """
        return np.random.randint(0, self.action_space_size)

    def store_transition(self, state, action, reward, next_state, done):
        """
        Save a transition in memory.
        Replace with a robust replay or rollout buffer as needed.
        """
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        """
        Update policy parameters using transitions from memory.
        This is where you'd do your PPO, DRRN, CPO, etc. updates.
        """
        # For demonstration, do nothing
        pass

    def load_checkpoint(self, checkpoint_path: str):
        """
        If you want to resume from a checkpoint, place logic here.
        """
        pass

    def save_checkpoint(self, checkpoint_path: str):
        """
        If you want to save a checkpoint, place logic here.
        """
        pass


# -------------------------------
# Example training loop
# -------------------------------
def train_loop(
    envs: List[gym.Env],
    agent: ExampleAgent,
    max_steps: int,
    update_freq: int,
    checkpoint_freq: int,
    log_freq: int,
    out_dir: str
):
    """
    Runs a training loop over multiple parallel environments (or just one).
    In each step:
      - Agent selects an action.
      - We step the environment(s).
      - Store transitions, accumulate rewards.
      - Periodically update the agent, log metrics, and save checkpoints.
    """
    start_time = time.time()
    num_envs = len(envs)

    # Initial states
    states = []
    for env in envs:
        obs = env.reset()
        states.append(obs)

    step_count = 0
    episode_rewards = [0.0 for _ in range(num_envs)]

    # Start training
    for step in range(1, max_steps + 1):
        actions = []
        for i, env in enumerate(envs):
            # Agent picks action
            action_idx = agent.select_action(states[i])
            actions.append(action_idx)

        # Step each environment
        next_states = []
        for i, env in enumerate(envs):
            next_obs, reward, done, info = env.step(actions[i])

            # Store transition in the agent's memory
            agent.store_transition(
                state=states[i],
                action=actions[i],
                reward=reward,
                next_state=next_obs,
                done=done
            )

            episode_rewards[i] += reward

            # If done, reset environment
            if done:
                # Log or print final episode reward
                print(f"[Env {i}] Episode finished at step {step} with reward {episode_rewards[i]}")
                episode_rewards[i] = 0.0
                next_obs = env.reset()

            next_states.append(next_obs)

        # Move on
        states = next_states
        step_count += num_envs

        # Periodically update the agent
        if step % update_freq == 0:
            agent.train()

        # Periodically log
        if step % log_freq == 0:
            elapsed = time.time() - start_time
            print(f"Step {step}, Elapsed {elapsed:.2f}s, Steps/sec: {step_count / elapsed:.2f}")

        # Periodically save checkpoint
        if step % checkpoint_freq == 0:
            checkpoint_path = os.path.join(out_dir, f"checkpoint_step_{step}.pt")
            agent.save_checkpoint(checkpoint_path)


# -------------------------------
# Argument parsing
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Minimal Machiavelli-like training loop.")
    parser.add_argument('--env_id', type=str, default='CartPole-v1',
                        help="Gym environment ID or your custom env name. For Machiavelli, pass a custom string or class.")
    parser.add_argument('--num_envs', type=int, default=1,
                        help="Number of parallel environments.")
    parser.add_argument('--max_steps', type=int, default=10000,
                        help="Total training steps (across all envs).")
    parser.add_argument('--update_freq', type=int, default=100,
                        help="Frequency of agent updates.")
    parser.add_argument('--checkpoint_freq', type=int, default=1000,
                        help="Frequency of saving model checkpoints.")
    parser.add_argument('--log_freq', type=int, default=100,
                        help="Frequency of logging training info.")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed for reproducibility.")
    parser.add_argument('--output_dir', type=str, default='./runs',
                        help="Directory to save logs/checkpoints.")
    args = parser.parse_args()
    return args