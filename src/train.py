import os
import time
import argparse
import numpy as np
from collections import deque
from typing import List

import torch
import gym  

from agent.base import RandomAgent

def train_loop(
    envs: List[gym.Env],
    agent: RandomAgent,
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


def main():
    args = parse_args()

    # Make output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create environment(s). For Machiavelli, you might do:
    #   envs = [MachiavelliEnv(game=args.env_id) for _ in range(args.num_envs)]
    # Or for standard gym:
    envs = [gym.make(args.env_id) for _ in range(args.num_envs)]

    # Get action space size (for discrete action envs)
    # In Machiavelli or other text envs, you might do:
    #   agent = ExampleAgent(action_space_size=envs[0].action_space.n)
    action_size = envs[0].action_space.n
    agent = RandomAgent(action_space_size=action_size)

    # Optionally load from a previous checkpoint
    # agent.load_checkpoint('path/to/checkpoint.pt')

    # Run training
    train_loop(
        envs=envs,
        agent=agent,
        max_steps=args.max_steps,
        update_freq=args.update_freq,
        checkpoint_freq=args.checkpoint_freq,
        log_freq=args.log_freq,
        out_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
