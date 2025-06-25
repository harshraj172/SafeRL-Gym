"""
Train a DRRN agent on the Machiavelli environment.

USAGE:

    DRRN:
    python -m src.train --max_steps 10000 --game kung-fu --agent_type DRRN --output_dir ./results/Machiavelli/DRRN/microsoft_deberta-v3-xsmall/kung-fu

    PPO LLM:
    python -m src.train --max_steps 1000 --game kung-fu --agent_type PPO_LLM --lm_name google/gemma-3-1b-it --output_dir ./results/Machiavelli/PPO_LLM/google_gemma-3-1b-it/kung-fu
"""

import os
import time
import json
import random
import argparse
import logging
from typing import List

import numpy as np

# Patch for NumPy versions that do not expose np.bool8.
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import torch
import gym

from tqdm.auto import tqdm
from src.agent.base import BaseAgent
from src.agent.drrn import DRRNAgent
from src.agent.ppo_llm import PPOLLMAgent
from src.agent.ppo_lag_llm import PPOLagLLMAgent
from src.env.machiavelli.machiavelli_env import MachiavelliEnv, build_state
from src.env.base import BaseEnv
from src.env.test import SuperEasyEnv
from src import logger
from src.utils import (
    read_json,
    ReplayMemory,
    PrioritizedReplayMemory,
    Transition,
    State,
)

logging.getLogger().setLevel(logging.CRITICAL)


def configure_logger(log_dir, suffix, add_tb=1, add_wb=1, args=None):
    logger.configure(log_dir, format_strs=["log"])
    global tb
    log_types = [
        logger.make_output_format("log", log_dir, log_suffix=suffix),
        logger.make_output_format("json", log_dir, log_suffix=suffix),
        logger.make_output_format("stdout", log_dir, log_suffix=suffix),
    ]
    if add_tb:
        log_types += [
            logger.make_output_format("tensorboard", log_dir, log_suffix=suffix)
        ]
    if add_wb:
        log_types += [
            logger.make_output_format("wandb", log_dir, log_suffix=suffix, args=args)
        ]
    tb = logger.Logger(log_dir, log_types)
    global log
    log = logger.log


def parse_args():
    """Parses command-line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", default="./results/", type=str, help="Directory to save results"
    )
    parser.add_argument(
        "--game", type=str, default="kung-fu", help="game present in the env"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--env", default="Machiavelli", type=str, help="environment name"
    )
    parser.add_argument(
        "--num_envs",
        default=2, # 8 for DRRN, 2 for PPO_LLM
        type=int,
        help="Number of instances of env (for diverse rollouts)",
    )
    parser.add_argument(
        "--agent_type",
        default="DRRN",
        type=str,
        help="Type of agent to use: DRRN | PPO_LLM | PPO_LLM_LAG",
    )
    parser.add_argument("--max_steps", default=5000, type=int)
    parser.add_argument("--update_freq", default=1, type=int)
    parser.add_argument("--checkpoint_freq", default=1000, type=int)
    parser.add_argument("--eval_freq", default=5000, type=int)
    parser.add_argument("--log_freq", default=100, type=int)
    parser.add_argument("--memory_size", default=10000, type=int)
    parser.add_argument("--priority_fraction", default=0.5, type=float)
    parser.add_argument("--batch_size", default=16, type=int)

    # logger
    parser.add_argument("--tensorboard", default=0, type=int)
    parser.add_argument("--wandb", default=0, type=int)
    parser.add_argument("--wandb_project", default="machiavelli_drrn_agent", type=str)

    # Machiavelli specific
    parser.add_argument(
        "--eps",
        default=None,
        type=float,
        help="None: ~ softmax act_value; else eps-greedy-exploration",
    )
    parser.add_argument(
        "--eps_top_k",
        default=-1,
        type=int,
        help="-1: uniform exploration; 0: ~ softmax lm_value; >0: ~ uniform(top k w.r.t. lm_value)",
    )
    parser.add_argument(
        "--alpha",
        default=0,
        type=float,
        help="act_value = -alpha * bert_value + q_value; used to penalize powerful actions",
    )
    parser.add_argument(
        "--logit_threshold",
        default=0.5,
        type=float,
        help="threshold for logit to be considered as a high utility action",
    )
    parser.add_argument(
        "--reg_weights",
        default="0,0,0",
        type=str,
        help="comma-separated string of weights for the model regularizer",
    )
    parser.add_argument("--beta", default=1, type=float, help="temperature for softmax")

    # DRRN specific
    parser.add_argument("--gamma", default=0.9, type=float)
    parser.add_argument("--learning_rate", default=0.0001, type=float)
    parser.add_argument("--clip", default=5, type=float)
    parser.add_argument("--embedding_dim", default=128, type=int)
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--lm_name", default="microsoft/deberta-v3-xsmall", type=str)
    parser.add_argument(
        "--regularizer_lm_name", default="microsoft/deberta-v3-large", type=str
    )
    parser.add_argument(
        "--lm_top_k",
        default=30,
        type=int,
        help="when >0, use lm top-k actions in place of jericho action detection",
    )
    parser.add_argument("--lm_type", default="gpt", help="gpt | ngram")
    parser.add_argument("--lm_path", default="gpt2")
    parser.add_argument("--lm_dict", default="")
    parser.add_argument(
        "--cost_limit",
        default=100.0,
        type=float,
        help="cost limit for lagging agents",
    )
    return parser.parse_args()


def log_episode_stats(tb, info, step):
    tb.logkv_mean("EpisodeScore", info["game_state"]["points"])
    tb.logkv("EpisodeLength", step)
    tb.logkv_mean("NumAch", len(info["game_state"]["unlocked_achievements"]))
    tb.logkv_mean("Cost", info["game_state"]["cumulative_cost"])
    tb.logkv_mean("Reward", info["game_state"]["reward"])


def log_step_stats(log, step, action_str, reward, info, done):
    log(f">> Action{step}: {action_str}")
    log(
        f"Reward{step}: {reward}, Score {info['game_state']['points']}, Num ach {len(info['game_state']['unlocked_achievements'])}, Done {done}\n"
    )
    log(
        f"Safety cost {info['game_state']['cumulative_cost']}, Reward {info['game_state']['reward']}"
    )


def train(
    agent,
    envs,
    max_steps,
    update_freq,
    eval_freq,
    checkpoint_freq,
    log_freq,
    args,
    lm=None,
):
    start = time.time()

    obs, rewards, dones, infos, transitions = [], [], [], [], []
    max_score = 0

    # Initialize each environment
    for env_i, env in enumerate(envs):
        ob, info = env.reset(seed=args.seed)
        obs.append(ob)
        rewards.append(0)
        dones.append(False)
        infos.append(info)
        transitions.append([])

    # Convert raw text obs -> agent states
    states = build_state(agent._tokenize, obs, infos)

    # Each environment must provide the valid actions in some form
    valid_ids = [
        [agent._tokenize(a) for a in info["game_state"]["choice_texts"]]
        for info in infos
    ]

    for step in tqdm(range(1, max_steps + 1)):
        # act
        action_ids, action_idxs, _ = agent.act(states, valid_ids, lm, args)
        action_strs = [
            info["game_state"]["choice_texts"][idx]
            for info, idx in zip(infos, action_idxs)
        ]

        # step
        next_obs, next_rewards, next_dones, next_infos = [], [], [], []
        costs = []
        for i, (env, action_str, action_idx) in enumerate(
            zip(envs, action_strs, action_idxs)
        ):
            ob, reward, done, info = env.step(action_idx)
            costs.append(info["game_state"]["safety_cost"])
            if env.curr_points() >= max_score:  # new high score experienced
                max_score = env.curr_points()
                agent.memory.clear_alpha() if hasattr(agent, "memory") else None
            if done:
                log_episode_stats(tb, info, step)
                if env.curr_points() >= max_score:  # put in alpha queue
                    for transition in transitions[i]:
                        agent.observe(transition, is_prior=True)
                ob, info = env.reset(seed=args.seed)
                next_obs, next_rewards, next_dones, next_infos = (
                    next_obs + [ob],
                    next_rewards + [0],
                    next_dones + [False],
                    next_infos + [info],
                )
            else:
                next_obs, next_rewards, next_dones, next_infos = (
                    next_obs + [ob],
                    next_rewards + [reward],
                    next_dones + [done],
                    next_infos + [info],
                )
        rewards, dones, infos = next_rewards, next_dones, next_infos

        # Logging for envs[0]
        log_step_stats(log, step, action_strs[0], rewards[0], infos[0], dones[0])

        next_states = build_state(
            agent._tokenize, next_obs, infos, prev_obs=obs, prev_acts=action_strs
        )
        next_valids = [
            [agent._tokenize(a) for a in info["game_state"]["choice_texts"]]
            for info in infos
        ]
        for (
            state,
            act,
            rew,
            valid_action_id,
            next_state,
            valids,
            done,
            transition,
            cost,
        ) in zip(
            states,
            action_ids,
            rewards,
            valid_ids,
            next_states,
            next_valids,
            dones,
            transitions,
            costs,
        ):
            if len(act) > 0:  # not [] (i.e. reset)
                transition.append(
                    Transition(
                        state=state,
                        act=act,
                        reward=rew,
                        valid_acts=valid_action_id,
                        next_state=next_state,
                        next_acts=valids,
                        done=done,
                        cost=cost,
                    )
                )
                agent.observe(transition[-1])  # , is_prior=(rew != 0)

        # Advance
        obs, states, valid_ids = next_obs, next_states, next_valids

        if step % log_freq == 0:
            tb.logkv("FPS", round((step * args.num_envs) / (time.time() - start), 2))
            tb.logkv("Max score seen", max_score)
            tb.logkv("Step", step)
            tb.dumpkvs()
        if step % update_freq == 0:
            loss = agent.update()
            tb.logkv_mean("Loss", loss)
        if step == 1 or step % checkpoint_freq == 0:
            savedir = "./checkpoints_utility/"
            os.makedirs(savedir, exist_ok=True)
            if "LLM" in args.agent_type:
                # Save only model weights for PPO_LLM
                torch.save(
                    agent.actor.state_dict(),
                    savedir
                    + "{}_{}_{}_game{}_seed_{}_actor.pt".format(
                        args.env,
                        args.agent_type,
                        args.lm_name.replace("/", "_"),
                        args.game,
                        args.seed
                    ),
                )
                torch.save(
                    agent.critic.state_dict(),
                    savedir
                    + "{}_{}_{}_game{}_seed_{}_critic.pt".format(
                        args.env,
                        args.agent_type,
                        args.lm_name.replace("/", "_"),
                        args.game,
                        args.seed
                    ),
                )
            else:
                torch.save(
                    agent,
                    savedir
                    + "{}_{}_{}_game{}_seed_{}.pt".format(
                        args.env,
                        args.agent_type,
                        args.lm_name.replace("/", "_"),
                        args.game,
                        args.seed
                    ),
                )
            # save args as json
            with open(
                savedir
                + "{}_{}_{}_game{}_seed_{}.json".format(
                    args.env, args.agent_type, args.lm_name.replace("/", "_"), args.game,
                    args.seed
                ),
                "w",
            ) as f:
                json.dump(vars(args), f, indent=4)
        del action_ids, action_idxs
        if step % 1000 == 0:
            print("empty cache")
            torch.cuda.empty_cache()


def main():
    # Parse args
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = os.path.join(
        args.output_dir,
        args.env,
        args.agent_type,
        args.lm_name.replace("/", "_"),
        args.game,
    )
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Configure logger
    suffix = args.env
    configure_logger(output_dir, suffix, args.tensorboard, args.wandb, args)

    # Create environments
    envs = []
    lm = None
    if args.env.lower() == "machiavelli":
        game2beta = read_json("src/env/machiavelli/game2beta.json")
        weights = [float(w) for w in args.reg_weights.split(",")]
        if weights == [0, 0, 0]:
            print("No regularization")

        else:
            lm = PowerRegularizer(
                args.regularizer_lm_name,
                "models/multi_deberta-v3-large_1e-05_32_3.pkl",
                args.reg_weights,
                utility_thresh=args.logit_threshold,
            )

        envs = [
            MachiavelliEnv(
                game=args.game,
                desired_achievements=None,
            )
            for _ in range(args.num_envs)
        ]
        args.stats_dim = len(envs[0].player.user_vars)
    elif args.env.lower() == "test":
        # Test environment for debugging
        envs = [SuperEasyEnv() for _ in range(args.num_envs)]
        args.stats_dim = 3  # Example stats dimension for SuperEasyEnv
    else:
        if args.games:
            for g in args.games:
                envs.append(BaseEnv(game=g, seed=args.seed))
        else:
            # Single custom environment with default
            envs = [
                BaseEnv(game="BaseGame", seed=args.seed) for _ in range(args.num_envs)
            ]

    # Initialize the agent
    # Possibly pass environment-specific dimensions
    if args.agent_type == "DRRN":
        agent = DRRNAgent(args)
    elif args.agent_type == "PPO_LLM":
        agent = PPOLLMAgent(args=args, max_reward=envs[0].max_reward)
    elif args.agent_type == "PPO_LLM_LAG":
        agent = PPOLagLLMAgent(
            args=args, 
            max_reward=envs[0].max_reward,
            cost_limit=envs[0].two_sigma_min_cost,
            max_cost=envs[0].two_sigma_max_cost
        )
    else:
        # Random Agent
        agent = BaseAgent(args)

    # Train
    train(
        agent=agent,
        envs=envs,
        max_steps=args.max_steps,
        update_freq=args.update_freq,
        eval_freq=args.eval_freq,
        checkpoint_freq=args.checkpoint_freq,
        log_freq=args.log_freq,
        args=args,
        lm=lm,
    )


if __name__ == "__main__":
    main()
