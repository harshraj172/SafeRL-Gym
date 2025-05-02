import os
import time
import random
import argparse
import logging

import numpy as np
import torch
import gym
from tqdm.auto import tqdm

from agent.base import BaseAgent
from agent.drrn import DRRNAgent
from agent.ppo import PPOAgent
from env.machiavelli.machiavelli_env import MachiavelliEnv, build_state
import logger
from utils import read_json, Transition

# suppress noisy logs
logging.getLogger().setLevel(logging.CRITICAL)

def configure_logger(log_dir, suffix, add_tb=True, add_wb=True, args=None):
    logger.configure(log_dir, format_strs=['log'])
    global tb, log
    tb = logger.Logger(
        log_dir,
        [logger.make_output_format('log', log_dir, log_suffix=suffix),
         logger.make_output_format('json', log_dir, log_suffix=suffix),
         logger.make_output_format('stdout', log_dir, log_suffix=suffix)]
        + ([logger.make_output_format('tensorboard', log_dir, log_suffix=suffix)] if add_tb else [])
        + ([logger.make_output_format('wandb', log_dir, log_suffix=suffix, args=args)] if add_wb else [])
    )
    log = logger.log

def parse_args():
    p = argparse.ArgumentParser()
    # general
    p.add_argument('--output_dir',      default='./machiavelli/game/agent/logs/')
    p.add_argument('--game',            default='alexandria', type=str)
    p.add_argument('--seed',            default=0,           type=int)
    p.add_argument('--env',             default='Machiavelli', type=str)
    p.add_argument('--num_envs',        default=8,           type=int)
    p.add_argument('--agent_type',      default='PPO',       choices=['DRRN','PPO','Random'])
    p.add_argument('--max_steps',       default=50000,       type=int)
    p.add_argument('--update_freq',     default=1,           type=int)
    p.add_argument('--checkpoint_freq', default=1000,        type=int)
    p.add_argument('--log_freq',        default=100,         type=int)
    # logging
    p.add_argument('--tensorboard', default=0, type=int)
    p.add_argument('--wandb',       default=0, type=int)
    p.add_argument('--wandb_project', default='machiavelli_drrn_agent', type=str)
    # machiavelli-specific
    p.add_argument('--eps',            default=None, type=float)
    p.add_argument('--eps_top_k',      default=-1,   type=int)
    p.add_argument('--alpha',          default=0,    type=float)
    p.add_argument('--logit_threshold',default=0.5,  type=float)
    p.add_argument('--reg_weights',    default='0,0,0', type=str)
    p.add_argument('--beta',           default=1,    type=float)
    # common algos
    p.add_argument('--gamma',         default=0.9,   type=float)
    p.add_argument('--learning_rate', default=1e-4,  type=float)
    p.add_argument('--hidden_dim',    default=128,   type=int)
    p.add_argument('--lm_name',       default='microsoft/deberta-v3-large', type=str)
    p.add_argument('--memory_size',     default=10000, type=int)
    p.add_argument('--priority_fraction', default=0.5, type=float)
    p.add_argument('--batch_size',      default=16, type=int)
    return p.parse_args()

def train(agent, envs, args, lm=None):
    start = time.time()
    # per‐env buffers of raw transitions (for passing into agent.observe)
    transitions = [[] for _ in range(args.num_envs)]
    max_score = 0

    # initial reset
    obs, rewards, dones, infos = [], [], [], []
    for i, env in enumerate(envs):
        o, info = env.reset(seed=args.seed)
        obs.append(o)
        rewards.append(0)
        dones.append(False)
        infos.append(info)

    # initial states & valid tokens
    states    = build_state(agent._tokenize, obs, infos)
    valid_ids = [[agent._tokenize(a) for a in info['game_state']['choice_texts']]
                 for info in infos]

    for step in tqdm(range(1, args.max_steps + 1)):

        # 1) act
        action_texts, action_idxs, _ = agent.act(states, valid_ids, lm, args)
        # map idxs back to actual strings for logging & next step
        action_strs = [
            infos[i]['game_state']['choice_texts'][action_idxs[i]]
            for i in range(args.num_envs)
        ]

        # 2) step through each env
        next_obs, next_rewards, next_dones, next_infos = [], [], [], []
        for i, (env, act_idx) in enumerate(zip(envs, action_idxs)):
            o2, r2, done2, info2 = env.step(act_idx)

            # track high‐score and clear alpha‐memory if improved
            if env.curr_points() >= max_score:
                max_score = env.curr_points()
                if hasattr(agent, 'memory'):
                    agent.memory.clear_alpha()

            # if done, log episodic score, and reset
            if done2:
                tb.logkv_mean('EpisodeScore', info2['game_state']['points'])
                o2, info2 = env.reset(seed=args.seed)
                next_rewards.append(r2)
                next_dones.append(True)
                next_obs.append(o2)
                next_infos.append(info2)
            else:
                next_obs.append(o2)
                next_rewards.append(r2)
                next_dones.append(False)
                next_infos.append(info2)

        # 3) logging for env‐0
        log(f">> Action{step}: {action_strs[0]}")
        log(f"Reward{step}: {next_rewards[0]}, "
            f"Score {infos[0]['game_state']['points']}, "
            f"Ach {len(infos[0]['game_state']['unlocked_achievements'])}, "
            f"Done {next_dones[0]}\n")

        # 4) build next states & valids
        next_states = build_state(agent._tokenize, next_obs, next_infos,
                                  prev_obs=obs, prev_acts=action_strs)
        next_valids = [[agent._tokenize(a) for a in info['game_state']['choice_texts']]
                       for info in next_infos]

        # 5) store transitions & call observe
        for i in range(args.num_envs):
            tr = Transition(
                states[i],
                action_texts[i],
                next_rewards[i],
                next_states[i],
                next_valids[i],
                next_dones[i]
            )
            transitions[i].append(tr)
            agent.observe(tr)

            # clear local buffer on done so we don't leak old transitions
            if next_dones[i]:
                transitions[i].clear()

        # 6) advance
        obs, states, valid_ids, infos = (
            next_obs, next_states, next_valids, next_infos
        )

        # 7) periodic logging & update
        if step % args.log_freq == 0:
            tb.logkv("FPS", round((step * args.num_envs) / (time.time() - start), 2))
            tb.logkv("Max score seen", max_score)
            tb.logkv("Step", step)
            tb.dumpkvs()

        if step % args.update_freq == 0:
            agent.update()

        if step % args.checkpoint_freq == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(agent, os.path.join(
                args.output_dir,
                f"{args.env}_game{args.game}_step{step}.pt"
            ))

        # free up GPU memory occasionally
        if step % 1000 == 0:
            torch.cuda.empty_cache()

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    configure_logger(args.output_dir, args.env,
                     bool(args.tensorboard), bool(args.wandb), args)

    # build environments
    if args.env.lower() == 'machiavelli':
        read_json  # to ensure import
        envs = [MachiavelliEnv(game=args.game, desired_achievements=None)
                for _ in range(args.num_envs)]
        args.stats_dim = len(envs[0].player.user_vars)
        lm = None
    else:
        envs = [gym.make(args.env) for _ in range(args.num_envs)]
        lm   = None

    # initialize agent
    if args.agent_type == 'DRRN':
        agent = DRRNAgent(args)
    elif args.agent_type == 'PPO':
        agent = PPOAgent(args)
    else:
        agent = BaseAgent(args)

    train(agent, envs, args, lm)

if __name__ == '__main__':
    main()
