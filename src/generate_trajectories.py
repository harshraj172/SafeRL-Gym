"""
Generate trajectories for a given agent and environment.
    Usage:
        python generate_trajectories.py -a .drrn -t ./trajectories/<agent_name>
"""
import argparse
from pathlib import Path
from typing import List
import traceback

from tqdm import tqdm
import torch

from agent.base import BaseAgent
from env.machiavelli.config import MachiavelliConfig
from env.machiavelli.machiavelli_env import MachiavelliEnv, Trajectory, build_state

cfg = MachiavelliConfig()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_trajectories(agent, env, outdir, num_episodes=1_000, max_traj_length=1000):
    print(f"------ Testing {agent} on {env.game} ------")

    for episode_idx in tqdm(range(num_episodes)):
        # Start new episode.
        obs, info = env.reset(seed=episode_idx)

        rew = 0
        done = False
        step_count = 0
        while not done:
            # Convert raw text obs -> agent states
            states = build_state(agent._tokenize, obs, [info])

            # print(obs, rewards, dones, infos, transitions)
            # print(states)
            
            # Each environment must provide the valid actions in some form
            valid_ids = [
                [agent._tokenize(a) for a in info['game_state']['choice_texts']]
            ]
            action_ids, action_idxs, _ = agent.act(
                states, valid_ids
            )
            obs, rew, done, info = env.step(action_idxs[0])

            step_count += 1
            # Terminate early if we've reached the maximum trajectory length
            if step_count >= max_traj_length:
                print("Reached maximum trajectory length! Terminating early.")
                done = True

        print(env.trajectory)

        # Save trajectory to file
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = Path(outdir) / f"{env.trajectory.timestamp}.pkl"
        outfile.parent.mkdir(parents=True, exist_ok=True)
        env.trajectory.save(outfile)

    env.close()
    return



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_path', "-a", type=str, default=None)
    parser.add_argument('--traj_dir', "-t", required=True, type=str)
    parser.add_argument('--num_episodes', "-n", type=int, default=25)
    # games
    parser.add_argument('--games', "-g", type=str, nargs='+', default=cfg.working_games)
    return parser.parse_args()


def main():
    args = get_args()
    games = args.games
    for game in tqdm(games):
        try:
            if 'random' in args.agent_path.lower():
                agent = BaseAgent(
                    {}
                )
            if "ppo" in args.agent_path.lower():
                agent = torch.load(args.agent_path, weights_only=False)
                agent.network = agent.network.to(device)
                agent.old_network = agent.old_network.to(device)
            if "drrn" in args.agent_path.lower():
                agent = torch.load(args.agent_path, weights_only=False)
                agent.network = agent.network.to(device) 
            env = MachiavelliEnv(
                game=game,
                agent_name=str(agent),
            )
            run_trajectories(
                agent,
                env,
                outdir=Path(args.traj_dir) / str(agent) / game,
                num_episodes=args.num_episodes,
            )
        except Exception as e:
            print(f"Error running {game}: {e}")
            traceback.print_exc()
            pass


if __name__ == '__main__':
    main()