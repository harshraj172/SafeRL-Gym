import argparse
from pathlib import Path
from typing import List

from tqdm import tqdm
import torch

from env.machiavelli.config import MachiavelliConfig
from env.machiavelli.machiavelli_env import MachiavelliEnv, Trajectory

cfg = MachiavelliConfig()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_trajectories(agent, env, outdir, num_episodes=25, max_traj_length=1000):
    print(f"------ Testing {agent} on {env.game} ------")

    for episode_idx in tqdm(range(num_episodes)):
        # Start new episode.
        obs, info = env.reset(seed=episode_idx)

        rew = 0
        done = False
        step_count = 0
        while not done:
            action = agent.get_action(obs, rew, done, info)
            obs, rew, done, info = env.step(action)

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
    return parser.parse_args()


def main():
    args = get_args()
    games = cfg.games_test
    for game in tqdm(games):
        agent = torch.load(args.agent_path, weights_only=False)
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


if __name__ == '__main__':
    main()