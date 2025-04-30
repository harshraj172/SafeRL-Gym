import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from env.machiavelli.machiavelli_env import Trajectory


def load_results(traj_dir: str):
    print(f"Loading results for {traj_dir}...")
    traj_dir = Path(traj_dir)
    
    df_rows = []
    for traj_path in tqdm(sorted(list(traj_dir.glob("**/*.pkl")))):
        traj = Trajectory.load(traj_path)
        reward =  traj.get_points()
        safety_cost = traj.get_mach_counts()['violations.Σ'] 
        df_rows.append({
            "game": traj.game_name,
            "traj_dir": traj_dir,
            "steps": len(traj),
            "safety_cost": safety_cost,
            "reward": reward,
        })
    df = pd.DataFrame(df_rows)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj_dir', "-t", required=True, type=str)
    parser.add_argument('--results_file', "-r", required=True, type=str)
    args = parser.parse_args()
    
    df = load_results(args.traj_dir)
    print(df)
    df.to_csv(args.results_file, index=False)
