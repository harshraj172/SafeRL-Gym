"""
Generate trajectories for a given agent and environment.
    Usage:
        python -m src.evaluate_trajectories -a .drrn -t ./trajectories/<agent_name>
"""
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.metrics import corr
from src.env.machiavelli.machiavelli_env import Trajectory


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
    # make dir
    results_dir = Path(args.results_file).parent
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # save to csv
    df.to_csv(args.results_file, index=False)
    # compute correlation costs
    game_cost = {}
    for game in df['game'].unique():
        game_df = df[df['game'] == game]
        print(f"Processing game: {game}")
        
        
        pos_cost_reward_corr = corr(
            game_df['safety_cost'].values, 
            game_df['reward'].values, 
            rectified=True
        )
        cost_reward_corr = corr(
            game_df['safety_cost'].values, 
            game_df['reward'].values, 
            rectified=True
        )
        game_cost[game] = {
            "cost_reward_corr": cost_reward_corr,
            "pos_cost_reward_corr": pos_cost_reward_corr,
        }
    # save game costs to json
    game_cost_file = results_dir / "game_costs_corr.json"
    with open(game_cost_file, 'w') as f:
        import json
        json.dump(game_cost, f, indent=4)
    print(f"Game costs saved to {game_cost_file}")


