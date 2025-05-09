"""
SafeRL-Gym Scripts

Usage:
    python src/main.py --plot-metrics results/Machiavelli/DRRN/microsoft_deberta-v3-xsmall/kung-fu/progressMachiavelli.json
"""
import argparse
import pandas as pd
from utils import read_json, plot_metrics

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="SafeRL-Gym Scripts")
    parser.add_argument(
        "--plot-metrics",
        type=str,
        required=True,
        help="Generate and save plots for Step vs EpisodeScore, Reward, and Cost using provided JSON path."
    )

    # Parse arguments
    args = parser.parse_args()

    # Load JSON data into a DataFrame
    data = read_json(args.plot_metrics)
    df = pd.DataFrame(data)

    # Call plot_metrics if --plot-metrics is specified
    if args.plot_metrics:
        env = args.plot_metrics.split("/")[1]
        agent = args.plot_metrics.split("/")[2]
        lm_name = args.plot_metrics.split("/")[3]
        game = args.plot_metrics.split("/")[-2]
        plot_info = {"env": env, "agent": agent, "lm_name": lm_name, "game": game}
        plot_metrics(df, plot_info=plot_info)

if __name__ == "__main__":
    main()