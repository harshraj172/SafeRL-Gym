# SafeRL-Gym

A repository for SafeRL experiments on text-based environments.

## Overview
SafeRL-Gym provides a framework for training reinforcement learning agents in text-based environments. The repository supports various environments and agent architectures, including DRRN-based agents. The primary entry point for training is the `src/train.py` script.

## Installation
### Requirements
Ensure you have Python installed. Recommended version: **Python 3.11**

Install dependencies:
```sh
pip install -r requirements.txt
```

## Usage
### Training an Agent
To start training an agent in the Machiavelli environment, run:
```sh
python -m src.train --env Machiavelli --agent_type PPO --game i-cyborg
```

## Evaluating the Agent
To evaluate the agent after training run:
```sh
python -m src.generate_trajectories -a ./checkpoints_utility/Machiavelli_PPO_microsoft_deberta-v3-xsmall_gamealexandria.pt -t ./trajectories
python -m src.evaluate_trajectories -t ./trajectories -r ./results.json
```
**Note**: Ensure to download the Machiavelli game data. Here are the steps:
- The data is available through [Google Drive](https://drive.google.com/file/d/19PXa2bgjkfFfTTI3EZIT3-IJ_vxrV0Rz/view).
- The password is `machiavelli`.
- Place the data at the top-level of this repo as `./game_data/`. (You should now have a folder structure as described in Repo structure.)

## Model & Environment Support
### Supported Agents
- **DRRN**: Deep Reinforcement Relevance Network (DRRN) based agent.
- **PPO**: PPO based agent.
- **Random**: Random action agent for baseline comparison.

### Supported Environments
- **MachiavelliEnv**: Custom environment with a focus on ethical reinforcement learning.
- **BaseEnv**: Generic environment for customizable experiments.

