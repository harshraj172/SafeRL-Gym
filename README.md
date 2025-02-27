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
python src/train.py --env Machiavelli --agent_type DRRN --game alexandria
```

## Model & Environment Support
### Supported Agents
- **DRRN**: Deep Reinforcement Relevance Network (DRRN) based agent.
- **Random**: Random action agent for baseline comparison.

### Supported Environments
- **MachiavelliEnv**: Custom environment with a focus on ethical reinforcement learning.
- **BaseEnv**: Generic environment for customizable experiments.

