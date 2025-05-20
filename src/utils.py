import json
import os
from collections import namedtuple
import random
import matplotlib.pyplot as plt

State = namedtuple('State', ('obs', 'description', 'inventory', 'state'))
Transition = namedtuple('Transition', ('state', 'act', 'reward', 'next_state', 'next_acts', 'done', 'cost'))


def read_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def plot_metrics(df, plot_info, color_map=None, marker='o'):
    """
    Plots Step vs multiple metrics ('EpisodeScore', 'Reward', 'Cost') from a DataFrame 
    and saves the plots to a results folder.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        plot_info (dict): Details about the plot--'env', 'agent', 'lm_name', and 'game'.
        color_map (dict): A dictionary mapping metrics to colors (optional).
        marker (str): The marker style for the plot points.
    """
    # Default color map if none is provided
    if color_map is None:
        color_map = {
            'EpisodeScore': 'blue',
            'Reward': 'green',
            'Cost': 'red'
        }
    
    # Create the title prefix and folder path
    folder_path = os.path.join("results", plot_info['env'], plot_info['agent'], plot_info['lm_name'], plot_info['game'], "plots")
    title_prefix = f"{plot_info['agent']} {plot_info['game']}"
    
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)
    
    # Metrics to plot
    metrics = ['EpisodeScore', 'Reward', 'Cost']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(df['Step'], df[metric], label=metric, marker=marker, color=color_map.get(metric, 'blue'))
        plt.xlabel('Step')
        plt.ylabel(metric)
        plt.title(f"{title_prefix}: {metric}")
        plt.grid()
        plt.legend()
        
        # Save the plot to the results folder
        file_name = f"Step_vs_{metric}.png"
        file_path = os.path.join(folder_path, file_name)
        plt.savefig(file_path)
        plt.close()  # Close the plot to free memory
        print(f"Plot saved to {file_path}")

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayMemory(object):
    def __init__(self, capacity, priority_fraction):
        self.priority_fraction = priority_fraction
        self.alpha_capacity = int(capacity * priority_fraction)
        self.beta_capacity = capacity - self.alpha_capacity
        self.alpha_memory, self.beta_memory = [], []
        self.alpha_position, self.beta_position = 0, 0
    
    def clear_alpha(self):
        self.alpha_memory = []
        self.alpha_position = 0

    def push(self, transition, is_prior=False):
        """Saves a transition."""
        if self.priority_fraction == 0.0:
            is_prior = False
        if is_prior:
            if len(self.alpha_memory) < self.alpha_capacity:
                self.alpha_memory.append(None)
            self.alpha_memory[self.alpha_position] = transition
            self.alpha_position = (self.alpha_position + 1) % self.alpha_capacity
        else:
            if len(self.beta_memory) < self.beta_capacity:
                self.beta_memory.append(None)
            self.beta_memory[self.beta_position] = transition
            self.beta_position = (self.beta_position + 1) % self.beta_capacity

    def sample(self, batch_size):
        if self.priority_fraction == 0.0:
            from_beta = min(batch_size, len(self.beta_memory))
            res = random.sample(self.beta_memory, from_beta)
        else:
            from_alpha = min(int(self.priority_fraction * batch_size), len(self.alpha_memory))
            from_beta = min(batch_size - int(self.priority_fraction * batch_size), len(self.beta_memory))
            res = random.sample(self.alpha_memory, from_alpha) + random.sample(self.beta_memory, from_beta)
        random.shuffle(res)
        return res

    def __len__(self):
        return len(self.alpha_memory) + len(self.beta_memory)