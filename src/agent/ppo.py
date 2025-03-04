import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModel
from agent.base import BaseAgent
from utils import ReplayMemory, PrioritizedReplayMemory, Transition, State

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPOPolicyNetwork(nn.Module):
    """
    Policy network for PPO (actor).
    """
    def __init__(self, lm_name, hidden_dim, action_dim):
        super(PPOPolicyNetwork, self).__init__()
        self.embedding = AutoModel.from_pretrained(lm_name)
        embedding_dim = self.embedding.get_input_embeddings().embedding_dim
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.embedding(**x)['last_hidden_state'][:, 0, :]  # Use [CLS] token
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)


class PPOValueNetwork(nn.Module):
    """
    Value network for PPO (critic).
    """
    def __init__(self, lm_name, hidden_dim):
        super(PPOValueNetwork, self).__init__()
        self.embedding = AutoModel.from_pretrained(lm_name)
        embedding_dim = self.embedding.get_input_embeddings().embedding_dim
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(**x)['last_hidden_state'][:, 0, :]  # Use [CLS] token
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PPOAgent(BaseAgent):
    """
    PPO Agent for the Machiavelli environment.
    """
    def __init__(self, args):
        super(PPOAgent, self).__init__(args)
        self.gamma = args.gamma
        self.clip_epsilon = 0.2  # PPO clip parameter
        self.entropy_coef = 0.01  # Entropy coefficient
        self.lr = args.learning_rate
        self.hidden_dim = args.hidden_dim
        self.action_dim = 10  # Example: 10 possible actions

        # Initialize policy and value networks
        self.policy = PPOPolicyNetwork(args.lm_name, self.hidden_dim, self.action_dim).to(device)
        self.value = PPOValueNetwork(args.lm_name, self.hidden_dim).to(device)

        # Optimizer
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=self.lr
        )

    def act(self, states, poss_acts, lm=None, eps=None, alpha=0, beta=1, k=-1):
        """
        Sample an action from the policy.
        """
        # states = [self._tokenize(state) for state in states]
        # states = {k: torch.cat([x[k] for x in states]) for k in states[0].keys()}
        states = State(*zip(*states))
        print(states)
        probs = self.policy(states)
        dist = Categorical(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions.tolist(), actions.tolist(), log_probs.tolist()

    def update(self):
        """
        Update the policy and value networks using PPO.
        """
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of transitions
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Convert states to tokenized inputs
        states = [self._tokenize(state) for state in batch.state]
        states = {k: torch.cat([x[k] for x in states]) for k in states[0].keys()}

        # Compute returns and advantages
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=device)
        values = self.value(states).squeeze()
        next_values = self.value(self._tokenize(batch.next_state[0])).squeeze()
        returns = rewards + self.gamma * next_values * (1 - dones)
        advantages = returns - values

        # Evaluate actions
        actions = torch.tensor(batch.act, device=device)
        log_probs, entropy, values = self.evaluate(states, actions)

        # Compute PPO loss
        ratio = (log_probs - torch.tensor(batch.log_prob, device=device)).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values.squeeze(), returns)
        entropy_loss = -entropy.mean()

        # Total loss
        loss = policy_loss + 0.5 * value_loss + self.entropy_coef * entropy_loss

        # Update networks
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, states, actions):
        """
        Evaluate the actions under the current policy.
        """
        probs = self.policy(states)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.value(states)
        return log_probs, entropy, values

