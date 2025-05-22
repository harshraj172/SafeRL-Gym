import gc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModel
from agent.base import BaseAgent
import copy
from collections import namedtuple
from env.machiavelli.utils import clean
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batch_encoded_tokens(x):
    """ Takes a list of tokenized sequences and returns a batch of padded sequences
        token_list: list of tokenized sequences
    """
    return {
        'input_ids': torch.nested.to_padded_tensor(torch.nested.nested_tensor([n['input_ids'].squeeze() for n in x]), 0),
        'attention_mask': torch.nested.to_padded_tensor(torch.nested.nested_tensor([n['attention_mask'].squeeze() for n in x]), 0)
    }

def layer_init(layer, std=1, bias_const=0.0):
    torch.nn.init.normal_(layer.weight, mean=0.0, std=std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPONetwork(nn.Module):
    def __init__(self, lm_name, stats_dim, hidden_dim):
        super().__init__()
        self.embedding = AutoModel.from_pretrained(lm_name).to(device)
        # freeze the embedding layer
        for param in self.embedding.parameters():
            param.requires_grad = False
        self.embedding.eval()
        self.stats_dim = stats_dim
        embedding_dim = self.embedding.config.hidden_size
        initializer_range = self.embedding.config.initializer_range  # Get from LM config
        
        # Single hidden layer to combine transformer outputs and inventory
        self.hidden = nn.Linear(2 * embedding_dim + stats_dim, hidden_dim)
        
        # Policy head takes both state and action embeddings
        self.policy_head = nn.Linear(hidden_dim + embedding_dim, 1)
        
        
        self.value_head = nn.Linear(hidden_dim, 1)
        
        
        
        # for layer in self.policy_head:
        #     if isinstance(layer, nn.Linear):
        #         layer_init(layer)
        # for layer in self.value_head:
        #     if isinstance(layer, nn.Linear):
        #         layer_init(layer)

        # Initialize value head weights with normal distribution
        self.value_head.weight.data.normal_(mean=0.0, std=0.02)
        self.value_head.bias.data.zero_()
        # for layer in self.policy_head:
        #     if isinstance(layer, nn.Linear):
        #         layer_init(layer, std=1)

        # for layer in self.value_head:
        #     if isinstance(layer, nn.Linear):
        #         layer_init(layer, std=0.02)
        self.policy_head.weight.data.normal_(mean=0.0, std=1)
        self.policy_head.bias.data.zero_()

    def forward(self, states, actions=None):
        """Forward pass handling both state and action encoding"""
        # Batch the encodings from states
        obs_encodings = batch_encoded_tokens([state.obs for state in states])
        desc_encodings = batch_encoded_tokens([state.description for state in states])
        
        # Ensure all inventory tensors are truncated/padded to stats_dim
        inventory_tensors = torch.stack([
            state.inventory[:self.stats_dim] if len(state.inventory) >= self.stats_dim 
            else F.pad(state.inventory, (0, self.stats_dim - len(state.inventory)))
            for state in states
        ]).to(device).float()
        
        # Get state embeddings from transformer
        obs_emb = self.embedding(**obs_encodings)['last_hidden_state'][:, 0, :]  # CLS token
        desc_emb = self.embedding(**desc_encodings)['last_hidden_state'][:, 0, :]
    
        # Combine state features
        state_features = F.relu(self.hidden(torch.cat([
            obs_emb, desc_emb, inventory_tensors
        ], dim=1)))
        
        # Get value estimate
        value = self.value_head(state_features)
        
        if actions is None:
            return state_features, value
            
        # Handle variable-length action lists
        act_sizes = [len(a) for a in actions]
        # Flatten actions into one list
        flat_actions = [act for act_list in actions for act in act_list]
        
        # Get embeddings for all actions
        act_encodings = batch_encoded_tokens(flat_actions)
        # Get action embeddings from transformer
        act_emb = self.embedding(**act_encodings)['last_hidden_state'][:, 0, :]
        # Expand states to match flattened actions
        expanded_states = torch.cat([state_features[i].repeat(size, 1) 
                                   for i, size in enumerate(act_sizes)], dim=0)
        # Get action scores
        action_scores = self.policy_head(torch.cat([expanded_states, act_emb], dim=1)).squeeze(-1)

        # Split scores back into lists per batch element
        action_scores = action_scores.split(act_sizes)
        return action_scores, value

    @torch.no_grad()
    # TODO: for a trained policy maybe we should take argmax?
    def act(self, states, poss_acts, lm=None, **kwargs):
        """Get action scores and sample from policy"""
        action_scores, values = self.forward(states, poss_acts)
        
        # Convert scores to probabilities and sample
        action_probs = [F.softmax(scores, dim=0) for scores in action_scores]
        distributions = [Categorical(probs) for probs in action_probs]
        actions = [dist.sample() for dist in distributions]
        
        log_probs = [dist.log_prob(act) for dist, act in zip(distributions, actions)]
        return actions, log_probs, values

# Define a PPO-specific transition tuple
PPOTransition = namedtuple('PPOTransition', 
    ('state', 'action', 'log_prob', 'value', 'reward', 'done', 'next_state', 'next_acts', 'advantage', 'returns', 'cost', 
    'cost_returns'))

class PPOAgent(BaseAgent):
    def __init__(self, args):
        super().__init__(args)
        self.network = PPONetwork(args.lm_name, args.stats_dim, args.hidden_dim).to(device)
        self.old_network = copy.deepcopy(self.network)
        self.tokenizer = AutoTokenizer.from_pretrained(args.lm_name)
        self.stats_dim = args.stats_dim
        
        # PPO specific parameters
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.001
        self.gamma = args.gamma
        self.gae_lambda = 0.95
        self.ppo_epochs = 4
        self.max_segment_length = 256 # 64 for unfrozen
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=args.learning_rate)
        
        # Store transitions for each environment
        self.transitions = [[] for _ in range(args.num_envs)]
        self.completed_episodes = [[]]

    def _tokenize(self, text):
        encoding = self.tokenizer(
            clean(text), padding=True, truncation=True, return_tensors="pt").to(device)
        encoding['length'] = torch.tensor(
            [encoding['input_ids'].size(1)]).to(device)
        return encoding

    def act(self, states, poss_acts, lm=None, args=None):
        """Select actions and return log probabilities and values"""
        with torch.no_grad():
            act_idxs, log_probs, values = self.old_network.act(states, poss_acts, lm)
        
        # Store the action data for use in observe() - we'll associate it with the most recent state
        self.last_actions = []
        for env_idx, (act_idx, log_prob, value) in enumerate(zip(act_idxs, log_probs, values)):
            self.last_actions.append({
                'action': act_idx.item(),
                'log_prob': log_prob.item(),
                'value': value.item()
            })
        
        # Return the actions as expected by the training loop
        act_ids = [poss_acts[batch][idx] for batch, idx in enumerate(act_idxs)]
        action_idxs = [idx.item() for idx in act_idxs]
        return act_ids, action_idxs, log_probs

    def observe(self, transition, is_prior=False):
        """Store transition in memory"""
        env_idx = transition.state.env_idx if hasattr(transition.state, 'env_idx') else 0
        
        # Get action info from our stored actions (if available)
        action_info = None
        if hasattr(self, 'last_actions') and env_idx < len(self.last_actions):
            action_info = self.last_actions[env_idx]
        
        # Create a completed transition
        ppo_transition = PPOTransition(
            state=transition.state,
            action=action_info['action'] if action_info else None,
            log_prob=action_info['log_prob'] if action_info else None,
            value=action_info['value'] if action_info else None,
            reward=transition.reward,
            done=transition.done,
            next_state=transition.next_state,
            next_acts=transition.next_acts,
            cost=transition.cost, # Safety cost of taking a given action
            cost_returns=None,
            advantage=None,
            returns=None
        )
        
        # Store the transition
        self.transitions[env_idx].append(ppo_transition)
        self.completed_episodes[-1].append(ppo_transition)
        
        # If we reach a done state or max segment length, complete the episode
        if transition.done or len(self.transitions[env_idx]) >= self.max_segment_length:
            self.completed_episodes.append([])
            self.transitions[env_idx] = []

    def _compute_truncated_advantages(self, episode, is_terminal):
        """Compute advantages with bootstrapping for truncated episodes"""
        # Skip transitions with None values
        valid_transitions = [t for t in episode if t.value is not None]
        
        if not valid_transitions:
            print("Warning: No valid transitions with values found in episode")
            return np.array([]), np.array([])
        
        rewards = [t.reward for t in valid_transitions]
        values = [t.value for t in valid_transitions]
        dones = [t.done for t in valid_transitions]
        costs = [t.cost for t in valid_transitions]
        
        advantages = np.zeros(len(rewards), dtype=np.float32)
        returns = np.zeros(len(rewards), dtype=np.float32)
        cost_returns = np.zeros(len(costs), dtype=np.float32)
        
        # If not terminal, bootstrap value from last state
        if not is_terminal and len(valid_transitions) > 0:
            # Get value estimate for last state
            with torch.no_grad():
                last_state = valid_transitions[-1].next_state
                if last_state is not None:  # Check if next_state exists
                    _, next_value = self.network([last_state], None)
                    next_value = next_value.item()
                else:
                    next_value = 0
        else:
            next_value = 0
        
        next_advantage = 0
        next_return = 0  # Initialize next_return
        next_cost = 0
        
        for t in reversed(range(len(rewards))):
            # Safely convert done to 0 or 1, handling any type
            done_mask = 0 if dones[t] else 1
            
            if dones[t]:
                next_return = 0
                next_advantage = 0
                next_cost = 0
            
            # Use done_mask instead of (1 - int(dones[t]))
            returns[t] = rewards[t] + self.gamma * next_return * done_mask
            delta = rewards[t] + self.gamma * next_value * done_mask - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * next_advantage * done_mask
            cost_returns[t] = costs[t] + self.gamma * next_cost * done_mask
            
            next_return = returns[t]
            next_value = values[t]
            next_advantage = advantages[t]
            next_cost = cost_returns[t]
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        
        # Update episode data with new advantages and returns
        for i in range(len(episode)):
            episode[i] = PPOTransition(
                state=episode[i].state,
                action=episode[i].action,
                log_prob=episode[i].log_prob,
                value=episode[i].value,
                reward=episode[i].reward,
                done=episode[i].done,
                next_state=episode[i].next_state,
                next_acts=episode[i].next_acts,
                advantage=advantages[i],
                returns=returns[i],
                cost=episode[i].cost,
                cost_returns=cost_returns[i]
            )
        
        return advantages, returns

    def update(self):
        """Update policy using PPO"""
        if not self.completed_episodes:
            return 0.0  # No data to update from
        
        total_loss = 0
        
        # Now process all completed episodes
        for episode in self.completed_episodes:
            # Compute advantages for this episode if not already done
            if len(episode) == 0:
                continue
            if episode[0].advantage is None:
                self._compute_truncated_advantages(episode, episode[-1].done)
            
            # Get episode data
            states = [t.state for t in episode]
            actions = torch.tensor([t.action for t in episode], device=device, dtype=torch.long)
            old_log_probs = torch.tensor([t.log_prob for t in episode], device=device, dtype=torch.float32)
            advantages = torch.tensor([t.advantage for t in episode], device=device, dtype=torch.float32)
            returns = torch.tensor([t.returns for t in episode], device=device, dtype=torch.float32)
            
            # Multiple optimization epochs
            for _ in range(self.ppo_epochs):
                state_batch = states
                action_scores, values = self.network(state_batch)
                dist = Categorical(F.softmax(action_scores, dim=-1))
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                
                # Calculate ratios and surrogate losses
                ratios = torch.exp(new_log_probs - old_log_probs)
                pg_loss1 = -advantages * ratios
                pg_loss2 = -advantages * torch.clamp(ratios, 1-self.clip_epsilon, 1+self.clip_epsilon)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                
                # PPO policy and value losses
                # v_loss_unclipped = (values.squeeze() - returns)**2
                # v_clipped = b_values + torch.clamp(
                #         values - b_values,
                #         -self.clip_epsilon,
                #         self.clip_epsilon,
                # )
                # v_loss_clipped = (v_clipped - returns) ** 2
                # v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                
                # v_loss = 0.5 * v_loss_max.mean()
                v_loss = 0.5 * F.mse_loss(values.squeeze(), returns)
                entropy_loss = -self.entropy_coef * entropy
                loss = pg_loss + v_loss + entropy_loss
                
                # Update network
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
                total_loss += pg_loss.item()
        

        # Update old network
        self.old_network.load_state_dict(self.network.state_dict())
        
        # Clear completed episodes
        self.completed_episodes = [[]]
        return total_loss

class PPOLagAgent(PPOAgent):
    """Proximal Policy Optimization with Lagrange multiplier for constraint handling"""
    def __init__(self, args):
        super().__init__(args)
        self.lagrange_multiplier = torch.tensor(1.0, requires_grad=True, device=device)
        self.lagrange_lr = 0.001
        self.lagrange_optimizer = optim.Adam([{'params': [self.lagrange_multiplier], 'lr': self.lagrange_lr}])
        
    def update(self):
        """Update policy using PPO"""
        if not self.completed_episodes:
            return 0.0  # No data to update from
        
        total_loss = 0
        
        # Now process all completed episodes
        for episode in self.completed_episodes:
            # Compute advantages for this episode if not already done
            if len(episode) == 0:
                continue
            if episode[0].advantage is None:
                self._compute_truncated_advantages(episode, episode[-1].done)
            
            # Get episode data
            states = [t.state for t in episode]
            actions = torch.tensor([t.action for t in episode], device=device, dtype=torch.long)
            old_log_probs = torch.tensor([t.log_prob for t in episode], device=device, dtype=torch.float32)
            advantages = torch.tensor([t.advantage for t in episode], device=device, dtype=torch.float32)
            returns = torch.tensor([t.returns for t in episode], device=device, dtype=torch.float32)
            cost_returns = torch.tensor([t.cost_returns for t in episode], device=device, dtype=torch.float32)
            costs = torch.tensor([t.cost for t in episode], device=device, dtype=torch.float32)
            # Multiple optimization epochs
            for _ in range(self.ppo_epochs):
                state_batch = states
                action_scores, values = self.network(state_batch)
                dist = Categorical(F.softmax(action_scores, dim=-1))
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                
                # Calculate ratios and surrogate losses
                ratios = torch.exp(new_log_probs - old_log_probs)
                advantages = advantages
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
                
                # PPO policy and value losses
                policy_loss = -(torch.min(surr1, surr2).mean())
                value_loss = F.mse_loss(values.squeeze(), returns)
                entropy_loss = -self.entropy_coef * entropy
                
                loss = (policy_loss - self.lagrange_multiplier.detach() * cost_returns.mean()) + 0.5 * value_loss + entropy_loss
                
                # Update network
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
                total_loss += policy_loss.item()
                # Update Lagrange multiplier to maximize the loss
            
                self.lagrange_optimizer.zero_grad()
                loss = -(self.lagrange_multiplier * cost_returns.mean())
                loss.backward()
                self.lagrange_optimizer.step()


        # Update old network
        self.old_network.load_state_dict(self.network.state_dict())
        
        # Clear completed episodes
        self.completed_episodes = []
        
        return total_loss