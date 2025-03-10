import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModel
from agent.base import BaseAgent
from utils import ReplayMemory, PrioritizedReplayMemory, Transition, State
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPONetwork(nn.Module):
    """
    Combined policy and value network for PPO, adapted for Machiavelli environment.
    """
    def __init__(self, lm_name, stats_dim, hidden_dim):
        super(PPONetwork, self).__init__()
        self.embedding = AutoModel.from_pretrained(lm_name)
        embedding_dim = self.embedding.get_input_embeddings().embedding_dim
        self.stats_dim = stats_dim
        
        # Shared layers for each input type
        self.obs_fc = nn.Linear(embedding_dim, hidden_dim)
        self.desc_fc = nn.Linear(embedding_dim, hidden_dim)
        self.inv_fc = nn.Linear(stats_dim, hidden_dim)
        
        # Combine features
        self.shared_fc = nn.Linear(3 * hidden_dim, hidden_dim)
        
        # Policy head
        self.policy_fc = nn.Linear(hidden_dim, hidden_dim)
        
        # Value head
        self.value_fc = nn.Linear(hidden_dim, hidden_dim)
        self.value_out = nn.Linear(hidden_dim, 1)

    def forward(self, state_input, action_input=None):
        # Encode each state component
        obs_encoding = self._mean_pooling(
            self.embedding(**state_input['obs'])['last_hidden_state'],
            state_input['obs']['attention_mask']
        )
        desc_encoding = self._mean_pooling(
            self.embedding(**state_input['description'])['last_hidden_state'],
            state_input['description']['attention_mask']
        )
        
        # Process each component
        obs_features = torch.relu(self.obs_fc(obs_encoding))
        desc_features = torch.relu(self.desc_fc(desc_encoding))
        inv_features = torch.relu(self.inv_fc(state_input['inventory']))
        
        # Combine features
        combined = torch.cat([obs_features, desc_features, inv_features], dim=-1)
        shared_features = torch.relu(self.shared_fc(combined))
        
        # Policy branch
        policy_hidden = torch.relu(self.policy_fc(shared_features))
        
        # Value branch
        value_hidden = torch.relu(self.value_fc(shared_features))
        value = self.value_out(value_hidden)

        if action_input is not None:
            # Encode actions
            action_output = self.embedding(**action_input)
            action_encodings = self._mean_pooling(
                action_output['last_hidden_state'],
                action_input['attention_mask']
            )
            action_scores = torch.matmul(policy_hidden, action_encodings.t())
            return action_scores, value
        
        return value

    def _mean_pooling(self, token_embeddings, attention_mask):
        """Mean pooling of token embeddings, taking attention mask into account"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class PPOAgent(BaseAgent):
    def __init__(self, args):
        super(PPOAgent, self).__init__(args)
        self.network = PPONetwork(args.lm_name, args.stats_dim, args.hidden_dim).to(device)
        self.old_network = copy.deepcopy(self.network)
        self.tokenizer = AutoTokenizer.from_pretrained(args.lm_name)
        self.stats_dim = args.stats_dim
        
        # PPO specific parameters
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        self.gamma = args.gamma
        self.gae_lambda = 0.95  # For GAE
        self.batch_size = args.batch_size
        self.ppo_epochs = 4     # Multiple epochs of optimization for each batch
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=args.learning_rate)
        self.memory = ReplayMemory(args.memory_size) if not args.priority else PrioritizedReplayMemory(args.memory_size)
        self.transitions = []
        
        # Store additional information needed for PPO
        self.trajectory_data = []

    def _prepare_state_input(self, states):
        """Convert states to network input format"""
        obs_encodings = [self._tokenize(state.obs) for state in states]
        desc_encodings = [self._tokenize(state.description) for state in states]
        inv_encodings = [torch.tensor(state.inventory[:self.stats_dim], device=device).float() for state in states]
        
        return {
            'obs': batch_encoded_tokens(obs_encodings),
            'description': batch_encoded_tokens(desc_encodings),
            'inventory': torch.stack(inv_encodings)
        }

    def _tokenize(self, text):
        """Match DRRN's tokenization"""
        encoding = self.tokenizer(
            clean(text), padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        encoding['length'] = torch.tensor([encoding['input_ids'].size(1)]).to(device)
        return encoding

    def act(self, states, poss_acts, lm=None, eps=None, alpha=0, beta=1, k=-1):
        """Match DRRN's act interface while maintaining PPO functionality"""
        with torch.no_grad():
            state_batch = self._prepare_state_input(states)
            action_batch = self._prepare_action_input(poss_acts)
            
            action_scores, values = self.old_network(state_batch, action_batch)
            action_probs = F.softmax(action_scores, dim=-1)
            
            # Sample actions
            dist = Categorical(action_probs)
            act_idxs = dist.sample()
            log_probs = dist.log_prob(act_idxs)
            
            # Store trajectory data
            for i, (state, poss_act) in enumerate(zip(states, poss_acts)):
                self.trajectory_data.append({
                    'state': state,
                    'poss_acts': poss_act,
                    'action_idx': act_idxs[i].item(),
                    'log_prob': log_probs[i].item(),
                    'value': values[i].item()
                })
            
            # Format output to match DRRN
            act_ids = [poss_acts[batch][idx] for batch, idx in enumerate(act_idxs)]
            
            return act_ids, act_idxs.tolist(), log_probs.tolist()

    def observe(self, state, act_ids, log_prob, reward, next_state, done, info):
        """Store transition and update trajectory data"""
        transition = Transition(state, act_ids[0], log_prob, reward, next_state, done, info)
        
        # Update the last trajectory entry with reward and done status
        if self.trajectory_data:
            self.trajectory_data[-1]['reward'] = reward
            self.trajectory_data[-1]['done'] = done
            self.trajectory_data[-1]['next_state'] = next_state
        
        if done:
            # Calculate GAE advantages and returns
            self._compute_advantages()
            
            # Store complete trajectories in memory
            for data in self.trajectory_data:
                self.memory.push(Transition(
                    data['state'], 
                    data['action_idx'], 
                    data['log_prob'],
                    data['return'],  # Use computed returns instead of raw rewards
                    data.get('next_state', None),
                    data['done'],
                    {'valid_acts': data['poss_acts'], 'advantage': data['advantage']}
                ))
            
            # Clear trajectory data for next episode
            self.trajectory_data = []
            
            # Update old network to match current network
            self.old_network.load_state_dict(self.network.state_dict())

    def _compute_advantages(self):
        """Compute GAE advantages and returns for the current trajectory"""
        rewards = [data['reward'] for data in self.trajectory_data]
        values = [data['value'] for data in self.trajectory_data]
        dones = [data['done'] for data in self.trajectory_data]
        
        # Compute GAE and returns
        advantages = []
        returns = []
        gae = 0
        
        # Add a final value estimate for incomplete trajectories
        if not self.trajectory_data[-1]['done'] and 'next_state' in self.trajectory_data[-1]:
            with torch.no_grad():
                next_state = self._prepare_state_input([self.trajectory_data[-1]['next_state']])
                next_value = self.network(next_state).item()
        else:
            next_value = 0
            
        # Compute returns and advantages
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t+1]
                
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])  # Return = advantage + value
        
        # Normalize advantages
        adv_mean = sum(advantages) / len(advantages)
        adv_std = torch.tensor(advantages).std().item()
        normalized_advantages = [(a - adv_mean) / (adv_std + 1e-8) for a in advantages]
        
        # Store in trajectory data
        for i in range(len(self.trajectory_data)):
            self.trajectory_data[i]['advantage'] = normalized_advantages[i]
            self.trajectory_data[i]['return'] = returns[i]

    def update(self):
        """Update policy using PPO with multiple optimization epochs"""
        if len(self.memory) < self.batch_size:
            return None

        total_loss = 0
        
        # Sample once, then do multiple optimization steps
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Prepare inputs (done once for efficiency)
        states = self._prepare_state_input(batch.state)
        valid_actions = [t.info['valid_acts'] for t in transitions]
        actions_batch = self._prepare_action_input(valid_actions)
        
        # Get advantages from stored info
        advantages = torch.tensor([t.info['advantage'] for t in transitions], device=device, dtype=torch.float32)
        returns = torch.tensor(batch.reward, device=device, dtype=torch.float32)
        old_log_probs = torch.tensor(batch.log_prob, device=device, dtype=torch.float32)
        actions = torch.tensor(batch.act, device=device, dtype=torch.long)
        
        # Multiple optimization epochs
        for _ in range(self.ppo_epochs):
            # Forward pass with current policy
            action_scores, values = self.network(states, actions_batch)
            action_probs = F.softmax(action_scores, dim=-1)
            
            # Calculate action log probabilities
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # PPO policy loss
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values.squeeze(), returns)
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # After updates, update old policy to current policy
        self.old_network.load_state_dict(self.network.state_dict())
        
        return total_loss / self.ppo_epochs

    def _prepare_action_input(self, actions):
        """Convert actions to network input format"""
        flat_actions = [act for acts in actions for act in acts]
        inputs = self.tokenizer(
            flat_actions, 
            padding=True, 
            truncation=True,
            return_tensors="pt"
        )
        return {k: v.to(device) for k, v in inputs.items()}

