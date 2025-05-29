import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.base import BaseAgent


class ActorNetwork(nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, state: str, actions: list[str], device, requires_grad=False):
        # Returns log-probs for each action given the state
        batch_size = len(actions)
        states = [state] * batch_size
        state_tokens = self.tokenizer(states, return_tensors="pt", padding=True).to(
            device
        )
        action_tokens = self.tokenizer(
            actions, return_tensors="pt", padding="longest", padding_side="right"
        ).to(device)
        action_tokens["input_ids"] = action_tokens["input_ids"][:, 1:]  # Remove BOS
        action_tokens["input_ids"] = torch.cat(
            (
                action_tokens["input_ids"],
                torch.zeros((batch_size, 1), dtype=torch.long, device=device),
            ),
            dim=1,
        )
        tokenized_length = action_tokens["attention_mask"].sum(dim=1)
        action_tokens["input_ids"][torch.arange(batch_size), tokenized_length - 1] = (
            1  # EOS
        )

        inputs = {
            "input_ids": torch.cat(
                (state_tokens["input_ids"], action_tokens["input_ids"]), dim=1
            ),
            "attention_mask": torch.cat(
                (state_tokens["attention_mask"], action_tokens["attention_mask"]), dim=1
            ),
        }
        inputs = {key: value.to(device) for key, value in inputs.items()}

        if requires_grad:
            outputs = self.model(**inputs)
        else:
            with torch.no_grad():
                outputs = self.model(**inputs)
        logits = outputs.logits
        logits = torch.clamp(logits, min=-100, max=100)
        k = state_tokens["input_ids"].shape[1]
        m = action_tokens["input_ids"].shape[1]
        next_token_logits = logits[:, k - 1 : -1, :]
        next_token_logprobs = F.log_softmax(next_token_logits, dim=-1)
        action_token_logprobs_ = next_token_logprobs[
            :, torch.arange(m), action_tokens["input_ids"][0]   # might be bug if batch_size > 1
        ]
        action_token_logprobs = action_token_logprobs_ * action_tokens["attention_mask"]
        return action_token_logprobs.sum(dim=-1)

    def create_distribution(self, state: str, actions: list[str], device):
        logits = self.forward(state, actions, device)
        return torch.distributions.Categorical(logits=logits)
        
        # might wanna change this to a multinomial distribution if actions are not mutually exclusive
        # return torch.distributions.multinomial.Multinomial(total_count=1,logits=logits)

    def create_conditional_logprobs(
        self, state: str, actions: list[str], device, probs=False, requires_grad=True
    ) -> torch.TensorType:
        absolute_logprobs = self.forward(
            state, actions, device, requires_grad=requires_grad
        )
        conditional_logprobs = F.log_softmax(absolute_logprobs, dim=-1)
        if probs:
            return conditional_logprobs, torch.exp(conditional_logprobs)
        else:
            return conditional_logprobs


class CriticNetwork(nn.Module):
    def __init__(self, model, tokenizer, input_dim):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.simple_nn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(), 
        )

    def forward(self, state: str, device):
        inputs = self.tokenizer(state, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        # FOR TESTING ONLY: Use the first token's embedding from layer 1
        hidden_states = outputs.hidden_states[1][:, 0, :].to(device)  # shape: [batch, hidden_size]
        # hidden_states = outputs.hidden_states[1].to(device)  # shape: [batch, hidden_size]
        self.simple_nn = self.simple_nn.to(device)
        x = self.simple_nn(hidden_states)
        return x.squeeze()


class PPOAgent(BaseAgent):
    def __init__(self, env, actor, critic, device, args=None, **kwargs):
        """
        PPOAgent for Machiavelli environment, inheriting from BaseAgent.
        - env: Machiavelli environment instance
        - actor: ActorNetwork
        - critic: CriticNetwork
        - device: torch device
        - args: argparse.Namespace or dict for BaseAgent (optional)
        - kwargs: PPO hyperparameters
        """
        if args is not None:
            super().__init__(args)
        self.env = env
        self.actor = actor
        self.critic = critic
        self.device = device
        self.gamma = kwargs.get("gamma", 0.99)
        self.clip_epsilon = kwargs.get("clip_epsilon", 0.1)
        self.lambda_gae = kwargs.get("lambda_gae", 0.95)
        # Remove duplicate parameters from actor and critic
        params = list({id(p): p for p in list(self.actor.parameters()) + list(self.critic.parameters())}.values())
        self.optimizer = torch.optim.Adam(
            params,
            lr=kwargs.get("learning_rate", 1e-5),
        )

    def choose_action(self, state, logprob_flag=False, return_idx=False):
        """
        Choose an action from the current state using the actor network.
        Returns (action, logprob) if logprob_flag else just action.
        If return_idx is True, also returns the action index.
        """
        info = self.env._get_info()
        actions = info['game_state']['choice_texts']
        dist = self.actor.create_distribution(state, actions, self.device)
        action_idx = dist.sample().to(self.device).item()
        action = actions[action_idx]
        logprob = dist.log_prob(torch.tensor(action_idx, device=self.device)).item()
        if return_idx:
            if logprob_flag:
                return action, logprob, action_idx
            else:
                return action, action_idx
        if logprob_flag:
            return action, logprob
        else:
            return action

    def collect_trajectories(self, batch_size):
        """
        Collects a batch of trajectories (state, action, reward, done, next_state, logprob).
        Resets env if done.
        """
        trajectories = []
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            state = reset_result[0]
        else:
            state = reset_result
        for _ in range(batch_size):
            info = self.env._get_info()
            available_actions = info['game_state']['choice_texts']
            action, logprob, action_idx = self.choose_action(state, logprob_flag=True, return_idx=True)
            next_state, reward, done, _ = self.env.step(action_idx)
            trajectories.append((state, action, reward, done, next_state, logprob, available_actions))
            state = next_state if not done else self.env.reset()
            if isinstance(state, tuple):
                state = state[0]
        return trajectories

    def compute_gaes(self, trajectories, returns_flag=False):
        """
        Compute Generalized Advantage Estimates (GAEs) for each trajectory.
        Returns (gaes, returns) if returns_flag else just gaes.
        """
        gaes = []
        returns = []
        gae = 0
        values = [] if returns_flag else None
        for i in reversed(range(len(trajectories))):
            state, _, reward, done, next_state, *_ = trajectories[i]
            value = self.critic(state, self.device)
            if returns_flag:
                values.insert(0, value)
            next_value = 0 if done else self.critic(next_state, self.device)
            delta = reward + self.gamma * next_value - value
            gae = delta + self.gamma * self.lambda_gae * gae * (1 - done)
            gaes.insert(0, gae)
            returns.insert(0, gae + value)
        gaes_tensor = torch.tensor(gaes, dtype=torch.float32)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        if returns_flag:
            return gaes_tensor, returns_tensor
        else:
            return gaes_tensor

    def update(self, trajectories):
        """
        Performs a PPO update using the collected trajectories.
        Computes policy loss and value loss, then optimizes.
        """
        gaes, returns = self.compute_gaes(trajectories, returns_flag=True)
        gaes = gaes.to(self.device)
        returns = returns.to(self.device)
        states = [t[0] for t in trajectories]
        actions = [t[1] for t in trajectories]
        old_logprobs = torch.tensor([t[5] for t in trajectories], dtype=torch.float32, device=self.device)
        actions_per_state = [t[6] for t in trajectories]
        actions_idx = [acts.index(a) for acts, a in zip(actions_per_state, actions)]
        # Policy loss
        logprobs = torch.stack([
            self.actor.create_distribution(s, acts, self.device).log_prob(torch.tensor(a_idx, device=self.device))
            for s, acts, a_idx in zip(states, actions_per_state, actions_idx)
        ])
        ratio = torch.exp(logprobs - old_logprobs)
        surrogate1 = ratio * gaes
        surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * gaes
        policy_loss = -torch.min(surrogate1, surrogate2).mean()
        # Value loss
        values = torch.stack([self.critic(s, self.device) for s in states])
        value_loss = F.mse_loss(values, returns)
        # Optimize
        loss = policy_loss + value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
