from collections import namedtuple
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from src.agent.base import BaseAgent
from src.env.machiavelli.utils import clean

PPOLLMTransition = namedtuple(
    "LLMTransition",
    (
        "state",
        "action",
        "log_prob",
        "value",
        "reward",
        "valid_actions",
        "done",
        "next_state",
        "next_acts",
        "advantage",
        "returns",
        "cost",
    ),
)

TrajectoryStep = namedtuple(
    "TrajectoryStep",
    [
        "state",
        "action",
        "reward",
        "done",
        "next_state",
        "logprob",
        "available_actions",
    ],
)


def format_state(state):
    return state.state
    # return f"Observations: {state.obs}\nDescription: {state.description}\nInventory: {state.inventory}"


def masked_mean(
    values: torch.Tensor, mask: torch.Tensor, axis: Optional[bool] = None
) -> torch.Tensor:
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


def masked_var(
    values: torch.Tensor, mask: torch.Tensor, unbiased: bool = True
) -> torch.Tensor:
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(
    values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True
) -> torch.Tensor:
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


class ActorNetwork(nn.Module):
    """
    Actor network for PPO-LLM agent.
    Computes action log-probabilities given a state and a list of possible actions using a language model.
    """

    def __init__(self, model, tokenizer):
        """
        Args:
            model: Pretrained language model (e.g., HuggingFace transformer).
            tokenizer: Corresponding tokenizer for the language model.
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = 10.0  # Temperature for scaling log-probabilities
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, state: str, actions: list[str], device, requires_grad=True):
        """
        Compute log-probabilities for each action given the state.
        Args:
            state: Input state as a string.
            actions: List of possible actions as strings.
            device: torch.device to run computation on.
            requires_grad: If True, enables gradient computation.
        Returns:
            torch.Tensor: Log-probabilities for each action.
        """
        batch_size = len(actions)
        states = [format_state(state)] * batch_size
        state_tokens = self.tokenizer(states, return_tensors="pt", padding=True)
        action_tokens = self.tokenizer(
            actions, return_tensors="pt", padding="longest", padding_side="right"
        )

        state_tokens = {k: v.to(device) for k, v in state_tokens.items()}
        action_tokens = {k: v.to(device) for k, v in action_tokens.items()}
        action_tokens["input_ids"] = action_tokens["input_ids"][:, 1:]
        eos_token_id = self.tokenizer.eos_token_id
        eos_tensor = torch.full(
            (batch_size, 1), eos_token_id, dtype=torch.long, device=device
        )
        action_tokens["input_ids"] = torch.cat(
            (action_tokens["input_ids"], eos_tensor), dim=1
        )
        # tokenized_length = action_tokens["attention_mask"].sum(dim=1)
        # action_tokens["input_ids"][torch.arange(batch_size, device=device), tokenized_length - 1] = (
        #     1  # EOS
        # )

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
        # Logits: [batch_size, total_seq_len, vocab_size]
        logits = outputs.logits

        # Split lengths
        k = state_tokens["input_ids"].shape[1]  # state length
        m = action_tokens["input_ids"].shape[1]  # action length

        # Next-token logits: [batch_size, m, vocab_size]
        next_token_logits = logits[
            :, k - 1 : -1, :
        ]  # aligns each action token with prediction from previous

        # Apply log-softmax over vocab
        log_probs = F.log_softmax(next_token_logits, dim=-1)

        # Action token ids: [batch_size, m]
        action_token_indices = action_tokens["input_ids"].to(
            device=device, dtype=torch.long
        )

        # Gather log-probs for the actual action tokens
        batch_size = action_token_indices.shape[0]
        token_indices = torch.arange(m, device=device)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)

        # Gather action token log-probs: shape [batch_size, m]
        action_token_logprobs = log_probs[
            batch_indices, token_indices.unsqueeze(0), action_token_indices
        ]

        # Mask padding tokens
        masked_logprobs = masked_mean(
            action_token_logprobs, action_tokens["attention_mask"], axis=1
        )  # Mean over action tokens, ignoring padding: better for variable-length actions
        # Return summed log-probs over tokens (or mean, scaled as needed)
        return masked_logprobs / self.temperature

    def create_distribution(self, state: str, actions: list[str], device):
        """
        Create a categorical distribution over actions given the state.
        Args:
            state: Input state as a string.
            actions: List of possible actions as strings.
            device: torch.device to run computation on.
        Returns:
            torch.distributions.Categorical: Distribution over actions.
        """
        logits = self.forward(state, actions, device)
        return torch.distributions.Categorical(logits=logits)


class CriticNetwork(nn.Module):
    """
    Enhanced Critic network for PPO-LLM agent with GRU encoders.
    Estimates the value of a given state using multiple GRU encoders for different state aspects.
    """

    def __init__(self, tokenizer, embedding, hidden_dim=128):
        super().__init__()
        self.tokenizer = tokenizer
        # Language model for embeddings
        self.embedding = embedding
        embedding_dim = self.embedding.get_input_embeddings().embedding_dim

        # Value estimation layers
        self.hidden = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.value_estimator = nn.Linear(hidden_dim, 1)
        self.hidden_dim = hidden_dim

    def _set_device(self, device):
        """
        Move the model components to the specified device.
        Args:
            device: torch.device to move the model to.
        """
        self.embedding = self.embedding.to(device)
        self.state_encoder = self.state_encoder.to(device)
        self.hidden = self.hidden.to(device)
        self.value_estimator = self.value_estimator.to(device)

    def forward(self, state_data, device):
        tokens = self.tokenizer(
            state_data.state,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_attention_mask=True,
        )

        tokens = {k: v.to(device) for k, v in tokens.items()}

        # Optional: detach if embedding is frozen
        with torch.no_grad():
            state_out = self.embedding(**tokens).last_hidden_state

        # CLS
        state_out = state_out[:, 0, :]

        # Forward through MLP
        z = self.hidden(state_out)
        value = self.value_estimator(z)

        return value.squeeze(-1).squeeze(
            -1
        )  # Remove the last dimension for scalar output


class PPOLLMAgent(BaseAgent):
    """
    PPO agent using language models for both actor and critic.
    Handles action selection, experience storage, advantage estimation, and PPO updates.
    """

    def __init__(self, args=None, **kwargs):
        """
        Initialize the PPO-LLM agent.
        Args:
            args: Namespace or dict with environment and model parameters.
            kwargs: Additional PPO hyperparameters.
        """
        if args is not None:
            super().__init__(args)

        device = (
            torch.device(f"cuda:{torch.cuda.current_device()}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        assert device.type == "cuda", "PPO_LLM requires a GPU for training."
        self.device = device
        self.loss = kwargs.get("loss", "a2c")  # 'a2c' or 'ppo'

        self.tokenizer = (
            args.tokenizer
            if hasattr(args, "tokenizer")
            else AutoTokenizer.from_pretrained(args.lm_name, model_max_length=512)
        )
        actor_model = (
            args.actor_model
            if hasattr(args, "actor_model")
            else AutoModelForCausalLM.from_pretrained(args.lm_name)
        )

        self.actor = ActorNetwork(actor_model, self.tokenizer).to(self.device)
        embedding = AutoModel.from_pretrained(
            "microsoft/deberta-v3-xsmall", output_hidden_states=True
        )

        self.critic = CriticNetwork(
            embedding=embedding,
            tokenizer=AutoTokenizer.from_pretrained(
                "microsoft/deberta-v3-xsmall", model_max_length=512
            ),
        ).to(self.device)

        self.gamma = kwargs.get("gamma", 0.99)
        self.clip_epsilon = kwargs.get("clip_epsilon", 0.5)
        self.lambda_gae = kwargs.get("lambda_gae", 0.95)
        self.entropy_coef = kwargs.get("entropy_coef", 0.01)

        params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(
            params,
            lr=kwargs.get("learning_rate", 3e-5),
        )

        self.trajectories = []
        self.episode_buffer = [[] for _ in range(args.num_envs)]
        self.completed_episodes = []
        self.max_segment_length = kwargs.get("max_segment_length", 8)
        self.last_actions = []

        self.lag = kwargs.get(
            "lag", False
        )  # Whether to use Lagrangian multiplier for cost
        self.cost_limit = kwargs.get(
            "cost_limit", 25.0
        )  # Maximum allowed cost per episode
        self.lagrange_lr = kwargs.get("lagrange_lr", 0.01)  # Learning rate for lambda
        self.lagrange_init = kwargs.get(
            "lagrange_init", 1.0
        )  # Initial value for lambda
        self.lagrange_max = kwargs.get("lagrange_max", 100.0)  # Max value for lambda
        self.lagrange_min = kwargs.get("lagrange_min", 0.0)  # Min value for lambda
        self.lagrange_lambda = torch.tensor(
            self.lagrange_init, device=self.device, requires_grad=False
        )
        self.clip_valuef = kwargs.get(
            "clip_valuef", 100.0
        )  # Clipping value for value loss

        self.vf_coef = kwargs.get("vf_coef", 0.25)

    def _tokenize(self, text):
        """
        Tokenize input text using the actor's tokenizer.
        Args:
            text: Input string or list of strings.
        Returns:
            dict: Tokenized tensors with input_ids and attention_mask
        """
        return text
        # if isinstance(text, str):
        #     text = [text]

        # encoding = self.tokenizer(
        #     [clean(t) for t in text], padding=True, truncation=True, return_tensors="pt"
        # ).to(self.device)

        # encoding["length"] = encoding["attention_mask"].sum(dim=1)

        # return encoding

    @torch.no_grad()
    def act(self, states, poss_acts, lm=None, *args, **kwargs):
        """
        Select actions for a batch of states and possible actions.
        Args:
            states: List of state strings.
            poss_acts: List of lists of possible actions for each state.
            lm: (Unused) Placeholder for compatibility.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        Returns:
            tuple: (actions, action indices, log-probabilities)
        """
        act_ids = []
        action_idxs = []
        log_probs = []
        self.last_actions = []
        for i, (state, actions) in enumerate(zip(states, poss_acts)):
            action, logprob, action_idx = self._choose_action(
                state, actions, logprob_flag=True, return_idx=True
            )
            act_ids.append(action)
            action_idxs.append(action_idx)
            log_probs.append(logprob)
            self.last_actions.append(
                {
                    "action": action_idx,
                    "log_prob": logprob,
                    "value": self.critic(state, self.device).item(),
                }
            )
        return act_ids, action_idxs, log_probs

    def observe(self, transition, is_prior=False):
        """
        Store a transition in the episode buffer and handle episode completion.

        Args:
            transition: An object with at least the following attributes:
                - state: The current environment state (may have 'env_idx' attribute for multi-env).
                - reward: The reward received after taking the action (float).
                - done: Whether the episode has ended (bool).
                - next_state: The next environment state after the action.
                - valid_acts: (Optional) List of possible actions for the next state.
                - next_acts: (Optional) List of possible actions for the next state.
                - cost: (Optional) Cost associated with the transition (float, default 0).
            is_prior: Unused. Present for compatibility with some interfaces.
        """
        if is_prior:
            # If this is a prior transition, we do not store it in the episode buffer.
            return
        env_idx = transition.state.env_idx
        action_info = (
            self.last_actions[env_idx]
            if hasattr(self, "last_actions") and env_idx < len(self.last_actions)
            else None
        )

        ppo_llm_transition = PPOLLMTransition(
            state=transition.state,
            action=action_info["action"] if action_info else None,
            log_prob=action_info["log_prob"] if action_info else None,
            value=action_info["value"] if action_info else None,
            reward=transition.reward,
            valid_actions=getattr(transition, "valid_acts", None),
            done=transition.done,
            next_state=transition.next_state,
            next_acts=getattr(transition, "next_acts", None),
            cost=getattr(transition, "cost", 0),
            advantage=None,
            returns=None,
        )
        self.episode_buffer[env_idx].append(ppo_llm_transition)
        if (
            transition.done
            or len(self.episode_buffer[env_idx]) >= self.max_segment_length
        ):
            # End of episode, store the completed episode
            self.completed_episodes.append(self.episode_buffer[env_idx])
            self.episode_buffer[env_idx] = []

    def compute_actions(self, s, va, a):
        dist = self.actor.create_distribution(s, va, self.device)
        if a > len(va) - 1:
            # Action index {a} out of bounds for valid actions {va}.
            return torch.tensor(0.0, device=self.device), torch.tensor(
                0.0, device=self.device
            )
        return dist.log_prob(a), dist.entropy()

    def _compute_advantages(self, episode, gamma=None, lambda_gae=None):
        gamma = gamma if gamma is not None else self.gamma
        lambda_gae = lambda_gae if lambda_gae is not None else self.lambda_gae

        rewards = [t.reward for t in episode]
        values = [t.value for t in episode]
        dones = [t.done for t in episode]
        costs = [t.cost for t in episode]

        # 🔁 Bootstrap if the episode was truncated
        if dones[-1]:
            next_value = 0
        else:
            with torch.no_grad():
                final_state = episode[-1].state
                next_value = self.critic(final_state, self.device).item()

        advantages = np.zeros(len(rewards), dtype=np.float32)
        next_advantage = 0

        for t in reversed(range(len(rewards))):
            not_done = 1.0 - float(dones[t])
            delta = rewards[t] + gamma * next_value * not_done - values[t]
            advantages[t] = delta + gamma * lambda_gae * next_advantage * not_done
            if self.lag:
                # Adjust advantage based on cost
                advantages[t] -= costs[t]
            next_value = values[t]
            next_advantage = advantages[t]

        returns = np.array(advantages) + np.array(values)

        advantages = masked_whiten(
            torch.tensor(advantages, device=self.device, dtype=torch.float32),
            torch.tensor(
                [1.0] * len(advantages), device=self.device, dtype=torch.float32
            ),
        )

        for i in range(len(episode)):
            episode[i] = episode[i]._replace(
                advantage=advantages[i], returns=returns[i]
            )
        # return advantages, returns

    def update(self):
        """
        Perform PPO update using completed episodes.
        Returns:
            float: Total loss over all episodes.
        """
        if not self.completed_episodes:
            return 0.0
        total_loss = 0.0
        for episode in self.completed_episodes:
            # 4 updates
            num_updates = 1
            for update in range(num_updates):
                if len(episode) == 0:
                    continue
                if episode[0].advantage is None:
                    self._compute_advantages(episode)
                states = [t.state for t in episode]
                actions = torch.tensor(
                    [t.action for t in episode], device=self.device, dtype=torch.long
                )
                valid_actions = [t.valid_actions for t in episode]
                old_log_probs = torch.tensor(
                    [t.log_prob for t in episode],
                    device=self.device,
                    dtype=torch.float32,
                )
                advantages = torch.tensor(
                    [t.advantage for t in episode],
                    device=self.device,
                    dtype=torch.float32,
                )
                returns = torch.tensor(
                    [t.returns for t in episode],
                    device=self.device,
                    dtype=torch.float32,
                )
                costs = torch.tensor(
                    [t.cost for t in episode], device=self.device, dtype=torch.float32
                )
                episode_cost = costs.sum().item()

                action_logprob_entropies = [
                    # logprob, entropy
                    self.compute_actions(s, va, a)
                    for s, va, a in zip(states, valid_actions, actions)
                ]
                logprobs = torch.stack(
                    [logprob for logprob, _ in action_logprob_entropies]
                )
                entropies = torch.stack(
                    [entropy for _, entropy in action_logprob_entropies]
                )

                values = torch.stack([self.critic(s, self.device) for s in states])
                policy_loss = None

                if self.loss == "a2c":
                    # A2C loss
                    policy_loss = -(logprobs * advantages).mean()
                elif self.loss == "ppo":
                    # PPO loss
                    # Calculate the surrogate losses
                    ratio = torch.exp(logprobs - old_log_probs)
                    surrogate1 = ratio * advantages
                    surrogate2 = (
                        torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                        * advantages
                    )
                    policy_loss = -torch.min(surrogate1, surrogate2).mean()

                # PPO-style value loss clipping
                vpred_clipped = torch.clamp(
                    values,
                    returns - self.clip_valuef,
                    returns + self.clip_valuef,
                )
                vf_losses1 = (values - returns) ** 2
                vf_losses2 = (vpred_clipped - returns) ** 2
                value_loss = torch.max(vf_losses1, vf_losses2).mean()
                entropy_loss = self.entropy_coef * entropies.mean()
                loss = policy_loss + self.vf_coef * value_loss - entropy_loss
                print(
                    f"Update {update + 1}/{num_updates}, "
                    f"Policy Loss: {policy_loss.item():.4f}, "
                    f"Value Loss: {value_loss.item():.4f}, "
                    f"Entropy Loss: {entropy_loss.item():.4f}, "
                    f"Total Loss: {loss.item():.4f}, "
                    f"Episode Cost: {episode_cost:.2f}"
                )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    max_norm=10.0,
                )
                # print grad norms
                actor_grad_norm_mean = 0.0
                for name, param in self.actor.named_parameters():
                    if param.grad is not None:
                        actor_grad_norm_mean += param.grad.norm().item()
                actor_grad_norm_mean /= len(list(self.actor.parameters()))
                critic_grad_norm_mean = 0.0
                for name, param in self.critic.named_parameters():
                    if param.grad is not None:
                        critic_grad_norm_mean += param.grad.norm().item()
                critic_grad_norm_mean /= len(list(self.critic.parameters()))
                print(
                    f"Actor grad norm: {actor_grad_norm_mean:.4f}, Critic grad norm: {critic_grad_norm_mean:.4f}"
                )
                self.optimizer.step()
                total_loss += loss.item()

                if self.lag:
                    with torch.no_grad():
                        self.lagrange_lambda += self.lagrange_lr * (
                            episode_cost - self.cost_limit
                        )
                        self.lagrange_lambda.clamp_(
                            self.lagrange_min, self.lagrange_max
                        )

        self.completed_episodes = []
        # Clear the episode buffer
        self.episode_buffer = [[] for _ in range(len(self.episode_buffer))]
        return total_loss

    def _choose_action(self, state, actions=None, logprob_flag=False, return_idx=False):
        """
        Choose an action from the current state using the actor network.
        Args:
            state: Input state as a string.
            actions: List of possible actions as strings (optional).
            logprob_flag: If True, also return log-probability.
            return_idx: If True, also return action index.
        Returns:
            tuple or str: Action (and optionally log-prob, index).
        """
        if not actions:
            raise ValueError(
                "No possible actions provided to _choose_action. "
                "The caller must supply a non-empty list of possible actions."
            )
        dist = self.actor.create_distribution(state, actions, self.device)
        action_idx = dist.sample().item()
        action = actions[action_idx]
        logprob = dist.log_prob(torch.tensor(action_idx, device=self.device)).item()
        if return_idx:
            return (
                (action, logprob, action_idx) if logprob_flag else (action, action_idx)
            )
        return (action, logprob) if logprob_flag else action
