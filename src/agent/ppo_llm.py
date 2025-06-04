from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
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

    def _set_device(self, device):
        """
        Move the model to the specified device.
        Args:
            device: torch.device to move the model to.
        """
        self.model = self.model.to(device)

    def forward(self, state: str, actions: list[str], device, requires_grad=False):
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
        self._set_device(device)
        assert (
            self.model.device.type == device.type
        ), f"Actor network device type {self.model.device.type} does not match expected device type {device.type}"
        batch_size = len(actions)
        states = [state] * batch_size
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
        logits = outputs.logits
        logits = torch.clamp(logits, min=-100, max=100)
        k = state_tokens["input_ids"].shape[1]
        m = action_tokens["input_ids"].shape[1]
        next_token_logits = logits[:, k - 1 : -1, :]
        # next_token_logprobs = F.log_softmax(next_token_logits, dim=-1)
        action_token_indices = action_tokens["input_ids"].to(
            device=device, dtype=torch.long
        )
        batch_indices = torch.arange(action_token_indices.shape[0], device=device)
        token_indices = torch.arange(m, device=device)
        # action_token_logprobs_ = next_token_logprobs[
        # batch_indices.unsqueeze(1), token_indices.unsqueeze(0), action_token_indices
        # ]
        action_token_logits = next_token_logits[
            batch_indices.unsqueeze(1), token_indices.unsqueeze(0), action_token_indices
        ]
        # action_token_logprobs = action_token_logprobs_ * action_tokens["attention_mask"]
        action_token_logits = action_token_logits * action_tokens["attention_mask"]
        return action_token_logits.sum(dim=-1)

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

    def create_conditional_logprobs(
        self, state: str, actions: list[str], device, probs=False, requires_grad=True
    ) -> torch.TensorType:
        """
        Compute conditional log-probabilities (and probabilities) for actions given the state.
        Args:
            state: Input state as a string.
            actions: List of possible actions as strings.
            device: torch.device to run computation on.
            probs: If True, also returns probabilities.
            requires_grad: If True, enables gradient computation.
        Returns:
            torch.Tensor or (torch.Tensor, torch.Tensor): Log-probabilities (and probabilities if probs=True).
        """
        absolute_logprobs = self.forward(
            state, actions, device, requires_grad=requires_grad
        )
        conditional_logprobs = F.log_softmax(absolute_logprobs, dim=-1)
        if probs:
            return conditional_logprobs, torch.exp(conditional_logprobs)
        else:
            return conditional_logprobs


class CriticNetwork(nn.Module):
    """
    Critic network for PPO-LLM agent.
    Estimates the value of a given state using a language model and a small neural network.
    """

    def __init__(self, model, tokenizer, input_dim):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.attn_proj = nn.Linear(model.config.hidden_size, 1, bias=False)
        self.simple_nn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _set_device(self, device):
        """
        Move the model and value head to the specified device.
        Args:
            device: torch.device to move the model to.
        """
        self.model = self.model.to(device)
        self.simple_nn = self.simple_nn.to(device)
        self.attn_proj = self.attn_proj.to(device)

    def forward(self, state: str, device):
        """
        Estimate the value of the given state.
        Args:
            state: Input state as a string.
            device: torch.device to run computation on.
        Returns:
            torch.Tensor: Estimated value of the state.
        """
        self._set_device(device)
        assert (
            self.model.device.type == device.type
        ), f"Critic network device type {self.model.device.type} does not match expected device type {device.type}"
        inputs = self.tokenizer(state, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]  # [batch, seq_len, hidden]
        scores = self.attn_proj(last_hidden)  # [batch, seq_len, 1]
        weights = torch.softmax(scores.squeeze(-1), dim=1)  # [batch, seq_len]
        pooled = torch.einsum("bsh,bs->bh", last_hidden, weights.to(device))
        x = self.simple_nn(pooled)
        return x.flatten()[0]


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

        self.tokenizer = (
            args.tokenizer
            if hasattr(args, "tokenizer")
            else AutoTokenizer.from_pretrained(args.lm_name, model_max_length=512)
        )
        actor_model = (
            args.actor_model
            if hasattr(args, "actor_model")
            else AutoModel.from_pretrained(args.lm_name)
        )
        critic_model = (
            args.critic_model
            if hasattr(args, "critic_model")
            else AutoModel.from_pretrained(args.lm_name)
        )

        self.actor = ActorNetwork(actor_model, self.tokenizer)
        self.critic = CriticNetwork(
            critic_model, self.tokenizer, input_dim=actor_model.config.hidden_size
        )

        self.env = args.env

        self.gamma = kwargs.get("gamma", 0.99)
        self.clip_epsilon = kwargs.get("clip_epsilon", 0.1)
        self.lambda_gae = kwargs.get("lambda_gae", 0.95)

        params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(
            params,
            lr=kwargs.get("learning_rate", 1e-5),
        )

        self.trajectories = []
        self.episode_buffer = [[] for _ in range(args.num_envs)]
        self.completed_episodes = [[]]
        self.max_segment_length = kwargs.get("max_segment_length", 256)
        self.last_actions = []

    def _tokenize(self, text):
        """
        Tokenize input text using the actor's tokenizer.
        Args:
            text: Input string or list of strings.
        Returns:
            dict: Tokenized tensors with input_ids and attention_mask
        """
        if isinstance(text, str):
            text = [text]

        encoding = self.tokenizer(
            [clean(t) for t in text], padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        encoding["length"] = encoding["attention_mask"].sum(dim=1)

        return encoding

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
        env_idx = getattr(transition.state, "env_idx", 0)
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
        self.completed_episodes[-1].append(ppo_llm_transition)
        if transition.done or len(self.episode_buffer) >= self.max_segment_length:
            self.completed_episodes.append([])
            self.episode_buffer = []

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
                [t.log_prob for t in episode], device=self.device, dtype=torch.float32
            )
            advantages = torch.tensor(
                [t.advantage for t in episode], device=self.device, dtype=torch.float32
            )
            returns = torch.tensor(
                [t.returns for t in episode], device=self.device, dtype=torch.float32
            )

            logprobs = torch.stack(
                [
                    self.actor.create_distribution(s, va, self.device).log_prob(a)
                    for s, va, a in zip(states, valid_actions, actions)
                ]
            )
            values = torch.stack([self.critic(s, self.device) for s in states])
            ratio = torch.exp(logprobs - old_log_probs)
            surrogate1 = ratio * advantages
            surrogate2 = (
                torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                * advantages
            )
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            value_loss = F.mse_loss(values, returns)
            loss = policy_loss + value_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        self.completed_episodes = [[]]
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

    def _compute_advantages(self, episode, gamma=None, lambda_gae=None):
        """
        Compute GAE and returns for an episode. Updates episode transitions in-place.
        Args:
            episode: List of PPOLLMTransition objects for the episode.
            gamma: Discount factor (optional).
            lambda_gae: GAE lambda parameter (optional).
        Returns:
            tuple: (advantages, returns) as numpy arrays.
        """
        gamma = gamma if gamma is not None else self.gamma
        lambda_gae = lambda_gae if lambda_gae is not None else self.lambda_gae
        rewards = [t.reward for t in episode]
        values = [t.value for t in episode]
        dones = [t.done for t in episode]
        advantages = np.zeros(len(rewards), dtype=np.float32)
        returns = np.zeros(len(rewards), dtype=np.float32)
        next_value = 0
        next_advantage = 0
        for t in reversed(range(len(rewards))):
            done_mask = 0 if dones[t] else 1
            delta = rewards[t] + gamma * next_value * done_mask - values[t]
            advantages[t] = delta + gamma * lambda_gae * next_advantage * done_mask
            returns[t] = rewards[t] + gamma * next_value * done_mask
            next_value = values[t]
            next_advantage = advantages[t]
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for i in range(len(episode)):
            episode[i] = episode[i]._replace(
                advantage=advantages[i], returns=returns[i]
            )
        return advantages, returns

    # def _collect_trajectories(self, batch_size):
    #     """
    #     NOT USED IN TRAINING, USE FOR DEBUGGING!
    #     Collect a batch of trajectories by interacting with the environment.
    #     Args:
    #         batch_size: Number of transitions to collect.
    #     Returns:
    #         list: List of TrajectoryStep objects.
    #     """
    #     trajectories = []
    #     reset_result = self.env.reset()
    #     state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    #     for _ in range(batch_size):
    #         info = self.env._get_info()
    #         available_actions = info["game_state"]["choice_texts"]
    #         action, logprob, action_idx = self._choose_action(
    #             state, logprob_flag=True, return_idx=True
    #         )
    #         next_state, reward, done, _ = self.env.step(action_idx)
    #         trajectories.append(
    #             TrajectoryStep(
    #                 state=state,
    #                 action=action,
    #                 reward=reward,
    #                 done=done,
    #                 next_state=next_state,
    #                 logprob=logprob,
    #                 available_actions=available_actions,
    #             )
    #         )
    #         state = self.env.reset() if done else next_state
    #         if isinstance(state, tuple):
    #             state = state[0]
    #     self.trajectories = trajectories
    #     return trajectories

    # def _compute_gaes(self, trajectories, returns_flag=False):
    #     """
    #     NOT USED IN TRAINING, USE FOR DEBUGGING!
    #     Compute Generalized Advantage Estimates (GAEs) for a list of trajectories.
    #     Args:
    #         trajectories: List of TrajectoryStep objects.
    #         returns_flag: If True, also return value estimates.
    #     Returns:
    #         torch.Tensor or tuple: GAE tensor (and returns tensor if returns_flag=True).
    #     """
    #     gaes = []
    #     returns = []
    #     gae = 0
    #     values = [] if returns_flag else None
    #     for i in reversed(range(len(trajectories))):
    #         state, _, reward, done, next_state, *_ = trajectories[i]
    #         value = self.critic(state, self.device)
    #         if returns_flag:
    #             values.insert(0, value)
    #         next_value = 0 if done else self.critic(next_state, self.device)
    #         delta = reward + self.gamma * next_value - value
    #         gae = delta + self.gamma * self.lambda_gae * gae * (1 - done)
    #         gaes.insert(0, gae)
    #         returns.insert(0, gae + value)
    #     gaes_tensor = torch.tensor(gaes, dtype=torch.float32)
    #     returns_tensor = torch.tensor(returns, dtype=torch.float32)
    #     return (gaes_tensor, returns_tensor) if returns_flag else gaes_tensor
