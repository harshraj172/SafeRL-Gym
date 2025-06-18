from collections import namedtuple
from typing import Optional

from tqdm import tqdm
from src.agent.ppo_llm import CriticNetwork, PPOLLMAgent, RolloutDataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from src.agent.base import BaseAgent
from src.env.machiavelli.utils import clean
from torch.utils.data import DataLoader

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


class PIDLagrangianOptimizer:
    """
    PID-based Lagrangian multiplier optimizer for constraint satisfaction.
    """

    def __init__(self, kp=0.05, ki=0.0005, kd=0.1, alpha=0.9):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.alpha = alpha  # Moving average factor for derivative

        self.lambda_val = 1.0  # Lagrangian multiplier
        self.integral = 0.0
        self.prev_error = 0.0
        self.derivative_ma = 0.0  # Moving average of derivative

    def step(self, cost_val, cost_limit):
        """Update Lagrangian multiplier using PID control."""
        error = cost_val - cost_limit

        # Proportional term
        proportional = self.kp * error

        # Integral term (accumulated error)
        self.integral += error
        integral_term = self.ki * self.integral

        # Derivative term (rate of change)
        derivative = error - self.prev_error
        self.derivative_ma = (
            self.alpha * self.derivative_ma + (1 - self.alpha) * derivative
        )
        derivative_term = self.kd * self.derivative_ma

        # Update lambda
        self.lambda_val += proportional + integral_term + derivative_term
        self.lambda_val = max(0.0, self.lambda_val)  # Keep non-negative

        self.prev_error = error

    def get_lambda(self):
        return self.lambda_val

    def state_dict(self):
        return {
            "lambda_val": self.lambda_val,
            "integral": self.integral,
            "prev_error": self.prev_error,
            "derivative_ma": self.derivative_ma,
        }

    def load_state_dict(self, state):
        self.lambda_val = state["lambda_val"]
        self.integral = state["integral"]
        self.prev_error = state["prev_error"]
        self.derivative_ma = state["derivative_ma"]


class PPOLagLLMAgent(PPOLLMAgent):
    """
    PPO LLM agent with Lagrangian multiplier for safety constraints.
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


        self.cost_limit = kwargs.get(
            "cost_limit", 3.0
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

        self.clip_cost_valuef = kwargs.get("clip_cost_valuef", 100.0)

        cost_embedding = AutoModel.from_pretrained(
            "microsoft/deberta-v3-xsmall", output_hidden_states=True
        )
        self.cost_critic = CriticNetwork(
            embedding=cost_embedding,
            tokenizer=AutoTokenizer.from_pretrained(
                "microsoft/deberta-v3-xsmall", model_max_length=2048
            ),
        ).to(self.device)

        # Replace simple Lagrangian with PID optimizer
        self.pid_lagrangian = PIDLagrangianOptimizer(
            kp=kwargs.get("lagrange_kp", 0.05),
            ki=kwargs.get("lagrange_ki", 0.0005),
            kd=kwargs.get("lagrange_kd", 0.1),
        )

        # Update optimizer to include cost critic
        params = (
            list(self.actor.parameters())
            + list(self.critic.parameters())
            + list(self.cost_critic.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=kwargs.get("learning_rate", 3e-5))

        # PPO-Lag specific parameters
        self.rescaling = kwargs.get("rescaling", True)  # Rescaling trick
        self.cost_vf_coef = kwargs.get("cost_vf_coef", 0.25)

    @torch.no_grad()
    def act(self, states, poss_acts, lm=None, *args, **kwargs):
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

            # Get both value and cost value predictions
            value = self.critic(state, self.device).item()
            cost_value = self.cost_critic(state, self.device).item()

            self.last_actions.append(
                {
                    "action": action_idx,
                    "log_prob": logprob,
                    "value": value,
                    "cost_value": cost_value,
                }
            )

        return act_ids, action_idxs, log_probs

    @torch.no_grad()
    def _compute_advantages(self, episode, gamma=None, lambda_gae=None):
        gamma = gamma if gamma is not None else self.gamma
        lambda_gae = lambda_gae if lambda_gae is not None else self.lambda_gae

        rewards = [t.reward for t in episode]
        costs = [t.cost for t in episode]
        values = [t.value for t in episode]
        cost_values = [t.cost_value for t in episode]
        dones = [t.done for t in episode]

        # normalize rewards by max_reward
        rewards = np.array(rewards, dtype=np.float32) / self.max_reward

        # Bootstrap for both reward and cost
        if dones[-1]:
            next_value = 0
            next_cost_value = 0
        else:
            with torch.no_grad():
                final_state = episode[-1].state
                next_value = self.critic(final_state, self.device).item()
                next_cost_value = self.cost_critic(final_state, self.device).item()

        # Compute reward advantages and returns
        advantages = np.zeros(len(rewards), dtype=np.float32)
        next_advantage = 0

        for t in reversed(range(len(rewards))):
            not_done = 1.0 - float(dones[t])
            delta = rewards[t] + gamma * next_value * not_done - values[t]
            advantages[t] = delta + gamma * lambda_gae * next_advantage * not_done
            next_value = values[t]
            next_advantage = advantages[t]

        returns = np.array(advantages) + np.array(values)

        # Compute cost advantages and returns
        cost_advantages = np.zeros(len(costs), dtype=np.float32)
        next_cost_advantage = 0

        # normalize rewards by max_costs (TODO get this from the game)
        costs = np.array(costs, dtype=np.float32) / 10.0

        for t in reversed(range(len(costs))):
            not_done = 1.0 - float(dones[t])
            cost_delta = costs[t] + gamma * next_cost_value * not_done - cost_values[t]
            cost_advantages[t] = (
                cost_delta + gamma * lambda_gae * next_cost_advantage * not_done
            )
            next_cost_value = cost_values[t]
            next_cost_advantage = cost_advantages[t]

        cost_returns = np.array(cost_advantages) + np.array(cost_values)

        cost_advantages = masked_whiten(
            torch.tensor(cost_advantages, device=self.device, dtype=torch.float32),
            torch.tensor(
                [1.0] * len(cost_advantages), device=self.device, dtype=torch.float32
            ),
        )

        # Update episode with computed advantages and returns
        for i in range(len(episode)):
            episode[i] = episode[i]._replace(
                advantage=advantages[i],
                cost_advantage=cost_advantages[i],
                returns=returns[i],
                cost_returns=cost_returns[i],
            )

    def update(self):
        # 1. Gather transitions and build a DataLoader -------------------------------------------------
        transitions = self._collect_transitions()
        if len(transitions) == 0:
            return 0.0  # nothing to learn from

        # Build DataLoader
        loader = DataLoader(
            RolloutDataset(transitions),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=lambda x: x,  # No special collate function needed
        )

        total_loss = 0.0
        total_episode_cost = 0.0
        for update in range(self.num_updates):
            for batch in tqdm(loader):
                # Unpack the batch -------------------------------------------------------------------
                states = [t.state for t in batch]
                actions = torch.tensor([t.action for t in batch], device=self.device, dtype=torch.long)
                valid_actions = [t.valid_actions for t in batch]
                old_log_probs = torch.tensor([t.log_prob for t in batch], device=self.device, dtype=torch.float32)
                old_values = torch.tensor([t.value for t in batch], device=self.device, dtype=torch.float32)
                advantages = torch.tensor([t.advantage for t in batch], device=self.device, dtype=torch.float32)
                returns = torch.tensor([t.returns for t in batch], device=self.device, dtype=torch.float32)
                costs = torch.tensor([t.cost for t in batch], device=self.device, dtype=torch.float32)
                cost_advantages = torch.tensor(
                    [t.cost_advantage for t in batch],
                    device=self.device,
                    dtype=torch.float32,
                )
                cost_returns = torch.tensor(
                    [t.cost_returns for t in batch],
                    device=self.device,
                    dtype=torch.float32,
                )
                episode_cost = costs.sum().item()
                total_episode_cost += episode_cost

                # 2. Forward pass --------------------------------------------------------------------
                action_logprob_entropies = [
                    self.compute_actions(s, va, a)
                    for s, va, a in zip(states, valid_actions, actions)
                ]
                logprobs = torch.stack([lp for lp, _ in action_logprob_entropies])
                entropies = torch.stack([ent for _, ent in action_logprob_entropies])
                values = torch.stack([self.critic(s, self.device) for s in states])

                # 3. Normalize advantages ------------------------------------------------------------
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(logprobs - old_log_probs)
                surrogate1 = ratio * advantages
                surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                # Get value and cost value predictions
                values = torch.stack([self.critic(s, self.device) for s in states])
                cost_values = torch.stack(
                    [self.cost_critic(s, self.device) for s in states]
                )
                # 5. Value loss ----------------------------------------------------------------------
                vpred_clipped = torch.clamp(values, old_values - self.clip_valuef, old_values + self.clip_valuef)
                vf_losses1 = (values - returns) ** 2
                vf_losses2 = (vpred_clipped - returns) ** 2
                value_loss = torch.max(vf_losses1, vf_losses2).mean()
                # 6. Entropy bonus ------------------------------------------------------------------
                entropy_loss = self.entropy_coef * entropies.mean()

                # 7. PPO-Lag safety loss (cost part)
                cost_surrogate = ratio * cost_advantages
                safety_loss = torch.mean(cost_surrogate) * self.pid_lagrangian.get_lambda()

                # Rescaling trick (Alg. 1 from Stooke et al.)
                if self.rescaling:
                    rescaling_factor = 1.0 / (self.pid_lagrangian.get_lambda() + 1.0)
                else:
                    rescaling_factor = 1.0

                # Total policy loss
                policy_loss = rescaling_factor * (policy_loss + safety_loss)


                # Cost value loss with clipping
                cost_vpred_clipped = torch.clamp(
                    cost_values,
                    cost_returns - self.clip_cost_valuef,
                    cost_returns + self.clip_cost_valuef,
                )
                cost_vf_losses1 = (cost_values - cost_returns) ** 2
                cost_vf_losses2 = (cost_vpred_clipped - cost_returns) ** 2
                cost_value_loss = torch.max(cost_vf_losses1, cost_vf_losses2).mean()

                # Total loss
                total_loss_episode = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.cost_vf_coef * cost_value_loss
                    - entropy_loss
                )

                # Backward pass
                self.optimizer.zero_grad()
                total_loss_episode.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters())
                    + list(self.critic.parameters())
                    + list(self.cost_critic.parameters()),
                    max_norm=10.0,
                )
                self.optimizer.step()

                total_loss += total_loss_episode.item()

                print(
                    f"Episode Cost: {episode_cost:.2f}, "
                    f"Lambda: {self.pid_lagrangian.get_lambda():.4f}, "
                    f"Policy Loss: {policy_loss.item():.4f}, "
                    f"Safety Loss: {safety_loss.item():.4f}, "
                    f"Value Loss: {value_loss.item():.4f}, "
                    f"Cost Value Loss: {cost_value_loss.item():.4f}"
                )
            print(
                f"[Update {update+1}/{self.num_updates}] Loss: {total_loss_episode.item():.4f} | "
                f"Policy {policy_loss.item():.4f} | Value {value_loss.item():.4f} | "
                f"Entropy {entropy_loss.item():.4f} | Mean batch cost {costs.mean().item():.2f}"
            )


        # Update Lagrangian multiplier based on average episode cost
        avg_episode_cost = total_episode_cost / max(len(self.completed_episodes), 1)
        self.pid_lagrangian.step(avg_episode_cost, self.cost_limit)

        self.completed_episodes = []
        # Clear the episode buffer
        self.episode_buffer = [[] for _ in range(len(self.episode_buffer))]
        return total_loss / (self.num_updates * len(loader))
