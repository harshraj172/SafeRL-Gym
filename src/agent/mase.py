import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Callable
from loguru import logger

from src.agent.ppo_llm import PPOLLMAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPOMASEAgent(PPOLLMAgent):
    def __init__(
        self,
        args,
        emergency_action: str = "emergency",
    ):
        """
        Initialize the MASE agent with the given arguments.
        :param args: The argparse Namespace with hyperparameters/settings.
        :param prompt_lb: Prompt for lower bound ethics score.
        :param prompt_ub: Prompt for upper bound ethics score.
        :param judge_llm_func: Function to call the LLM for safety assessment and gamma estimation.

        :return: None
        """

        super().__init__(args)
        self.b_h = args.b_h
        self.c = args.c
        self.emergency_action = emergency_action
        self._gamma = args.gamma_mase

    @torch.no_grad()
    def _choose_action(
        self,
        state,
        actions,
        safe_actions=None,
        logprob_flag=False,
        return_idx=False,
    ):
        if safe_actions is None or len(safe_actions) == 0:
            if return_idx and logprob_flag:
                return (self.emergency_action, float("-inf"), -1)
            elif return_idx:
                return (self.emergency_action, -1)
            elif logprob_flag:
                return (self.emergency_action, float("-inf"))
            else:
                return self.emergency_action

        # Filter the distribution to only include safe actions
        safe_action_idxs = [actions.index(act) for act in safe_actions]
        dist = self.actor.create_distribution(state, actions, self.device)
        logits = dist.logits.clone()
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[safe_action_idxs] = False
        logits[mask] = float("-inf")
        filtered_dist = Categorical(logits=logits)
        # Sample from the filtered distribution
        action_idx = filtered_dist.sample().item()
        action = actions[action_idx]
        logprob = filtered_dist.log_prob(torch.tensor(action_idx, device=self.device)).item()
        del dist, logits, filtered_dist  # Free memory
        
        if return_idx and logprob_flag:
            return (action, logprob, action_idx)
        elif return_idx:
            return (action, action_idx)
        elif logprob_flag:
            return (action, logprob)
        else:
            return action

    @torch.no_grad()
    def act(self, states, poss_acts, lm=None, safe_actions=None):
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
            result = self._choose_action(
                state,
                actions=actions,
                safe_actions=safe_actions[i] if safe_actions else None,
                logprob_flag=True,
                return_idx=True,
            )
            # Ensure we always get 3 values for this specific call
            if len(result) == 3:
                action, logprob, action_idx = result
            else:
                # Fallback for unexpected return sizes
                action, logprob, action_idx = result[0], float("-inf"), -1
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

    def safe_actions(self, state, action_options, safety_scores):
        """
        Determine safe actions based on the current state and action options.
        :param state: The current state of the environment.
        :param action_options: List of possible actions.
        :param safety_scores: Safety scores for each action.
        :return: List of safe actions.
        """
        return [
            action
            for action, score in zip(action_options, safety_scores)
            if score < self.b_h
        ]

    def gamma_mase(self, state, action):
        """
        Estimate the gamma value for the current state and action.
        This method should be implemented to interact with an LLM or other gamma estimation mechanism.
        :param state: The current state of the environment.
        :param action: The action to evaluate.
        :param eps: Epsilon value for uncertainty estimation, when estimation is not possible.
        :return: Estimated gamma value.
        """
        return self._gamma
