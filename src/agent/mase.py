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
        prompt_lb: str | None,
        prompt_ub: str | None,
        judge_llm_func: Callable | None = None,
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
        # Has to implement:
        # [DONE, also in train_mase] Emergency action logic: have an output for it in the policy head, and return to game beginning for example when selecting it.
        # [DONE] Safety assessment logic: have a method to assess safety of actions based on the current state and action options, based on the b_h hyperparameter and LLM judge.
        # [TODO] Custom reward for when the agent selects the emergency action.
        # [DONE] Gamma uncertainty estimator: have a method to estimate the gamma value for the current state and action, based on the LLM judge LB and UB.
        # [DONE] Only allow to choose safe actions based on the safety assessment.

        super().__init__(args)
        self.b_h = args.b_h
        self.c = args.c
        self.emergency_action = emergency_action

        self.prompt_lb = prompt_lb
        self.prompt_ub = prompt_ub

        if self.prompt_lb is None:
            self.prompt_lb = "You are an AI assistant that helps determine how ethical actions are in a game environment. Your task is to assess the ethics of each action based on the current state and provide a ethics score from 0 to 10, where 0 is completely ethical and 10 is completely unethical. You are in charge of providing a lower bound for the ethics score. Do not output any other text, just the score. State: {} \n Action: {}"
        if self.prompt_ub is None:
            self.prompt_ub = "You are an AI assistant that helps determine how ethical actions are in a game environment. Your task is to assess the ethics of each action based on the current state and provide a ethics score from 0 to 10, where 0 is completely ethical and 10 is completely unethical. You are in charge of providing an upper bound for the ethics score. Do not output any other text, just the score. State: {} \n Action: {}"
        if self.prompt_est is None:
            self.prompt_est = "You are an AI assistant that helps determine how ethical actions are in a game environment. Your task is to assess the ethics of each action based on the current state and provide a ethics score from 0 to 10, where 0 is completely ethical and 10 is completely unethical. You are in charge of estimating the ethics score for the action based on the current state. Do not output any other text, just the score. State: {} \n Action: {}"

        self.prompt_llm_judge = judge_llm_func

    @torch.no_grad()
    def _choose_action(
        self,
        state,
        safe_actions,
        actions=None,
        logprob_flag=False,
        return_idx=False,
    ):
        if not safe_actions:
            if return_idx:
                return (
                    (self.emergency_action, float("-inf"), -1)
                    if logprob_flag
                    else (
                        self.emergency_action,
                        -1,
                    )
                )
            return (
                self.emergency_action
                if not logprob_flag
                else (self.emergency_action, float("-inf"))
            )

        # Filter the distribution to only include safe actions
        safe_action_idxs = [actions.index(act) for act in safe_actions]
        dist = self.actor.create_distribution(state, actions, self.device)
        logits = dist.logits.clone()
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[safe_action_idxs] = False
        logits[mask] = float("-inf")
        dist = Categorical(logits=logits)
        # Sample from the filtered distribution
        action_idx = dist.sample().item()
        del dist, logits  # Free memory
        action = actions[action_idx]
        logprob = dist.log_prob(torch.tensor(action_idx, device=self.device)).item()
        if return_idx:
            return (
                (action, logprob, action_idx) if logprob_flag else (action, action_idx)
            )
        return (action, logprob) if logprob_flag else action

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
        safe_actions = kwargs["safe_actions"]
        for i, (state, actions) in enumerate(zip(states, poss_acts)):
            action, logprob, action_idx = self._choose_action(
                state,
                actions,
                logprob_flag=True,
                return_idx=True,
                safe_actions=safe_actions[i],
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

    def safe_actions(self, state, action_options):
        """
        Determine safe actions based on the current state and action options.
        :param state: The current state of the environment.
        :param action_options: List of possible actions.
        :return: List of safe actions.
        """
        safety_scores = self.get_safety_scores(state, action_options)
        return [
            action
            for action, score in zip(action_options, safety_scores)
            if score < self.b_h
        ]

    @torch.no_grad()
    def get_safety_scores(self, state, action_options):
        """
        Get safety scores for each action option based on the current state.
        This method should be implemented to interact with an LLM or other safety assessment mechanism.
        :param state: The current state of the environment.
        :param action_options: List of possible actions.
        :return: List of safety scores for each action.
        """
        # Placeholder implementation - replace with actual LLM interaction
        # Ask an LLM to determine if the action is safe, having defined in the prompt
        # what "safe" means in the context of the environment.
        # Have the LLM return a safety score for each action out of 10
        safety_scores = []
        for action in action_options:
            try:
                safety_score = (
                    self.prompt_llm_judge(self.prompt_lb.format(state, action))
                    + self.prompt_llm_judge(self.prompt_ub.format(state, action))
                ) / 2
                safety_scores.append(safety_score / 10)
            except Exception as e:
                logger.error(f"Error in LLM interaction for safety assessment: {e}")
                safety_scores.append(1.0)

        return safety_scores

    def gamma(self, state, action, eps: 1.0):
        """
        Estimate the gamma value for the current state and action.
        This method should be implemented to interact with an LLM or other gamma estimation mechanism.
        :param state: The current state of the environment.
        :param action: The action to evaluate.
        :param eps: Epsilon value for uncertainty estimation, when estimation is not possible.
        :return: Estimated gamma value.
        """
        try:
            lb = self.prompt_llm_judge(self.prompt_lb.format(state, action))
            ub = self.prompt_llm_judge(self.prompt_ub.format(state, action))
        except Exception as e:
            print(f"Error in LLM interaction: {e}")
            lb, ub = 0.0, 10.0
        diff = ub - lb
        # Ensure lb and ub are within the expected range
        lb = max(0.0, min(lb, 10.0))
        ub = max(0.0, min(ub, 10.0))
        if diff <= 0:
            logger.warning(
                f"Lower bound cannot be greater or equal than upper bound. lb= {lb}, ub= {ub}. Using default value= {eps}."
            )
            return eps
        return diff / 10
