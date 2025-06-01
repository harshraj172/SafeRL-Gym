import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Callable
from loguru import logger

from src.agent.ppo import PPOAgent, PPONetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MASEPPONetwork(PPONetwork):
    def __init__(self, lm_name, stats_dim, hidden_dim):
        """
        Initialize the MASE PPO network.
        :param lm_name: Name of the language model to use.
        :param stats_dim: Dimension of the statistics input.
        :param hidden_dim: Dimension of the hidden layers.
        """
        super().__init__(lm_name, stats_dim, hidden_dim)
        # Additional initialization specific to MASE can be added here


class PPOMASEAgent(PPOAgent):
    def __init__(
        self,
        args,
        prompt_lb: str | None,
        prompt_ub: str | None,
        promp_est: str | None,
        judge_llm_func: Callable | None = None,
    ):
        """
        Initialize the MASE agent with the given arguments.
        :param args: The argparse Namespace with hyperparameters/settings.
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
        self.network = PPONetwork(args.lm_name, args.stats_dim, args.hidden_dim).to(
            device
        )

        self.prompt_lb = prompt_lb
        self.prompt_ub = prompt_ub
        self.prompt_est = promp_est

        if self.prompt_lb is None:
            self.prompt_lb = "You are an AI assistant that helps determine how ethical actions are in a game environment. Your task is to assess the ethics of each action based on the current state and provide a ethics score from 0 to 10, where 0 is completely ethical and 10 is completely unethical. You are in charge of providing a lower bound for the ethics score. Do not output any other text, just the score. State: {} \n Action: {}"
        if self.prompt_ub is None:
            self.prompt_ub = "You are an AI assistant that helps determine how ethical actions are in a game environment. Your task is to assess the ethics of each action based on the current state and provide a ethics score from 0 to 10, where 0 is completely ethical and 10 is completely unethical. You are in charge of providing an upper bound for the ethics score. Do not output any other text, just the score. State: {} \n Action: {}"
        if self.prompt_est is None:
            self.prompt_est = "You are an AI assistant that helps determine how ethical actions are in a game environment. Your task is to assess the ethics of each action based on the current state and provide a ethics score from 0 to 10, where 0 is completely ethical and 10 is completely unethical. You are in charge of estimating the ethics score for the action based on the current state. Do not output any other text, just the score. State: {} \n Action: {}"

        self.prompt_llm_judge = judge_llm_func

    @torch.no_grad()
    def mase_act(self, states, poss_acts):
        """Get action scores and sample from policy"""
        action_scores, values = self.network.forward(states, poss_acts)

        # Convert scores to probabilities and sample
        action_probs = [F.softmax(scores, dim=0) for scores in action_scores]
        # If an action is unsafe, set its probability to zero
        for batch, (state, probs) in enumerate(zip(states, action_probs)):
            action_options = poss_acts[batch]
            safe_actions = self.safe_actions(state, action_options, self.b_h)
            for prob_idx, action in enumerate(action_options):
                if action not in safe_actions:
                    probs[prob_idx] = 0.0
            # Normalize probabilities to ensure they sum to 1
            if probs.sum() > 0:
                action_probs[batch] = probs / probs.sum()
            else:
                # If no actions are safe, fallback to emergency action
                action_probs[batch] = None
        distributions = [
            Categorical(probs) if probs.sum() > 0 else None for probs in action_probs
        ]
        actions = [
            dist.sample() if dist is not None else None for dist in distributions
        ]

        log_probs = [
            (
                dist.log_prob(act)
                if dist is not None and act is not None
                else torch.tensor(float("-inf"))
            )
            for dist, act in zip(distributions, actions)
        ]
        return actions, log_probs, values

    def act(self, states, poss_acts, lm=None, eps=None, alpha="", beta=1, k=-1):
        """Select actions and return log probabilities and values"""
        with torch.no_grad():
            act_idxs, log_probs, values = self.mase_act(states, poss_acts, lm)

        # Store the action data for use in observe() - we'll associate it with the most recent state
        self.last_actions = []
        for env_idx, (act_idx, log_prob, value) in enumerate(
            zip(act_idxs, log_probs, values)
        ):
            if act_idx is None:
                log_prob = torch.tensor(float("-inf"))
                value = torch.tensor(0.0)
            self.last_actions.append(
                {
                    "action": (
                        "emergency" if act_idx is None else poss_acts[env_idx][act_idx]
                    ),
                    "log_prob": log_prob.item(),
                    "value": value.item(),
                }
            )

        # Return the actions as expected by the training loop
        act_ids = [
            poss_acts[batch][idx] if idx is not None else None
            for batch, idx in enumerate(act_idxs)
        ]
        action_idxs = [idx.item() if idx is not None else -1 for idx in act_idxs]
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
                safety_score = self.prompt_llm_judge(
                    self.prompt_est.format(state, action)
                )
                safety_scores.append(float(safety_score))
            except Exception as e:
                logger.error(f"Error in LLM interaction for safety assessment: {e}")
                safety_scores.append(10.0)

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
        return diff

    def mase_reward(state, action, reward): ...
