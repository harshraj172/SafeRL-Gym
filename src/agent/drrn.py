import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

# Suppose you already have DRRN and associated code from drrn.py
# from mypackage.drrn import DRRN
# from mypackage.drrn_agent_utils import ReplayMemory, Transition, build_state, etc.

class DRRNAgent:
    """
    A DRRN-based agent that exposes the same interface methods as RandomAgent:
        - select_action()
        - store_transition()
        - train()
        - save_checkpoint()
        - load_checkpoint()
    so it can be used interchangeably in your training loop.
    """

    def __init__(self,
                 action_space_size: int,
                 drrn_model: nn.Module,
                 replay_memory,
                 tokenizer_fn,
                 gamma=0.9,
                 learning_rate=1e-4,
                 clip_grad=5.0,
                 batch_size=16,
                 device='cpu'):
        """
        :param action_space_size: The max number of discrete actions the env can have (e.g. for Gym).
                                 (In a textual environment, you might handle the variable # of choices differently.)
        :param drrn_model:       The DRRN network instance (a PyTorch nn.Module).
        :param replay_memory:    A replay buffer object (like a PrioritizedReplayMemory).
        :param tokenizer_fn:     A function to tokenize text (for textual DRRN). For numeric envs, you may skip.
        :param gamma:            Discount factor.
        :param learning_rate:    Learning rate for the optimizer.
        :param clip_grad:        Gradient clipping value.
        :param batch_size:       Batch size for training.
        :param device:           'cpu' or 'cuda' device string.
        """
        self.action_space_size = action_space_size
        self.network = drrn_model
        self.replay_memory = replay_memory
        self.tokenizer_fn = tokenizer_fn

        self.gamma = gamma
        self.batch_size = batch_size
        self.clip_grad = clip_grad
        self.device = device

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=10000)  # might keep a small queue for debugging or stats

    def select_action(self, state) -> int:
        """
        Return an integer action index for the environment, just like RandomAgent does.

        If this were a textual environment with variable # of actions, you'd
        pass the dynamic action candidates to the DRRN. For a simple discrete environment,
        we might treat the set [0 .. action_space_size-1] as valid actions
        and compute Q-values for each.

        For demonstration, we'll just compute Q(s, a) for all a in [0..action_space_size-1],
        and pick argmax. (Or sample from some distribution if you prefer.)
        """
        # Build "dummy" states: in a textual environment you'd do something like:
        #   built_state = build_state(self.tokenizer_fn, state.obs, state.info, ...)
        # For a standard numeric environment, you might skip encoding or do a simpler transform
        # We'll do a naive transform:
        built_state = [self._simple_encode_state(state)]

        # Build action candidates for DRRN
        act_batch = []
        for a in range(self.action_space_size):
            # For textual environment: you might do tokenize_fn on each possible action text
            # For numeric: we do a placeholder
            act_batch.append(self._simple_encode_action(a))

        # Get Q-values [q_0, q_1, ..., q_{N-1}] from DRRN
        q_values = self.network(built_state, [act_batch])[0]  # the network returns a list of size=1 for the batch
        q_values = q_values.detach().cpu().numpy()

        # Epsilon-greedy or softmax could go here. For simplicity, pick argmax
        action_idx = int(np.argmax(q_values))
        return action_idx

    def store_transition(self, state, action, reward, next_state, done):
        """
        Same interface as RandomAgent, but we'll store into the DRRN memory structure.
        For text-based DRRN, we'd do the state/next_state building and store a Transition object.
        """
        # Convert to DRRN-like representation (here is a minimal numeric approach).
        s = self._simple_encode_state(state)
        a = [self._simple_encode_action(action)]  # DRRN expects a list
        # next_s can also have a dynamic set of possible actions
        next_s = self._simple_encode_state(next_state)
        # Next-state's action set is up to you. Here we just do the entire action space again:
        next_a = [self._simple_encode_action(i) for i in range(self.action_space_size)]

        # Transition: (state, act, reward, next_state, next_acts, done).
        # In the reference DRRN code, it might be a named tuple or a custom class.
        # We'll do something minimal:
        transition = (
            s,
            a[0],  # your DRRN might store the single chosen action in a nested list
            reward,
            next_s,
            next_a,
            done
        )

        # If your DRRN memory is a PrioritizedReplayMemory, call memory.push(transition).
        self.replay_memory.push(transition, is_prior=False)

        # Also store in the local memory so we can debug or see transitions
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        """
        The actual DRRN training step. Similar to 'update()' in your original code.
        We do a small batch from replay_memory, compute TD target, backprop.
        """
        if len(self.replay_memory) < self.batch_size:
            return

        transitions = self.replay_memory.sample(self.batch_size)
        # Convert them into torch-friendly form
        # Typically you might have: batch = Transition(*zip(*transitions)) in code
        # We'll do something minimal:

        state_batch, act_batch, reward_batch, next_state_batch, next_act_batch, done_batch = [], [], [], [], [], []
        for tr in transitions:
            s, a, r, ns, na, d = tr
            state_batch.append(s)
            act_batch.append([a])  # DRRN expects a list of valid actions
            reward_batch.append(r)
            next_state_batch.append(ns)
            next_act_batch.append(na)
            done_batch.append(d)

        # Compute Q(s', a') for all a'
        next_qvals = self.network(next_state_batch, next_act_batch)
        # For each in the batch, we want the max over next_qvals
        next_qvals = torch.tensor([vals.max().item() for vals in next_qvals], device=self.device)

        # (1 - done) to zero out next_q when done
        done_tensor = torch.tensor(done_batch, dtype=torch.float32, device=self.device)
        next_qvals = next_qvals * (1.0 - done_tensor)

        # Our target = r + gamma * next_q
        reward_tensor = torch.tensor(reward_batch, dtype=torch.float32, device=self.device)
        targets = reward_tensor + self.gamma * next_qvals

        # Now compute Q(s, a). We nest the chosen action in a list for each batch item:
        qvals = self.network(state_batch, act_batch)  # returns a list of length batch_size
        # Each qvals[i] is a 1-element tensor, so we cat them:
        qvals = torch.cat(qvals, dim=0)

        # Smooth L1 (Huber) or MSE
        loss = F.smooth_l1_loss(qvals, targets.detach())

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_grad)
        self.optimizer.step()

    def save_checkpoint(self, checkpoint_path: str):
        """
        Save model weights, optimizer, etc.
        """
        ckpt = {
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(ckpt, checkpoint_path)
        print(f"Saved DRRN checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model weights, optimizer, etc.
        """
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.network.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print(f"Loaded DRRN checkpoint from {checkpoint_path}")

    def _simple_encode_state(self, state):
        """
        Minimal numeric encoding for a Gym env state.
        In a text-based environment, you'd do something like:
           return tokenize_fn(state_string)
        For now, just a dummy zero vector or cast to a torch tensor.
        """
        if isinstance(state, np.ndarray):
            s = torch.tensor(state, dtype=torch.float32, device=self.device)
        elif isinstance(state, (int, float)):
            s = torch.tensor([state], dtype=torch.float32, device=self.device)
        else:
            # fallback: zero
            s = torch.zeros(10, dtype=torch.float32, device=self.device)
        return s

    def _simple_encode_action(self, action_idx):
        """
        Minimal numeric encoding for an action. For text, you'd do tokenize_fn(action_string).
        Here, we just return a 1D tensor with the action index or something similarly trivial.
        """
        return torch.tensor([action_idx], dtype=torch.float32, device=self.device)
