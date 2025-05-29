import torch
from transformers import AutoTokenizer

from src.utils import ReplayMemory, PrioritizedReplayMemory, Transition, State
from src.env.machiavelli.utils import clean

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaseAgent:
    """
    A RandomAgent that does takes random action in a text based env
    """

    def __init__(self, args):
        """
        Initialize with the same signature as DRRNAgent.
        :param args: The argparse Namespace with hyperparameters/settings.
        """
        # If you'd like to keep the same memory usage
        if hasattr(args, 'memory_size'):
            self.memory = PrioritizedReplayMemory(args.memory_size, args.priority_fraction)
        if hasattr(args, 'lm_name'):
            self.tokenizer = AutoTokenizer.from_pretrained(args.lm_name, model_max_length=512)

        self.configure = False
        self.lm = None
        self.alpha = 0
        self.beta = 1

    def _tokenize(self, text):
        if not hasattr(self, 'tokenizer'):
            return None
        encoding = self.tokenizer(
            clean(text), 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(device)
        encoding['length'] = torch.tensor(
            [encoding['input_ids'].size(1)], device=device
        )
        return encoding

    def _configure(self, lm, alpha, beta):
        """
        For adjusting weighting parameters for power regularization, etc.
        """
        self.lm = lm
        self.alpha = alpha
        self.beta = beta
        self.configure = True

    def train(self):
        """
        Called when the agent should prepare for training.
        In a random agent, we do nothing special here.
        """
        pass

    def test(self):
        """
        Called when the agent should prepare for testing/eval mode.
        In a random agent, we do nothing special here.
        """
        pass

    def observe(self, transition, is_prior=False):
        """
        Store a transition in the replay buffer (if you want to keep it).
        """
        self.memory.push(transition, is_prior)

    def act(self, states, poss_acts, lm=None, eps=None, alpha='', beta=1, k=-1):
        """
        Select a random action from poss_acts for each environment/state: (act_ids, idxs, None)
        """
        act_ids = []
        idxs = []
        for batch_idx, valid_actions in enumerate(poss_acts):
            if len(valid_actions) == 0:
                # If for some reason there are no valid actions, pick a dummy
                idx = 0
            else:
                idx = torch.randint(low=0, high=len(valid_actions), size=(1,)).item()
            act_ids.append(valid_actions[idx])
            idxs.append(idx)

        return act_ids, idxs, None

    def update(self):
        """
        In a random agent, we do not update any parameters (no training).
        """
        pass
