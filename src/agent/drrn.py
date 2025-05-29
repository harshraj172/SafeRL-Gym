import time
import numpy as np
import itertools
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from src.agent.base import BaseAgent
from src.env.machiavelli.utils import clean
from src.utils import ReplayMemory, PrioritizedReplayMemory, Transition, State

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch_encoded_tokens(x):
    """ Takes a list of tokenized sequences and returns a batch of padded sequences
        token_list: list of tokenized sequences
    """
    return {'input_ids': torch.nested.to_padded_tensor(torch.nested.nested_tensor([n['input_ids'].squeeze() for n in x]), 0),
            'attention_mask': torch.nested.to_padded_tensor(torch.nested.nested_tensor([n['attention_mask'].squeeze() for n in x]), 0)}

class DRRN(torch.nn.Module):
    """
        Deep Reinforcement Relevance Network - He et al. '16
    """

    def __init__(self, lm_name, stats_dim, hidden_dim):
        super(DRRN, self).__init__()
        self.embedding = AutoModel.from_pretrained(
            lm_name, output_hidden_states=True)
        embedding_dim = self.embedding.get_input_embeddings().embedding_dim
        self.obs_encoder = nn.GRU(embedding_dim, hidden_dim)
        self.look_encoder = nn.GRU(embedding_dim, hidden_dim)
        self.inv_encoder = nn.GRU(stats_dim, hidden_dim)
        self.act_encoder = nn.GRU(embedding_dim, hidden_dim)
        self.hidden = nn.Linear(4 * hidden_dim, hidden_dim)
        self.act_scorer = nn.Linear(hidden_dim, 1)

    # def reset(self, idx):
    #     self.obs_hidden[:,idx,:] = torch.zeros(1, self.hidden_dim, device=device)

    def packed_rnn(self, x, rnn, stats=False, hidden=None):
        """ Runs the provided rnn on the input x. Takes care of packing/unpacking.
            x: list of unpadded input sequences
            Returns a tensor of size: len(x) x hidden_dim
        """
        if stats:
            x = [xi[:rnn.input_size] for xi in x] # TAYLOR - for some reason has longer inputs than rnn input size at times, truncating here
            inp = torch.stack(x).unsqueeze(0)
            _, out = rnn(inp)
            return out.squeeze(0)
        
        lengths = torch.cat([n['length'] for n in x])
        x = batch_encoded_tokens(x)
        # Run the embedding layer
        # Time x Batch x EncDim
        embed = self.embedding(**x)['last_hidden_state'].permute(1, 0, 2).detach()
        out, _ = rnn(embed)

        # Unpack
        idx = (lengths - 1).view(-1, 1).expand(len(lengths), out.size(2)).unsqueeze(0)
        out = out.gather(0, idx).squeeze(0)
        del idx, lengths
        return out

    def forward(self, state_batch, act_batch):
        """
            Batched forward pass.
            obs_id_batch: iterable of unpadded sequence ids
            act_batch: iterable of lists of unpadded admissible command ids
            Returns a tuple of tensors containing q-values for each item in the batch
        """
        # Zip the state_batch into an easy access format
        # print("state_batch:", state_batch)
        state = State(*zip(*state_batch))
        # This is number of admissible commands in each element of the batch
        act_sizes = [len(a) for a in act_batch]
        # Combine next actions into one long list
        act_batch = list(itertools.chain.from_iterable(act_batch))
        act_out = self.packed_rnn(act_batch, self.act_encoder)
        # Encode the various aspects of the state
        obs_out = self.packed_rnn(state.obs, self.obs_encoder, hidden='obs')
        look_out = self.packed_rnn(state.description, self.look_encoder)
        inv_out = self.packed_rnn(
            state.inventory, self.inv_encoder, stats=True)

        state_out = torch.cat((obs_out, look_out, inv_out), dim=1)
        # Expand the state to match the batches of actions
        state_out = torch.cat([state_out[i].repeat(j, 1)
                              for i, j in enumerate(act_sizes)], dim=0)
        z = torch.cat((state_out, act_out), dim=1)  # Concat along hidden_dim
        z = F.relu(self.hidden(z))
        act_values = self.act_scorer(z).squeeze(-1)
        # Split up the q-values by batch
        return act_values.split(act_sizes)

    @torch.no_grad()
    def act(self, states, valid_ids, lm=None, **kwargs):
        """ Returns an action-string, optionally sampling from the distribution
            of Q-Values.
        """
        eps = kwargs.get("eps", None)
        alpha = kwargs.get("alpha", 0)
        beta = kwargs.get("beta", 1)
        k = kwargs.get("k", -1)
        return_q = kwargs.get("return_q", False)
        q_values = self.forward(states, valid_ids)
        if alpha > 0 or (eps is not None and k != -1):  # need to use lm_values
            expand = [z for x in valid_ids for z in x]
            lm_values = lm(batch_encoded_tokens(expand))
            lm_values = torch.split(lm_values, [len(x) for x in valid_ids])
            act_values = [q_value - bert_value
                          for q_value, bert_value in zip(q_values, lm_values)]
        else:
            act_values = q_values
        if eps is None:  # sample ~ softmax(act_values)
            act_idxs = [torch.multinomial(
                F.softmax(vals / beta, dim=0), num_samples=1).item() for vals in act_values]
        else:  # w.p. eps, ~ softmax(act_values) | uniform(top_k(act_values)), w.p. (1-eps) arg max q_values
            if k == 0:  # soft sampling
                act_idxs = [torch.multinomial(
                    F.softmax(vals, dim=0), num_samples=1).item() for vals in lm_values]
            elif k == -1:
                act_idxs = [np.random.choice(range(len(vals)))
                            for vals in q_values]
            else:  # hard (uniform) sampling
                act_idxs = [np.random.choice(vals.topk(k=min(k, len(vals)), dim=0).indices.tolist()) for vals in
                            lm_values]
            act_idxs = [vals.argmax(dim=0).item() if np.random.rand() > eps else idx for idx, vals in
                        zip(act_idxs, q_values)]
        if return_q:
            return act_idxs, act_values
        return act_idxs

class DRRNAgent(BaseAgent):
    def __init__(self, args):
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.network = DRRN(args.lm_name, args.stats_dim,
                            args.hidden_dim).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(args.lm_name, model_max_length=512)
        # self.memory = ReplayMemory(args.memory_size)
        self.memory = PrioritizedReplayMemory(
            args.memory_size, args.priority_fraction)
        self.save_path = args.output_dir
        self.clip = args.clip
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=args.learning_rate)
        self.transitions = None
        self.prev_obs = None
        self.prev_acts = None
        self.configure = False

    def _tokenize(self, text):
        #text = [clean(t) for t in text]
        encoding = self.tokenizer(
            clean(text), padding=True, truncation=True, return_tensors="pt").to(device)
        encoding['length'] = torch.tensor(
            [encoding['input_ids'].size(1)]).to(device)
        return encoding

    def _configure(self, lm, alpha, beta):
        self.lm = lm
        self.alpha = alpha
        self.beta = beta
        self.configure = True
    
    def train(self):
        self.prev_obs = None
        self.prev_acts = None

    def test(self):
        self.prev_obs = None
        self.prev_acts = None

    def observe(self, transition, is_prior=False):
        self.memory.push(transition, is_prior)

    def act(self, states, poss_acts, lm=None, args=dict()):
        """ Returns a string action from poss_acts. """
        # add these details from model.json under checkpoints directory
        if type(args) == dict:
            eps, alpha, beta, k = args.get("eps", None), args.get("alpha", 0), args.get("beta", 1), args.get("k", -1)
            # idxs = self.network.act(states, poss_acts, lm=lm, **vars(args))
            idxs = self.network.act(states, poss_acts, lm=lm, eps=eps, alpha=alpha, beta=beta, k=k)
        else:
            idxs = self.network.act(states, poss_acts, lm=lm, eps=args.eps, alpha=args.alpha, beta=args.beta)
        act_ids = [poss_acts[batch][idx] for batch, idx in enumerate(idxs)]
        return act_ids, idxs, None
    
    def update(self):
        if len(self.memory) < self.batch_size:
            return

        self.transitions = transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Compute Q(s', a') for all a'
        # TODO: Use a target network???
        next_qvals = self.network(batch.next_state, batch.next_acts)
        # Take the max over next q-values
        next_qvals = torch.tensor([vals.max()
                                  for vals in next_qvals], device=device)
        # Zero all the next_qvals that are done
        next_qvals = next_qvals * \
            (1 - torch.tensor(batch.done, dtype=torch.float, device=device))
        targets = torch.tensor(
            batch.reward, dtype=torch.float, device=device) + self.gamma * next_qvals
        # Next compute Q(s, a)
        # Nest each action in a list - so that it becomes the only admissible cmd
        nested_acts = tuple([[a] for a in batch.act])
        qvals = self.network(batch.state, nested_acts)
        # Combine the qvals: Maybe just do a greedy max for generality
        qvals = torch.cat(qvals)

        # Compute TD loss (Huber loss)
        loss = F.smooth_l1_loss(qvals, targets.detach())

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.clip)
        self.optimizer.step()
        loss_val = loss.item()
        del next_qvals, qvals, targets, loss
        return loss_val