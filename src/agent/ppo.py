import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModel
from agent.base import BaseAgent
import copy
from collections import namedtuple
import numpy as np
from env.machiavelli.utils import clean

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------
# Utility: pad token sequences once per batch
# ----------------------------------------------------------------------------
def batch_encoded_tokens(encodings):
    """
    Convert a list of tokenizer outputs to padded tensors on device.
    Each encoding: dict with 'input_ids' and 'attention_mask' of shape [1, seq_len].
    """
    ids   = [e['input_ids'].squeeze(0)      for e in encodings]
    masks = [e['attention_mask'].squeeze(0) for e in encodings]
    padded_ids   = nn.utils.rnn.pad_sequence(ids,   batch_first=True, padding_value=0).to(device)
    padded_masks = nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0).to(device)
    return {'input_ids': padded_ids, 'attention_mask': padded_masks}

# ----------------------------------------------------------------------------
# PPO Network: transformer encoder + policy & value heads
# ----------------------------------------------------------------------------
class PPONetwork(nn.Module):
    def __init__(self, lm_name, stats_dim, hidden_dim):
        super().__init__()
        # Load pretrained LM
        self.encoder = AutoModel.from_pretrained(lm_name)

        # --- freeze first two transformer layers ---
        """
        base = getattr(self.encoder, 'base_model', self.encoder)
        if hasattr(base, 'encoder') and hasattr(base.encoder, 'layer'):
            for layer in base.encoder.layer[:2]:
                for p in layer.parameters():
                    p.requires_grad = False
        """
        # --- enable gradient checkpointing to save memory ---
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

        emb_dim     = self.encoder.config.hidden_size
        self.stats_dim = stats_dim

        # state feature head
        self.fc_state  = nn.Linear(2 * emb_dim + stats_dim, hidden_dim)
        # policy head: combine state_feat + action_emb
        self.fc_policy = nn.Linear(hidden_dim + emb_dim, 1)
        # value head: state_feat -> scalar
        self.fc_value  = nn.Linear(hidden_dim, 1)

        init_range = self.encoder.config.initializer_range
        nn.init.normal_(self.fc_value.weight, 0.0, init_range)
        nn.init.zeros_(self.fc_value.bias)

    def forward(self, states, actions=None):
        # encode observations & descriptions
        obs_enc  = batch_encoded_tokens([s.obs         for s in states])
        desc_enc = batch_encoded_tokens([s.description for s in states])
        # inventory vectors
        inv = torch.stack([
            s.inventory[:self.stats_dim] 
                if len(s.inventory) >= self.stats_dim 
                else F.pad(s.inventory, (0, self.stats_dim - len(s.inventory)))
            for s in states
        ], dim=0).float().to(device)

        # transformer embeddings (CLS token)
        obs_emb  = self.encoder(**obs_enc )['last_hidden_state'][:, 0, :]
        desc_emb = self.encoder(**desc_enc)['last_hidden_state'][:, 0, :]

        # state feature
        state_feat = F.relu(self.fc_state(torch.cat([obs_emb, desc_emb, inv], dim=1)))
        # value prediction
        value = self.fc_value(state_feat)

        if actions is None:
            return state_feat, value

        # policy logits for each candidate action
        sizes        = [len(a) for a in actions]
        flat_actions = [act for sub in actions for act in sub]
        act_enc      = batch_encoded_tokens(flat_actions)
        act_emb      = self.encoder(**act_enc)['last_hidden_state'][:, 0, :]
        # repeat state features to match actions
        expanded = torch.cat([state_feat[i].repeat(n, 1) for i, n in enumerate(sizes)], dim=0)
        logits_flat = self.fc_policy(torch.cat([expanded, act_emb], dim=1)).squeeze(-1)
        return logits_flat.split(sizes), value

    @torch.no_grad()
    def act(self, states, valid_acts, lm=None, args=None):
        logits, values = self.forward(states, valid_acts)
        probs   = [F.softmax(l, dim=0) for l in logits]
        dists   = [Categorical(p)     for p in probs]
        actions = [d.sample()         for d in dists]
        logps   = [d.log_prob(a)     for d, a in zip(dists, actions)]
        return actions, logps, values

# ----------------------------------------------------------------------------
# PPO Transition tuple
# ----------------------------------------------------------------------------
PPOTransition = namedtuple('PPOTransition', (
    'state','action','log_prob','value','reward','done',
    'next_state','next_acts','advantage','returns'
))

# ----------------------------------------------------------------------------
# PPO Agent
# ----------------------------------------------------------------------------
class PPOAgent(BaseAgent):
    def __init__(self, args):
        super().__init__(args)
        self.net      = PPONetwork(args.lm_name, args.stats_dim, args.hidden_dim).to(device)
        self.old_net  = copy.deepcopy(self.net)
        self.tokenizer= AutoTokenizer.from_pretrained(args.lm_name)

        # PPO hyperparameters
        self.clip_eps = 0.2
        self.ent_coef = 0.01
        self.gamma    = args.gamma
        self.lmbda    = 0.95
        self.epochs   = 4
        self.num_envs = args.num_envs

        # optimizer over all trainable params (heads + unfrozen transformer layers)
        params = [p for p in self.net.parameters() if p.requires_grad]
        self.opt   = optim.Adam(params, lr=args.learning_rate)

        # buffers
        self.buf       = [[] for _ in range(self.num_envs)]
        self.completed = []
        self._last     = None

    def _tokenize(self, text):
        enc = self.tokenizer(clean(text),
                             return_tensors='pt',
                             padding=True,
                             truncation=True).to(device)
        return enc

    def act(self, states, valid_acts, lm=None, args=None):
        acts, logps, vals = self.old_net.act(states, valid_acts)
        # cache for observe()
        self._last = (valid_acts, acts, logps, vals)
        texts = [valid_acts[i][a.item()] for i, a in enumerate(acts)]
        idxs  = [a.item()                     for a in acts]
        return texts, idxs, vals

    def observe(self, transition, is_prior=False):
        valids, acts, logps, vals = self._last
        idx = getattr(transition.state, 'env_idx', 0)
        a, lp, v = acts[idx].item(), logps[idx].item(), vals[idx].item()
        self.buf[idx].append(PPOTransition(
            transition.state,
            a, lp, v,
            transition.reward,
            transition.done,
            transition.next_state,
            valids[idx],
            None, None
        ))
        # flush on episode end, prioritized flag, or max segment length
        if transition.done:# or is_prior or len(self.buf[idx]) >= 64:
            if all(t.action is not None for t in self.buf[idx]):
                self.completed.append(self.buf[idx])
            self.buf[idx] = []

    def _gae(self, ep):
        T    = len(ep)
        rews = np.array([t.reward for t in ep], dtype=np.float32)
        vals = np.array([t.value  for t in ep], dtype=np.float32)
        dones= np.array([t.done   for t in ep], dtype=bool)

        adv  = np.zeros(T, np.float32)
        ret  = np.zeros(T, np.float32)

        next_v = 0.0 if ep[-1].done else self.net([ep[-1].next_state], None)[1].item()
        gae    = 0.0
        for i in reversed(range(T)):
            mask = 0.0 if dones[i] else 1.0
            delta= rews[i] + self.gamma * next_v * mask - vals[i]
            gae   = delta + self.gamma * self.lmbda * gae * mask
            adv[i]= gae
            next_v= vals[i]

        ret[-1] = rews[-1]
        for i in reversed(range(T-1)):
            ret[i] = rews[i] + self.gamma * ret[i+1] * (0.0 if dones[i] else 1.0)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret = (ret - ret.mean()) / (ret.std() + 1e-8)

        for i in range(T):
            ep[i] = ep[i]._replace(advantage=adv[i], returns=ret[i])

    def update(self):
        """
        Wait until each parallel env has produced one completed episode,
        then update on that batch of episodes.
        Returns (avg_policy_loss, avg_value_loss, avg_entropy).
        """
        if len(self.completed) < self.num_envs:
            return (0.0, 0.0, 0.0)

        # grab exactly one episode per env
        episodes = self.completed[:self.num_envs]
        self.completed = self.completed[self.num_envs:]

        # compute GAE per episode
        for ep in episodes:
            self._gae(ep)

        # flatten transitions
        batch = [t for ep in episodes for t in ep]
        N     = len(batch)

        # prepare tensors
        actions = torch.tensor([t.action    for t in batch], device=device)
        old_lps  = torch.tensor([t.log_prob  for t in batch], device=device)
        advs     = torch.tensor([t.advantage for t in batch], device=device)
        rets     = torch.tensor([t.returns   for t in batch], device=device)
        states   = [t.state    for t in batch]
        valids   = [t.next_acts for t in batch]

        total_p, total_v, total_e = 0.0, 0.0, 0.0

        # PPO epochs on this single transitions‐batch
        for _ in range(self.epochs):
            logits, vals = self.net(states, valids)
            dists   = [Categorical(logits=l) for l in logits]
            new_lps = torch.stack([dists[i].log_prob(actions[i]) for i in range(N)])
            ent     = torch.stack([d.entropy() for d in dists]).mean()

            ratios = torch.exp(new_lps - old_lps)
            s1     = ratios * advs
            s2     = torch.clamp(ratios, 1-self.clip_eps, 1+self.clip_eps) * advs

            p_loss = -torch.min(s1, s2).mean()
            v_loss = F.mse_loss(vals.squeeze(-1), rets)

            total_p += p_loss.item()
            total_v += v_loss.item()
            total_e += ent.item()

            loss = p_loss + 0.5 * v_loss - self.ent_coef * ent
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.opt.step()

        # sync old policy and compute averages
        self.old_net.load_state_dict(self.net.state_dict())
        denom = float(N * self.epochs)
        avg_p = total_p / denom
        avg_v = total_v / denom
        avg_e = total_e / denom

        print(f"P/V/E losses: {avg_p:.6f}, {avg_v:.6f}, {avg_e:.6f}")
        return (avg_p, avg_v, avg_e)