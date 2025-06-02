"""
USAGE:
    PYTHONPATH=. pytest tests/agent/test_ppo_llm.py
"""

import pytest
import copy
from unittest.mock import patch
from types import SimpleNamespace
import torch
from transformers import PreTrainedTokenizerBase, PretrainedConfig, PreTrainedModel
from src.env.machiavelli.machiavelli_env import MachiavelliEnv


class MockTokenizer(PreTrainedTokenizerBase):
    def __init__(self):
        super().__init__()
        self.vocab = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "Once": 3,
            "upon": 4,
            "a": 5,
            "time": 6,
            ",": 7,
            "there": 8,
            "was": 9,
            "king": 10,
            ".": 11,
            "queen": 12,
            "princess": 13,
        }
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2

    def __call__(
        self, texts, return_tensors=None, padding=False, padding_side="right", **kwargs
    ):
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize by splitting on whitespace and handling punctuation
        def tokenize(t):
            return t.replace(",", " ,").replace(".", " .").split()

        tokenized = [tokenize(t) for t in texts]
        max_len = max(len(toks) for toks in tokenized)
        input_ids = []
        attention_mask = []
        for toks in tokenized:
            # Use self.pad_token_id for unknown tokens instead of 99
            ids = [self.vocab.get(tok, self.pad_token_id) for tok in toks]
            if padding:
                pad_len = max_len - len(ids)
                if padding_side == "right":
                    ids = ids + [self.pad_token_id] * pad_len
                else:
                    ids = [self.pad_token_id] * pad_len + ids
            input_ids.append(ids)
            attention_mask.append([1 if i < len(toks) else 0 for i in range(max_len)])
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids):
        return " ".join(self.inv_vocab.get(i, "<unk>") for i in ids)

    def convert_ids_to_tokens(self, ids):
        # Accepts int or list of ints
        if isinstance(ids, int):
            return self.inv_vocab.get(ids, "<unk>")
        return [self.inv_vocab.get(i, "<unk>") for i in ids]

    def convert_tokens_to_ids(self, tokens):
        # Accepts str or list of str
        if isinstance(tokens, str):
            return self.vocab.get(tokens, 99)
        return [self.vocab.get(tok, 99) for tok in tokens]


class MockConfig(PretrainedConfig):
    def __init__(self):
        super().__init__()
        self.hidden_size = 8
        self.vocab_size = 20


class MockModel(PreTrainedModel):
    config_class = MockConfig

    def __init__(self, config=None):
        config = config or MockConfig()
        super().__init__(config)
        self.logits_bias = torch.nn.Parameter(torch.zeros(1))
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size

    def forward(
        self, input_ids=None, attention_mask=None, output_hidden_states=False, **kwargs
    ):
        batch, seq = input_ids.shape
        logits = (
            torch.randn(batch, seq, self.vocab_size, device=input_ids.device)
            + self.logits_bias
        )
        if output_hidden_states:
            hidden_states = [
                torch.randn(batch, seq, self.hidden_size, device=input_ids.device)
                for _ in range(13)
            ]
            return type(
                "MockOutput", (), {"logits": logits, "hidden_states": hidden_states}
            )()
        return type("MockOutput", (), {"logits": logits})()

    def to(self, device):
        return super().to(device)

    def generate(self, input_ids, max_length=20, **kwargs):
        # Generate random tokens for testing
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        gen_len = max_length - seq_len
        gen = torch.randint(
            0, self.vocab_size, (batch_size, gen_len), device=input_ids.device
        )
        return torch.cat([input_ids, gen], dim=1)


@pytest.fixture(scope="module")
def mock_resources():
    device = torch.device("cpu")
    tokenizer = MockTokenizer()
    config = MockConfig()
    model = MockModel(config)
    return device, tokenizer, model, config


@pytest.fixture(scope="module")
def machiavelli_env():
    return MachiavelliEnv(game="aegis-project-180-files")


def test_mock_actornetwork_forward(mock_resources):
    from src.agent.ppo_llm import ActorNetwork

    device, tokenizer, model, _ = mock_resources
    actor = ActorNetwork(model, tokenizer)
    state = "Once upon a"
    actions = ["time , there was a king .", "time , there was a queen ."]
    batch_logprob = actor.forward(state, actions, device)
    assert isinstance(batch_logprob, torch.Tensor)
    assert batch_logprob.shape == (2,)


def test_mock_criticnetwork_forward(mock_resources):
    from src.agent.ppo_llm import CriticNetwork

    device, tokenizer, model, config = mock_resources
    critic = CriticNetwork(model, tokenizer, config.hidden_size)
    state = "Once upon a time , there was a king ."
    output = critic(state, device)
    assert isinstance(output, torch.Tensor)


def test_mock_actornetwork_conditional_logprobs(mock_resources):
    from src.agent.ppo_llm import ActorNetwork

    device, tokenizer, model, _ = mock_resources
    actor = ActorNetwork(model, tokenizer)
    state = "Once upon a"
    actions = ["time , there was a king .", "time , there was a princess ."]
    conditional_logprobs = actor.create_conditional_logprobs(state, actions, device)
    assert isinstance(conditional_logprobs, torch.Tensor)
    assert conditional_logprobs.shape == (2,)
    summary = str(actor)
    print(summary)
    assert "CausalLM" in summary or "model" in summary


def test_gemma_actornetwork_conditional_logprobs(mock_resources):
    from src.agent.ppo_llm import ActorNetwork

    device, tokenizer, model, _ = mock_resources
    actor = ActorNetwork(model, tokenizer)
    state = "Once upon a"
    actions = ["time , there was a king .", "time , there was a princess ."]
    conditional_logprobs = actor.create_conditional_logprobs(state, actions, device)
    print(conditional_logprobs)
    assert isinstance(conditional_logprobs, torch.Tensor)
    assert conditional_logprobs.shape == (2,)
    assert torch.isfinite(conditional_logprobs).all()


def test_mock_criticnetwork_repr(mock_resources):
    from src.agent.ppo_llm import CriticNetwork

    _, tokenizer, model, config = mock_resources
    input_dim = config.hidden_size
    critic = CriticNetwork(model, tokenizer, input_dim)
    summary = str(critic)
    print(summary)
    assert "CriticNetwork" in summary or "simple_nn" in summary


def test_gemma_criticnetwork_forward(mock_resources):
    from src.agent.ppo_llm import CriticNetwork

    device, tokenizer, model, config = mock_resources
    input_dim = config.hidden_size
    critic = CriticNetwork(model, tokenizer, input_dim)
    state = "Once upon a time, there was a king."
    output = critic(state, device)
    print(output)
    assert (
        output.dim() == 0
        or output.shape == torch.Size([])
        or output.shape == torch.Size([1])
    )


def test_ppoagent_choose_action_machiavelli(mock_resources, machiavelli_env):
    from src.agent.ppo_llm import PPOLLMAgent

    device, tokenizer, model, config = mock_resources
    model2 = copy.deepcopy(model)
    args = SimpleNamespace(
        actor_model=model,
        critic_model=model2,
        tokenizer=tokenizer,
        env=machiavelli_env,
        num_envs=1,
    )
    with (
        patch("src.agent.ppo_llm.AutoModel.from_pretrained", return_value=model),
        patch(
            "src.agent.ppo_llm.AutoTokenizer.from_pretrained", return_value=tokenizer
        ),
    ):
        agent = PPOLLMAgent(args)
        state, _ = machiavelli_env.reset()
        action, logprob = agent._choose_action(state, logprob_flag=True)
        assert isinstance(action, (str, int))
        assert isinstance(logprob, float)


def test_ppoagent_collect_trajectories_machiavelli(mock_resources, machiavelli_env):
    from src.agent.ppo_llm import PPOLLMAgent

    device, tokenizer, model, config = mock_resources
    model2 = copy.deepcopy(model)
    args = SimpleNamespace(
        actor_model=model,
        critic_model=model2,
        tokenizer=tokenizer,
        env=machiavelli_env,
        num_envs=1,
    )
    with (
        patch("src.agent.ppo_llm.AutoModel.from_pretrained", return_value=model),
        patch(
            "src.agent.ppo_llm.AutoTokenizer.from_pretrained", return_value=tokenizer
        ),
    ):
        agent = PPOLLMAgent(args)
        batch_size = 3
        trajectories = agent._collect_trajectories(batch_size)
        assert len(trajectories) == batch_size
        states, actions, rewards, dones, next_states, logprobs, *_ = zip(*trajectories)
        assert all(isinstance(s, str) for s in states)
        assert all(isinstance(a, (str, int)) for a in actions)
        assert all(isinstance(r, (float, int)) for r in rewards)
        assert all(isinstance(d, bool) for d in dones)
        assert all(isinstance(ns, str) for ns in next_states)
        assert all(isinstance(lp, float) for lp in logprobs)


def test_ppollmagent_act(mock_resources, machiavelli_env):
    from src.agent.ppo_llm import PPOLLMAgent

    device, tokenizer, model, config = mock_resources
    model2 = copy.deepcopy(model)
    args = SimpleNamespace(
        actor_model=model,
        critic_model=model2,
        tokenizer=tokenizer,
        env=machiavelli_env,
        num_envs=1,
    )
    with (
        patch("src.agent.ppo_llm.AutoModel.from_pretrained", return_value=model),
        patch(
            "src.agent.ppo_llm.AutoTokenizer.from_pretrained", return_value=tokenizer
        ),
    ):
        agent = PPOLLMAgent(args)
        state, _ = machiavelli_env.reset()
        poss_acts = [machiavelli_env._get_info()["game_state"]["choice_texts"]]
        actions, action_idxs, logprobs = agent.act(
            [state], poss_acts, logprob_flag=True
        )
        assert isinstance(actions[0], (str, int))
        assert isinstance(logprobs[0], float)


def test_ppollmagent_observe(mock_resources, machiavelli_env):
    from src.agent.ppo_llm import PPOLLMAgent

    device, tokenizer, model, config = mock_resources
    model2 = copy.deepcopy(model)
    args = SimpleNamespace(
        actor_model=model,
        critic_model=model2,
        tokenizer=tokenizer,
        env=machiavelli_env,
        num_envs=1,
    )
    with (
        patch("src.agent.ppo_llm.AutoModel.from_pretrained", return_value=model),
        patch(
            "src.agent.ppo_llm.AutoTokenizer.from_pretrained", return_value=tokenizer
        ),
    ):
        agent = PPOLLMAgent(args)
        state, _ = machiavelli_env.reset()
        next_state, reward, done, _ = machiavelli_env.step(0)
        transition = SimpleNamespace(
            state=state,
            reward=reward,
            done=done,
            next_state=next_state,
            next_acts=None,
            cost=0,
        )
        agent.observe(transition)
        buffer = agent.episode_buffer[0]
        assert len(buffer) == 1
        t = buffer[0]
        assert isinstance(t.state, str)
        assert isinstance(t.reward, (float, int))
        assert isinstance(t.done, bool)
        assert isinstance(t.next_state, str)


def test_ppollmagent_update(mock_resources, machiavelli_env):
    from src.agent.ppo_llm import PPOLLMAgent

    device, tokenizer, model, config = mock_resources
    model2 = copy.deepcopy(model)
    args = SimpleNamespace(
        actor_model=model,
        critic_model=model2,
        tokenizer=tokenizer,
        env=machiavelli_env,
        num_envs=1,
    )
    with (
        patch("src.agent.ppo_llm.AutoModel.from_pretrained", return_value=model),
        patch(
            "src.agent.ppo_llm.AutoTokenizer.from_pretrained", return_value=tokenizer
        ),
    ):
        agent = PPOLLMAgent(args)
        state, _ = machiavelli_env.reset()
        poss_acts = [machiavelli_env._get_info()["game_state"]["choice_texts"]]
        agent.act([state], poss_acts)
        next_state, reward, done, _ = machiavelli_env.step(0)
        transition = SimpleNamespace(
            state=state,
            reward=reward,
            done=done,
            next_state=next_state,
            next_acts=None,
            cost=0,
        )
        agent.observe(transition)
        loss = agent.update()
        assert isinstance(loss, float)
