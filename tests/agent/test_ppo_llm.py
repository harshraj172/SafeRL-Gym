"""
USAGE:
    PYTHONPATH=$(pwd) pytest tests/agent/test_ppo_llm.py
"""

import torch
import pytest
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from src.agent.ppo_llm import ActorNetwork, CriticNetwork, PPOLLMAgent
from src.env.machiavelli.machiavelli_env import MachiavelliEnv


@pytest.fixture(scope="module")
def gemma_resources():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it", device=device)
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-1b-it", attn_implementation="eager"
    ).to(device)
    config = AutoConfig.from_pretrained("google/gemma-3-1b-it")
    return device, tokenizer, model, config


@pytest.fixture(scope="module")
def machiavelli_env():
    return MachiavelliEnv(game="aegis-project-180-files")


def test_gemma_actornetwork_forward(gemma_resources):
    device, tokenizer, model, _ = gemma_resources
    actor = ActorNetwork(model, tokenizer)
    state = "Once upon a"
    action1 = " time, there was a king."
    action2 = " time, there was a queen. To be "
    actions = [action1, action2]
    batch_logprob = actor.forward(state, actions, device)
    assert isinstance(batch_logprob, torch.Tensor)
    assert batch_logprob.shape == (2,)
    assert torch.isfinite(batch_logprob).all()


def test_gemma_actornetwork_repr(gemma_resources):
    device, tokenizer, model, _ = gemma_resources
    actor = ActorNetwork(model, tokenizer)
    summary = str(actor)
    print(summary)
    assert "CausalLM" in summary or "model" in summary


def test_gemma_actornetwork_conditional_logprobs(gemma_resources):
    device, tokenizer, model, _ = gemma_resources
    actor = ActorNetwork(model, tokenizer)
    state = "Once upon a"
    action1 = " time, there was a king."
    action2 = " time, there was a princess."
    actions = [action1, action2]
    conditional_logprobs = actor.create_conditional_logprobs(state, actions, device)
    print(conditional_logprobs)
    assert isinstance(conditional_logprobs, torch.Tensor)
    assert conditional_logprobs.shape == (2,)
    assert torch.isfinite(conditional_logprobs).all()


def test_gemma_criticnetwork_repr(gemma_resources):
    _, tokenizer, model, config = gemma_resources
    input_dim = config.hidden_size  # Use the model's hidden size as input_dim
    critic = CriticNetwork(model, tokenizer, input_dim)
    summary = str(critic)
    print(summary)
    assert "CriticNetwork" in summary or "simple_nn" in summary


def test_gemma_criticnetwork_forward(gemma_resources):
    device, tokenizer, model, config = gemma_resources
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


def test_ppoagent_choose_action_machiavelli(gemma_resources, machiavelli_env):
    device, tokenizer, model, config = gemma_resources
    actor = ActorNetwork(model, tokenizer)
    critic = CriticNetwork(model, tokenizer, config.hidden_size)
    agent = PPOLLMAgent(machiavelli_env, actor, critic, device)
    state, _ = machiavelli_env.reset()
    action, logprob = agent.choose_action(state, logprob_flag=True)
    assert isinstance(action, (str, int))
    assert isinstance(logprob, float)


def test_ppoagent_collect_trajectories_machiavelli(gemma_resources, machiavelli_env):
    device, tokenizer, model, config = gemma_resources
    actor = ActorNetwork(model, tokenizer)
    critic = CriticNetwork(model, tokenizer, config.hidden_size)
    agent = PPOLLMAgent(machiavelli_env, actor, critic, device)
    batch_size = 3
    trajectories = agent.collect_trajectories(batch_size)
    assert len(trajectories) == batch_size
    states, actions, rewards, dones, next_states, logprobs, *_ = zip(*trajectories)
    assert all(isinstance(s, str) for s in states)
    assert all(isinstance(a, (str, int)) for a in actions)
    assert all(isinstance(r, (float, int)) for r in rewards)
    assert all(isinstance(d, bool) for d in dones)
    assert all(isinstance(ns, str) for ns in next_states)
    assert all(isinstance(lp, float) for lp in logprobs)


def test_ppoagent_compute_gaes_machiavelli(gemma_resources, machiavelli_env):
    device, tokenizer, model, config = gemma_resources
    actor = ActorNetwork(model, tokenizer)
    critic = CriticNetwork(model, tokenizer, config.hidden_size)
    agent = PPOLLMAgent(machiavelli_env, actor, critic, device)
    batch_size = 3
    trajectories = agent.collect_trajectories(batch_size)
    gaes, returns = agent.compute_gaes(trajectories, returns_flag=True)
    assert gaes.shape == (batch_size,)
    assert returns.shape == (batch_size,)


def test_ppoagent_update_machiavelli(gemma_resources, machiavelli_env):
    device, tokenizer, model, config = gemma_resources
    actor = ActorNetwork(model, tokenizer)
    critic = CriticNetwork(model, tokenizer, config.hidden_size)
    agent = PPOLLMAgent(machiavelli_env, actor, critic, device)
    batch_size = 3
    trajectories = agent.collect_trajectories(batch_size)
    loss = agent.update(trajectories)
    assert isinstance(loss, float)
