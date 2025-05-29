"""
    USAGE:
        PYTHONPATH=$(pwd) pytest tests/agent/test_ppo_llm.py
"""

import torch
import pytest
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from src.agent.ppo_llm import ActorNetwork, CriticNetwork, PPOAgent
from src.env.machiavelli.machiavelli_env import MachiavelliEnv



@pytest.fixture(scope="module")
def gemma_resources():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it", device=device)
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", attn_implementation='eager').to(device)
    config = AutoConfig.from_pretrained("google/gemma-3-1b-it")
    return device, tokenizer, model, config

@pytest.fixture(scope="module")
def machiavelli_env():
    # Use a real game name from your metadata
    env = MachiavelliEnv(game="aegis-project-180-files")
    return env

def test_gemma_actornetwork_forward(gemma_resources):
    device, tokenizer, model, _ = gemma_resources
    actor = ActorNetwork(model, tokenizer)
    state = "Once upon a"
    action1 = " time, there was a king."
    action2 = " time, there was a queen. To be "
    actions = [action1, action2]
    batch_logprob = actor.forward(state, actions, device)
    print(batch_logprob)
    torch.testing.assert_close(batch_logprob, torch.tensor([ -42.2574, -149.5201], device=device), rtol=1e-3, atol=1e-3)


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
    torch.testing.assert_close(conditional_logprobs, torch.tensor([-0.1708, -1.8513], device=device), rtol=1e-3, atol=1e-3)


def test_gemma_criticnetwork_repr(gemma_resources):
    device, tokenizer, model, config = gemma_resources
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
    assert output.dim() == 0 or output.shape == torch.Size([]) or output.shape == torch.Size([1])
    assert 0.0 <= output.item() <= 1.0  # Output should be in [0, 1] due to sigmoid


def test_ppoagent_choose_action_machiavelli(gemma_resources, machiavelli_env):
    device, tokenizer, model, config = gemma_resources
    actor = ActorNetwork(model, tokenizer)
    critic = CriticNetwork(model, tokenizer, config.hidden_size)
    agent = PPOAgent(machiavelli_env, actor, critic, device)
    state, _ = machiavelli_env.reset()
    action, logprob = agent.choose_action(state, logprob_flag=True)
    assert isinstance(action, str) or isinstance(action, int)
    assert isinstance(logprob, float)

def test_ppoagent_collect_trajectories_machiavelli(gemma_resources, machiavelli_env):
    device, tokenizer, model, config = gemma_resources
    actor = ActorNetwork(model, tokenizer)
    critic = CriticNetwork(model, tokenizer, config.hidden_size)
    agent = PPOAgent(machiavelli_env, actor, critic, device)
    batch_size = 3
    trajectories = agent.collect_trajectories(batch_size)
    assert len(trajectories) == batch_size
    for t in trajectories:
        state, action, reward, done, next_state, logprob, *_ = t
        assert isinstance(state, str)
        assert isinstance(action, str) or isinstance(action, int)
        assert isinstance(reward, float) or isinstance(reward, int)
        assert isinstance(done, bool)
        assert isinstance(next_state, str)
        assert isinstance(logprob, float)

def test_ppoagent_compute_gaes_machiavelli(gemma_resources, machiavelli_env):
    device, tokenizer, model, config = gemma_resources
    actor = ActorNetwork(model, tokenizer)
    critic = CriticNetwork(model, tokenizer, config.hidden_size)
    agent = PPOAgent(machiavelli_env, actor, critic, device)
    batch_size = 3
    trajectories = agent.collect_trajectories(batch_size)
    gaes, returns = agent.compute_gaes(trajectories, returns_flag=True)
    assert gaes.shape == (batch_size,)
    assert returns.shape == (batch_size,)

def test_ppoagent_update_machiavelli(gemma_resources, machiavelli_env):
    device, tokenizer, model, config = gemma_resources
    actor = ActorNetwork(model, tokenizer)
    critic = CriticNetwork(model, tokenizer, config.hidden_size)
    agent = PPOAgent(machiavelli_env, actor, critic, device)
    batch_size = 3
    trajectories = agent.collect_trajectories(batch_size)
    loss = agent.update(trajectories)
    assert isinstance(loss, float)