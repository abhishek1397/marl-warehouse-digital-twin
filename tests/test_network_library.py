"""Comprehensive test suite for marl/networks PyTorch neural network library."""

import os
import shutil
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn

from gymnasium.spaces import Box, Dict as GymDict

from marl.networks import (
    ActorNetwork,
    BaseNetwork,
    CategoricalDistribution,
    CNNFeatureExtractor,
    CriticNetwork,
    FeatureExtractor,
    IndependentDistribution,
    MLP,
    NetworkFactory,
    NormalDistribution,
    PolicyNetwork,
    SharedActorCritic,
    ValueNetwork,
    compute_grad_norm,
    get_activation_fn,
    init_kaiming,
    init_orthogonal,
    init_weights,
    init_xavier,
    inspect_layers,
    to_device,
)


@pytest.fixture
def temp_dir():
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path, ignore_errors=True)


def test_base_network_functionality(temp_dir) -> None:
    mlp = MLP(input_dim=10, output_dim=2)
    assert mlp.device.type == "cpu"

    counts = mlp.count_parameters()
    assert counts["total"] > 0
    assert counts["trainable"] == counts["total"]

    stats = mlp.weight_statistics()
    assert len(stats) > 0
    assert "network.0.weight" in stats

    summary = mlp.get_summary()
    assert "MLP" in summary

    save_path = os.path.join(temp_dir, "mlp.pt")
    mlp.save(save_path)
    assert os.path.exists(save_path)

    mlp2 = MLP(input_dim=10, output_dim=2)
    mlp2.load(save_path)
    assert torch.allclose(mlp.network[0].weight, mlp2.network[0].weight)

    with pytest.raises(FileNotFoundError):
        mlp.load(os.path.join(temp_dir, "non_existent.pt"))


def test_mlp_activations_and_variations() -> None:
    for act in ["relu", "tanh", "elu", "gelu", "leaky_relu", "sigmoid", "identity", "none"]:
        fn = get_activation_fn(act)
        assert isinstance(fn, nn.Module)

    with pytest.raises(ValueError):
        get_activation_fn("invalid_act_name")

    # LayerNorm & Dropout
    mlp_ln = MLP(input_dim=16, output_dim=4, use_layer_norm=True, dropout=0.1)
    out_ln = mlp_ln(torch.randn(8, 16))
    assert out_ln.shape == (8, 4)

    # BatchNorm
    mlp_bn = MLP(input_dim=16, output_dim=4, use_batch_norm=True)
    mlp_bn.eval()
    out_bn = mlp_bn(torch.randn(8, 16))
    assert out_bn.shape == (8, 4)

    # Residual Connection
    mlp_res = MLP(input_dim=16, output_dim=16, hidden_dims=[16], use_residual=True)
    out_res = mlp_res(torch.randn(8, 16))
    assert out_res.shape == (8, 16)


def test_initializations() -> None:
    linear = nn.Linear(10, 5)
    init_orthogonal(linear)
    init_xavier(linear)
    init_kaiming(linear)

    for init_type in ["orthogonal", "xavier", "kaiming", "normal", "uniform"]:
        mod = nn.Linear(10, 5)
        init_weights(mod, init_type=init_type)


def test_cnn_feature_extractor() -> None:
    cnn = CNNFeatureExtractor(input_channels=1, output_dim=64)
    img_4d = torch.randn(4, 1, 20, 20)
    out_4d = cnn(img_4d)
    assert out_4d.shape == (4, 64)

    img_3d = torch.randn(4, 20, 20)
    out_3d = cnn(img_3d)
    assert out_3d.shape == (4, 64)


def test_feature_extractor() -> None:
    # Vector Box
    fe_vec = FeatureExtractor(observation_space=Box(low=0, high=1, shape=(32,)), output_dim=64)
    assert fe_vec(torch.randn(4, 32)).shape == (4, 64)
    assert fe_vec.output_dim == 64

    # 2D Box
    fe_2d = FeatureExtractor(observation_space=Box(low=0, high=1, shape=(20, 20)), output_dim=64)
    assert fe_2d(torch.randn(4, 20, 20)).shape == (4, 64)

    # GymDict Space
    dict_space = GymDict({
        "pos": Box(low=0, high=10, shape=(2,)),
        "battery": Box(low=0, high=100, shape=(1,)),
    })
    fe_dict = FeatureExtractor(observation_space=dict_space, output_dim=64)
    obs_sample = {"pos": torch.randn(4, 2), "battery": torch.randn(4, 1)}
    assert fe_dict(obs_sample).shape == (4, 64)

    with pytest.raises(ValueError):
        FeatureExtractor(observation_space="unsupported_space_type")

    with pytest.raises(TypeError):
        fe_vec([1, 2, 3])


def test_distributions() -> None:
    logits = torch.randn(4, 8)
    cat_dist = CategoricalDistribution(logits=logits)
    assert cat_dist.sample().shape == (4,)
    assert cat_dist.mode().shape == (4,)
    assert cat_dist.log_prob(torch.tensor([0, 1, 2, 3])).shape == (4,)
    assert cat_dist.entropy().shape == (4,)
    assert cat_dist.probs.shape == (4, 8)

    mean = torch.zeros(4, 3)
    std = torch.ones(4, 3)
    norm_dist = NormalDistribution(loc=mean, scale=std)
    n_sample = norm_dist.sample()
    assert n_sample.shape == (4, 3)
    assert norm_dist.mode().shape == (4, 3)
    assert norm_dist.log_prob(n_sample).shape == (4,)
    assert norm_dist.entropy().shape == (4,)

    indep = IndependentDistribution(norm_dist.dist)
    assert indep.sample().shape == (4, 3)
    assert indep.log_prob(n_sample).shape == (4,)
    assert indep.entropy().shape == (4,)


def test_actor_and_critic_networks() -> None:
    actor = ActorNetwork(observation_space=32, action_dim=8)
    obs = torch.randn(4, 32)

    dist = actor(obs)
    assert isinstance(dist, CategoricalDistribution)

    act_stoch, log_p_stoch = actor.sample_action(obs, deterministic=False)
    act_det, log_p_det = actor.sample_action(obs, deterministic=True)
    assert act_stoch.shape == (4,)

    # Critic V(s)
    critic_v = CriticNetwork(observation_space=32, action_dim=None)
    val_v = critic_v(obs)
    assert val_v.shape == (4, 1)

    # Critic Q(s, a)
    critic_q = CriticNetwork(observation_space=32, action_dim=8)
    val_q = critic_q(obs, action=act_det)
    assert val_q.shape == (4, 1)


def test_shared_actor_critic() -> None:
    shared = SharedActorCritic(observation_space=32, action_dim=8)
    obs = torch.randn(4, 32)

    dist, values = shared(obs)
    assert values.shape == (4, 1)

    actions = dist.sample()
    val_eval, log_p, ent = shared.evaluate_actions(obs, actions)
    assert val_eval.shape == (4, 1)
    assert log_p.shape == (4,)
    assert ent.shape == (4,)


def test_value_and_policy_networks() -> None:
    val_net = ValueNetwork(observation_space=32)
    obs = torch.randn(4, 32)
    assert val_net(obs).shape == (4, 1)

    policy = PolicyNetwork(observation_space=32, action_dim=8, use_shared_critic=False)
    act, log_p = policy.act(obs)
    pred = policy.predict(obs)
    assert act.shape == (4,)
    assert pred.shape == (4,)

    _, log_p_eval, ent_eval = policy.evaluate_actions(obs, act)
    assert log_p_eval.shape == (4,)

    policy_shared = PolicyNetwork(observation_space=32, action_dim=8, use_shared_critic=True)
    vals, log_p_s, ent_s = policy_shared.evaluate_actions(obs, act)
    assert vals.shape == (4, 1)


def test_network_factory() -> None:
    mlp = NetworkFactory.create("mlp", input_dim=10, output_dim=2)
    assert isinstance(mlp, MLP)

    actor = NetworkFactory.create("actor", observation_space=32, action_dim=8)
    assert isinstance(actor, ActorNetwork)

    critic = NetworkFactory.create("critic", observation_space=32)
    assert isinstance(critic, CriticNetwork)

    shared = NetworkFactory.create("shared_actor_critic", observation_space=32)
    assert isinstance(shared, SharedActorCritic)

    class CustomNet(BaseNetwork):
        def forward(self, x): return x

    NetworkFactory.register("custom", CustomNet)
    assert isinstance(NetworkFactory.create("custom"), CustomNet)

    with pytest.raises(KeyError):
        NetworkFactory.create("unknown_network")


def test_network_utilities() -> None:
    mlp = MLP(input_dim=10, output_dim=2)
    out = mlp(torch.randn(2, 10))
    loss = out.sum()
    loss.backward()

    norm = compute_grad_norm(mlp)
    assert norm >= 0.0

    layers = inspect_layers(mlp)
    assert len(layers) > 0

    t_dict = {"a": torch.randn(2, 2)}
    moved = to_device(t_dict, torch.device("cpu"))
    assert moved["a"].device.type == "cpu"
