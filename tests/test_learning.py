from types import SimpleNamespace

import pytest
import torch

import astrocites.learning as learning
from astrocites.learning import WeightDependentPostPre


def _make_rule_connection(weights=None, wmin=0.1, wmax=0.8):
    if weights is None:
        weights = torch.full((2, 2), 0.5)
    source = SimpleNamespace(
        batch_size=1,
        s=torch.tensor([[1, 0]], dtype=torch.bool),
        x=torch.tensor([[1.0, 0.0]]),
    )
    target = SimpleNamespace(
        s=torch.tensor([[0, 1]], dtype=torch.bool),
        x_neg=torch.tensor([[0.0, 1.0]]),
    )
    connection = SimpleNamespace(
        w=torch.nn.Parameter(weights.clone(), requires_grad=False),
        source=source,
        target=target,
        wmin=wmin,
        wmax=wmax,
    )
    return connection


def test_stdp_lookup_uses_last_valid_column_and_maps_out_of_range_to_zero():
    connection = _make_rule_connection()
    rule = WeightDependentPostPre(connection, nu=[10, 10])
    weight = torch.tensor([0.5, 0.5, 0.5])
    deltas = torch.tensor([59.0, 60.0, -61.0])

    values = rule._stdp_lookup(weight, deltas)

    assert values[0] == rule.STDP_base[5, 119]
    assert values[1] == rule.STDP_base[5, 0]
    assert values[2] == rule.STDP_base[5, 0]
    assert rule.delta_w_custom_single(torch.tensor(0.5), torch.tensor(59.0)) == rule.STDP_base[5, 119]


def test_every_stdp_update_clamps_weights_to_configured_bounds():
    connection = _make_rule_connection()
    rule = WeightDependentPostPre(connection, nu=[10, 10])
    rule.STDP_base = rule.STDP_base.clone()
    rule.STDP_base.fill_(100.0)

    rule.update(current_position=0, adjacent_positions=[1])
    assert connection.w.min() >= connection.wmin
    assert connection.w.max() <= connection.wmax
    assert connection.w[0, 1] == connection.wmax

    rule.STDP_base = rule.STDP_base.clone()
    rule.STDP_base.fill_(-100.0)
    rule.update(current_position=0, adjacent_positions=[1])
    assert connection.w.min() >= connection.wmin
    assert connection.w.max() <= connection.wmax
    assert connection.w[0, 1] == connection.wmin


def test_stdp_clamp_preserves_structurally_forbidden_zero_weights():
    connection = _make_rule_connection(torch.tensor([[0.5, 0.0], [0.0, 0.5]]))
    connection.structural_mask = torch.tensor([[True, False], [False, True]])
    rule = WeightDependentPostPre(connection, nu=[10, 10])
    rule.STDP_base = rule.STDP_base.clone()
    rule.STDP_base.fill_(100.0)

    rule.update(current_position=0, adjacent_positions=[1])

    assert connection.w[0, 1] == 0
    assert connection.w[1, 0] == 0
    allowed = connection.w[connection.structural_mask]
    assert allowed.min() >= connection.wmin
    assert allowed.max() <= connection.wmax


def test_missing_stdp_table_is_a_hard_error(monkeypatch, tmp_path):
    missing_path = tmp_path / "missing-STDP.txt"
    monkeypatch.setattr(
        learning,
        "os",
        SimpleNamespace(
            path=SimpleNamespace(
                dirname=lambda _path: str(tmp_path),
                join=lambda *_parts: str(missing_path),
            )
        ),
    )

    with pytest.raises(FileNotFoundError, match="Required STDP lookup table not found"):
        WeightDependentPostPre(_make_rule_connection(), nu=[10, 10])


def test_main_reference_rule_skips_post_stdp_clamp_and_mask():
    connection = _make_rule_connection()
    connection.enforce_post_stdp_bounds = False
    connection.apply_structural_mask_during_stdp = False
    connection.main_reference_stdp = True
    connection.structural_mask = torch.zeros((2, 2), dtype=torch.bool)
    rule = WeightDependentPostPre(connection, nu=[10, 10])
    rule.STDP_base = rule.STDP_base.clone()
    rule.STDP_base.fill_(1.0)
    rule.update(current_position=0, adjacent_positions=[1])
    assert connection.w[0, 1] > connection.wmax
    assert connection.w[0, 1] > 0


def test_generic_batch_stdp_path_uses_vectorized_lookup():
    connection = _make_rule_connection()
    rule = WeightDependentPostPre(connection, nu=[10, 10])
    rule.STDP_base = rule.STDP_base.clone()
    rule.STDP_base.fill_(0.05)
    before = connection.w.detach().clone()
    rule.update()
    assert connection.last_stdp_update_raw.shape == connection.w.shape
    assert not torch.equal(connection.w, before)


def test_stdp_table_is_loaded_once_per_process(monkeypatch):
    learning._load_stdp_table.cache_clear()
    loadtxt = learning.np.loadtxt
    calls = []
    def counted_loadtxt(path):
        calls.append(path)
        return loadtxt(path)
    monkeypatch.setattr(learning.np, "loadtxt", counted_loadtxt)
    try:
        WeightDependentPostPre(_make_rule_connection(), nu=[10, 10])
        WeightDependentPostPre(_make_rule_connection(), nu=[10, 10])
        assert len(calls) == 1
    finally:
        learning._load_stdp_table.cache_clear()
