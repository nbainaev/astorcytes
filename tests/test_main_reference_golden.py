"""Golden micro-tests against the authors' executable notebook in main.

The notebook is loaded from the pinned main commit without executing notebook
cells at top level. Only class definitions needed for the model are executed.
"""
import ast
import json
import subprocess

import numpy as np
import pytest
import torch

from astrocites.connection import Connection as PlausibleConnection
from astrocites.learning import NoOp as PlausibleNoOp
from astrocites.learning import WeightDependentPostPre as PlausibleSTDP
from astrocites.nodes import Input as PlausibleInput
from astrocites.nodes import LIFNodes as PlausibleLIF
from astrocites.nodes import Nodes as PlausibleNodes

MAIN_NOTEBOOK_COMMIT = "f31a8ec8925f4dfa60e7b04f65a879cfbf2d4bef"
MAIN_CLASS_NAMES = {
    "Nodes", "Input", "LIFNodes", "LearningRule", "WeightDependentPostPre",
    "NoOp", "Connection", "NetworkMonitor", "Network",
}


@pytest.fixture(scope="module")
def main_reference():
    notebook = subprocess.check_output(
        ["git", "show", f"{MAIN_NOTEBOOK_COMMIT}:main.ipynb"], text=True
    )
    source = "".join(json.loads(notebook)["cells"][0]["source"])
    tree = ast.parse(source)
    selected = [
        node for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        or isinstance(node, ast.ClassDef) and node.name in MAIN_CLASS_NAMES
    ]
    namespace = {}
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, "main.ipynb:reference-classes", "exec"), namespace)
    return namespace


def _connection_pair(main_reference, *, weight=None, invert=True):
    main_nodes = main_reference["Nodes"]
    main_connection = main_reference["Connection"]
    main_noop = main_reference["NoOp"]
    main_source = main_nodes(n=2, traces=True)
    main_target = main_nodes(n=2, traces=True)
    source = PlausibleNodes(n=2, traces=True)
    target = PlausibleNodes(n=2, traces=True)
    if weight is None:
        weight = torch.tensor([[0.0, 10.0], [0.0, 0.0]])
    common = dict(
        impulse_amplitude=0.5, impulse_amplitude_2=0.5,
        impulse_length=40, impulse_shape_factor=0.9, invert=invert,
        w=weight.clone(), wmin=0.0, wmax=20.0,
    )
    reference = main_connection(
        main_source, main_target, update_rule=main_noop, **common
    )
    plausible = PlausibleConnection(
        source, target, update_rule=PlausibleNoOp, **common
    )
    return main_source, main_target, reference, source, target, plausible


def _connection_trace(main_reference, event_times, duration=110):
    main_source, _, reference, source, _, plausible = _connection_pair(main_reference)
    reference_rows = []
    plausible_rows = []
    for tick in range(duration):
        event = tick in event_times
        main_source.s.zero_()
        source.s.zero_()
        main_source.s[0, 0] = event
        source.s[0, 0] = event
        y_ref = reference.compute(main_source.s)
        y_new = plausible.compute(source.s)
        reference_rows.append(torch.cat((
            y_ref.reshape(-1), reference.a_pre.clone(),
            reference.impulse_state.clone(),
        )))
        plausible_rows.append(torch.cat((
            y_new.reshape(-1), plausible.a_pre.clone(),
            plausible.impulse_state.clone(),
        )))
    return torch.stack(reference_rows), torch.stack(plausible_rows)


def test_input_refractory_matches_notebook_static_and_spike_semantics(main_reference):
    reference = main_reference["Input"](n=1, traces=True, thresh=7, rest=0, reset=0, refrac=40)
    plausible = PlausibleInput(n=1, traces=True, thresh=7, rest=0, reset=0, refrac=40)
    ref_times, new_times = [], []
    for tick in range(1000):
        raw = torch.tensor([[0.015]])
        reference.forward(raw)
        plausible.forward(raw)
        if reference.s[0, 0]:
            ref_times.append(tick)
        if plausible.s[0, 0]:
            new_times.append(tick)
        assert torch.equal(reference.s, plausible.s)
        assert torch.equal(reference.refrac_count, plausible.refrac_count)
        assert torch.allclose(reference.x, plausible.x)
    expected = list(range(0, 1000, 40))
    assert ref_times == expected
    assert new_times == expected

    # A prerecorded Bernoulli event train is already spikes; the input layer
    # applies the same refractory gate in both implementations.
    events = {0, 1, 39, 40, 80}
    reference.reset_()
    plausible.reset_()
    ref_times, new_times = [], []
    for tick in range(100):
        raw = torch.tensor([[float(tick in events)]])
        reference.forward(raw)
        plausible.forward(raw)
        if reference.s[0, 0]:
            ref_times.append(tick)
        if plausible.s[0, 0]:
            new_times.append(tick)
        assert torch.equal(reference.s, plausible.s)
    assert ref_times == new_times == [0, 40, 80]


@pytest.mark.parametrize("separation", [1, 10, 39, 40, 41, 100])
def test_synaptic_impulse_gate_matches_notebook_for_two_spikes(main_reference, separation):
    both_ref, both_new = _connection_trace(main_reference, {0, separation})
    one_ref, one_new = _connection_trace(main_reference, {0})
    assert torch.allclose(both_ref, both_new, atol=1e-7, rtol=0)
    if separation < 40:
        # The second presynaptic event is suppressed while the 40-step impulse
        # state is active; it does not launch an overlapping kernel.
        assert torch.allclose(both_ref, one_ref, atol=1e-7, rtol=0)
        assert torch.allclose(both_new, one_new, atol=1e-7, rtol=0)
    else:
        # At and after the kernel boundary a new impulse is accepted.
        assert both_ref[separation, -2] > 0
        assert both_new[separation, -2] > 0


def test_single_spike_current_and_lif_trace_match_notebook(main_reference):
    main_source, main_target, main_conn, source, target, plausible_conn = _connection_pair(
        main_reference, weight=torch.tensor([[0.0, 10.0], [0.0, 0.0]]),
    )
    reference_lif = main_reference["LIFNodes"](
        n=2, traces=True, thresh=7 * torch.ones(2), rest=0, reset=0, refrac=40, dt=1,
    )
    plausible_lif = PlausibleLIF(
        n=2, traces=True, thresh=7 * torch.ones(2), rest=0, reset=0, refrac=40, dt=1,
    )
    reference_lif.compute_decays(1)
    plausible_lif.compute_decays(1)
    for tick in range(50):
        event = tick == 0
        main_source.s.zero_()
        source.s.zero_()
        main_source.s[0, 0] = event
        source.s[0, 0] = event
        current_ref = main_conn.compute(main_source.s)
        current_new = plausible_conn.compute(source.s)
        reference_lif.forward(current_ref)
        plausible_lif.forward(current_new)
        assert torch.allclose(current_ref, current_new, atol=1e-7, rtol=0)
        assert torch.allclose(reference_lif.v, plausible_lif.v, atol=1e-7, rtol=0)
        assert torch.equal(reference_lif.s, plausible_lif.s)
        assert torch.allclose(reference_lif.x, plausible_lif.x, atol=1e-7, rtol=0)
        assert torch.allclose(reference_lif.refrac_count, plausible_lif.refrac_count)


def test_astrocyte_g_ca_threshold_and_reset_match_notebook(main_reference):
    reference = main_reference["LIFNodes"](
        n=2, traces=True, thresh=7 * torch.ones(2), rest=0, reset=0, refrac=40, dt=1,
        enable_astrocyte=True, alpha=0.001, k=0.2,
    )
    plausible = PlausibleLIF(
        n=2, traces=True, thresh=7 * torch.ones(2), rest=0, reset=0, refrac=40, dt=1,
        enable_astrocyte=True, alpha=0.001, k=0.2,
    )
    reference.compute_decays(1)
    plausible.compute_decays(1)
    for tick in range(12):
        prev = torch.tensor([[float(tick in {0, 1}), 0.0]])
        reference.prev_layer_s = prev.clone()
        plausible.prev_layer_s = prev.clone()
        zero = torch.zeros(1, 2)
        reference.forward(zero)
        plausible.forward(zero)
        assert torch.allclose(reference.G, plausible.G, atol=1e-7, rtol=0)
        assert torch.equal(reference.Ca, plausible.Ca)
        assert torch.allclose(reference.thresh, plausible.thresh, atol=1e-7, rtol=0)
        assert torch.equal(reference.s, plausible.s)
    assert reference.Ca[0] > 0
    reference.reset_()
    plausible.reset_()
    assert torch.equal(reference.G, plausible.G)
    assert torch.equal(reference.Ca, plausible.Ca)
    assert reference.Ca[0] > 0  # Ca persists through an action/network reset.


def test_stdp_lookup_matches_valid_notebook_offsets_and_fixes_column_120(main_reference):
    main_source, main_target, _, source, target, _ = _connection_pair(
        main_reference, weight=torch.full((2, 2), 0.5), invert=False,
    )
    ref_rule = main_reference["WeightDependentPostPre"](
        main_reference["Connection"](
            main_source, main_target,
            update_rule=main_reference["NoOp"], w=torch.full((2, 2), 0.5),
            nu=[10, 10], wmin=0.0, wmax=2.0,
        ),
        nu=[10, 10],
    )
    new_rule = PlausibleSTDP(
        PlausibleConnection(
            source, target, update_rule=PlausibleNoOp,
            w=torch.full((2, 2), 0.5), nu=[10, 10], wmin=0.0, wmax=2.0,
        ),
        nu=[10, 10],
    )
    for delta in (-60, -20, -1, 1, 20, 59):
        old_value = ref_rule.delta_w_custom_single(torch.tensor(0.5), torch.tensor(float(delta)))
        new_value = new_rule.delta_w_custom_single(torch.tensor(0.5), torch.tensor(float(delta)))
        assert torch.equal(old_value, new_value)
    # Notebook admits second_index==120, then indexes past its [101,120] table.
    with pytest.raises(IndexError):
        ref_rule.delta_w_custom_single(torch.tensor(0.5), torch.tensor(60.0))
    assert torch.equal(
        new_rule.delta_w_custom_single(torch.tensor(0.5), torch.tensor(60.0)),
        new_rule.STDP_base[5, 0],
    )


def _build_comparison_network(main_reference, reference, astro=True):
    if reference:
        Input = main_reference["Input"]
        LIF = main_reference["LIFNodes"]
        ConnectionClass = main_reference["Connection"]
        WeightRule = main_reference["WeightDependentPostPre"]
        NoOpRule = main_reference["NoOp"]
        NetworkClass = main_reference["Network"]
        MonitorClass = main_reference["NetworkMonitor"]
    else:
        from astrocites.connection import Connection as ConnectionClass
        from astrocites.learning import NoOp as NoOpRule
        from astrocites.learning import WeightDependentPostPre as WeightRule
        from astrocites.network import Network as NetworkClass, NetworkMonitor as MonitorClass
        from astrocites.nodes import Input, LIFNodes as LIF
    count = 9
    x = Input(n=count, traces=True, thresh=7, rest=0, reset=0, refrac=40)
    y = LIF(n=count, traces=True, thresh=7 * torch.ones(count), rest=0,
            reset=0, refrac=40, dt=1)
    inh = LIF(n=count, traces=True, thresh=7 * torch.ones(count), rest=0,
              reset=0, refrac=40, dt=1, enable_astrocyte=astro,
              alpha=0.001, k=0.2)
    weights = torch.full((count, count), 0.001)
    for source in range(count):
        for target in range(count):
            if abs(source // 3 - target // 3) + abs(source % 3 - target % 3) == 1:
                weights[source, target] = 0.65
    identity = torch.eye(count)
    xy = ConnectionClass(
        x, y, impulse_amplitude=0.5, impulse_amplitude_2=0.5,
        impulse_length=40, impulse_shape_factor=0.9, invert=True,
        update_rule=WeightRule, w=weights.clone(), nu=[10, 10],
        wmin=0.001, wmax=1.0, weight_decay=0, post_spike_weight_decay=0.005,
    )
    xi = ConnectionClass(
        x, inh, impulse_amplitude=0.5, impulse_amplitude_2=0.5,
        impulse_length=40, impulse_shape_factor=0.9, invert=True,
        update_rule=NoOpRule, w=identity.clone(), nu=[1, 1],
        wmin=-100, wmax=1, weight_decay=0,
    )
    iy = ConnectionClass(
        inh, y, impulse_amplitude=0.5, impulse_amplitude_2=0.5,
        impulse_length=40, impulse_shape_factor=0.9, invert=True,
        update_rule=NoOpRule, w=-identity.clone(), nu=[1, 1],
        wmin=-100, wmax=1, weight_decay=0,
    )
    net = NetworkClass(dt=1)
    net.add_layer(x, "X")
    net.add_layer(y, "Y")
    net.add_layer(inh, "I")
    net.add_connection(xy, "X", "Y")
    net.add_connection(xi, "X", "I")
    net.add_connection(iy, "I", "Y")
    monitor = MonitorClass(
        net, state_vars=("v", "s", "G", "Ca", "x", "x_neg", "a_pre", "impulse_state")
    )
    net.add_monitor(monitor, "Network")
    return net, monitor, xy


def test_full_one_window_network_trace_and_action_reset_match_notebook(main_reference):
    reference, reference_monitor, reference_xy = _build_comparison_network(
        main_reference, reference=True, astro=True,
    )
    plausible, plausible_monitor, plausible_xy = _build_comparison_network(
        main_reference, reference=False, astro=True,
    )
    static_input = torch.zeros(1, 9)
    static_input[0, 4] = 0.015
    kwargs = dict(
        inpts={"X": static_input}, time=250,
        injects_v={"I": torch.full((9,), 0.02)},
        current_position=4, adjacent_positions=[1, 3, 5, 7],
        conn_XY=reference_xy,
    )
    reference.run(**kwargs)
    plausible.run(**{**kwargs, "conn_XY": plausible_xy, "enable_stdp": True})
    old = reference_monitor.get()
    new = plausible_monitor.get()
    assert old.keys() == new.keys()
    for layer_name in old:
        assert old[layer_name].keys() == new[layer_name].keys()
        for variable in old[layer_name]:
            assert torch.allclose(
                old[layer_name][variable].float(),
                new[layer_name][variable].float(),
                atol=1e-6, rtol=0,
            ), (layer_name, variable)
    assert reference.layers["I"].Ca.max() > 0
    reference.reset_()
    plausible.reset_()
    for layer_name in ("X", "Y", "I"):
        for field in ("v", "refrac_count", "x", "x_neg", "G", "Ca"):
            if hasattr(reference.layers[layer_name], field):
                assert torch.allclose(
                    getattr(reference.layers[layer_name], field).float(),
                    getattr(plausible.layers[layer_name], field).float(),
                ), (layer_name, field)
    assert torch.equal(reference.layers["I"].G, torch.zeros_like(reference.layers["I"].G))
    assert reference.layers["I"].Ca.max() > 0
    assert torch.equal(reference_xy.impulse_state, torch.zeros_like(reference_xy.impulse_state))
    assert torch.equal(plausible_xy.impulse_state, torch.zeros_like(plausible_xy.impulse_state))


def _stdp_pair_result(main_reference, delta, reference):
    if reference:
        Nodes = main_reference["Nodes"]
        ConnectionClass = main_reference["Connection"]
        Rule = main_reference["WeightDependentPostPre"]
    else:
        from astrocites.connection import Connection as ConnectionClass
        from astrocites.learning import WeightDependentPostPre as Rule
        from astrocites.nodes import Nodes
    source, target = Nodes(n=2, traces=True), Nodes(n=2, traces=True)
    weight = torch.full((2, 2), 0.5)
    conn = ConnectionClass(
        source, target, update_rule=Rule, w=weight.clone(), nu=[10, 10],
        wmin=0.0, wmax=2.0, post_spike_weight_decay=0.005,
    )
    if not reference:
        conn.main_reference_stdp = True
    if delta < 0:
        source.s[0, 0] = True
        target.x_neg[0, 1] = torch.exp(torch.tensor(float(delta) / 20.0))
    else:
        source.x[0, 0] = torch.exp(torch.tensor(-float(delta) / 20.0))
        target.s[0, 1] = True
    conn.update(current_position=0, adjacent_positions=[1])
    return conn.w.detach().clone()


@pytest.mark.parametrize("delta", [-60, -20, -1, 1, 20, 59])
def test_stdp_pre_post_pair_update_matches_notebook_for_valid_offsets(main_reference, delta):
    old = _stdp_pair_result(main_reference, delta, reference=True)
    new = _stdp_pair_result(main_reference, delta, reference=False)
    assert torch.allclose(old, new, atol=1e-7, rtol=0)


def test_main_reference_selected_action_stdp_gain_is_exact():
    from astrocites.experiment import _apply_selected_action_stdp_gain

    source, target = PlausibleNodes(n=2), PlausibleNodes(n=2)
    conn = PlausibleConnection(
        source, target, update_rule=PlausibleNoOp,
        w=torch.full((2, 2), 0.5), wmin=0, wmax=2,
    )
    before = conn.w.detach().clone()
    conn.w.data[0, 1] += 0.2
    conn.w.data[0, 0] += 0.1
    _apply_selected_action_stdp_gain(conn, before, 0, 1, 1.5)
    assert torch.allclose(conn.w[0, 1], torch.tensor(0.8))
    assert torch.allclose(conn.w[0, 0], torch.tensor(0.6))


def test_action_selection_matches_notebook_argmax_mask_and_tie_rule():
    from astrocites.experiment import _select_action_from_counts

    mask = np.array([1, 1, 0, 0])
    action, ties = _select_action_from_counts(
        [10, 5, 0, 0], mask, current_position=3,
        rng=np.random.RandomState(0),
    )
    assert action == 0
    assert ties == 1
    action, ties = _select_action_from_counts(
        [5, 10, 0, 0], mask, current_position=3,
        rng=np.random.RandomState(0),
    )
    assert action == 1
    assert ties == 1
    for counts in ([7, 7, 0, 0], [0, 0, 0, 0]):
        expected = np.random.RandomState(42).choice([0, 1])
        action, ties = _select_action_from_counts(
            counts, mask, current_position=3,
            rng=np.random.RandomState(42),
        )
        assert action == expected
        assert ties == 2
    action, ties = _select_action_from_counts(
        [9, 9, 0, 0], mask, current_position=0,
        rng=np.random.RandomState(0),
    )
    assert action == 1
    assert ties == 1
