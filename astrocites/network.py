import torch

from astrocites.learning import NoOp


class NetworkMonitor:
    def __init__(self, network, state_vars=('v', 's', 'w')):
        self.network = network
        self.state_vars = state_vars
        self.recording = {}
        self._targets = None
        self.reset_()

    def _build_targets(self):
        targets = []
        for name, layer in self.network.layers.items():
            for var in self.state_vars:
                if hasattr(layer, var):
                    targets.append((name, layer, var))
        for conn_key, conn in self.network.connections.items():
            for var in self.state_vars:
                if hasattr(conn, var):
                    targets.append((conn_key, conn, var))
        self._targets = targets

    def record(self):
        if self._targets is None:
            self._build_targets()
        recording = self.recording
        for name, obj, var in self._targets:
            data = getattr(obj, var)
            if var == 's':
                data = data.float()
            recording.setdefault(name, {}).setdefault(var, []).append(data.detach().clone())

    def get(self):
        result = {}
        for key, val_dict in self.recording.items():
            result[key] = {}
            for var, data_list in val_dict.items():
                if data_list:
                    result[key][var] = torch.stack(data_list)
        return result

    def reset_(self):
        self.recording = {}


class SpikeCountMonitor:
    """Accumulate per-layer spike totals without retaining every timestep."""
    def __init__(self, network, layer_names=("X", "Y", "I")):
        self.network = network
        self.layer_names = layer_names
        self.reset_()

    def record(self):
        for name in self.layer_names:
            layer = self.network.layers[name]
            self.counts[name].add_(layer.s.to(dtype=torch.long))

    def get(self):
        return {name: {"s": counts.clone()} for name, counts in self.counts.items()}

    def reset_(self):
        self.counts = {
            name: torch.zeros_like(self.network.layers[name].s, dtype=torch.long)
            for name in self.layer_names
        }


class Network(torch.nn.Module):
    def __init__(self, dt=1.0, batch_size=1, learning=True):
        super().__init__()
        self.dt = dt
        self.batch_size = batch_size
        self.learning = learning
        self.layers = torch.nn.ModuleDict()
        self.connections = torch.nn.ModuleDict()
        self.monitors = {}

    def add_layer(self, layer, name):
        self.layers[name] = layer
        if hasattr(layer, 'compute_decays'):
            layer.compute_decays(self.dt)
        layer.set_batch_size(self.batch_size)

    def add_connection(self, connection, source, target):
        key = f"{source}_{target}"
        self.connections[key] = connection

    def add_monitor(self, monitor, name):
        self.monitors[name] = monitor

    def _get_inputs(self, layers=None):
        inpts = {}
        if layers is None:
            layers = self.layers.keys()
        for layer_name in layers:
            if layer_name not in inpts:
                layer = self.layers[layer_name]
                inpts[layer_name] = torch.zeros(self.batch_size, *layer.shape)
        for conn_key, connection in self.connections.items():
            parts = conn_key.split('_')
            if len(parts) >= 2:
                src = parts[0]
                tgt = parts[1]
                if tgt in layers:
                    source_output = connection.compute(self.layers[src].s)
                    if source_output.dim() == 1:
                        source_output = source_output.unsqueeze(0)
                    inpts[tgt] += source_output
        return inpts

    def run(self, inpts, time, injects_v=None, current_position=None, adjacent_positions=None, conn_XY=None, enable_stdp=True, diagnostics=None, optimized_connections=True, **kwargs):
        timesteps = int(time / self.dt)
        injects_v = injects_v or {}
        layer_items = list(self.layers.items())
        # resolve connection routing once instead of re-parsing keys every timestep
        routes = []
        for conn_key, connection in self.connections.items():
            parts = conn_key.split('_')
            if len(parts) >= 2 and parts[1] in self.layers:
                routes.append((conn_key, connection, self.layers[parts[0]], parts[1]))
        # input buffers are reused across timesteps; layers never mutate their input
        current_inpts = {name: torch.zeros(self.batch_size, *layer.shape) for name, layer in layer_items}
        for t_step in range(timesteps):
            for buf in current_inpts.values():
                buf.zero_()
            for conn_key, connection, source_layer, tgt in routes:
                if optimized_connections and connection is conn_XY and current_position is not None:
                    source_output = connection.compute(
                        source_layer.s, active_source=current_position,
                    )
                else:
                    source_output = connection.compute(source_layer.s, optimized=optimized_connections)
                if source_output.dim() == 1:
                    source_output = source_output.unsqueeze(0)
                current_inpts[tgt] += source_output
                if diagnostics is not None and diagnostics.get("record_currents", True):
                    current_stats = diagnostics.setdefault("synaptic_current_l1", {})
                    current_stats[conn_key] = current_stats.get(conn_key, 0.0) + source_output.abs().sum()
            for layer_name, input_data in inpts.items():
                if layer_name in current_inpts:
                    if len(input_data.shape) == 3:
                        current_inpts[layer_name] += input_data[t_step]
                    else:
                        current_inpts[layer_name] += input_data
            for name, layer in layer_items:
                if name in injects_v:
                    inject_voltage = injects_v[name]
                    if len(inject_voltage.shape) == 1:
                        layer.v += inject_voltage
                    else:
                        layer.v += inject_voltage[t_step]
                if name in ['Y', 'I']:
                    layer.forward(current_inpts[name], current_position=current_position, adjacent_positions=adjacent_positions)
                else:
                    layer.forward(current_inpts[name])
            if diagnostics is not None:
                if diagnostics.get("record_currents", True):
                    current_stats = diagnostics.setdefault("net_current_l1", {})
                    for name, current in current_inpts.items():
                        current_stats[name] = current_stats.get(name, 0.0) + current.abs().sum()
                spike_stats = diagnostics.setdefault("spike_counts", {})
                for name, layer in layer_items:
                    spike_stats[name] = spike_stats.get(name, torch.zeros_like(layer.s, dtype=torch.long))
                    spike_stats[name].add_(layer.s.to(dtype=torch.long))
                    if getattr(layer, "enable_astrocyte", False):
                        diagnostics["astro_neuron_steps"] = diagnostics.get("astro_neuron_steps", 0) + int(layer.Ca.numel())
                        diagnostics["astro_active_neuron_steps"] = diagnostics.get("astro_active_neuron_steps", 0) + int((layer.Ca > 0).sum().item())
                        threshold_drop = (layer.initial_thresh - layer.thresh).float()
                        diagnostics["astro_threshold_drop_sum"] = diagnostics.get("astro_threshold_drop_sum", 0.0) + float(threshold_drop.sum().item())
            for _conn_key, connection, source_layer, _tgt in routes:
                if getattr(connection.target, 'enable_astrocyte', False):
                    # clone: layer spike buffers are reused across timesteps, but the
                    # astrocyte must see the spikes as of when they were stored
                    connection.target.__dict__['prev_layer_s'] = source_layer.s.clone()
                if connection is conn_XY:
                    if enable_stdp:
                        connection.update(current_position=current_position,
                                          adjacent_positions=adjacent_positions,
                                          learning=self.learning, **kwargs)
                        if diagnostics is not None:
                            raw_update = getattr(connection, "last_stdp_update_raw", None)
                            if raw_update is not None:
                                diagnostics["stdp_raw_update_l1"] = (
                                    diagnostics.get("stdp_raw_update_l1", 0.0)
                                    + float(raw_update.abs().sum().item())
                                )
                elif not isinstance(connection.update_rule, NoOp):
                    connection.update(learning=self.learning, **kwargs)
            for monitor in self.monitors.values():
                monitor.record()
        if diagnostics is not None:
            for name in ("synaptic_current_l1", "net_current_l1"):
                diagnostics[name] = {
                    key: float(value.item()) if torch.is_tensor(value) else float(value)
                    for key, value in diagnostics.get(name, {}).items()
                }
            diagnostics["spike_counts"] = {
                name: value.squeeze(0).detach().cpu()
                for name, value in diagnostics.get("spike_counts", {}).items()
            }
            diagnostics["astro_active_fraction"] = (
                diagnostics.get("astro_active_neuron_steps", 0)
                / diagnostics["astro_neuron_steps"]
                if diagnostics.get("astro_neuron_steps", 0) else 0.0
            )
            diagnostics["astro_mean_threshold_drop"] = (
                diagnostics.get("astro_threshold_drop_sum", 0.0)
                / diagnostics["astro_neuron_steps"]
                if diagnostics.get("astro_neuron_steps", 0) else 0.0
            )

    def reset_(self):
        for layer in self.layers.values():
            layer.reset_()
        for connection in self.connections.values():
            connection.reset_()
        for monitor in self.monitors.values():
            monitor.reset_()
