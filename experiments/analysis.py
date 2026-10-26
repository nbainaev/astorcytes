"""Seed-level analysis, publication figures and machine-readable source data."""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import gzip, hashlib, json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from . import engine as e

NAMES = dict(
    mlp="MLP local e-prop",
    random="Random walk",
    noastro="No Astro",
    astro="AstroRL",
    trace="AstroRL (Trace)",
)
SHORT = dict(mlp="MLP", random="Random", noastro="No Astro", astro="AstroRL", trace="Trace")
COLORS = dict(mlp="#0072B2", random="#666666", noastro="#D55E00", astro="#009E73", trace="#AA4499")
MARKERS = dict(mlp="o", random="s", noastro="^", astro="D", trace="v")
STYLES = dict(mlp="-", random=":", noastro="--", astro="-.", trace="-")
LABELS = dict(
    coverage="Coverage",
    coverage_auc_50="Coverage AUC (horizon 50)",
    backtracking_rate="Backtracking / valid moves",
    collision_rate="Collision rate",
    revisit_rate="Revisit rate",
    novelty_rate="New cells / action",
    exploration_efficiency="Unique moves / possible unique moves",
    loop2_rate="Two-action recurrence",
    loop4_rate="Four-action recurrence",
    state_action_coverage="State-action coverage",
    spatial_entropy="Spatial entropy / log(free cells)",
    prediction_accuracy="Nearest-code cell accuracy",
    prediction_balanced_accuracy="Balanced nearest-code cell accuracy",
    prediction_mse="Next-input-code MSE",
    reward01="Min–max normalized reward",
    optimal_over_steps="Optimal / actual steps",
    path_efficiency="Goal-adjusted path efficiency",
    success="Goal reached (auxiliary)",
    steps="Actual actions / episode",
    entropy="Action entropy (nats)",
    training_steps="Total training actions",
    prediction_move_accuracy="Nearest-code cell accuracy: moves",
    prediction_collision_accuracy="Nearest-code cell accuracy: collisions",
    persistence_accuracy="Persistence baseline accuracy",
    action_discrimination="Between-action prediction variance",
)
FIG = None
captions = {}
layout_checks = {}


def ci(values, seed=44019, draws=5000):
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
        scalar = True
    else:
        scalar = False
    if len(x) == 1:
        low = high = x[0]
    else:
        rng = np.random.default_rng(seed)
        weights = rng.multinomial(len(x), np.full(len(x), 1 / len(x)), size=draws) / len(x)
        samples = weights @ x
        low, high = np.nanquantile(samples, [0.025, 0.975], axis=0)
    mean = np.nanmean(x, axis=0)
    low = np.where((low > mean) & (low - mean < 1e-12), mean, low)
    high = np.where((high < mean) & (mean - high < 1e-12), mean, high)
    return (float(mean[0]), float(low[0]), float(high[0])) if scalar else (mean, low, high)


def save(fig, name, caption):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    overflow = []
    unrendered = set()
    for ax in fig.axes:
        for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = sorted(limits)
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not lo - 1e-12 <= tick.get_loc() <= hi + 1e-12:
                    unrendered.update((tick.label1, tick.label2))
    for item in fig.findobj(matplotlib.text.Text):
        if item in unrendered or not item.get_visible() or not item.get_text():
            continue
        box = item.get_window_extent(renderer)
        if (
            box.x0 < canvas.x0 - 2
            or box.y0 < canvas.y0 - 2
            or box.x1 > canvas.x1 + 2
            or box.y1 > canvas.y1 + 2
        ):
            overflow.append(item.get_text())
    layout_checks[name] = dict(
        text_outside_canvas=overflow,
        width_mm=fig.get_figwidth() * 25.4,
        height_mm=fig.get_figheight() * 25.4,
    )
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=600, facecolor="white", transparent=False)
    captions[name] = caption
    plt.close(fig)


def smooth(y, window=20):
    y = np.asarray(y, float)
    out = np.empty_like(y)
    for j in range(len(y)):
        out[j] = np.nanmean(y[max(0, j - window + 1) : j + 1])
    return out


def map_title(spec):
    return (
        dict(open="Open", obstacles="Obstacles", rooms="Rooms", branching="Branching")[
            spec["family"]
        ]
        + f" {spec['size']}×{spec['size']}"
    )


def legend(fig):
    handles = [
        plt.Line2D(
            [],
            [],
            color=COLORS[k],
            linestyle=STYLES[k],
            marker=MARKERS[k],
            markersize=3,
            label=SHORT[k],
        )
        for k in e.KINDS
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=5, frameon=False)


def curves(training, specs, metric, by_steps=False):
    fig, axes = plt.subplots(
        2, 4, figsize=(180 / 25.4, 112 / 25.4), layout="constrained", sharey=True
    )
    for row, group in enumerate(e.GROUPS):
        for col, spec in enumerate(specs):
            ax = axes[row, col]
            for kind in e.KINDS:
                frame = training[
                    (training.group == group)
                    & (training.kind == kind)
                    & (training["map"] == spec["name"])
                ]
                sequences = []
                maxsteps = []
                for seed, values in frame.groupby("seed", sort=True):
                    values = values.sort_values("episode")
                    y = smooth(values[metric].values)
                    if by_steps:
                        maxsteps.append(values.training_steps.iloc[-1])
                        sequences.append((values.training_steps.values, y))
                    else:
                        sequences.append(y)
                if by_steps:
                    grid = np.linspace(max(x[0][0] for x in sequences), min(maxsteps), 100)
                    data = np.asarray([np.interp(grid, x, y) for x, y in sequences])
                    xx = grid / 1000
                else:
                    data = np.asarray(sequences)
                    xx = np.arange(1, data.shape[1] + 1)
                mean, lo, hi = ci(data, draws=2000)
                ax.plot(
                    xx,
                    mean,
                    color=COLORS[kind],
                    ls=STYLES[kind],
                    lw=1.0,
                    marker=MARKERS[kind],
                    ms=2,
                    markevery=max(1, len(xx) // 6),
                    mew=0.4,
                    label=SHORT[kind],
                )
                ax.fill_between(xx, lo, hi, color=COLORS[kind], alpha=0.14, lw=0)
            ax.set_title(map_title(spec))
            ax.grid(alpha=0.18, lw=0.4)
            if col == 0:
                ax.set_ylabel(
                    ("No RL\n" if group == "no_rl" else "RL\n") + LABELS.get(metric, metric)
                )
            if row == 1:
                ax.set_xlabel("Environment actions (×1000)" if by_steps else "Training episodes")
            if metric not in ("prediction_mse", "entropy", "steps"):
                ax.set_ylim(0, 1)
    legend(fig)
    save(
        fig,
        f"training_{metric}" + ("_actions" if by_steps else ""),
        f"{LABELS.get(metric, metric)} during training. Trailing 20-episode means within each seed; mean and pointwise 95% seed-bootstrap CI, n=10. "
        + (
            "Within-seed linear interpolation of smoothed observations to 100 action-count positions, with no extrapolation; each model ends at its shortest seed budget."
            if by_steps
            else "No final seed or checkpoint was selected."
        )
        + " No-RL behavior curves of MLP, No Astro and random walk coincide by design where the metric depends only on trajectories; all source rows are retained.",
    )


def development_curves(dev, specs):
    for metric in ("prediction_balanced_accuracy", "prediction_mse"):
        fig, axes = plt.subplots(
            2, 4, figsize=(180 / 25.4, 110 / 25.4), layout="constrained", sharey=True
        )
        for row, g in enumerate(e.GROUPS):
            for col, s in enumerate(specs):
                ax = axes[row, col]
                for k in e.KINDS:
                    frame = dev[(dev.group == g) & (dev.kind == k) & (dev["map"] == s["name"])]
                    table = frame.pivot(index="seed", columns="episode", values=metric)
                    mean, lo, hi = ci(table.values)
                    ax.plot(
                        table.columns,
                        mean,
                        color=COLORS[k],
                        ls=STYLES[k],
                        marker=MARKERS[k],
                        ms=2.5,
                    )
                    ax.fill_between(table.columns, lo, hi, color=COLORS[k], alpha=0.14, lw=0)
                ax.set_title(map_title(s))
                ax.grid(alpha=0.18)
                if col == 0:
                    ax.set_ylabel(("No RL\n" if g == "no_rl" else "RL\n") + LABELS[metric])
                if row == 1:
                    ax.set_xlabel("Training episodes")
                if metric.endswith("accuracy"):
                    ax.set_ylim(0, 1)
        legend(fig)
        save(
            fig,
            f"common_probe_learning_{metric}",
            "Common-input prediction diagnostic at episodes 0,100,…,600; 512 forced-random transitions per seed/checkpoint. Mean and 95% seed-bootstrap CI, n=10; lines connect measured checkpoints without smoothing. Development probes are separate from the final held-out probe.",
        )


def endpoint(frame, specs, metric, prefix="evaluation", technical=False):
    fig, axes = plt.subplots(
        2, 4, figsize=(180 / 25.4, 116 / 25.4), layout="constrained", sharey=True
    )
    for row, g in enumerate(e.GROUPS):
        for col, s in enumerate(specs):
            ax = axes[row, col]
            for j, k in enumerate(e.KINDS):
                data = frame[(frame.group == g) & (frame.kind == k) & (frame["map"] == s["name"])]
                modes = (
                    ("on", "off")
                    if k in ("astro", "trace") and prefix == "evaluation"
                    else tuple(sorted(set(data["mode"])))
                )
                for mode in modes:
                    values = data[data["mode"] == mode][metric].dropna().values
                    if not len(values):
                        continue
                    offset = (-0.12 if mode == "on" else 0.12) if len(modes) > 1 else 0.0
                    mean, lo, hi = ci(values)
                    x = j + offset
                    jitter = np.linspace(-0.055, 0.055, len(values))
                    ax.scatter(x + jitter, values, s=5, alpha=0.45, color=COLORS[k], linewidths=0)
                    ax.errorbar(
                        x,
                        mean,
                        yerr=[[mean - lo], [hi - mean]],
                        fmt=MARKERS[k],
                        ms=4,
                        elinewidth=0.8,
                        capsize=2,
                        color=COLORS[k],
                        mfc=COLORS[k]
                        if prefix != "evaluation" or mode == "on" or k not in ("astro", "trace")
                        else "white",
                        mew=0.8,
                    )
            ax.set_title(map_title(s))
            ax.set_xticks(range(5), [SHORT[k] for k in e.KINDS], rotation=40, ha="right")
            ax.grid(axis="y", alpha=0.18)
            if col == 0:
                ax.set_ylabel(("No RL\n" if g == "no_rl" else "RL\n") + LABELS.get(metric, metric))
            if metric not in (
                "training_steps",
                "prediction_mse",
                "steps",
                "entropy",
                "action_discrimination",
            ):
                ax.set_ylim(-0.025, 1.025)
    fig.suptitle(
        "Technical render check: 2 training episodes, 1 seed; preliminary"
        if technical
        else "Filled: memory on / absent; open: same checkpoint, memory off"
        if prefix == "evaluation"
        else "Separate common-input prediction probe",
        fontsize=8,
    )
    save(
        fig,
        f"{prefix}_{metric}",
        f"{prefix}: {LABELS.get(metric, metric)}. "
        + (
            "Technical render check only: real two-episode smoke runs, one seed, two evaluation episodes; no inferential interval."
            if technical
            else "Each small point is one independent seed; large points and intervals are means and 95% seed-bootstrap CI, n=10. Evaluation uses 100 episodes per seed and mode. Memory on/off reuse the same trained checkpoint, starts and action uniforms; each episode resets dynamic state."
        ),
    )


def occupancy(rows, specs, spatial):
    # Equal-seed probability normalization; never pool longer trajectories as extra replicates.
    panels = {}
    limits = {}
    for g in e.GROUPS:
        for s in specs:
            for k in e.KINDS:
                rr = [
                    r
                    for r in rows
                    if r["task"]["group"] == g
                    and r["task"]["kind"] == k
                    and r["task"]["map"]["name"] == s["name"]
                ]
                for phase in (
                    "train",
                    "early",
                    "late",
                    "eval_on",
                    "eval_off",
                    "eval_arrival_on",
                    "eval_arrival_off",
                    "probe",
                ):
                    values = []
                    for r in rr:
                        z = spatial[r["path"]]
                        if phase == "probe":
                            num = z["probe_error"]
                            den = z["probe_count"]
                            a = np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)
                        else:
                            if phase in ("train", "early", "late"):
                                occ = z["training_occupancy"]
                                a = (
                                    occ.sum(0)
                                    if phase == "train"
                                    else occ[0]
                                    if phase == "early"
                                    else occ[-1]
                                )
                            else:
                                mode = (
                                    "on"
                                    if phase.endswith("_on") and k in ("astro", "trace")
                                    else "off"
                                )
                                statistic = (
                                    "arrivals" if phase.startswith("eval_arrival") else "occupancy"
                                )
                                a = z[f"eval_{mode}_{statistic}"]
                            a = a / max(a.sum(), 1)
                        values.append(a)
                    count = np.sum(np.isfinite(values), axis=0)
                    total = np.nansum(values, axis=0)
                    mean = np.divide(
                        total, count, out=np.full(total.shape, np.nan), where=count > 0
                    )
                    panels[(g, s["name"], k, phase)] = mean
                    if phase != "probe":
                        limits[s["name"]] = max(limits.get(s["name"], 0), float(np.nanmax(mean)))
    for g in e.GROUPS:
        for phase in (
            "train",
            "eval_on",
            "eval_off",
            "eval_arrival_on",
            "eval_arrival_off",
            "probe",
        ):
            kinds = ("astro", "trace") if phase.endswith("_off") else e.KINDS
            fig, axes = plt.subplots(
                4,
                len(kinds),
                figsize=(180 / 25.4, (160 if len(kinds) == 5 else 190) / 25.4),
                layout="constrained",
                squeeze=False,
            )
            for i, s in enumerate(specs):
                norm = Normalize(0, 1 if phase == "probe" else limits[s["name"]])
                for j, k in enumerate(kinds):
                    ax = axes[i, j]
                    a = panels[(g, s["name"], k, phase)].copy()
                    mask = np.ones_like(a, dtype=bool)
                    for pos in s["free"]:
                        mask[tuple(pos)] = False
                    image = np.ma.array(a, mask=mask | ~np.isfinite(a))
                    cmap = plt.get_cmap("magma" if phase == "probe" else "cividis").copy()
                    cmap.set_bad("white" if phase == "probe" else "#c9c9c9")
                    im = ax.imshow(
                        image, cmap=cmap, norm=norm, interpolation="nearest", origin="upper"
                    )
                    if phase == "probe":
                        walls = np.ma.array(np.ones_like(a), mask=~mask)
                        ax.imshow(
                            walls,
                            cmap=matplotlib.colors.ListedColormap(["#c9c9c9"]),
                            vmin=0,
                            vmax=1,
                            interpolation="nearest",
                        )
                    goal = max(map(tuple, s["free"]))
                    ax.scatter(
                        goal[1],
                        goal[0],
                        marker="*",
                        s=30,
                        facecolor="white",
                        edgecolor="black",
                        linewidth=0.5,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if i == 0:
                        ax.set_title(SHORT[k])
                    if j == 0:
                        ax.set_ylabel(map_title(s))
                fig.colorbar(
                    im,
                    ax=axes[i, :].tolist(),
                    shrink=0.75,
                    aspect=18,
                    label="Prediction error fraction"
                    if phase == "probe"
                    else "Fraction of actions",
                )
            fig.suptitle(
                ("No RL" if g == "no_rl" else "RL")
                + " — "
                + dict(
                    train="training occupancy",
                    eval_on="evaluation: memory on / absent",
                    eval_off="evaluation: memory off",
                    eval_arrival_on="evaluation arrivals: memory on / absent",
                    eval_arrival_off="evaluation arrivals: memory off",
                    probe="common-probe prediction errors",
                )[phase],
                fontsize=9,
            )
            quantity = (
                "next-cell error fractions on shared forced-random inputs"
                if phase == "probe"
                else "normalized post-action arrival counts, including terminal goal visits"
                if phase.startswith("eval_arrival")
                else "normalized source-cell action counts (terminal goal has no subsequent action)"
            )
            save(
                fig,
                f"spatial_{g}_{phase}",
                f"{g}, {phase}. Mean of within-seed {quantity}, n=10, with common color limits across all compared conditions for each map. Walls gray; star marks the fixed goal. Probe unsampled free cells are white (missing), while zero error remains a data color. No spatial interpolation.",
            )
        for s in specs:
            fig, axes = plt.subplots(2, 5, figsize=(180 / 25.4, 83 / 25.4), layout="constrained")
            for i, phase in enumerate(("early", "late")):
                for j, k in enumerate(e.KINDS):
                    ax = axes[i, j]
                    a = panels[(g, s["name"], k, phase)]
                    mask = np.ones_like(a, dtype=bool)
                    for p in s["free"]:
                        mask[tuple(p)] = False
                    cmap = plt.get_cmap("cividis").copy()
                    cmap.set_bad("#c9c9c9")
                    im = ax.imshow(
                        np.ma.array(a, mask=mask),
                        cmap=cmap,
                        norm=Normalize(0, limits[s["name"]]),
                        interpolation="nearest",
                    )
                    goal = max(map(tuple, s["free"]))
                    ax.scatter(
                        goal[1],
                        goal[0],
                        marker="*",
                        s=25,
                        c="white",
                        edgecolors="black",
                        linewidth=0.4,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if i == 0:
                        ax.set_title(SHORT[k])
                    if j == 0:
                        ax.set_ylabel("Episodes 1–100" if i == 0 else "Episodes 501–600")
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Fraction of actions")
            fig.suptitle(f"{g}: {map_title(s)}", fontsize=9)
            save(
                fig,
                f"occupancy_change_{g}_{s['family']}",
                f"Early versus late training occupancy, {g}, {map_title(s)}. Each seed is normalized within its 100-episode window before averaging. Shared map-specific color limits, no interpolation; n=10 independent seeds.",
            )
    np.savez_compressed(
        e.OUT / "spatial_group_means.npz",
        **{"__".join(key): value for key, value in panels.items()},
    )


def paired_statistics(evalframe, probeframe, specs):
    results = []

    def compare(a, b, family, description, metric, group, mapname, kind):
        common = sorted(set(a.index) & set(b.index))
        x = (a.loc[common] - b.loc[common]).values.astype(float)
        mean, lo, hi = ci(x)
        signs = np.array(list(__import__("itertools").product((-1.0, 1.0), repeat=len(x))))
        p = float(np.mean(np.abs(signs @ x / len(x)) >= abs(x.mean()) - 1e-14))
        results.append(
            dict(
                family=family,
                comparison=description,
                group=group,
                map=mapname,
                kind=kind,
                metric=metric,
                n=len(x),
                difference=mean,
                ci_low=lo,
                ci_high=hi,
                p=p,
            )
        )

    active = evalframe[(~evalframe.kind.isin(["astro", "trace"])) | (evalframe["mode"] == "on")]
    for g in e.GROUPS:
        metrics = (
            ("reward01", "optimal_over_steps")
            if g == "rl"
            else ("coverage", "backtracking_rate", "prediction_balanced_accuracy")
        )
        for s in specs:
            for metric in metrics:
                frame = (
                    probeframe
                    if g == "no_rl" and metric == "prediction_balanced_accuracy"
                    else active
                )
                subset = frame[(frame.group == g) & (frame["map"] == s["name"])]
                base = subset[subset.kind == "noastro"].set_index("seed")[metric]
                for k in ("mlp", "random", "astro", "trace"):
                    a = subset[subset.kind == k].set_index("seed")[metric]
                    compare(
                        a, base, f"{g}_primary", "model minus No Astro", metric, g, s["name"], k
                    )
    for g in e.GROUPS:
        for s in specs:
            for k in ("astro", "trace"):
                f = evalframe[
                    (evalframe.group == g) & (evalframe["map"] == s["name"]) & (evalframe.kind == k)
                ]
                for metric in (
                    "reward01",
                    "optimal_over_steps",
                    "coverage",
                    "backtracking_rate",
                    "prediction_accuracy",
                ):
                    compare(
                        f[f["mode"] == "on"].set_index("seed")[metric],
                        f[f["mode"] == "off"].set_index("seed")[metric],
                        "memory_ablation",
                        "same checkpoint: memory on minus off",
                        metric,
                        g,
                        s["name"],
                        k,
                    )
    for family in {r["family"] for r in results}:
        rr = sorted([r for r in results if r["family"] == family], key=lambda r: r["p"])
        last = 0
        for j, r in enumerate(rr):
            last = max(last, min(1.0, r["p"] * (len(rr) - j)))
            r["holm_p"] = last
    e.write(e.OUT / "paired_comparisons.json", results)
    pd.DataFrame(results).to_csv(e.OUT / "paired_comparisons.csv", index=False)
    for metric in (
        "reward01",
        "optimal_over_steps",
        "coverage",
        "backtracking_rate",
        "prediction_accuracy",
    ):
        fig, axes = plt.subplots(
            1, 2, figsize=(180 / 25.4, 100 / 25.4), layout="constrained", sharex=True
        )
        for ax, g in zip(axes, e.GROUPS):
            rr = [
                r
                for r in results
                if r["family"] == "memory_ablation" and r["group"] == g and r["metric"] == metric
            ]
            for i, r in enumerate(rr):
                ax.errorbar(
                    r["difference"],
                    i,
                    xerr=[[r["difference"] - r["ci_low"]], [r["ci_high"] - r["difference"]]],
                    fmt=MARKERS[r["kind"]],
                    color=COLORS[r["kind"]],
                    capsize=2,
                    ms=4,
                )
            ax.set_yticks(
                range(len(rr)), [r["map"].split("_")[0] + " / " + SHORT[r["kind"]] for r in rr]
            )
            ax.axvline(0, color="black", ls=":", lw=0.7)
            ax.grid(axis="x", alpha=0.2)
            ax.set_title("No RL" if g == "no_rl" else "RL")
            ax.set_xlabel("Memory on − memory off")
        save(
            fig,
            f"memory_ablation_{metric}",
            f"Within-checkpoint inference effect on {LABELS[metric]}; paired mean difference and paired 95% seed-bootstrap CI, n=10, matched 100-episode evaluation starts/action uniforms. Exact sign-flip p-values and family-wise Holm corrections are in paired_comparisons.csv; no significance stars are substituted for uncertainty.",
        )
    return results


def tradeoffs(frame, specs):
    active = frame[(~frame.kind.isin(["astro", "trace"])) | (frame["mode"] == "on")]
    fig, axes = plt.subplots(
        2, 4, figsize=(180 / 25.4, 115 / 25.4), layout="constrained", sharex=True, sharey=True
    )
    for i, g in enumerate(e.GROUPS):
        for j, s in enumerate(specs):
            ax = axes[i, j]
            for k in e.KINDS:
                f = active[(active.group == g) & (active.kind == k) & (active["map"] == s["name"])]
                ax.scatter(f.coverage, f.reward01, c=COLORS[k], marker=MARKERS[k], s=10, alpha=0.55)
                ax.scatter(
                    f.coverage.mean(),
                    f.reward01.mean(),
                    c=COLORS[k],
                    marker=MARKERS[k],
                    s=28,
                    edgecolors="black",
                    linewidths=0.5,
                )
            ax.set(xlim=(0, 1), ylim=(0, 1))
            ax.set_title(map_title(s))
            ax.grid(alpha=0.18)
            if i == 1:
                ax.set_xlabel("Coverage")
            if j == 0:
                ax.set_ylabel(("No RL\n" if g == "no_rl" else "RL\n") + "Normalized reward")
    legend(fig)
    save(
        fig,
        "exploration_navigation_tradeoff",
        "Exploration/navigation tradeoff on final frozen evaluation. Small points are independent seed means, large points are group centroids, n=10. No-RL rewards are diagnostic only; policy parameters were not trained with reward.",
    )


def checkpoint_check(row):
    import torch

    torch.set_num_threads(1)
    path = Path(row["path"]) / "checkpoint_0600.pt"
    state = torch.load(path, weights_only=False, map_location="cpu")
    assert state["episode"] == 600 and state["task"] == row["task"]
    repeated, *_ = e.evaluate(
        state["learner"], row["task"]["map"], row["task"]["seed"], count=100, split=8
    )
    assert repeated == row["final"], "Evaluation changed after checkpoint loading"
    return dict(
        group=row["task"]["group"],
        kind=row["task"]["kind"],
        map=row["task"]["map"]["name"],
        seed=row["task"]["seed"],
        evaluation_matches_checkpoint=True,
    )


def report():
    global FIG
    FIG = e.OUT / "figures"
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    rows = e.read(e.OUT / "final_index.json")
    assert len(rows) == 400
    check_rows = [r for r in rows if r["task"]["seed"] == 62000]
    assert len(check_rows) == 40
    with ProcessPoolExecutor(max_workers=30) as pool:
        reload_checks = list(pool.map(checkpoint_check, check_rows))
    e.write(e.OUT / "checkpoint_reload_checks.json", reload_checks)
    protocol = e.read(e.OUT / "protocol.json")
    specs = protocol["maps"]
    training = []
    development = []
    evaluation = []
    evaluation_episodes = []
    probes = []
    spatial = {}
    audit = dict(
        final_runs=400,
        all_600_episodes=True,
        all_evaluations_frozen=True,
        all_no_rl_policies_frozen=True,
        matched_snn_initialization=True,
        matched_evaluation_starts=True,
        matched_common_probe=True,
    )
    for r in rows:
        t = r["task"]
        base = dict(kind=t["kind"], group=t["group"], map=t["map"]["name"], seed=t["seed"])
        assert r["episodes"] == 600 and r["final"]["count"] == 100 and r["final"]["weights_frozen"]
        if t["group"] == "no_rl" or t["kind"] == "random":
            assert r["metadata"]["initial_policy_hash"] == r["metadata"]["final_policy_hash"]
        assert (
            r["learning_audit"]["max_policy_core_gradient"] == 0
            and r["learning_audit"]["predictor_changed"]
        )
        with gzip.open(Path(r["path"]) / "metrics.json.gz", "rt") as f:
            data = json.load(f)
        assert len(data["training"]) == 600 and max(x["steps"] for x in data["training"]) <= 50
        training.extend(dict(**base, **x) for x in data["training"])
        development.extend(dict(**base, **x) for x in data["development"])
        for mode, episodes in data["evaluation"].items():
            evaluation_episodes.extend(dict(**base, mode=mode, **x) for x in episodes)
        for mode, metrics in r["final"]["evaluation"].items():
            evaluation.append(
                dict(**base, mode=mode, **metrics, training_steps=r["training_steps"])
            )
        probes.append(dict(**base, mode="probe", **r["final"]["probe"]))
        with np.load(Path(r["path"]) / "spatial.npz") as z:
            spatial[r["path"]] = {k: z[k].copy() for k in z.files}
    for s in specs:
        for seed in protocol["final_seeds"]:
            rr = [
                r
                for r in rows
                if r["task"]["map"]["name"] == s["name"] and r["task"]["seed"] == seed
            ]
            assert len({r["start_hash"] for r in rr}) == 1
            assert len({r["final"]["start_hash"] for r in rr}) == 1
            assert len({r["final"]["probe"]["source_hash"] for r in rr}) == 1
            snn = [r for r in rr if r["task"]["kind"] in ("noastro", "astro", "trace")]
            assert len({r["metadata"]["initial_backbone_hash"] for r in snn}) == 1
            assert len({r["metadata"]["initial_predictor_hash"] for r in snn}) == 1
            assert len({r["metadata"]["policy_parameter_count"] for r in snn}) == 1
            null = [
                r
                for r in rr
                if r["task"]["group"] == "no_rl"
                and r["task"]["kind"] in ("mlp", "random", "noastro")
            ]
            null.append(
                next(r for r in rr if r["task"]["group"] == "rl" and r["task"]["kind"] == "random")
            )
            assert len({r["training_steps"] for r in null}) == 1
            arrays = [spatial[r["path"]]["training_occupancy"] for r in null]
            assert all(np.array_equal(arrays[0], a) for a in arrays[1:]), (
                "Uniform null policies diverged"
            )
    audit["uniform_null_training_occupancy_identical"] = True
    audit["full_checkpoint_reload_evaluations"] = 40
    e.write(e.OUT / "design_checks.json", audit)
    train = pd.DataFrame(training)
    dev = pd.DataFrame(development)
    ev = pd.DataFrame(evaluation)
    probe = pd.DataFrame(probes)
    train.to_csv(e.OUT / "training_metrics.csv.gz", index=False, compression="gzip")
    pd.DataFrame(evaluation_episodes).to_csv(
        e.OUT / "evaluation_episode_metrics.csv.gz", index=False, compression="gzip"
    )
    dev.to_csv(e.OUT / "prediction_learning.csv", index=False)
    ev.to_csv(e.OUT / "evaluation_per_seed.csv", index=False)
    probe.to_csv(e.OUT / "probe_per_seed.csv", index=False)
    for key in (
        "coverage",
        "coverage_auc_50",
        "backtracking_rate",
        "collision_rate",
        "revisit_rate",
        "novelty_rate",
        "entropy",
        "prediction_accuracy",
        "prediction_mse",
        "reward01",
        "optimal_over_steps",
        "path_efficiency",
        "state_action_coverage",
        "spatial_entropy",
    ):
        curves(train, specs, key)
    for key in ("coverage", "prediction_accuracy", "reward01", "optimal_over_steps"):
        curves(train, specs, key, True)
    development_curves(dev, specs)
    for key in (
        "coverage",
        "coverage_auc_50",
        "exploration_efficiency",
        "backtracking_rate",
        "collision_rate",
        "revisit_rate",
        "novelty_rate",
        "state_action_coverage",
        "spatial_entropy",
        "loop2_rate",
        "loop4_rate",
        "entropy",
        "prediction_accuracy",
        "prediction_balanced_accuracy",
        "prediction_move_accuracy",
        "prediction_collision_accuracy",
        "prediction_mse",
        "persistence_accuracy",
        "action_discrimination",
        "reward01",
        "optimal_over_steps",
        "path_efficiency",
        "steps",
        "success",
        "training_steps",
    ):
        endpoint(ev, specs, key)
    for key in (
        "prediction_balanced_accuracy",
        "prediction_move_accuracy",
        "prediction_collision_accuracy",
        "prediction_mse",
    ):
        endpoint(probe, specs, key, "common_probe")
    occupancy(rows, specs, spatial)
    paired_statistics(ev, probe, specs)
    tradeoffs(ev, specs)
    # Hyperparameter selection and complete between-seed distributions remain inspectable.
    locked = e.read(e.OUT / "locked_parameters.json")
    pd.DataFrame(
        [dict(group=x["group"], kind=x["kind"], map=x["map"], **x["hp"]) for x in locked]
    ).to_csv(e.OUT / "locked_parameters.csv", index=False)
    budgets = []
    checkpoints = []
    for r in rows:
        t = r["task"]
        m = r["metadata"]
        budgets.append(
            dict(
                group=t["group"],
                kind=t["kind"],
                map=t["map"]["name"],
                seed=t["seed"],
                parameter_count=m["parameter_count"],
                trainable_parameter_count=m["trainable_parameter_count"],
                policy_parameter_count=m["policy_parameter_count"],
                episodes=600,
                training_steps=r["training_steps"],
                prediction_representation_updates=600,
                prediction_readout_updates=600 * t["hp"]["pred_epochs"],
                actor_optimizer_steps=600 if t["group"] == "rl" and t["kind"] != "random" else 0,
                nonzero_actor_updates=r["learning_audit"]["actor_updates"],
                seconds=r["seconds"],
            )
        )
        path = Path(r["path"]) / "checkpoint_0600.pt"
        checkpoints.append(
            dict(
                group=t["group"],
                kind=t["kind"],
                map=t["map"]["name"],
                seed=t["seed"],
                path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    pd.DataFrame(budgets).to_csv(e.OUT / "model_compute_budgets.csv", index=False)
    pd.DataFrame(checkpoints).to_csv(e.OUT / "checkpoint_manifest.csv", index=False)
    if (e.OUT / "validation_ranking.json").exists():
        validation = e.read(e.OUT / "validation_ranking.json")
        low = min([0.0] + [c["mean"] - c["sd"] for r in validation for c in r["candidates"]])
        high = max([1.0] + [c["mean"] + c["sd"] for r in validation for c in r["candidates"]])
        fig, axes = plt.subplots(2, 4, figsize=(180 / 25.4, 110 / 25.4), layout="constrained")
        for i, g in enumerate(e.GROUPS):
            for j, s in enumerate(specs):
                ax = axes[i, j]
                for n, k in enumerate(e.KINDS):
                    r = next(
                        r
                        for r in validation
                        if r["group"] == g and r["kind"] == k and r["map"] == s["name"]
                    )
                    for offset, c in zip((-0.1, 0.1), r["candidates"]):
                        ax.errorbar(
                            n + offset,
                            c["mean"],
                            yerr=c["sd"],
                            fmt=MARKERS[k],
                            ms=3,
                            color=COLORS[k],
                            mfc=COLORS[k] if offset < 0 else "white",
                            capsize=2,
                        )
                ax.set_title(map_title(s))
                ax.set_xticks(range(5), [SHORT[k] for k in e.KINDS], rotation=40, ha="right")
                ax.set_ylim(low - 0.02, high + 0.02)
                if j == 0:
                    ax.set_ylabel(
                        ("No RL\n" if g == "no_rl" else "RL\n") + "Within-model objective ± SD"
                    )
        save(
            fig,
            "hyperparameter_validation",
            "Two final candidate configurations per group/model/map, evaluated on five independent validation seeds at the full 600-episode budget. Filled marker: selected by mean − 0.25 SD; open marker: other candidate. Intervals show sample SD, not CI, and are not truncated. Objectives differ between groups and for the random walk: compare candidates within a model, not heights across models. Final seeds were not used for selection.",
        )
        # Text report uses seed means/SD and all memory modes, never pooled episodes as n.
    lines = [
        "# Две группы: prediction/exploration без RL и обучение с RL",
        "",
        "Все 400 итоговых обучений завершены. 600 эпизодов, максимум 50 попыток движения; эпизод завершается при первом достижении фиксированной цели. 10 независимых seed на каждую группу × модель × карту.",
        "Отдельная evaluation: 100 новых эпизодов на seed и режим памяти, веса заморожены. Для AstroRL и Trace проведены парные memory ON/OFF evaluation одного checkpoint.",
        "",
        "Параметры заданы в configs/conditions.json. При запуске tune выполняется отдельный подбор на seed, не пересекающихся с финальными. Настройки текущего запуска сохранены в locked_parameters.csv.",
        "",
        "MLP: чистая ReLU policy 2→64→64→4 с локальными eligibility и фиксированным прямым feedback; отдельный action-conditioned prediction MLP использует тот же локальный принцип. Это feedforward e-prop-style/DFA, не оригинальное рекуррентное e-prop Bellec. Random walk равновероятно выбирает четыре действия и обучает Bellec predictor; RL не меняет его политику. SNN-модели No Astro/AstroRL/Trace имеют одинаковые архитектуры; RL-голова detached.",
        "",
        "В таблицах — среднее ± выборочное SD по 10 seed. На графиках — 95% bootstrap CI по seed и отдельные seed-точки. 100 эпизодов внутри seed не считаются независимыми экспериментальными репликами.",
        "",
        "Общий prediction probe задаёт одинаковые наблюдения и действия, поэтому inference-переключатель memory не меняет его входы или predictor. Он показан один раз на checkpoint; ON/OFF prediction-метрики посещённых агентом траекторий приведены отдельно. Короткие forced-random probe-контексты не прерываются на цели и могут включать её как исходное состояние: это дополнительный offline-тест, а не замена 100 обычным goal-terminated эпизодам evaluation.",
        "",
        "Награда R=success−0,01T; min–max=(R+0,5)/(1,5−0,01d*). Отношение d*/T показывается вместе с success·d*/T: при недостижении цели обычное отношение может вводить в заблуждение. Success rate — вспомогательная метрика.",
        "",
    ]
    aggregate = []
    for g in e.GROUPS:
        lines += ["## " + ("Без RL" if g == "no_rl" else "С RL"), ""]
        for s in specs:
            lines += [
                "### " + map_title(s),
                "",
                "| Модель / память | Reward01 | d*/T | success·d*/T | Coverage | Backtracking | Prediction accuracy |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
            for k in e.KINDS:
                f = ev[(ev.group == g) & (ev.kind == k) & (ev["map"] == s["name"])]
                for mode in sorted(set(f["mode"]), reverse=True):
                    z = f[f["mode"] == mode]
                    cells = []
                    rr = dict(group=g, kind=k, map=s["name"], mode=mode, n=len(z))
                    for key in (
                        "reward01",
                        "optimal_over_steps",
                        "path_efficiency",
                        "coverage",
                        "backtracking_rate",
                        "prediction_accuracy",
                    ):
                        values = z[key].values
                        mean, lo, hi = ci(values)
                        sd = float(np.std(values, ddof=1))
                        cells.append(f"{mean:.3f} ± {sd:.3f}")
                        rr[key] = dict(mean=mean, sd=sd, ci_low=lo, ci_high=hi)
                    aggregate.append(rr)
                    lines.append(
                        "| "
                        + NAMES[k]
                        + (" / " + mode.upper() if k in ("astro", "trace") else "")
                        + " | "
                        + " | ".join(cells)
                        + " |"
                    )
            lines += [""]
    lines += [
        "## Основные графики",
        "",
        "![Coverage learning](figures/training_coverage.png)",
        "",
        "![Common-input prediction learning](figures/common_probe_learning_prediction_balanced_accuracy.png)",
        "",
        "![Final evaluation reward](figures/evaluation_reward01.png)",
        "",
        "![Final evaluation path efficiency](figures/evaluation_optimal_over_steps.png)",
        "",
        "![Memory influence](figures/memory_ablation_reward01.png)",
        "",
        "![No-RL evaluation occupancy](figures/spatial_no_rl_eval_on.png)",
        "",
        "![RL evaluation occupancy](figures/spatial_rl_eval_on.png)",
        "",
        "Полный список графиков, подписей, единиц и преобразований — FIGURE_GUIDE.md. Все модели/seed сохранены, в том числе неудачные.",
        "",
    ]
    lines += ["## Что учитывать при интерпретации", ""] + [
        f"- {x}" for x in protocol["limitations"]
    ]
    lines += [
        "",
        "## Файлы и воспроизводимость",
        "",
        "training_metrics.csv.gz — скалярные метрики всех обучающих эпизодов; evaluation_per_seed.csv и probe_per_seed.csv — финальные результаты; paired_comparisons.csv — парные разницы/интервалы/точные sign-flip p и Holm-коррекции. model_compute_budgets.csv содержит параметры, реальные переходы, optimizer steps и время; checkpoint_manifest.csv — пути и SHA256 весов. Отдельные episode-метрики, occupancy и checkpoint_0600.pt хранятся в каждом final-каталоге. По одному checkpoint на каждую группу/модель/карту (40 checkpoint) повторно загружены и полностью воспроизвели 100-эпизодную evaluation.",
        "Occupancy нормирована на число фактических действий внутри каждого seed, затем усреднена по seed. Стены маскированы, цель отмечена звездой, цветовые пределы одинаковы для одной карты во всех сравниваемых условиях. Отдельно показаны обучение, ранние/поздние эпизоды и evaluation ON/OFF.",
        "PDF/SVG — векторные графики, PNG — 600 dpi; общий стиль шириной 180 мм. Требования конкретного журнала не заданы. Для каждого рисунка сохранены подпись и данные. Сырые tuning-каталоги удалялись после записи компактных результатов; итоговые скалярные данные и checkpoint сохранены.",
        "",
    ]
    (e.OUT / "REPORT_RU.md").write_text("\n".join(lines))
    (e.OUT / "FIGURE_GUIDE.md").write_text(
        "# Figure guide\n\n"
        + "\n\n".join(
            f"## {name}\n\n{caption}\n\n[PDF](figures/{name}.pdf) · [SVG](figures/{name}.svg) · [PNG](figures/{name}.png)"
            for name, caption in captions.items()
        )
    )
    e.write(e.OUT / "aggregate.json", aggregate)
    e.write(e.OUT / "figure_captions.json", captions)
    from PIL import Image

    image_checks = {}
    for path in FIG.glob("*.png"):
        with Image.open(path) as im:
            dpi = im.info.get("dpi")
            assert dpi and min(dpi) > 599
            opaque = im.mode != "RGBA" or im.getchannel("A").getextrema() == (255, 255)
            assert opaque
            image_checks[path.name] = dict(pixels=im.size, dpi=dpi, mode=im.mode, opaque=opaque)
    e.write(e.OUT / "figure_layout_checks.json", layout_checks)
    e.write(e.OUT / "figure_image_checks.json", image_checks)
    print("REPORT AND", len(captions), "FIGURES COMPLETE", flush=True)


if __name__ == "__main__":
    report()
