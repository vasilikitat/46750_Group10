"""Matplotlib figures for the input data and the optimisation results.

Every function returns the ``Figure`` and optionally saves it, so the same code works in a
script (``python main.py``) and in a notebook (``plot_schedule(results, data);``).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_loader import InputData
from .model import Results


def _finish(fig: plt.Figure, save_to: Path | str | None) -> plt.Figure:
    fig.tight_layout()
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150)
    return fig


def plot_inputs(data: InputData, save_to: Path | str | None = None) -> plt.Figure:
    """Hourly prices (with tariffs) and available PV / load preferences, side by side."""
    h = data.hours
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))

    ax1.step(h, data.energy_price, where="mid", label="energy price", color="k")
    ax1.step(h, data.energy_price + data.import_tariff, where="mid", ls="--", label="price + import tariff")
    ax1.step(h, data.energy_price - data.export_tariff, where="mid", ls=":", label="price - export tariff")
    ax1.set(xlabel="hour", ylabel="DKK/kWh", title="Electricity prices")
    ax1.legend(fontsize=8)

    ax2.fill_between(h, data.pv_available, step="mid", alpha=0.4, color="orange", label="PV available")
    ax2.axhline(data.load_max_kWh, color="C3", ls="--", label="max load")
    if data.load_min_kWh > 0:
        ax2.axhline(data.load_min_kWh, color="C3", ls=":", label="min load")
    if data.reference_load is not None:
        ax2.step(h, data.reference_load, where="mid", color="C0", label="reference load")
    ax2.set(xlabel="hour", ylabel="kWh/h", title="PV and load preferences")
    ax2.legend(fontsize=8)
    fig.suptitle(f"Input data - {data.question}", fontsize=11)
    return _finish(fig, save_to)


def plot_schedule(results: Results, data: InputData, save_to: Path | str | None = None) -> plt.Figure:
    """Optimal schedule: load, PV used, import/export, with prices on a second axis."""
    hr = results.hourly
    h = hr.index.to_numpy()
    fig, ax = plt.subplots(figsize=(11, 4.2))
    
    width = 0.8
    if "load" in hr:
        ax.bar(h, hr["load"], width, color="C0", alpha=0.7, label="load")
    if "pv" in hr:
        ax.bar(h, -hr["pv"], width, color="orange", alpha=0.7, label="PV used (negative = generation)")
    if "pv_available" in hr and "pv" in hr:
        ax.step(h, -hr["pv_available"], where="mid", color="orange", ls="--", lw=1, label="PV available")
    if "import" in hr and "export" in hr:
        ax.plot(h, hr["import"] - hr["export"], "k.-", label="net import (+) / export (-)")
    if "reference_load" in hr:
        ax.step(h, hr["reference_load"], where="mid", color="C0", ls=":", label="reference load")
      
    ax.axhline(0, color="grey", lw=0.8)
    ax.set(xlabel="hour", ylabel="kWh/h", title=f"Optimal schedule - {results.question} (cost {results.objective:.1f} DKK)")

    ax2 = ax.twinx()
    ax2.step(h, hr["price"], where="mid", color="C3", lw=1.2, label="energy price")
    ax2.step(h, data.energy_price + data.import_tariff, where="mid", color="red", ls="--", lw=1, label="price + import tariff")
    ax2.step(h, data.energy_price - data.export_tariff, where="mid", color="red", ls=":", lw=1, label="price - export tariff")
    ax2.plot([0,23],[data.consumption_utility,data.consumption_utility])
    ax2.set_ylabel("DKK/kWh", color="C3")

    lines, labels = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(lines + l2, labels + lb2, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    return _finish(fig, save_to)


def plot_duals(results: Results, data: InputData, save_to: Path | str | None = None) -> plt.Figure:
    """Hourly dual variables (all ``dual_*`` columns) against the price signals."""
    hr = results.hourly
    dual_cols = [c for c in hr.columns if c.startswith("dual_")]
    fig, ax = plt.subplots(figsize=(11, 4))
    h = hr.index.to_numpy()
    for c in dual_cols:
        ax.step(h, hr[c], where="mid", label=c.removeprefix("dual_"))
    ax.step(h, data.energy_price + data.import_tariff, where="mid", color="grey", ls="--", lw=1, label="price + import tariff")
    ax.step(h, data.energy_price - data.export_tariff, where="mid", color="grey", ls=":", lw=1, label="price - export tariff")
    ax.set(xlabel="hour", ylabel="DKK/kWh", title=f"Dual variables - {results.question}")
    ax.legend(fontsize=8, ncol=3)
    return _finish(fig, save_to)


def plot_scenario_comparison(
    runs: dict[str, Results], metric: str = "objective", save_to: Path | str | None = None
) -> plt.Figure:
    """Bar chart of one metric across scenarios. ``metric`` is ``"objective"`` or the name of an
    hourly column whose daily sum is compared (e.g. ``"import"``, ``"export"``, ``"load"``)."""
    names = list(runs)
    if metric == "objective":
        values = [r.objective for r in runs.values()]
        ylabel = "daily cost [DKK]"
    else:
        values = [r.hourly[metric].sum() for r in runs.values()]
        ylabel = f"daily {metric} [kWh]"
    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(names)), 3.8))
    ax.bar(names, values, color="C0")
    ax.set(ylabel=ylabel, title=f"Scenario comparison - {metric}")
    ax.tick_params(axis="x", rotation=20)
    return _finish(fig, save_to)

def plot_sweep(df: pd.DataFrame, data: InputData, param_col: str = "c_L", save_to: Path | str | None = None) -> plt.Figure:
    """Line plots of summary metrics against c^L parameter,
    with vertical markers at the smallest/largest effective import and export prices, which
    bound the c^L thresholds where the optimal load's behaviour changes."""
    metrics = ["procurement_cost", "disutility", "daily_load", "total_deviation"]
    titles = ["Procurement cost [DKK]", "Disutility [DKK]", "Daily load [kWh]", "Total deviation [kWh]"]

    lo = (data.energy_price - data.export_tariff).min()
    hi = (data.energy_price + data.import_tariff).max()

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.flatten()

    for ax, metric, title in zip(axes, metrics, titles):
        ax.plot(df[param_col], df[metric], "o-", color="C0")
        ax.axvline(lo, color="grey", ls=":", lw=1, label="min (price - export tariff)")
        ax.axvline(hi, color="grey", ls="--", lw=1, label="max (price + import tariff)")
        ax.set(xlabel=param_col, ylabel=title, title=title)

    axes[0].legend(fontsize=7)
    fig.suptitle("Sweep over " + param_col, fontsize=11)
    return _finish(fig, save_to)


def plot_results_overview(
    results: Results,
    data: InputData,
    save_to: Path | str | None = None,
) -> plt.Figure:
    """
    Combined optimisation-results figure with three vertically stacked panels:

    1) Optimal schedule
    2) Prices, utility and PV production cost
    3) Dual variables, with price signals shown faintly for comparison

    The hourly value at timestamp t is plotted over the interval [t, t+1),
    using bar(..., align="edge") and stairs(..., edges).
    """
    hr = results.hourly

    # Hour starts, e.g. 0,1,...,23
    h = hr.index.to_numpy(dtype=float)

    # Hour edges, e.g. 0,1,...,24
    edges = np.arange(int(h[0]), int(h[-1]) + 2)

    # Effective prices
    market_price = np.asarray(data.energy_price)
    import_price = np.asarray(data.energy_price) + data.import_tariff
    export_price = np.asarray(data.energy_price) - data.export_tariff

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(13, 11), sharex=True
    )
    # Identify pricing regimes
    # For Q2/Q3 (disutility-based models), consumption_utility is None: skip the
    # regime comparison entirely rather than crash, and shade nothing.
    has_utility = data.consumption_utility is not None
    if has_utility:
        high_utility_regime = data.consumption_utility > import_price
        low_utility_regime = data.consumption_utility < export_price
    else:
        high_utility_regime = np.zeros_like(h, dtype=bool)
        low_utility_regime = np.zeros_like(h, dtype=bool)
    
    # Shade hourly regions
    high_label_added = False
    low_label_added = False
    # ================================================================
    # 1. OPTIMAL SCHEDULE
    # ================================================================
    for i in range(len(h)):

        # u^L > p_imp  -> high-consumption / importing regime
        if high_utility_regime[i]:
            ax1.axvspan(
                edges[i],
                edges[i + 1],
                color="grey",
                alpha=0.15,
                zorder=0,
                label=(
                    r"$u^L > p_t^{imp}$"
                    if not high_label_added else None
                ),
            )
            high_label_added = True
            ax2.axvspan(
                edges[i],
                edges[i + 1],
                color="grey",
                alpha=0.15,
                zorder=0,
                label=(
                    r"$u^L > p_t^{imp}$"
                    if not high_label_added else None
                ),
            )
            high_label_added = True
            ax3.axvspan(
                edges[i],
                edges[i + 1],
                color="grey",
                alpha=0.15,
                zorder=0,
                label=(
                    r"$u^L > p_t^{imp}$"
                    if not high_label_added else None
                ),
            )
            high_label_added = True
    
        # u^L < p_exp  -> low-consumption / exporting regime
        elif low_utility_regime[i]:
            ax1.axvspan(
                edges[i],
                edges[i + 1],
                color="lightcoral",
                alpha=0.15,
                zorder=0,
                label=(
                    r"$u^L < p_t^{exp}$"
                    if not low_label_added else None
                ),
            )
            low_label_added = True
            ax2.axvspan(
                edges[i],
                edges[i + 1],
                color="lightcoral",
                alpha=0.15,
                zorder=0,
                label=(
                    r"$u^L < p_t^{exp}$"
                    if not low_label_added else None
                ),
            )
            low_label_added = True
            ax3.axvspan(
                edges[i],
                edges[i + 1],
                color="lightcoral",
                alpha=0.15,
                zorder=0,
                label=(
                    r"$u^L < p_t^{exp}$"
                    if not low_label_added else None
                ),
            )
            low_label_added = True

    # PV production - yellow bars
    if "pv" in hr.columns:
        ax1.bar(
            h,
            hr["pv"].to_numpy(),
            width=1.0,
            align="edge",
            color="yellow",
            alpha=0.65,
            label="PV production",
            zorder=1,
        )

    # PV availability - thick dashed orange line
    ax1.stairs(
        np.asarray(data.pv_available),
        edges,
        color="orange",
        linestyle="--",
        linewidth=2.8,
        label="PV available",
        zorder=5,
    )

    # Load - solid blue line
    if "load" in hr.columns:
        ax1.stairs(
            hr["load"].to_numpy(),
            edges,
            color="blue",
            linewidth=2.0,
            label="Load",
            zorder=4,
        )

    # Import - black step line + circle markers
    if "import" in hr.columns:
        ax1.stairs(
            hr["import"].to_numpy(),
            edges,
            color="black",
            linewidth=1.5,
            linestyle="--",
            label="Import",
            zorder=6,
        )
        ax1.plot(
            h,
            hr["import"].to_numpy(),
            linestyle="none",
            color="black",
            marker="o",
            markersize=4,
            zorder=7,
        )

    # Export - black step line + x markers, plotted negative
    if "export" in hr.columns:
        export_vals = -hr["export"].to_numpy()
        ax1.stairs(
            export_vals,
            edges,
            color="black",
            linewidth=1.5,
            linestyle="--",
            label="Export",
            zorder=6,
        )
        ax1.plot(
            h,
            export_vals,
            linestyle="none",
            color="black",
            marker="x",
            markersize=5,
            zorder=7,
        )

    # Load limits - plotted last
    ax1.hlines(
        data.load_max_kWh,
        edges[0],
        edges[-1],
        colors="blue",
        linestyles="--",
        linewidth=1.3,
        alpha=0.8,
        label="Max load",
        zorder=8,
    )
    ax1.hlines(
        data.load_min_kWh,
        edges[0],
        edges[-1],
        colors="blue",
        linestyles=":",
        linewidth=1.3,
        alpha=0.8,
        label="Min load",
        zorder=8,
    )

    # Zero line because export is negative
    ax1.axhline(0, color="black", linewidth=0.8, alpha=0.4, zorder=2)

    ax1.set_ylabel("Energy [kWh/h]")
    ax1.set_title("Optimal schedule")
    ax1.grid(alpha=0.2)
    ax1.legend(fontsize=8, ncol=4, loc="center left")

    # ================================================================
    # 2. PRICES, UTILITY AND PV COST
    # ================================================================

    ax2.stairs(
        market_price,
        edges,
        color="black",
        linewidth=1.5,
        label="Market price",
    )
    ax2.stairs(
        import_price,
        edges,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Import price",
    )
    ax2.stairs(
        export_price,
        edges,
        color="green",
        linestyle=":",
        linewidth=1.8,
        label="Export price",
    )
    
    # Utility of consumption - only for Q1-style data (see has_utility above)
    if has_utility:
        ax2.hlines(
            data.consumption_utility,
            edges[0],
            edges[-1],
            colors="blue",
            linestyles="-.",
            linewidth=1.6,
            label="Consumption utility",
        )

    # PV marginal production cost
    ax2.hlines(
        data.pv_marginal_cost,
        edges[0],
        edges[-1],
        colors="orange",
        linestyles="-.",
        linewidth=1.6,
        label="PV marginal cost",
    )

    ax2.set_ylabel("DKK/kWh")
    ax2.set_title("Prices, utility and PV cost")
    ax2.grid(alpha=0.2)
    ax2.legend(fontsize=8, ncol=3, loc="upper center")

    # ================================================================
    # 3. DUAL VARIABLES
    # ================================================================

    dual_cols = [c for c in hr.columns if c.startswith("dual_")]

    for c in dual_cols:
        ax3.stairs(
            hr[c].to_numpy(),
            edges,
            linewidth=1.8,
            label=c.removeprefix("dual_").replace("_", " "),
        )

    # Faint price signals for comparison
    ax3.stairs(
        market_price,
        edges,
        color="grey",
        linewidth=1.0,
        alpha=0.35,
        label="Market price",
    )
    ax3.stairs(
        import_price,
        edges,
        color="grey",
        linestyle="--",
        linewidth=1.0,
        alpha=0.35,
        label="Import price",
    )
    ax3.stairs(
        export_price,
        edges,
        color="grey",
        linestyle=":",
        linewidth=1.0,
        alpha=0.35,
        label="Export price",
    )

    ax3.set_xlabel("Hour")
    ax3.set_ylabel("DKK/kWh")
    ax3.set_title("Dual variables")
    ax3.grid(alpha=0.2)
    ax3.legend(fontsize=8, ncol=4, loc="upper center")

    # ================================================================
    # GENERAL LAYOUT
    # ================================================================

    for ax in (ax1, ax2, ax3):
        ax.set_xlim(edges[0], edges[-1])

    ax3.set_xticks(edges)
    fig.suptitle(f"Optimisation results - {results.question}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    return _finish(fig, save_to)
