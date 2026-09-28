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

# ================================================================================================
# Question 3.(f) — sensitivity analysis on the minimum daily energy requirement E^min
# ================================================================================================
# Only for the Q3 consumer (quadratic disutility + Σ_t l_t >= E^min). Three panels, as in the Week-4
# exercise "Nodal prices and congestion":
#   1. λ_t at representative hours vs E^min   (analogue of the nodal prices λ_i)
#   2. μ of the E^min requirement vs E^min     (analogue of the line multipliers ν)
#   3. net daily utility vs E^min              (analogue of the total cost)
# The E^min ranges in which λ_t and μ keep the same behaviour ("regimes") are detected automatically.
 
def _check_q3(data: InputData) -> None:
    if data.question != "Q3":
        raise ValueError(f"The E^min sensitivity analysis applies to the Q3 model only (got {data.question}).")
 
 
def sweep_min_daily_energy(data: InputData, E_min_values) -> pd.DataFrame:
    """Solve the Q3 model for each E^min (everything else as in ``data``). One row per value: objective,
    μ, daily load, hours above the reference, and λ_t for all hours (columns ``lambda_<t>``)."""
    _check_q3(data)
    from .model import MinEnergyConsumerModel          # local imports: only needed for Q3
    from .scenarios import set_load_preferences
 
    rows = []
    for E_min in E_min_values:
        data_i = set_load_preferences(data, min_daily_energy_kWh=E_min)
        results = MinEnergyConsumerModel(data_i).build().solve()
        hr = results.hourly
        row = {
            "E_min": E_min,
            "objective": results.objective,
            "mu": abs(results.duals.get("min_daily_energy", 0.0)),
            "daily_load": hr["load"].sum(),
            "hours_above_reference": int((hr["load"] > hr["reference_load"] + 1e-6).sum()),
        }
        row.update({f"lambda_{t}": v for t, v in enumerate(hr["dual_balance"].to_numpy())})
        rows.append(row)
    return pd.DataFrame(rows)
 
 
def detect_Emin_regimes(
    df: pd.DataFrame,
    c_Q: float | None = None,
    quadratic_factor: float = 2.0,
    lam_tol: float = 1e-4,
    slope_tol: float = 1e-3,
    ramp_tol: float = 0.05,
) -> pd.DataFrame:
    """Regimes of E^min in which (i) the slope dμ/dE^min and (ii) the set of RAMPING hours stay the same.
 
    A ramping hour has its load pinned (e.g. PV fully used, no import), so its λ_t moves one-for-one with μ.
    Any other movement of λ_t is a non-unique dual (degenerate hour): reported, but it does not define a regime.
    If ``c_Q`` is given, ``free_hours`` = quadratic_factor·c_Q / (dμ/dE^min) is the number of hours absorbing
    the requirement (quadratic_factor = 2 for c^Q (l−ℓ^ref)², 1 for ½ c^Q (l−ℓ^ref)²)."""
    lam_cols = [c for c in df.columns if c.startswith("lambda_")]
    E, mu = df["E_min"].to_numpy(), df["mu"].to_numpy()
    lam = df[lam_cols].to_numpy()
    dE, dmu, dlam = np.diff(E), np.diff(mu), np.diff(lam, axis=0)
    slopes = dmu / dE
    hours = np.array([int(c.removeprefix("lambda_")) for c in lam_cols])
 
    def same(a, b):
        return a["ramps"] == b["ramps"] and abs(a["slope"] - b["slope"]) <= slope_tol * max(1.0, abs(b["slope"]))
 
    groups = []
    for k in range(len(dE)):
        moving = np.abs(dlam[k]) > lam_tol
        ramp = moving & (dmu[k] > lam_tol) & (np.abs(dlam[k] - dmu[k]) <= ramp_tol * dmu[k])
        g = dict(first=k, last=k, slope=slopes[k], ramps=frozenset(hours[ramp].tolist()),
                 nonunique=hours[moving & ~ramp].tolist())
        if groups and same(g, groups[-1]):
            groups[-1]["last"] = k
            groups[-1]["nonunique"] += g["nonunique"]
        else:
            groups.append(g)
 
    # an interval containing a breakpoint mixes two regimes -> drop it, then re-merge identical neighbours
    groups = [g for i, g in enumerate(groups) if not (g["first"] == g["last"] and 0 < i < len(groups) - 1)]
    merged = []
    for g in groups:
        if merged and same(g, merged[-1]):
            merged[-1]["last"] = g["last"]
            merged[-1]["nonunique"] += g["nonunique"]
        else:
            merged.append(g)
 
    rows = []
    for i, g in enumerate(merged):
        start = E[0] if i == 0 else 0.5 * (E[merged[i - 1]["last"] + 1] + E[g["first"]])
        end = E[-1] if i == len(merged) - 1 else 0.5 * (E[g["last"] + 1] + E[merged[i + 1]["first"]])
        free = (round(quadratic_factor * c_Q / g["slope"], 1)
                if c_Q is not None and g["slope"] > slope_tol else None)
        rows.append(dict(
            E_from=start, E_to=end,
            mu_from=np.interp(start, E, mu), mu_to=np.interp(end, E, mu),
            dmu_dE=g["slope"], free_hours=free,
            ramping_hours=sorted(g["ramps"]),
            # flagged in >= 2 intervals (a ramp ending mid-interval gives a single spurious flag)
            nonunique_lambda=sorted({h for h in g["nonunique"] if g["nonunique"].count(h) >= 2} - g["ramps"]),
        ))
    return pd.DataFrame(rows)
 
 
def plot_Emin_sweep(
    df: pd.DataFrame,
    regimes: pd.DataFrame,
    representative_hours: dict[int, str],
    save_to: Path | str | None = None,
) -> plt.Figure:
    """Three panels vs E^min: λ_t at representative hours, μ, net daily utility. Every other regime is
    shaded and each breakpoint is marked with a dotted line."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 3.8))
    x = df["E_min"].to_numpy()
    styles = [dict(color="#4C78A8", ls="-", lw=2.5), dict(color="#54A24B", ls="--", lw=2),
              dict(color="#E45756", ls=":", lw=2), dict(color="#B279A2", ls="-.", lw=2),
              dict(color="#F58518", ls="-", lw=1.5), dict(color="#72B7B2", ls="--", lw=1.5)]
 
    for (h, label), st in zip(representative_hours.items(), styles):
        ax1.plot(x, df[f"lambda_{h}"], label=label, **st)
    ax1.set_title(r"Hourly prices $\lambda_t$ [DKK/kWh]")
 
    ax2.plot(x, df["mu"], color="#E45756", lw=2, label=r"$\mu$ ($E^{min}$ requirement)")
    ax2.set_title(r"Multiplier $\mu$ of $E^{min}$ [DKK/kWh]")
 
    ax3.plot(x, df["objective"], color="#4C78A8", lw=2, label="net daily utility")
    ax3.set_title("Net daily utility [DKK]")
 
    for ax in (ax1, ax2, ax3):
        for i, r in regimes.iterrows():
            if i % 2:
                ax.axvspan(r["E_from"], r["E_to"], color="grey", alpha=0.08, lw=0)
            if i > 0:
                ax.axvline(r["E_from"], color="grey", ls=":", lw=0.8)
        ax.set_xlabel(r"$E^{min}$ [kWh]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    return _finish(fig, save_to)
 
 
def Emin_sensitivity(
    data: InputData,
    E_max: float = 60.0,
    step: float = 0.25,
    representative_hours: dict[int, str] | None = None,
    c_Q: float | None = None,
    quadratic_factor: float = 2.0,
    save_to: Path | str | None = None,
) -> tuple[plt.Figure, pd.DataFrame, pd.DataFrame]:
    """Q3.(f) in one call: sweep E^min on a uniform grid (plus the assignment's E^min), detect the regimes,
    plot the three panels. Returns (figure, sweep table, regime table). Raises for any question other than Q3."""
    _check_q3(data)
    if representative_hours is None:
        representative_hours = {17: r"$\lambda_{17}$", 5: r"$\lambda_{5}$",
                                16: r"$\lambda_{16}$", 19: r"$\lambda_{19}$ (unaffected)"}
    E_min_values = np.unique(np.append(np.arange(0, E_max + step, step), data.min_daily_energy_kWh))
    df = sweep_min_daily_energy(data, E_min_values)
    regimes = detect_Emin_regimes(df, c_Q=c_Q, quadratic_factor=quadratic_factor)
    fig = plot_Emin_sweep(df, regimes, representative_hours, save_to=save_to)
    return fig, df, regimes

def plot_Emin_vs_unconstrained(
    results_q3: Results,
    results_2c: Results,
    data: InputData,
    save_to: Path | str | None = None,
) -> plt.Figure:
    """Question 3.(e), same style as ``plot_schedule``: optimal Q3 schedule (load, PV used, net import) against
    the reference profile and the load of the unconstrained consumer of 2.(c) on the same data, with the
    effective prices and the multiplier μ of the E^min requirement on the second axis."""
    _check_q3(data)
    hr = results_q3.hourly
    h = hr.index.to_numpy()
    mu = abs(results_q3.duals.get("min_daily_energy", 0.0))
    fig, ax = plt.subplots(figsize=(11, 4.2))
 
    width = 0.8
    ax.bar(h, hr["load"], width, color="C0", alpha=0.7, label=f"load with $E^{{min}}$ ({hr['load'].sum():.1f} kWh)")
    if "pv" in hr:
        ax.bar(h, -hr["pv"], width, color="orange", alpha=0.7, label="PV used (negative = generation)")
    ax.step(h, -np.asarray(data.pv_available), where="mid", color="orange", ls="--", lw=1, label="PV available")
    if "import" in hr and "export" in hr:
        ax.plot(h, hr["import"] - hr["export"], "k.-", label="net import (+) / export (-)")
    ax.step(h, np.asarray(data.reference_load), where="mid", color="C0", ls=":", lw=1.8,
            label=f"reference load ({np.sum(data.reference_load):.1f} kWh)")
    ax.step(h, results_2c.hourly["load"].to_numpy(), where="mid", color="#54A24B", ls="--", lw=1.8,
            label=f"2.(c) load, no $E^{{min}}$ ({results_2c.hourly['load'].sum():.1f} kWh)")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set(xlabel="hour", ylabel="kWh/h",
           title=f"Optimal schedule - Q3 vs 2.(c) ($E^{{min}}$ = {data.min_daily_energy_kWh:.0f} kWh)")
 
    ax2 = ax.twinx()
    ax2.step(h, data.energy_price, where="mid", color="C3", lw=1.2, label="energy price")
    ax2.step(h, data.energy_price + data.import_tariff, where="mid", color="red", ls="--", lw=1, label="price + import tariff")
    ax2.step(h, data.energy_price - data.export_tariff, where="mid", color="red", ls=":", lw=1, label="price - export tariff")
    ax2.axhline(mu, color="purple", ls="-.", lw=1.4, label=rf"$\mu$ = {mu:.2f} DKK/kWh")
    ax2.set_ylabel("DKK/kWh", color="C3")
 
    lines, labels = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(lines + l2, labels + lb2, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    return _finish(fig, save_to)
 