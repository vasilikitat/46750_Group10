"""Entry point: load one question's data, build and solve the model, save results and figures.

    python main.py                          # base case of Q1_caseA
    python main.py --question Q2_linear     # another case
    python main.py --scenarios              # also run the example sensitivity scenarios

Results (CSV, TXT, PNG) are written to ``results/<question>/``. Extend ``run_scenarios``
with your own scenarios, or add a new function per question, as your analysis grows.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from src.data_loader import load_question, list_questions
from src.model import FlexibleConsumerModel, LinearDisutilityModel, QuadraticDisutilityModel, MinEnergyConsumerModel, BatteryConsumerModel, Results
from src.plotting import plot_duals, plot_inputs, plot_scenario_comparison, plot_schedule, plot_results_overview, plot_sweep, Emin_sensitivity, plot_Emin_vs_unconstrained, plot_sweep_quadratic
from src.scenarios import scale_prices, scale_pv, set_tariffs, set_linear_disutility, set_quadratic_disutility

RESULTS_DIR = Path(__file__).resolve().parent / "results"

def select_model(data):
    """Pick the correct model class according to the case data needed."""
    if data.battery_capacity_kWh is not None:
        return BatteryConsumerModel          # Q3_battery
    if data.min_daily_energy_kWh is not None:
        return MinEnergyConsumerModel        # Q3
    if data.quadratic_disutility is not None:
        return QuadraticDisutilityModel      # Q2_quadratic
    if data.linear_disutility is not None:
        return LinearDisutilityModel         # Q2_linear
    return FlexibleConsumerModel             # Q1_caseA, Q1_caseB

def run_base_case(question: str, out: Path, show: bool) -> Results | None:
    data = load_question(question)
    print(data.summary(), "\n")
    plot_inputs(data, save_to=out / "inputs.png")

    #Hardcode propably?
    #model = FlexibleConsumerModel(data).build()

    model_cls = select_model(data)
    model = model_cls(data).build()

    try:
        results = model.solve()
    except NotImplementedError as e:
        print(f"[skipped] {e}")
        return None

    print(results, "\n")
    results.save(out)
    plot_schedule(results, data, save_to=out / "schedule.png")
    plot_results_overview(results, data, save_to=out / "results_overview.png")
    plot_duals(results, data, save_to=out / "duals.png")
    if show:
        matplotlib.pyplot.show()
    return results


def run_scenarios(question: str, out: Path) -> dict[str, Results]:
    """Example sensitivity analysis. Replace with the scenarios you design in Question 1.g."""
    base = load_question(question)
    scenarios = {
        "base": base,
        "flat_prices": scale_prices(base, factor=0.0, keep_mean=True),
        "double_spread": scale_prices(base, factor=2.0, keep_mean=True),
        "no_tariffs": set_tariffs(base, import_tariff=0.0, export_tariff=0.0),
        "no_pv": scale_pv(base, factor=0.0),
    }
    runs: dict[str, Results] = {}
    for name, data in scenarios.items():
        results = FlexibleConsumerModel(data).build().solve()
        results.save(out, tag=name)
        runs[name] = results
        print(f"{name:>14}: cost {results.objective:8.2f} DKK | import {results.hourly['import'].sum():5.1f} kWh"
              f" | export {results.hourly['export'].sum():5.1f} kWh")
    plot_scenario_comparison(runs, "objective", save_to=out / "scenarios_cost.png")
    return runs


def sweep_linear_disutility(base_question: str = "Q2_linear", c_L_values=None) -> pd.DataFrame:
    """Sweep the linear disutility coefficient c^L and report summary metrics per run (Question 2.b.iv)."""
    base = load_question(base_question)
    if c_L_values is None:
        c_L_values = np.linspace(0.0, 3.5, 30)

    rows = []
    for c_L in c_L_values:
        data = set_linear_disutility(base, c_L=c_L)
        results = LinearDisutilityModel(data).build().solve()

        at_min = np.isclose(results.hourly["load"], data.load_min_kWh, atol=1e-4)
        wants_more = results.hourly["reference_load"] > data.load_min_kWh

        rows.append({
            "c_L": c_L,
            "procurement_cost": results.procurement_cost,
            "disutility": results.meta["disutility"],
            "daily_load": results.hourly["load"].sum(),
            "daily_pv": results.hourly["pv"].sum(),
            "daily_import": results.hourly["import"].sum(),
            "daily_export": results.hourly["export"].sum(),
            "total_deviation": (results.hourly["dev_pos"] + results.hourly["dev_neg"]).sum(),
            "hours_load_min_binding": int((at_min & wants_more).sum()),
        })

    return pd.DataFrame(rows)


def sweep_quadratic_disutility(base_question: str = "Q2_quadratic", c_Q_values=None) -> pd.DataFrame:
    """Sweep the quadratic disutility coefficient c^Q and report summary metrics per run (Question 2.c.iv)."""
    base = load_question(base_question)
    if c_Q_values is None:
        c_Q_values = np.geomspace(0.01, 20, 30)

    rows = []
    for c_Q in c_Q_values:
        data = set_quadratic_disutility(base, c_Q=c_Q)
        results = QuadraticDisutilityModel(data).build().solve()

        at_min = np.isclose(results.hourly["load"], data.load_min_kWh, atol=1e-4)
        wants_more = results.hourly["reference_load"] > data.load_min_kWh

        rows.append({
            "c_Q": c_Q,
            "procurement_cost": results.procurement_cost,
            "disutility": results.meta["disutility"],
            "daily_load": results.hourly["load"].sum(),
            "daily_pv": results.hourly["pv"].sum(),
            "daily_import": results.hourly["import"].sum(),
            "daily_export": results.hourly["export"].sum(),
            "total_deviation": (results.hourly["load"] - results.hourly["reference_load"]).abs().sum(),
            "hours_load_min_binding": int((at_min & wants_more).sum()),
        })

    return pd.DataFrame(rows)

def sweep_min_daily_energy(question: str, out: Path) -> None:
    """Sensitivity analysis on E^min for Question 3.(f): lambda_t, mu and net utility vs E^min."""
    data = load_question(question)
    fig, df_Emin, regimes = Emin_sensitivity(
        data,
        c_Q=data.quadratic_disutility,     # the Q3 c^Q from the data -> free_hours column
        quadratic_factor=2.0,              # 2 for c^Q (l - l_ref)^2, 1 for 1/2 c^Q (l - l_ref)^2
        save_to=out / "Emin_sweep.png",
    )
    df_Emin.to_csv(out / "Emin_sweep.csv", index=False)
    regimes.to_csv(out / "Emin_regimes.csv", index=False)
    print("\n--- E^min regimes ---")
    print(regimes.round(3).to_string(index=False))
    matplotlib.pyplot.close(fig)


def main() -> None:
    
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--question", default="Q1_caseA", choices=list_questions(), help="data case to use")
    parser.add_argument("--scenarios", action="store_true", help="also run the example sensitivity scenarios")
    parser.add_argument("--sweep", action="store_true", help="also run the c_L (Q2_linear) or c_Q (Q2_quadratic) sweep for Question 2.(b).iv / 2.(c).iv")
    parser.add_argument("--show", action="store_true", help="open the figures in a window")
    
    args = parser.parse_args()

    out = RESULTS_DIR / args.question
    out.mkdir(parents=True, exist_ok=True)
    if not args.show:
        matplotlib.use("Agg")

    base = run_base_case(args.question, out, args.show)
    if args.scenarios and base is not None:
        run_scenarios(args.question, out)
    if args.sweep:
        if args.question == "Q2_quadratic":
            sweep_df = sweep_quadratic_disutility()
            sweep_df.to_csv(out / "cQ_sweep.csv", index=False)
            plot_sweep_quadratic(sweep_df, load_question("Q2_quadratic"), save_to=out / "cQ_sweep.png")
        else:
            base_data = load_question("Q2_linear")
            sweep_df = sweep_linear_disutility()
            sweep_df.to_csv(out / "cL_sweep.csv", index=False)
            plot_sweep(sweep_df, base_data, save_to=out / "cL_sweep.png")
    if base is not None and select_model(load_question(args.question)) is MinEnergyConsumerModel:
        data_q3 = load_question(args.question)
        results_2c = QuadraticDisutilityModel(data_q3).build().solve()   # same data, no E^min
        plot_Emin_vs_unconstrained(base, results_2c, data_q3, save_to=out / "schedule_vs_2c.png")
        sweep_min_daily_energy(args.question, out)
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()
