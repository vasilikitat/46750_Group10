"""Optimisation model of a single flexible consumer, implemented with gurobipy.

The class below separates the three steps you will repeat for every question:

    model = FlexibleConsumerModel(data)   # 1. hand over the input data
    model.build()                         # 2. declare variables, objective, constraints
    results = model.solve()               # 3. optimise and collect primal AND dual values

``build()`` is the only method you need to complete for Question 1; the other questions
are variations of it (a different objective, an extra constraint). Copy this file or
subclass ``FlexibleConsumerModel`` and override ``build()`` to keep one model per question.

Two conventions make the dual variables easy to read out afterwards:

* Every constraint family is stored in ``self.con`` under a descriptive name, e.g.
  ``self.con["balance"] = self.m.addConstrs(...)``. ``solve()`` then returns the dual value
  (shadow price, Gurobi attribute ``Pi``) of every constraint in ``self.con`` automatically.
* Bounds that you want a dual for must be written as explicit constraints (``addConstr``),
  not as variable bounds (``lb=``/``ub=``). Gurobi reports the sensitivity of a variable
  bound in the reduced cost (``RC``), not in ``Pi``.
* Duals of quadratic constraints (``m.addQConstr``) are read from ``QCPi`` and require ``QCPDual = 1`` (set below).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from .data_loader import InputData


@dataclass
class Results:
    """Primal and dual solution of one model run."""

    question: str
    status: str
    objective: float
    hourly: pd.DataFrame                   # one row per hour: variables, prices, hourly duals
    utility: float                         
    procurement_cost: float                
    duals: dict[str, float] = field(default_factory=dict)   # duals of non-hourly constraints
    meta: dict = field(default_factory=dict)                 # anything else worth keeping (scenario name, ...)

    def save(self, folder: Path | str, tag: str = "") -> None:
        """Write ``hourly`` to CSV and the scalar values to a small text file."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"{self.question}{'_' + tag if tag else ''}"
        self.hourly.to_csv(folder / f"{stem}_hourly.csv", index_label="hour")
        with open(folder / f"{stem}_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"status    : {self.status}\nobjective : {self.objective:.4f} DKK\n")
            for k, v in self.duals.items():
                f.write(f"dual[{k}] : {v:.4f}\n")

    def __str__(self) -> str:
        cols = [c for c in self.hourly.columns if not c.startswith("dual_")]
        return (
            f"status: {self.status} | objective: {self.objective:.2f} DKK "
            f"(utility: {self.utility:.2f} DKK, procurement cost: {self.procurement_cost:.2f} DKK)\n"
            f"daily totals (kWh): " + ", ".join(f"{c}={self.hourly[c].sum():.1f}" for c in cols if c in ("import", "export", "load", "pv"))
            + (f"\nduals: {self.duals}" if self.duals else "")
        )


def _dual(c) -> float:
    """Dual value of a linear (``Pi``) or quadratic (``QCPi``, requires QCPDual=1) constraint."""
    return c.QCPi if isinstance(c, gp.QConstr) else c.Pi


class FlexibleConsumerModel:
    """Consumption problem of one consumer over a 24-hour horizon (Question 1); extend it for Questions 2 and 3."""

    def __init__(self, data: InputData, name: str = "flexible_consumer", verbose: bool = False):
        self.data = data
        self.T = range(data.n_hours)
        self.m = gp.Model(name)
        self.m.Params.OutputFlag = 1 if verbose else 0
        self.m.Params.QCPDual = 1          # only relevant if you add a quadratic constraint (none is needed in Assignment 1)
        self.var: dict[str, gp.tupledict | gp.Var] = {}   # decision variables by name
        self.con: dict[str, gp.tupledict | gp.Constr] = {}  # constraints by name (duals read from here)

    # ------------------------------------------------------------------ 2. build
    def build(self) -> "FlexibleConsumerModel":
        """Declare decision variables, objective and constraints.

        TODO (Question 1): complete this method with the variables, objective and constraints
        of the problem you formulated in Question 1. Keep the naming pattern below so
        that ``solve()`` can return the primal and dual values automatically.
        """
        d, m, T = self.data, self.m, self.T

        # --- Decision variables --------------------------------------------------------
        # TODO: identify and declare the decision variables of your formulation.
        # Store every variable family in self.var["<name>"]: solve() then returns its hourly
        # values automatically as a column of results.hourly.
        # Pattern for hourly variables (one per hour):
        #   self.var["<name>"] = m.addVars(T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="<name>")
        # Pattern for a single (daily) variable:
        #   self.var["<name>"] = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="<name>")
        # Notes:
        # * gurobipy indexes the names automatically: name="<name>" in addVars(T, ...) creates
        #   <name>[0], <name>[1], ..., <name>[23] - no need to build per-hour names yourself.
        # * vtype: the same format takes GRB.BINARY or GRB.INTEGER if you ever need them (the
        #   problem then becomes a MILP and dual values are no longer defined; solve() skips them).
        # * lb defaults to 0 in gurobipy: a free variable needs an explicit lb=-GRB.INFINITY, and
        #   a bound you want a dual for must be an explicit constraint, not lb=/ub= (see the README).
        # * naming the families "import", "export", "load", "pv" makes the standard plots of
        #   src/plotting.py work out of the box.

        self.var["load"] = m.addVars(T, lb=0, name="load")
        # how much to consume 
        self.var["pv"] = m.addVars(T, lb=0, name="pv")
        # how much to use from the sun
        self.var["import"] = m.addVars(T, lb=0, name="import")
        # how much to buy
        self.var["export"] = m.addVars(T, lb=0, name="export")
        #how much to sell
        
        # --- Objective ---------------------------------------------------------------
        # TODO: express the objective function and its direction (GRB.MINIMIZE or GRB.MAXIMIZE):
        #   m.setObjective(gp.quicksum(<expression in t> for t in T), <direction>)
        # The input-data attributes (with units) are documented in src/data_loader.py (InputData).

        
        m.setObjective(
    gp.quicksum(
        d.consumption_utility*self.var["load"][t]- d.pv_marginal_cost*self.var["pv"][t]-(d.energy_price[t] + d.import_tariff)*self.var["import"][t]+(d.energy_price[t] - d.export_tariff)*self.var["export"][t]
        for t in T
    ),
    GRB.MAXIMIZE
)
        # --- Constraints -------------------------------------------------------------
        # TODO: add the constraints of your formulation.
        # Pattern for hourly constraints (one per hour, duals returned as a 24-vector; names are
        # indexed automatically, like for the variables):
        #   self.con["<name>"] = m.addConstrs(
        #       (<lhs expression> - <rhs expression> <= 0 for t in T), name="<name>")
        # Pattern for a single constraint (dual returned as a scalar):
        #   self.con["<name>"] = m.addConstr(<lhs expression> - <rhs expression> <= 0, name="<name>")

        self.con["balance"] = m.addConstrs(
    (self.var["load"][t] == self.var["pv"][t] + self.var["import"][t] - self.var["export"][t] for t in T), name="balance")

        self.con["pv_limit"] = m.addConstrs((self.var["pv"][t] <= d.pv_available[t] for t in T) , name="pv_limit")
        self.con["load_min"] = m.addConstrs((self.var["load"][t] >= d.load_min_kWh for t in T) , name="load_min")
        self.con["load_max"] = m.addConstrs((self.var["load"][t] <= d.load_max_kWh for t in T) , name="load_max")        
        m.update()
        return self

    # ------------------------------------------------------------------ 3. solve
    def solve(self) -> Results:
        """Optimise and return primal values, objective and dual values."""
        m = self.m
        m.update()
        if m.NumConstrs == 0 and m.NumQConstrs == 0:
            raise NotImplementedError(
                "The model has no constraints: complete FlexibleConsumerModel.build() in src/model.py first."
            )
        m.optimize()
        status = _status_name(m.Status)
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Optimisation ended with status {status}. Check the model (m.computeIIS() helps for infeasibility).")
        return self._extract_results(status)

    # --------------------------------------------------------------- extraction
    def _extract_results(self, status: str) -> Results:
        d, T = self.data, list(self.T)
        hourly = pd.DataFrame(index=pd.Index(T, name="hour"))
        hourly["price"] = d.energy_price
        hourly["pv_available"] = d.pv_available
        if d.reference_load is not None:
            hourly["reference_load"] = d.reference_load

        # Primal values: every hourly variable family in self.var becomes a column
        for name, v in self.var.items():
            if isinstance(v, gp.tupledict):
                hourly[name] = [v[t].X for t in T]
        scalars = {name: v.X for name, v in self.var.items() if isinstance(v, gp.Var)}

        # Dual values: every constraint family in self.con becomes a 'dual_<name>' column or scalar
        duals: dict[str, float] = {}
        for name, c in self.con.items():
            try:
                if isinstance(c, gp.tupledict):
                    hourly[f"dual_{name}"] = [_dual(c[t]) for t in T]
                else:
                    duals[name] = _dual(c)
            except (AttributeError, gp.GurobiError):
                # No duals available (e.g. model with integer variables)
                pass
                
        utility = (d.consumption_utility * hourly["load"]).sum()
        procurement_cost = (d.pv_marginal_cost * hourly["pv"]
            + (hourly["price"] + d.import_tariff) * hourly["import"]
            - (hourly["price"] - d.export_tariff) * hourly["export"]
        ).sum()

        return Results(
            question=d.question,
            status=status,
            objective=self.m.ObjVal,
            hourly=hourly,
            duals=duals,
            utility=utility,
            procurement_cost=procurement_cost,
            meta={"scalar_variables": scalars},
        )


_STATUS = {
    GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE", GRB.UNBOUNDED: "UNBOUNDED",
    GRB.INF_OR_UNBD: "INF_OR_UNBD", GRB.TIME_LIMIT: "TIME_LIMIT", GRB.SUBOPTIMAL: "SUBOPTIMAL",
    GRB.NUMERIC: "NUMERIC", GRB.INTERRUPTED: "INTERRUPTED",
}


def _status_name(code: int) -> str:
    return _STATUS.get(code, f"STATUS_{code}")


class LinearDisutilityModel(FlexibleConsumerModel):
    """Question 2.(b): linear disutility from deviating (in absolute value) from a reference load profile.

    The non-differentiable |load - reference_load| term is reformulated as an LP using two
    continuous auxiliary variables, dev_pos and dev_neg (overshoot and undershoot), linked to
    load via load = reference_load + dev_pos - dev_neg. No complementarity constraint is needed:
    since both enter the objective with a positive cost, the optimizer never sets both positive.
    """

    def build(self) -> "LinearDisutilityModel":
        d, m, T = self.data, self.m, self.T

        self.var["load"] = m.addVars(T, lb=0, name="load")
        self.var["pv"] = m.addVars(T, lb=0, name="pv")
        self.var["import"] = m.addVars(T, lb=0, name="import")
        self.var["export"] = m.addVars(T, lb=0, name="export")
        self.var["dev_pos"] = m.addVars(T, lb=0, name="dev_pos")
        self.var["dev_neg"] = m.addVars(T, lb=0, name="dev_neg")

        m.setObjective(
            gp.quicksum(
                -d.linear_disutility * (self.var["dev_pos"][t] + self.var["dev_neg"][t])
                - d.pv_marginal_cost * self.var["pv"][t]
                - (d.energy_price[t] + d.import_tariff) * self.var["import"][t]
                + (d.energy_price[t] - d.export_tariff) * self.var["export"][t]
                for t in T
            ),
            GRB.MAXIMIZE
        )

        self.con["balance"] = m.addConstrs(
            (self.var["load"][t] == self.var["pv"][t] + self.var["import"][t] - self.var["export"][t] for t in T),
            name="balance")
        self.con["pv_limit"] = m.addConstrs((self.var["pv"][t] <= d.pv_available[t] for t in T), name="pv_limit")
        self.con["load_min"] = m.addConstrs((self.var["load"][t] >= d.load_min_kWh for t in T), name="load_min")
        self.con["load_max"] = m.addConstrs((self.var["load"][t] <= d.load_max_kWh for t in T), name="load_max")
        self.con["deviation_link"] = m.addConstrs(
            (self.var["load"][t] == d.reference_load[t] + self.var["dev_pos"][t] - self.var["dev_neg"][t] for t in T),
            name="deviation_link")

        m.update()
        return self

    def _extract_results(self, status: str) -> Results:
        d, T = self.data, list(self.T)
        hourly = pd.DataFrame(index=pd.Index(T, name="hour"))
        hourly["price"] = d.energy_price
        hourly["pv_available"] = d.pv_available
        hourly["reference_load"] = d.reference_load

        for name, v in self.var.items():
            if isinstance(v, gp.tupledict):
                hourly[name] = [v[t].X for t in T]
        scalars = {name: v.X for name, v in self.var.items() if isinstance(v, gp.Var)}

        duals: dict[str, float] = {}
        for name, c in self.con.items():
            try:
                if isinstance(c, gp.tupledict):
                    hourly[f"dual_{name}"] = [_dual(c[t]) for t in T]
                else:
                    duals[name] = _dual(c)
            except (AttributeError, gp.GurobiError):
                pass

        disutility = (d.linear_disutility * (hourly["dev_pos"] + hourly["dev_neg"])).sum()
        procurement_cost = (d.pv_marginal_cost * hourly["pv"]
            + (hourly["price"] + d.import_tariff) * hourly["import"]
            - (hourly["price"] - d.export_tariff) * hourly["export"]
        ).sum()

        return Results(
            question=d.question,
            status=status,
            objective=self.m.ObjVal,
            hourly=hourly,
            duals=duals,
            utility=-disutility,
            procurement_cost=procurement_cost,
            meta={"scalar_variables": scalars, "disutility": disutility},
        )


class QuadraticDisutilityModel(FlexibleConsumerModel):
    """Question 2.(c): quadratic disutility from deviating from a reference load profile."""

    def build(self) -> "QuadraticDisutilityModel":
        d, m, T = self.data, self.m, self.T

        self.var["load"] = m.addVars(T, lb=0, name="load")
        self.var["pv"] = m.addVars(T, lb=0, name="pv")
        self.var["import"] = m.addVars(T, lb=0, name="import")
        self.var["export"] = m.addVars(T, lb=0, name="export")

        m.setObjective(
            gp.quicksum(
                -d.quadratic_disutility*(self.var["load"][t] - d.reference_load[t])**2
                - d.pv_marginal_cost*self.var["pv"][t]
                - (d.energy_price[t] + d.import_tariff)*self.var["import"][t]
                + (d.energy_price[t] - d.export_tariff)*self.var["export"][t]
                for t in T
            ),
            GRB.MAXIMIZE
        )

        self.con["balance"] = m.addConstrs(
            (self.var["load"][t] == self.var["pv"][t] + self.var["import"][t] - self.var["export"][t] for t in T),
            name="balance")
        self.con["pv_limit"] = m.addConstrs((self.var["pv"][t] <= d.pv_available[t] for t in T), name="pv_limit")
        self.con["load_min"] = m.addConstrs((self.var["load"][t] >= d.load_min_kWh for t in T), name="load_min")
        self.con["load_max"] = m.addConstrs((self.var["load"][t] <= d.load_max_kWh for t in T), name="load_max")

        m.update()
        return self
    
    def _extract_results(self, status: str) -> Results:
        d, T = self.data, list(self.T)
        hourly = pd.DataFrame(index=pd.Index(T, name="hour"))
        hourly["price"] = d.energy_price
        hourly["pv_available"] = d.pv_available
        hourly["reference_load"] = d.reference_load

        for name, v in self.var.items():
            if isinstance(v, gp.tupledict):
                hourly[name] = [v[t].X for t in T]
        scalars = {name: v.X for name, v in self.var.items() if isinstance(v, gp.Var)}

        duals: dict[str, float] = {}
        for name, c in self.con.items():
            try:
                if isinstance(c, gp.tupledict):
                    hourly[f"dual_{name}"] = [_dual(c[t]) for t in T]
                else:
                    duals[name] = _dual(c)
            except (AttributeError, gp.GurobiError):
                pass

        disutility = (d.quadratic_disutility * (hourly["load"] - hourly["reference_load"])**2).sum()
        procurement_cost = (d.pv_marginal_cost * hourly["pv"]
            + (hourly["price"] + d.import_tariff) * hourly["import"]
            - (hourly["price"] - d.export_tariff) * hourly["export"]
        ).sum()

        return Results(
            question=d.question,
            status=status,
            objective=self.m.ObjVal,
            hourly=hourly,
            duals=duals,
            utility=-disutility,
            procurement_cost=procurement_cost,
            meta={"scalar_variables": scalars, "disutility": disutility},
        )
