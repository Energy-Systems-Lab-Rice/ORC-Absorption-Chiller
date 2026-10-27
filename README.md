# ORCSolarOld

Steady-state thermodynamic models of two solar-driven heat engines:

- **Organic Rankine Cycle (ORC)** with a regenerator (HXGR), for turning solar heat into
  power. You can compare working fluids such as heptane and octane.
- **Single-effect LiBr/H2O absorption chiller**, for turning solar heat into cooling.
  It is checked against the published model it is based on.

All fluid properties come from [CoolProp](http://www.coolprop.org/).

## Setup

Requires Python 3.10 or newer.

```bash
pip install CoolProp numpy matplotlib
```

## Running

Each script runs straight through and saves its figures as PNGs in `pycache/`. That
folder is created automatically, and re-running a script overwrites the old images.

| Script | What it does | Output in `pycache/` |
|---|---|---|
| `python orc_system_builder.py` | ORC efficiency vs turbine inlet temperature for several fluids, and a T-s diagram of each fluid's cycle | `efficiency_vs_inlet_temp_fluids.png`, `ts_diagram_fluids.png` |
| `python orc_efficiency_plot.py` | Efficiency vs inlet temperature for n-heptane, plotted against a set of calibration points | `efficiency_vs_inlet_temp_NHEPTANE.png` |
| `python absorption_chiller_builder.py` | Solves one solar-driven chiller case (95 °C hot water, 30 °C cooling water, 9 °C chilled water), prints every state point, the heat duties, the COP and an energy-balance check, and draws the cycle's T-s diagram | `absorption_ts_diagram.png` |
| `python absorption_chiller_verification.py` | Checks the chiller model against Ghatos et al. (2021): Table 4 and Figs 3-9, plus energy balance and Carnot checks. Prints pass/fail and exits non-zero on failure | `verify_fig3_*.png` ... `verify_fig9_*.png` |

To change the inputs (fluids, temperatures, efficiencies, cooling load), edit the values
in each script's `main()`.

## Layout

```
components/
  orc_reheat.py           ORC cycle solver: OrcCycle(T1, T4, fluid, ...)
  absorption_chiller.py   Absorption chiller solver: solve(DesignSpec(...)) -> CycleResult
plots/
  TsPlot.py               Saturation dome for the ORC T-s diagram
  efficiencyPlot.py       ORC efficiency vs inlet temperature sweep
*_builder.py, *_plot.py   Runner scripts (see table above)
absorption_chiller_verification.py
pycache/                  Generated figures
```

The solvers in `components/` only calculate. They don't print or plot anything, so they
can be imported into other scripts. The runner scripts handle the printing and plotting.

## Using the absorption chiller model in your own code

```python
from components.absorption_chiller import DesignSpec, solve

r = solve(DesignSpec(T_gen=85, T_evap=5, T_cond=35, T_abs=35, Q_evap=10))
print(r.COP, r.Q_gen, r.Q_cond, r.Q_abs)
```

The model uses SI units throughout: temperatures in °C, pressures in kPa, heat flows in
kW, and LiBr concentrations in percent (0-100).

`solve()` raises an error if the inputs describe a cycle that can't work. One example
is a generator too cold to separate any refrigerant. Another is a solution above 65 % LiBr,
where the LiBr would crystallize. To sweep through the crystallization region and get
warnings instead of errors, pass `strict=False`.

`to_design_spec()` builds the inputs from external water temperatures (hot water,
cooling water, chilled water) instead of internal cycle temperatures.

## References

- Lansing, F.L. (1976). *Computer Modeling of a Single-Stage Lithium Bromide/Water
  Absorption Refrigeration Unit.* JPL DSN Progress Report 42-32, 247-257.
- Ghatos, S., Taha-Janan, M., Mehdari, A. (2021). *Thermodynamic model of a single stage
  H2O-LiBr absorption cooling.* E3S Web of Conferences 234, 00091.
  https://doi.org/10.1051/e3sconf/202123400091
