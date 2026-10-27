"""
Runner for the single-effect LiBr/H2O absorption chiller model.

The cycle model itself lives in components/absorption_chiller.py; this file
only drives it and prints/plots the results, so the component stays pure and
importable from anywhere (an annual simulation, a sweep, a test).

It does two things:
  1. Demonstrates the external-coupling seam: hot water in from a solar loop
     and cooling water in from a tower, mapped to internal cycle temperatures,
     and checks the First Law closes.
  2. Plots a T-s (temperature-entropy) diagram of that cycle under ./pycache/.

Run:
    python absorption_chiller_builder.py

For the checks against the published model (Ghatos et al. 2021, Table 4 and
Figs 3-9), run absorption_chiller_verification.py instead.

Requirements: matplotlib (for the plots) and CoolProp. The water/steam side
of the solver (components.absorption_chiller.solve) calls CoolProp directly
for real IAPWS-95 properties; only the LiBr solution correlations remain
standard-library-only, because CoolProp's own LiBr model uses a different,
incompatible enthalpy reference (see the module note in absorption_chiller.py
above h_sol).
"""

import os

import matplotlib.pyplot as plt

from components.absorption_chiller import (
    Approaches,
    solve,
    state_entropy,
    to_design_spec,
)

# Figures are saved here as PNGs rather than popped up in blocking windows, so
# the script runs straight through; re-running overwrites the previous images.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pycache")

TON_REFRIGERATION_KW = 3.516853  # 1 ton = 3024 kcal/hr, for the Lansing case


def save_fig(name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path)
    plt.close()
    print("saved", path)


def report(result, title):
    """Print the full state table, the duties and the First Law residual."""
    print("=" * 76)
    print(title)
    print("=" * 76)
    s = result.spec
    print(f"  inputs   T_gen={s.T_gen} C  T_evap={s.T_evap} C  "
          f"T_cond={s.T_cond} C  T_abs={s.T_abs} C  "
          f"eps_shx={s.eps_shx}  Q_evap={s.Q_evap} kW")
    print(f"  P_lo = {result.P_lo:7.4f} kPa      P_hi = {result.P_hi:7.4f} kPa")
    print(f"  X_abs= {result.X_abs:7.3f} %LiBr    X_gen= {result.X_gen:7.3f} %LiBr"
          f"    f = {result.f:.4f}")
    print()
    print("   pt      T[C]     P[kPa]    X[%]      h[kJ/kg]     m[kg/s]")
    for i in range(1, 11):
        p = result.states[i]
        print(f"   {i:2d}  {p.T:9.2f} {p.P:10.4f} {p.X:7.3f} "
              f"{p.h:13.3f} {p.m:11.5f}")
    print()
    print(f"  Q_evap={result.Q_evap:8.3f}   Q_gen={result.Q_gen:8.3f}   "
          f"Q_cond={result.Q_cond:8.3f}   Q_abs={result.Q_abs:8.3f}   "
          f"W_pump={result.W_pump:.5f}   [kW]")
    print(f"  COP={result.COP:.4f}    COP_carnot={result.COP_carnot:.4f}    "
          f"relative performance={result.eta_rel:.4f}")

    # The energy balance is the only structural error detector in the model: a
    # wrong enthalpy still gives a plausible COP, but it will not close.
    closed = abs(result.energy_closure) < 1e-9
    print(f"  First Law (Q_gen+Q_evap+W_pump-Q_cond-Q_abs) = "
          f"{result.energy_closure:+.3e} kW   -> "
          f"{'closes' if closed else 'DOES NOT CLOSE -- the model is wrong'}")
    for w in result.warnings:
        print(f"  warning: {w}")
    print()
    return closed


def plot_ts_diagram(result, filename, title):
    """T-s (temperature-entropy) diagram for one CycleResult.

    Two closed loops are drawn: the LiBr solution loop (points 1-6, back to
    1) and the refrigerant loop (points 7-10, back to 7). They are two
    different substances with unrelated entropy scales -- refrigerant
    entropy runs in single digits (kJ/kg.K), solution entropy in tenths --
    so each gets its own x-axis, sharing the y-axis (temperature) both
    loops have in common. Entropy comes from
    components.absorption_chiller.state_entropy (CoolProp); the solver
    itself never touches entropy.
    """
    solution_pts = (1, 2, 3, 4, 5, 6, 1)      # closes back to state 1
    refrigerant_pts = (7, 8, 9, 10, 7)        # closes back to state 7

    def path(indices):
        T = [result.states[i].T for i in indices]
        s = [state_entropy(result.states[i]) for i in indices]
        return T, s

    # Constrained layout makes room for the second (top) x-axis; the default
    # margins only leave space for the title and cut it off.
    fig, ax_sol = plt.subplots(layout="constrained")
    ax_ref = ax_sol.twiny()

    # Points 1/2 (pump work) and 5/6 (solution valve) sit almost on top of
    # each other -- correctly, since both steps are nearly isentropic -- so
    # labels get an offset instead of sitting right on the marker.
    def label(ax, i, T, s, color):
        ax.annotate(str(i), (s, T), color=color, textcoords="offset points",
                   xytext=(6, 6), fontsize=9)

    T_sol, s_sol = path(solution_pts)
    sol_line, = ax_sol.plot(s_sol, T_sol, "o-", color="tab:blue")
    for i in solution_pts[:-1]:
        label(ax_sol, i, result.states[i].T, state_entropy(result.states[i]), "tab:blue")
    ax_sol.set_xlabel("Solution entropy [kJ/(kg.K)]  (points 1-6)", color="tab:blue")
    ax_sol.tick_params(axis="x", colors="tab:blue")

    T_ref, s_ref = path(refrigerant_pts)
    ref_line, = ax_ref.plot(s_ref, T_ref, "s-", color="tab:orange")
    for i in refrigerant_pts[:-1]:
        label(ax_ref, i, result.states[i].T, state_entropy(result.states[i]), "tab:orange")
    ax_ref.set_xlabel("Refrigerant entropy [kJ/(kg.K)]  (points 7-10)", color="tab:orange")
    ax_ref.tick_params(axis="x", colors="tab:orange")

    ax_sol.set_ylabel("Temperature [C]")
    ax_sol.set_title(title)
    ax_sol.grid(True, alpha=0.3)
    # Inside the axes, in the empty space under the loops, so it can't cover
    # the bottom axis label the way a figure-level legend did.
    ax_sol.legend([sol_line, ref_line], ["LiBr solution loop", "Refrigerant loop"],
                  loc="center", bbox_to_anchor=(0.72, 0.3))
    save_fig(filename)


def main():
    # --- 1. Calling the Solver ------------------
    # calling the solver and giving temperature and heat inputs needed
    # T is in C, Q is in KW
    spec = to_design_spec(T_hot_in=95.0, T_cool_in=30.0, T_chw_out=9.0,
                          Q_evap=10.0, approaches=Approaches())
    result = solve(spec)
    report(result,
           "Solar-driven case: 95 C hot water, 30 C cooling water, 9 C chilled")

    # --- 2. T-s diagram of the cycle ------------------------------
    plot_ts_diagram(result, "absorption_ts_diagram.png",
                    "T-s diagram for Absorbtion Chiller")


if __name__ == "__main__":
    main()
