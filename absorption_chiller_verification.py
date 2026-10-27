"""
Verification of components/absorption_chiller.py against the published model it
is built from:

  Ghatos, S., Taha-Janan, M., Mehdari, A. (2021) "Thermodynamic model of a
  single stage H2O-LiBr absorption cooling", E3S Web of Conf. 234, 00091.

It runs four independent checks and exits non-zero if any fails:
  1. Table 4 -- all ten state points (T, P, X, h, m) at the paper's design
     point, compared one by one.
  2. Figs 3-9 -- every parametric sweep in the paper, re-run through solve()
     and overlaid on the paper's curves. Each figure is saved under ./pycache/
     as verify_fig*.png and scored by RMS / max deviation.
  3. First Law -- the energy balance closes at every point of every sweep.
  4. Second Law -- COP stays below the Carnot COP at every point.

Run:
    python absorption_chiller_verification.py

Operating conditions. The paper states only some of the conditions behind each
figure, so the missing ones were inferred by fitting this model to the paper's
own curves, and they are cross-checked: the figures overlap (Fig 4 at
T_cond=40 is Fig 5 at T_evap=5 is Fig 7 at E=0.7 is Fig 8's Q_gen), and they
all agree on one base case:
    T_evap = 5 C, T_cond = 40 C, T_abs = 35 C, E = 0.7, Q_evap = 1.5 kW
  * T_abs = 35 C: the text says "evaporator temperature fixed at 5 C & 35 C";
    the 35 is the absorber. T_abs = 35 fits Fig 4 to 0.002 RMS; 30 or 32 do
    not come close.
  * Fig 6: the text says T_cond = 35 C, but the plotted data only fits
    T_cond = 40 C (0.004 RMS against 0.06 for 35) -- including the collapse to
    COP 0.05 at T_abs = 39 C. Treated as a typo in the text.
  * Q_evap = 1.5 kW: read off Figs 8 and 9 (and consistent with the 0.0006
    kg/s refrigerant flow in Table 4).

The paper's curves were digitized by eye from the published figures, so they
carry about +/-0.003 COP and +/-0.005 kW of reading error. The tolerances below
allow for that, plus one known modelling difference: the paper computes the
superheated vapour enthalpy h7 from an unstated fit that runs ~16 kJ/kg above
CoolProp's IAPWS-95, which lowers its Q_cond by ~0.01 kW and its COP by ~0.004.
"""

import sys

import matplotlib.pyplot as plt

from absorption_chiller_builder import save_fig
from components.absorption_chiller import (
    X_SAFE_MAX,
    DesignSpec,
    NoSeparationError,
    T_from_h,
    solve,
)

BASE = dict(T_evap=5.0, T_cond=40.0, T_abs=35.0, eps_shx=0.7, Q_evap=1.5)

COP_TOL = 0.01      # max |model - paper| COP at any digitized point
Q_TOL = 0.02        # kW, same for the heat duties

# Fixed categorical order: the three generator temperatures (or condenser
# temperatures in Fig 3, or the four duties in Figs 8-9) always take these
# slots in this order.
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")

# ---------------------------------------------------------------------------
# Published data
# ---------------------------------------------------------------------------

# Table 4, as printed: point -> (T [C], P [kPa], X [%LiBr], h [kJ/kg], m [kg/s]).
# Paper numbering differs from Lansing's, which this model uses:
TABLE4 = {
    1: (32.0, 4.7576, 0.0, 133.92, 0.0006),        # condenser outlet
    2: (5.0, 0.8725, 0.0, 133.92, 0.0006),         # after refrigerant valve
    3: (5.0, 0.8725, 0.0, 2509.93, 0.0006),        # evaporator outlet
    4: (28.0, 0.8725, 51.1061, 57.7252, 0.0044),   # absorber outlet
    5: (28.0, 4.7576, 51.1061, 57.7276, 0.0044),   # pump outlet
    6: (56.0, 4.7576, 51.1061, 112.32, 0.0044),    # SHX outlet -> generator
    7: (75.0, 4.7576, 0.0, 2657.30, 0.0006),       # vapour from generator
    8: (75.0, 4.7576, 59.6910, 181.91, 0.0038),    # generator outlet
    9: (42.1, 4.7576, 59.6910, 118.14, 0.0038),    # SHX outlet -> absorber
    10: (42.1, 0.8725, 59.6910, 118.14, 0.0038),   # after solution valve
}
PAPER_TO_MODEL = {1: 8, 2: 9, 3: 10, 4: 1, 5: 2, 6: 3, 7: 7, 8: 4, 9: 5, 10: 6}
TABLE4_SPEC = DesignSpec(T_gen=75.0, T_evap=5.0, T_cond=32.0, T_abs=28.0,
                         eps_shx=0.7, Q_evap=1.5)

# Table 4 entries the model is not expected to reproduce, and why.
TABLE4_KNOWN = {
    (6, "T"): "the paper's own h6 = 112.32 inverts to 53.2 C, not 56",
    (7, "h"): "paper uses an unstated superheat fit; this model uses IAPWS-95",
}

# Digitized figure data: {series value: [paper COP or Q at each x]}.
TG = (80.0, 90.0, 100.0)

FIG3_X = [80.0 + 2.0 * i for i in range(11)]                    # T_gen
FIG3 = {34.0: [0.767, 0.771, 0.774, 0.775, 0.776, 0.777, 0.777, 0.777, 0.777, 0.777, 0.777],
        35.0: [0.759, 0.765, 0.769, 0.771, 0.773, 0.774, 0.774, 0.775, 0.775, 0.775, 0.775],
        39.0: [0.705, 0.728, 0.742, 0.750, 0.755, 0.759, 0.761, 0.763, 0.764, 0.764, 0.765]}

FIG4_X = [34.0 + i for i in range(9)]                           # T_cond
FIG4 = {80.0: [0.767, 0.759, 0.750, 0.739, 0.724, 0.705, 0.677, 0.636, 0.565],
        90.0: [0.777, 0.774, 0.770, 0.766, 0.763, 0.759, 0.754, 0.748, 0.742],
        100.0: [0.777, 0.775, 0.773, 0.771, 0.769, 0.766, 0.763, 0.760, 0.756]}

FIG5_X = [5.0 + i for i in range(12)]                           # T_evap
FIG5 = {80.0: [0.677, 0.711, 0.735, 0.753, 0.767, 0.779, 0.790, 0.800, 0.807, 0.814, 0.821, 0.828],
        90.0: [0.754, 0.762, 0.770, 0.777, 0.783, 0.789, 0.795, 0.801, 0.806, 0.811, 0.816, 0.821],
        100.0: [0.762, 0.767, 0.772, 0.777, 0.782, 0.786, 0.791, 0.796, 0.800, 0.804, 0.808, 0.812]}

FIG6_X = [20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 39.0]  # T_abs
FIG6 = {80.0: [0.805, 0.797, 0.789, 0.783, 0.775, 0.760, 0.738, 0.702, 0.640, 0.448, 0.047],
        90.0: [0.800, 0.792, 0.787, 0.783, 0.778, 0.772, 0.766, 0.760, 0.752, 0.740, 0.730],
        100.0: [0.795, 0.790, 0.785, 0.781, 0.777, 0.773, 0.768, 0.764, 0.760, 0.755, 0.752]}

FIG7_X = [0.1 * i for i in range(11)]                           # eps_shx
FIG7 = {80.0: [0.470, 0.492, 0.516, 0.542, 0.570, 0.601, 0.637, 0.677, 0.723, 0.775, 0.835],
        90.0: [0.640, 0.653, 0.667, 0.684, 0.701, 0.718, 0.736, 0.755, 0.775, 0.794, 0.815],
        100.0: [0.680, 0.690, 0.702, 0.714, 0.725, 0.737, 0.750, 0.762, 0.775, 0.790, 0.803]}

FIG8_X = FIG3_X                                                 # T_gen
FIG8 = {"Q_evap": [1.5] * 11,
        "Q_gen": [2.215, 2.100, 2.045, 2.020, 2.000, 1.990, 1.980, 1.975, 1.972, 1.970, 1.968],
        "Q_cond": [1.600, 1.602, 1.604, 1.605, 1.607, 1.608, 1.610, 1.611, 1.612, 1.613, 1.615],
        "Q_abs": [2.110, 2.000, 1.945, 1.915, 1.895, 1.885, 1.875, 1.865, 1.862, 1.858, 1.855]}

FIG9_X = FIG5_X                                                 # T_evap, T_gen = 100
FIG9 = {"Q_evap": [1.5] * 12,
        "Q_gen": [1.968, 1.957, 1.945, 1.933, 1.921, 1.909, 1.897, 1.886, 1.875, 1.865, 1.855, 1.845],
        "Q_cond": [1.615, 1.614, 1.612, 1.611, 1.610, 1.608, 1.607, 1.606, 1.604, 1.603, 1.602, 1.600],
        "Q_abs": [1.855, 1.845, 1.833, 1.823, 1.812, 1.800, 1.790, 1.780, 1.770, 1.762, 1.753, 1.745]}

DUTY_LABELS = {"Q_evap": "Q_evap", "Q_gen": "Q_gen", "Q_cond": "Q_cond", "Q_abs": "Q_abs"}

# ---------------------------------------------------------------------------
# Running the model
# ---------------------------------------------------------------------------

# Every result solve() returns during the sweeps, for the First/Second Law
# checks at the end.
ALL_RESULTS = []


def run(**overrides):
    """solve() at the base case with some inputs overridden.

    strict=False so the sweeps can follow the paper past the 65 %LiBr
    crystallization limit (Figs 3, 7, 8 all go there); those points are drawn
    dotted. Returns
    None where the generator cannot separate refrigerant at all.
    """
    try:
        r = solve(DesignSpec(**{**BASE, **overrides, "strict": False}))
    except NoSeparationError:
        return None
    ALL_RESULTS.append(r)
    return r


def in_safe_window(r):
    return r.X_gen <= X_SAFE_MAX


def fine(xs, n=8):
    """xs refined n-fold, so the model draws as a smooth line between the
    paper's points."""
    out = []
    for a, b in zip(xs, xs[1:]):
        out += [a + (b - a) * k / n for k in range(n)]
    return out + [xs[-1]]


def draw_model(ax, xs, results, value, color, label):
    """Model curve: solid inside the safe window, dotted outside it."""
    pts = [(x, value(r), in_safe_window(r)) for x, r in zip(xs, results) if r]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], ":", lw=2, color=color)
    ax.plot([p[0] for p in pts], [p[1] if p[2] else float("nan") for p in pts],
            "-", lw=2, color=color, label=label)


def draw_paper(ax, xs, ys, color):
    ax.plot(xs, ys, "o", ms=7, mfc="white", mec=color, mew=1.5)


def finish(ax, xlabel, ylabel, title, filename):
    # Style key for line vs. marker, appended to the series legend.
    ax.plot([], [], "-", lw=2, color="0.3", label="this model")
    ax.plot([], [], ":", lw=2, color="0.3", label="model, above 65 %LiBr (crystallization)")
    ax.plot([], [], "o", ms=7, mfc="white", mec="0.3", mew=1.5,
            label="Ghatos et al. (2021)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, color="0.9")
    ax.legend(fontsize=8, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    save_fig(filename)


def score(name, pairs, tol):
    """pairs: [(model, paper), ...] -> summary row (name, rms, max, ok)."""
    diffs = [m - p for m, p in pairs]
    rms = (sum(d * d for d in diffs) / len(diffs)) ** 0.5
    worst = max(abs(d) for d in diffs)
    return name, rms, worst, tol, worst <= tol


# ---------------------------------------------------------------------------
# Figure sweeps. Each returns a score row.
# ---------------------------------------------------------------------------


def cop_sweep(fig, xs_paper, data, swept, series, xlabel, title, filename):
    """Shared driver for the five COP-vs-something figures."""
    fig_ax = plt.subplots(figsize=(6.4, 4.6))[1]
    pairs = []
    for color, (s, ys) in zip(COLORS, data.items()):
        xs = fine(xs_paper)
        draw_model(fig_ax, xs, [run(**{swept: x, series: s}) for x in xs],
                   lambda r: r.COP, color, f"{series} = {s:g}")
        draw_paper(fig_ax, xs_paper, ys, color)
        for x, y in zip(xs_paper, ys):
            r = run(**{swept: x, series: s})
            pairs.append((r.COP if r else 0.0, y))
    finish(fig_ax, xlabel, "COP", title, filename)
    return score(fig, pairs, COP_TOL)


def duty_sweep(fig, xs_paper, data, swept, fixed, xlabel, title, filename):
    """Shared driver for the two heat-duty figures (8 and 9)."""
    ax = plt.subplots(figsize=(6.4, 4.6))[1]
    xs = fine(xs_paper)
    results = [run(**{swept: x}, **fixed) for x in xs]
    at_paper = [run(**{swept: x}, **fixed) for x in xs_paper]
    pairs = []
    for color, (key, ys) in zip(COLORS, data.items()):
        draw_model(ax, xs, results, lambda r, k=key: getattr(r, k), color,
                   DUTY_LABELS[key])
        draw_paper(ax, xs_paper, ys, color)
        pairs += [(getattr(r, key), y) for r, y in zip(at_paper, ys)]
    finish(ax, xlabel, "Heat transfer rate [kW]", title, filename)
    return score(fig, pairs, Q_TOL)


def figures():
    base = "T_evap=5 C, T_abs=35 C, E=0.7"
    return [
        cop_sweep("Fig 3  COP vs T_gen", FIG3_X, FIG3, "T_gen", "T_cond",
                  "Generator temperature [C]",
                  f"Fig 3 - COP vs generator temperature\n({base})",
                  "verify_fig3_cop_vs_tgen.png"),
        cop_sweep("Fig 4  COP vs T_cond", FIG4_X, FIG4, "T_cond", "T_gen",
                  "Condenser temperature [C]",
                  f"Fig 4 - COP vs condenser temperature\n({base})",
                  "verify_fig4_cop_vs_tcond.png"),
        cop_sweep("Fig 5  COP vs T_evap", FIG5_X, FIG5, "T_evap", "T_gen",
                  "Evaporator temperature [C]",
                  "Fig 5 - COP vs evaporator temperature\n"
                  "(T_cond=40 C, T_abs=35 C, E=0.7)",
                  "verify_fig5_cop_vs_tevap.png"),
        cop_sweep("Fig 6  COP vs T_abs", FIG6_X, FIG6, "T_abs", "T_gen",
                  "Absorber temperature [C]",
                  "Fig 6 - COP vs absorber temperature\n"
                  "(T_evap=5 C, T_cond=40 C, E=0.7)",
                  "verify_fig6_cop_vs_tabs.png"),
        cop_sweep("Fig 7  COP vs E", FIG7_X, FIG7, "eps_shx", "T_gen",
                  "Solution heat exchanger effectiveness E [-]",
                  "Fig 7 - COP vs SHX effectiveness\n"
                  "(T_evap=5 C, T_cond=40 C, T_abs=35 C)",
                  "verify_fig7_cop_vs_eps.png"),
        duty_sweep("Fig 8  Q vs T_gen", FIG8_X, FIG8, "T_gen", {},
                   "Generator temperature [C]",
                   "Fig 8 - Heat transfer rates vs generator temperature\n"
                   "(T_evap=5 C, T_cond=40 C, T_abs=35 C, E=0.7, Q_evap=1.5 kW)",
                   "verify_fig8_duties_vs_tgen.png"),
        duty_sweep("Fig 9  Q vs T_evap", FIG9_X, FIG9, "T_evap", {"T_gen": 100.0},
                   "Evaporator temperature [C]",
                   "Fig 9 - Heat transfer rates vs evaporator temperature\n"
                   "(T_gen=100 C, T_cond=40 C, T_abs=35 C, E=0.7, Q_evap=1.5 kW)",
                   "verify_fig9_duties_vs_tevap.png"),
    ]


# ---------------------------------------------------------------------------
# Table 4
# ---------------------------------------------------------------------------

# Per-quantity tolerances. P is relative (CoolProp vs the paper's Eq. 12 fit);
# m is half a unit in the paper's last printed digit.
TABLE4_TOL = {"T": 0.05, "P": 0.002, "X": 0.001, "h": 0.5, "m": 0.00005}


def check_table4():
    r = solve(TABLE4_SPEC)
    print("=" * 76)
    print("Table 4 -- state points at T_gen=75, T_evap=5, T_cond=32, T_abs=28 C")
    print("=" * 76)
    print("  paper pt (model pt)  quantity      paper        model     diff")
    ok = True
    for p, values in TABLE4.items():
        s = r.states[PAPER_TO_MODEL[p]]
        for q, paper, model in zip("TPXhm", values, (s.T, s.P, s.X, s.h, s.m)):
            diff = model - paper
            err = abs(diff) / paper if q == "P" else abs(diff)
            known = TABLE4_KNOWN.get((p, q))
            if err <= TABLE4_TOL[q] and not known:
                continue    # only print what does not match exactly
            status = f"known: {known}" if known else "FAIL"
            ok = ok and bool(known)
            print(f"  {p:5d} ({PAPER_TO_MODEL[p]:2d})         {q:8s} "
                  f"{paper:11.4f}  {model:11.4f}  {diff:+8.4f}   {status}")
    print(f"  every other value (of 50) agrees within "
          f"T {TABLE4_TOL['T']} C, P {TABLE4_TOL['P']:.1%}, X {TABLE4_TOL['X']} %, "
          f"h {TABLE4_TOL['h']} kJ/kg, m {TABLE4_TOL['m']} kg/s")
    print(f"  (T at paper pt 6 from its own h6: "
          f"{T_from_h(TABLE4[6][3], TABLE4[6][2]):.2f} C)")

    # The paper's COP, recomputed from its own Table 4 with its Eq. (10).
    h = {i: v[3] for i, v in TABLE4.items()}
    f = TABLE4[8][2] / (TABLE4[8][2] - TABLE4[4][2])
    cop_paper = (h[3] - h[2]) / (h[7] + (f - 1) * h[8] - f * (h[6] + h[4] - h[5]))
    print(f"  COP: paper Eq.(10) on Table 4 = {cop_paper:.4f}, "
          f"model = {r.COP:.4f}  (diff {r.COP - cop_paper:+.4f}, "
          f"from the h7 difference)")
    print()
    ok = ok and abs(r.COP - cop_paper) <= COP_TOL
    ALL_RESULTS.append(r)
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ok = check_table4()

    rows = figures()
    print()
    print("=" * 76)
    print("Figures 3-9 -- model against digitized paper curves")
    print("=" * 76)
    print(f"  {'figure':24s} {'RMS':>8s} {'max':>8s} {'tol':>8s}")
    for name, rms, worst, tol, passed in rows:
        print(f"  {name:24s} {rms:8.4f} {worst:8.4f} {tol:8.3f}   "
              f"{'pass' if passed else 'FAIL'}")
        ok = ok and passed
    print()

    # Conservation laws, over every point solved above.
    worst_closure = max(abs(r.energy_closure) for r in ALL_RESULTS)
    over_carnot = [r for r in ALL_RESULTS if r.COP >= r.COP_carnot]
    first_ok = worst_closure < 1e-9
    second_ok = not over_carnot
    print(f"  First Law:  worst |closure| over {len(ALL_RESULTS)} points = "
          f"{worst_closure:.2e} kW   {'pass' if first_ok else 'FAIL'}")
    print(f"  Second Law: COP < COP_carnot at all {len(ALL_RESULTS)} points"
          if second_ok else
          f"  Second Law: {len(over_carnot)} points with COP >= COP_carnot   FAIL")
    ok = ok and first_ok and second_ok

    print()
    print("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
