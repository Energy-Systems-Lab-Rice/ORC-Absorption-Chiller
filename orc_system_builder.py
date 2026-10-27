"""
ORC (Organic Rankine Cycle) model with a heat-exchanger/regenerator (HXGR).

Translated from ORCModelReheat.ipynb.

Requirements:
    pip install CoolProp numpy matplotlib

Run:
    python orc_system_builder.py

Figures are saved as PNGs under ./pycache/ (created if missing) instead of
popping up blocking windows one at a time; re-running the script overwrites
the previous images.
"""

import os

import matplotlib.pyplot as plt

from components.orc_reheat import OrcCycle
from plots.TsPlot import TsPlot
from plots.efficiencyPlot import efficiencyPlot

# Figures are saved here (as PNGs) instead of popped up in blocking windows,
# so the script can run straight through and re-running it overwrites the
# previous images instead of piling up new ones.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pycache")


def save_fig(name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path)
    plt.close()
    print("saved", path)


def TK(T):
    """Converts C to K"""
    return T + 273.15


# OrcCycle (solves the 6 cycle states for a single fluid/inlet-temp/efficiency
# combination) now lives in components/orc_reheat.py; it's called below with
# TsPlot passed in so it can build the T-s diagram data without importing
# this module back. efficiencyPlot lives in plots/efficiencyPlot.py alongside
# TsPlot, since it's also plot-input-building code rather than core cycle math.


# ---------------------------------------------------------------------------
# Economic modelling, currently this is incorrect (kept from the notebook for
# reference; using cost estimates from the Kashif paper; all costs are per kg)
# ---------------------------------------------------------------------------
# C_cond = 12300 * (q_c/50000)**0.76
# C_gen = 1850000 * ((wdot_t)/11800000)**0.94
# C_pump = math.exp(3.3892 + 0.0536*math.log(wdot_p) + 0.1538*(math.log(wdot_p)**2))
# C_turb = math.exp(2.2476 + 1.4965*math.log(wdot_t) - 0.1618*(math.log(wdot_t)**2))
# C_hxgr = math.exp(4.3247 - 0.3030*math.log(h2a-h3) + 0.1634*(math.log(h2a-h3)**2))
# C_evap = math.exp(3.2138 + 0.2688*math.log(q_h) + 0.0796*(math.log(q_h)**2))
# print(C_cond,C_gen,C_pump,C_turb,C_hxgr,C_evap)
#
# plot cost vs mass flow rate (mass flow rate is a function of qdot = mdot*deltah)


def main():
    # find temp levels for different DNI parabolic troughs
    T1 = 185  # temp into the turbine estimate in C
    T4 = 35  # temp out of condenser also an estimate in C
    # also input 'TOLUENE','NOCTANE','CYCLOPENTANE','BENZENE', 'BUTANE'
    Fluids = ['HEPTANE', 'NOCTANE']
    nt_vals = [0.8, 0.85, 0.9]
    np_vals = [0.65, 0.75, 0.8]

    plt.figure()
    for F in Fluids:
        inlet_temperatures, efficiencies = efficiencyPlot(F, 0.85, 0.65)  # default values from Kashif paper
        plt.plot(inlet_temperatures, efficiencies, label=F)
    plt.xlabel("Inlet Temperatures (C)")
    plt.ylabel("Efficiency (%)")
    plt.title('Inlet Temp vs Efficiency for Selected Fluids')
    plt.legend()
    save_fig("efficiency_vs_inlet_temp_fluids.png")

    plt.figure()
    for F in Fluids:
        efficiency, T_vals, s_vals, combined_temps, combined_entropies = OrcCycle(T1, T4, F, False, 0.85, 0.65, TsPlot)
        line, = plt.plot(combined_entropies, combined_temps, label=F)
        plt.plot(s_vals, T_vals, 'o', color=line.get_color())
    plt.xlabel("s (J/K)")
    plt.ylabel("T (C)")
    plt.title('T-s diagram for Selected Fluids')
    plt.legend()
    save_fig("ts_diagram_fluids.png")


if __name__ == "__main__":
    main()
