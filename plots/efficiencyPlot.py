import math

from CoolProp.CoolProp import PropsSI

from components.orc_reheat import OrcCycle


def TC(T):
    """Converts K to C"""
    return T - 273.15


def efficiencyPlot(F, nt, np_):
    """Efficiency vs inlet temperature relation for the selected fluid.

    nt  = turbine isentropic efficiency
    np_ = pump isentropic efficiency (named np_ here to avoid shadowing numpy's `np`)
    """
    inlet_temperatures = []
    efficiencies = []
    highT = math.floor(TC(PropsSI('Tcrit', F)))

    for T1 in range(100, highT):
        T4 = 35  # condenser temp in C
        inlet_temperatures.append(T1)
        # ts_plot_fn omitted: we only need the efficiency here, and building the
        # T-s dome on every iteration would be needlessly expensive (see OrcCycle).
        efficiency, T_vals, s_vals, combined_temps, combined_entropies = OrcCycle(T1, T4, F, False, nt, np_)
        efficiencies.append(efficiency)

    return inlet_temperatures, efficiencies
