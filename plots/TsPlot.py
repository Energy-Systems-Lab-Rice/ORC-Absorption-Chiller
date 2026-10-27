import numpy as np
from CoolProp.CoolProp import PropsSI


def TC(T):
    """Converts K to C"""
    return T - 273.15


def TsPlot(F):
    """Builds the saturation-dome T-s diagram (temperature vs entropy) for fluid F."""
    P_min = PropsSI('P_min', F)
    P_max = PropsSI('Pcrit', F)
    pressures = np.logspace(np.log10(P_min), np.log10(P_max), 1000)

    temperatures = []
    temperatures2 = []
    entropies = []
    entropies2 = []

    for P in pressures:
        # Saturation state (we can plot for saturation states for simplicity)
        T_sat1 = PropsSI('T', 'P', P, 'Q', 0, F)  # Temperature at saturated liquid (Q=0)
        S_sat1 = PropsSI('S', 'P', P, 'Q', 0, F)
        T_sat2 = PropsSI('T', 'P', P, 'Q', 1, F)  # Temperature at saturated vapor (Q=1)
        S_sat2 = PropsSI('S', 'P', P, 'Q', 1, F)

        temperatures.append(TC(T_sat1))
        entropies.append(S_sat1)
        temperatures2.append(TC(T_sat2))
        entropies2.append(S_sat2)

    # Convert lists to arrays for plotting
    temperatures = np.array(temperatures)
    entropies = np.array(entropies)
    temperatures2 = np.flip(np.array(temperatures2))
    entropies2 = np.flip(np.array(entropies2))

    combined_temps = np.append(temperatures, temperatures2)
    combined_entropies = np.append(entropies, entropies2)

    return combined_temps, combined_entropies
