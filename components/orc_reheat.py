import CoolProp.CoolProp as CP
from CoolProp.CoolProp import PropsSI


def TC(T):
    """Converts K to C"""
    return T - 273.15


def TK(T):
    """Converts C to K"""
    return T + 273.15


def OrcCycle(T1, T4, F, verbose, nt, np_, ts_plot_fn=None):
    """Runs a single ORC cycle (with HXGR) for fluid F at inlet temp T1 / condenser temp T4.

    nt  = turbine isentropic efficiency
    np_ = pump isentropic efficiency (named np_ here to avoid shadowing numpy's `np`)
    verbose (was `T` in the notebook) prints the work/efficiency/state summary when True.
    ts_plot_fn = optional callable(F) -> (combined_temps, combined_entropies), used to
        build the saturation-dome T-s diagram data; passed in so this module doesn't need
        to know about the caller's plotting code. Building the dome is the expensive part
        of this function (1000 PropsSI-heavy points), so leave it as None (the default)
        when you only need the cycle states/efficiency, e.g. in a loop over many T1 values.
    """
    CP.set_reference_state(F, 'NBP')
    # define proper input parameters, give reasonable estimates for temp out of boiler
    # print('GWP100 is ', PropsSI(F,'GWP100'), 'ODP is ', PropsSI(F,'ODP'))
    P1 = PropsSI('P', 'T|gas', TK(T1), 'Q', 1, F)  # saturated vapor for a dry fluid, units Pa
    s1 = PropsSI('S', 'T|gas', TK(T1), 'P', P1, F)
    h1 = PropsSI('H', 'T|gas', TK(T1), 'P', P1, F)

    # THROUGH CONDENSER
    P4 = PropsSI('P', 'T|liquid', TK(T4), 'Q', 0, F)
    s4 = PropsSI('S', 'T|liquid', TK(T4), 'P', P4, F)
    h4 = PropsSI('H', 'T|liquid', TK(T4), 'P', P4, F)

    # THROUGH TURBINE
    P2 = P4
    h2s = PropsSI('H', 'P', P2, 'S', s1, F)
    h2a = h1 - nt * (h1 - h2s)
    s2 = PropsSI('S', 'H', h2a, 'P', P2, F)
    T2 = PropsSI('T', 'P', P2, 'S', s2, F)

    # THROUGH PUMP
    P5 = P1
    h5s = PropsSI('H', 'P', P5, 'S', s4, F)
    h5a = h4 - np_ * (h4 - h5s)
    T5 = PropsSI('T', 'P', P5, 'H', h5a, F)
    s5 = PropsSI('S', 'T|liquid', T5, 'P', P5, F)

    # THROUGH HXGR
    e = 0.9  # hxgr effectiveness, an estimate given by previous research
    P3 = P2
    P6 = P5
    T3 = TK(T2) - e * (TK(T2) - T5)
    h3 = PropsSI('H', 'P', P3, 'T', T3, F)
    h6 = h2a - h3 + h5a
    T6 = PropsSI('T', 'H', h6, 'P', P6, F)
    s3 = PropsSI('S', 'T', T3, 'P', P3, F)
    s6 = PropsSI('S', 'T', T6, 'P', P6, F)

    # work and efficiency calculations (single iteration)
    wdot_t = h1 - h2a
    wdot_p = h5a - h4
    q_h = h1 - h6
    q_c = h3 - h4
    efficiency = (-wdot_t + wdot_p) / -q_h

    if verbose:
        s12 = s2 - s1
        s23 = s3 - s2
        s34 = s4 - s3
        s45 = s5 - s4
        s56 = s6 - s5
        s61 = s1 - s6
        print('the efficiency for ' + F + ' is', efficiency * 100, 'percent')
        print('pump work: ', wdot_p / 1000, ' kJ/kg')
        print('turbine work: ', wdot_t / 1000, ' kJ/kg')
        print('Temperatures (1 to 6 in order): ', T1, TC(T2), TC(T3), T4, TC(T5), TC(T6))
        print('Pressures (1 to 6 in order): ', P1, P2, P3, P4, P5, P6)
        print('Entropies (1 to 6 in order): ', s1, s2, s3, s4, s5, s6)
        print('Change in Entropies: ', s12, s23, s34, s45, s56, s61)

    T_vals = (T1, TC(T2), TC(T3), T4, TC(T5), TC(T6))
    s_vals = (s1, s2, s3, s4, s5, s6)

    if ts_plot_fn is None:
        combined_temps, combined_entropies = None, None
    else:
        combined_temps, combined_entropies = ts_plot_fn(F)
    return efficiency, T_vals, s_vals, combined_temps, combined_entropies
