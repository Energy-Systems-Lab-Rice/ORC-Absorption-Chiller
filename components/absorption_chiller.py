"""
Single-effect LiBr/H2O absorption chiller -- steady-state thermodynamic model.

Algorithm from:
  Lansing, F.L. (1976) "Computer Modeling of a Single-Stage Lithium Bromide/
  Water Absorption Refrigeration Unit", JPL DSN Progress Report 42-32, 247-257.
Property correlations (SI) from:
  Ghatos, S., Taha-Janan, M., Mehdari, A. (2021) "Thermodynamic model of a
  single stage H2O-LiBr absorption cooling", E3S Web of Conf. 234, 00091.
Physical bounds from:
  Alefeld, G. & Radermacher, R. (1993) "Heat Conversion Systems", Ch. 4.

Needs CoolProp for the water/steam properties (real IAPWS-95, via PropsSI).
The LiBr solution side (properties/§4.2-style functions below) stays pure
standard library -- see their module note for why CoolProp can't cover that
part too. No numpy. Every function is pure: same inputs always give the same
outputs, and nothing here prints or plots. Call it from
absorption_chiller_builder.py.

State points follow Lansing Fig. 1:
   1  absorber outlet, saturated solution   @ (T_abs, P_lo)  X = X_abs
   2  solution pump outlet                  @ (T_abs, P_hi)
   3  SHX outlet, absorber side             @ (T3,    P_hi)  -> generator
   4  generator outlet, saturated solution  @ (T_gen, P_hi)  X = X_gen
   5  SHX outlet, generator side            @ (T5,    P_hi)
   6  solution valve outlet                 @ (T5,    P_lo)  -> absorber
   7  refrigerant vapour from generator     @ (T_gen, P_hi)  superheated
   8  condenser outlet, saturated liquid    @ (T_cond,P_hi)
   9  refrigerant valve outlet              @ (T_evap,P_lo)
  10  evaporator outlet, saturated vapour   @ (T_evap,P_lo)

Units are strict SI throughout: T in degC, P in kPa, h in kJ/kg, m in kg/s,
Q and W in kW, and X in PERCENT LiBr by mass (0-100, never a 0-1 fraction).

Naming: the words "strong" and "weak" are deliberately never used for the
solution streams. Lansing, Alefeld/Radermacher and Ghatos each define them
differently -- three sources, three conventions, guaranteed sign errors. The
streams are named for where they come from instead: X_abs leaves the absorber
(points 1-3), X_gen leaves the generator (points 4-6). X_gen > X_abs always.
"""

from dataclasses import asdict, dataclass
from math import isfinite

from CoolProp.CoolProp import PropsSI

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ChillerError(Exception):
    """Base class: the inputs describe a physically invalid cycle."""


class CrystallizationError(ChillerError):
    """A solution concentration went above the 65 %LiBr crystallization limit.

    Above ~65 % the LiBr can crystallize out of the cooled solution leaving
    the SHX and plug the machine.
    """


class NoSeparationError(ChillerError):
    """X_gen <= X_abs, so the generator boils off no refrigerant.

    Unconditional (it ignores strict=False) because the circulation ratio f
    is negative or infinite here: every downstream number is meaningless,
    not merely inaccurate.
    """


class PropertyRangeError(ChillerError, ValueError):
    """A correlation was called outside its published validity range.

    Also a ValueError, so callers that only guard against bad arguments
    still catch it.
    """


class ConvergenceError(ChillerError):
    """A bisection hit its iteration cap without converging."""


# ---------------------------------------------------------------------------
# Water / steam properties
# ---------------------------------------------------------------------------

# These five used to be closed-form fits transcribed from Ghatos Eqs. (12),
# (14), (15) plus a constant-cp superheat model from Lansing Eq. (3) -- see
# git history if you need them back. CoolProp's own water model (IAPWS-95)
# is more accurate than any of those fits and needs no coefficients, so it
# replaces them here. This is the one place in this file that needs CoolProp
# installed; the LiBr solution functions below intentionally do not (see
# their module note).


def Psat(T):
    """Saturation pressure of pure water [kPa] at T [degC]."""
    return PropsSI("P", "T", T + 273.15, "Q", 0, "Water") / 1000.0


def Tsat(P):
    """Saturation temperature of pure water [degC] at P [kPa]."""
    return PropsSI("T", "P", P * 1000.0, "Q", 0, "Water") - 273.15


def h_water_liq(T):
    """Saturated liquid water enthalpy [kJ/kg] at T [degC]."""
    return PropsSI("H", "T", T + 273.15, "Q", 0, "Water") / 1000.0


def h_water_vap(T):
    """Saturated water vapour enthalpy [kJ/kg] at T [degC]."""
    return PropsSI("H", "T", T + 273.15, "Q", 1, "Water") / 1000.0


def h_water_sup(T, P):
    """Superheated steam enthalpy [kJ/kg] at T [degC] and P [kPa].

    Needs both T and P (unlike the saturated-state functions above) because
    superheated steam is a genuinely two-property state -- there's no single
    saturation temperature to fall back on the way there was with the old
    constant-cp fit, which is exactly why that fit was the weakest
    correlation in the model.
    """
    return PropsSI("H", "T", T + 273.15, "P", P * 1000.0, "Water") / 1000.0


# ---------------------------------------------------------------------------
# LiBr/H2O solution properties
#
# Unlike the water/steam functions above, these stay closed-form correlations
# rather than CoolProp calls. CoolProp does have an INCOMP::LiBr fluid, but
# its enthalpy is on its own internal reference, unrelated to the ASHRAE/
# Kaita correlation Q_gen and Q_abs are built on below -- mixing the two
# would silently corrupt those duties (this is exactly the reference-state
# mismatch SPEC 10.3 warns about). state_entropy(), further down, uses
# INCOMP::LiBr for entropy only, which is reference-independent and safe.
# ---------------------------------------------------------------------------

# ASHRAE / Kaita solution enthalpy, h = A(X) + B(X).T + C(X).T^2.
# Ghatos Table 3 reprints this with two errors: the layout is given as
# h = U.T^2 + V.T + W, and C3 is printed negative. Neither reproduces their
# own Table 4 (the printed form gives h4 = -1.99e3, the sign-flipped form
# -149.7, against a tabulated 57.7252). The form below, with C3 POSITIVE,
# reproduces Table 4 to within 0.004 kJ/kg. Do not "fix" the sign back.
_A = (-2024.33, 163.309, -4.88161, 6.302948e-2, -2.913704e-4)
_B = (18.2829, -1.1691757, 3.248041e-2, -4.034184e-4, 1.8520569e-6)
_C = (-3.7008214e-2, 2.8877666e-3, -8.1313015e-5, 9.9116628e-7, -4.4441207e-9)

# Published validity window of the enthalpy correlation.
X_CORRELATION_MIN, X_CORRELATION_MAX = 40.0, 70.0
T_CORRELATION_MIN, T_CORRELATION_MAX = 15.0, 165.0

# Crystallization limit.
X_SAFE_MAX = 65.0


def _check_percent(X):
    """X is a percentage, 0-100. Getting this wrong returns plausible garbage
    rather than an obvious error, so every entry point checks it."""
    if not isfinite(X) or not 0.0 <= X <= 100.0:
        raise PropertyRangeError(f"X={X!r} is not a LiBr percentage in 0-100.")
    if 0.0 < X < 1.0:
        raise PropertyRangeError(
            f"X={X!r} looks like a mass fraction. This model takes percent: "
            f"pass {X * 100:.1f}, not {X!r}."
        )


def _poly(coeffs, X):
    return sum(c * X ** i for i, c in enumerate(coeffs))


def h_sol(T, X):
    """LiBr/H2O solution enthalpy [kJ/kg of solution]. ASHRAE (Kaita) form.

    X in percent, T in degC. Published range 40-70 %LiBr, 15-165 degC; the
    range is checked by the cycle guards rather than here, so that
    strict=False sweeps can still map the infeasible region.
    """
    _check_percent(X)
    return _poly(_A, X) + _poly(_B, X) * T + _poly(_C, X) * T ** 2


def T_from_h(h, X):
    """Solution temperature [degC] with enthalpy h at concentration X.

    Inverse of h_sol by bisection; h_sol is strictly increasing in T across
    the operating window, so this is safe.
    """
    _check_percent(X)
    return _bisect(lambda T: h_sol(T, X), h, -50.0, 250.0, "T_from_h")


# This one stays on Lansing's fit rather than following rho_sol/cp_sol to
# CoolProp
def X_equilibrium(T_sol, T_ref):
    """Equilibrium LiBr concentration [%] of a solution at T_sol [degC] whose
    vapour is in equilibrium with pure water at T_ref [degC]. Lansing Eq. (7).

    A linear-Duhring fit, +/-0.2 % over 50-65 %LiBr. T_ref is the saturation
    temperature of pure water at the same pressure as the solution -- so the
    absorber pairs with T_evap and the generator with T_cond.
    """
    return 100.0 * (49.04 + 1.125 * T_sol - T_ref) / (134.65 + 0.47 * T_sol)


# Fixed reference pressure for the two lookups below. Density and specific
# heat barely vary with pressure for a liquid this incompressible (verified:
# <0.001% change from 101 kPa to 500 kPa)
_REFERENCE_P_PA = 101325.0


def rho_sol(T, X):
    """LiBr solution density [kg/m3] at T [degC], X [%LiBr].

    Via CoolProp's INCOMP::LiBr (Patek & Klomfar 2006). Density has no
    reference-state ambiguity the way enthalpy does (see the module note
    above h_sol), so this is safe to source from CoolProp even though h_sol
    stays on the ASHRAE/Kaita fit. Matches the old Lee et al./Ghatos Eq. (18)
    correlation to within 0.5% at typical cycle conditions.
    """
    _check_percent(X)
    return PropsSI("D", "T", T + 273.15, "P", _REFERENCE_P_PA, f"INCOMP::LiBr[{X / 100.0}]")


def cp_sol(T, X):
    """LiBr solution specific heat [kJ/(kg.K)] at T [degC], X [%LiBr].

    Via CoolProp's INCOMP::LiBr. Ghatos Eq. (13), which this replaces, only
    depended on X; CoolProp's data has genuine (if fairly mild) T-dependence
    too. Not called by solve() (which uses the enthalpy form -- it avoids a
    linearisation error); this is here for an off-design NTU rating model
    later, per SPEC 9.1.
    """
    _check_percent(X)
    return PropsSI("C", "T", T + 273.15, "P", _REFERENCE_P_PA, f"INCOMP::LiBr[{X / 100.0}]") / 1000.0


# ---------------------------------------------------------------------------
# Shared numerics
# ---------------------------------------------------------------------------


def _bisect(fn, target, lo, hi, name, tol=1e-9, max_iter=200):
    """Solve fn(x) == target on [lo, hi] for a strictly increasing fn."""
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if fn(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            return 0.5 * (lo + hi)
    raise ConvergenceError(
        f"{name} did not converge to {tol} in {max_iter} iterations "
        f"(target={target!r}, bracket=[{lo!r}, {hi!r}])."
    )


# ---------------------------------------------------------------------------
# Inputs and results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DesignSpec:
    """The six design inputs of Lansing Sec. VII, plus two switches.

    Frozen (and therefore hashable) so an annual simulation can memoise
    solve() on it.
    """

    T_gen: float                # generator solution outlet, degC
    T_evap: float               # evaporator, degC
    T_cond: float               # condenser, degC
    T_abs: float                # absorber solution outlet, degC
    eps_shx: float = 0.7        # solution heat exchanger effectiveness, 0-1
    Q_evap: float = 10.0        # design cooling load, kW
    include_pump_work: bool = True
    strict: bool = True         # False: warn instead of raising, for sweeps


@dataclass(frozen=True, slots=True)
class StatePoint:
    """One of the ten cycle states. X = 0 for the pure-water points 7-10."""

    index: int
    T: float    # degC
    P: float    # kPa
    X: float    # %LiBr
    h: float    # kJ/kg
    m: float    # kg/s


@dataclass(frozen=True, slots=True)
class CycleResult:
    """Everything the solver computed, in SI. Q and W are in kW."""

    spec: DesignSpec
    states: dict            # {1..10: StatePoint}
    P_lo: float
    P_hi: float
    X_abs: float
    X_gen: float
    f: float                # circulation ratio, kg solution per kg refrigerant
    m_r: float
    m_abs: float
    m_gen: float
    Q_evap: float
    Q_gen: float
    Q_cond: float
    Q_abs: float
    W_pump: float
    COP: float
    COP_carnot: float
    eta_rel: float
    energy_closure: float   # kW, First Law residual: must be ~0
    warnings: tuple

    def to_dict(self):
        """Plain JSON-serialisable dict -- no dataclasses, no numpy.

        asdict recurses, so spec and every StatePoint come back as plain
        dicts too. json.dumps renders the integer state keys as strings.
        """
        return asdict(self)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def _validate(spec):
    """Reject malformed input with ValueError (not a ChillerError -- these
    are bad arguments, not infeasible cycles)."""
    for field in ("T_gen", "T_evap", "T_cond", "T_abs", "eps_shx", "Q_evap"):
        value = getattr(spec, field)
        if not isfinite(value):
            raise ValueError(f"{field}={value!r} is not a finite number.")
    if not 0.0 <= spec.eps_shx <= 1.0:
        raise ValueError(f"eps_shx={spec.eps_shx!r} must be in [0, 1].")
    if spec.Q_evap <= 0.0:
        raise ValueError(f"Q_evap={spec.Q_evap!r} must be positive.")
    if spec.T_evap >= spec.T_cond:
        raise ValueError(
            f"T_evap={spec.T_evap!r} must be below T_cond={spec.T_cond!r}."
        )
    if spec.T_abs >= spec.T_gen:
        raise ValueError(
            f"T_abs={spec.T_abs!r} must be below T_gen={spec.T_gen!r}."
        )


def _flag(strict, error_cls, message, warnings):
    """Raise in strict mode; otherwise record a warning and carry on."""
    if strict:
        raise error_cls(message)
    warnings.append(message)


def _check_envelope(spec, X_abs, X_gen, warnings):
    """Feasibility guards, run before anything divides by X_gen - X_abs.

    No separation is fatal whatever spec.strict says: f would be negative or
    infinite. The other two checks honour spec.strict, so that a sweep can
    still map the region past the crystallization limit (the Ghatos figures
    go well past 65 %LiBr, for instance).
    """
    if X_gen <= X_abs:
        raise NoSeparationError(
            f"X_gen={X_gen:.3f} % <= X_abs={X_abs:.3f} %: at T_gen={spec.T_gen} C "
            f"the generator cannot boil refrigerant out of the solution."
        )
    for name, X in (("X_abs", X_abs), ("X_gen", X_gen)):
        if X > X_SAFE_MAX:
            _flag(spec.strict, CrystallizationError,
                  f"{name}={X:.2f} % is above {X_SAFE_MAX} %LiBr: "
                  f"crystallization risk.", warnings)
        if not X_CORRELATION_MIN <= X <= X_CORRELATION_MAX:
            _flag(spec.strict, PropertyRangeError,
                  f"{name}={X:.2f} % is outside the h_sol correlation range "
                  f"{X_CORRELATION_MIN}-{X_CORRELATION_MAX} %LiBr.", warnings)
    for name in ("T_abs", "T_gen"):
        T = getattr(spec, name)
        if not T_CORRELATION_MIN <= T <= T_CORRELATION_MAX:
            _flag(spec.strict, PropertyRangeError,
                  f"{name}={T} C is outside the h_sol correlation range "
                  f"{T_CORRELATION_MIN}-{T_CORRELATION_MAX} C.", warnings)


def solve(spec):
    """Solve the single-effect LiBr/H2O cycle. One forward pass, no iteration
    over the cycle -- the only iterative numerics are CoolProp's own water
    solver and the one bisection in T_from_h (LiBr solution side).

    Raises ValueError for malformed input, and a ChillerError subclass for
    input that is well-formed but describes an infeasible cycle. It never
    returns nan.
    """
    _validate(spec)
    warnings = []

    # 1. Two pressure levels, set by the saturated refrigerant states.
    P_lo = Psat(spec.T_evap)
    P_hi = Psat(spec.T_cond)

    # 2. Equilibrium concentrations. Each pairs a SOLUTION temperature with
    #    the refrigerant saturation temperature at that same pressure.
    X_abs = X_equilibrium(spec.T_abs, spec.T_evap)
    X_gen = X_equilibrium(spec.T_gen, spec.T_cond)
    _check_envelope(spec, X_abs, X_gen, warnings)

    # 3. Circulation ratio: solution leaving the absorber per unit
    #    refrigerant. This is the generator LiBr species balance.
    f = X_gen / (X_gen - X_abs)
    if f > 30.0:
        warnings.append(
            f"circulation ratio f={f:.1f} is very high; pump power and SHX "
            f"losses will dominate."
        )

    # 4. Refrigerant loop.
    h8 = h_water_liq(spec.T_cond)               # saturated liquid, condenser
    h9 = h8                                     # isenthalpic expansion valve
    h10 = h_water_vap(spec.T_evap)              # saturated vapour, evaporator
    h7 = h_water_sup(spec.T_gen, P_hi)          # superheated, from generator

    # 5. Solution loop and the solution heat exchanger.
    h1 = h_sol(spec.T_abs, X_abs)
    v1 = 1.0 / rho_sol(spec.T_abs, X_abs)                       # m3/kg
    w_pump = v1 * (P_hi - P_lo) if spec.include_pump_work else 0.0   # kJ/kg
    h2 = h1 + w_pump                                            # T2 = T_abs
    h4 = h_sol(spec.T_gen, X_gen)

    # Effectiveness is defined on the generator-side stream because it
    # carries the minimum capacity rate (Lansing p. 250 verifies this).
    T5 = spec.T_gen - spec.eps_shx * (spec.T_gen - spec.T_abs)  # Lansing 20b
    h5 = h_sol(T5, X_gen)
    h6 = h5                                     # isenthalpic solution valve

    # Adiabatic SHX energy balance: f*(h3 - h2) = (f - 1)*(h4 - h5).
    # The enthalpy form is used rather than Lansing Eq. (21b), which needs cp
    # at two concentrations and adds a linearisation error for no benefit.
    h3 = h2 + ((f - 1.0) / f) * (h4 - h5)
    T3 = T_from_h(h3, X_abs)

    # 6. Mass flows.
    m_r = spec.Q_evap / (h10 - h9)      # refrigerant
    m_abs = f * m_r                     # absorber outlet -> generator
    m_gen = (f - 1.0) * m_r             # generator outlet -> absorber

    # 7. Component duties. Q_gen is the generator-only control volume, so the
    #    solution enters at state 3 (after the SHX). Lansing Eq. (26) uses a
    #    combined generator+SHX volume entering at state 2 and is numerically
    #    identical because the SHX is adiabatic -- but do not mix the two.
    Q_gen = m_r * h7 + m_gen * h4 - m_abs * h3
    Q_cond = m_r * (h7 - h8)
    Q_abs = m_r * h10 + m_gen * h6 - m_abs * h1
    W_pump = m_abs * w_pump

    # 8. Performance.
    COP = spec.Q_evap / (Q_gen + W_pump)
    COP_carnot = (((spec.T_gen - spec.T_abs) / (spec.T_gen + 273.15))
                  * ((spec.T_evap + 273.15) / (spec.T_cond - spec.T_evap)))
    # First Law over the whole machine. This is the model's only structural
    # error detector: a wrong enthalpy still yields a plausible COP, but it
    # will not close the energy balance.
    energy_closure = (Q_gen + spec.Q_evap + W_pump) - (Q_cond + Q_abs)

    states = _build_states(
        spec, P_lo, P_hi, X_abs, X_gen, T3, T5,
        (h1, h2, h3, h4, h5, h6, h7, h8, h9, h10),
        m_r, m_abs, m_gen,
    )
    return CycleResult(
        spec=spec, states=states, P_lo=P_lo, P_hi=P_hi,
        X_abs=X_abs, X_gen=X_gen, f=f,
        m_r=m_r, m_abs=m_abs, m_gen=m_gen,
        Q_evap=spec.Q_evap, Q_gen=Q_gen, Q_cond=Q_cond, Q_abs=Q_abs,
        W_pump=W_pump,
        COP=COP, COP_carnot=COP_carnot, eta_rel=COP / COP_carnot,
        energy_closure=energy_closure, warnings=tuple(warnings),
    )


def _build_states(spec, P_lo, P_hi, X_abs, X_gen, T3, T5, h, m_r, m_abs, m_gen):
    """Assemble the ten state points. h is (h1 ... h10) in order."""
    T = (spec.T_abs, spec.T_abs, T3, spec.T_gen, T5, T5,
         spec.T_gen, spec.T_cond, spec.T_evap, spec.T_evap)
    P = (P_lo, P_hi, P_hi, P_hi, P_hi, P_lo, P_hi, P_hi, P_lo, P_lo)
    X = (X_abs, X_abs, X_abs, X_gen, X_gen, X_gen, 0.0, 0.0, 0.0, 0.0)
    m = (m_abs, m_abs, m_abs, m_gen, m_gen, m_gen, m_r, m_r, m_r, m_r)
    return {
        i + 1: StatePoint(index=i + 1, T=T[i], P=P[i], X=X[i], h=h[i], m=m[i])
        for i in range(10)
    }


# ---------------------------------------------------------------------------
# External coupling -- the seam to a solar loop / cooling tower / chilled water
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Approaches:
    """Approach temperatures [K] between external fluids and the cycle.

    Lansing p. 252 argues against fixed UA products: he found a 350 % change
    in LiBr flow shifted the overall temperature pattern by only 6 %, so UA
    is not a fixed property of the machine. Fixed approaches are his
    recommendation instead, and the defaults below are his values for a
    water-cooled machine. An air-cooled evaporator wants roughly 20 K in and
    6 K out instead.
    """

    gen: float = 10.0    # T_hot_in  - T_gen
    cond: float = 3.0    # T_cond    - T_cool_in
    abs_: float = 3.0    # T_abs     - T_cool_in
    evap: float = 3.0    # T_chw_out - T_evap


def to_design_spec(T_hot_in, T_cool_in, T_chw_out, Q_evap,
                   eps_shx=0.7, approaches=Approaches()):
    """Map external fluid temperatures [degC] to internal cycle temperatures.

    The solver works in internal cycle temperatures; a solar loop or cooling
    tower speaks external fluid temperatures. Keeping the two apart is the
    whole point of this function -- do not conflate them.

    NOTE: the condenser and the absorber both reference T_cool_in, i.e. the
    cooling water branches are in PARALLEL (the configuration of Lansing
    Sec. VIII). Series cooling -- absorber first, then condenser, so the
    condenser sees water already warmed by the absorber -- is a different
    mapping and is not implemented; it is not a safe thing to guess.
    """
    return DesignSpec(
        T_gen=T_hot_in - approaches.gen,
        T_evap=T_chw_out - approaches.evap,
        T_cond=T_cool_in + approaches.cond,
        T_abs=T_cool_in + approaches.abs_,
        eps_shx=eps_shx,
        Q_evap=Q_evap,
    )


# ---------------------------------------------------------------------------
# Entropy -- for a T-s diagram only. solve() never calls this (entropy plays
# no role in the cycle solution), but CoolProp is already an import at the
# top of this file for the water/steam properties, so nothing extra is
# needed here to use it.
# ---------------------------------------------------------------------------


def state_entropy(state):
    """Specific entropy [kJ/(kg.K)] of one StatePoint, via CoolProp.

    Two different substances flow through this cycle -- pure refrigerant
    (points 7-10, X = 0) and LiBr/H2O solution (points 1-6) -- each with its
    own entropy reference, unrelated to the other's. Entropy is only
    meaningful for comparing points within the same loop, not across them.

    Refrigerant points (7-10): computed from this component's own h and P,
    via CoolProp's water model -- the same model h_water_* above already
    uses, so there's no reference-state question here at all. h (not T and
    P) is used because point 9, just after the refrigerant expansion valve,
    is a two-phase liquid/vapour mixture -- T and P alone can't tell
    CoolProp which one it's looking at, since they aren't independent on the
    saturation line.

    LiBr solution points (1-6): computed from T, P and X alone, via
    CoolProp's INCOMP::LiBr correlation (Lee et al. -- the same source as
    this component's own density equation, but not its enthalpy equation).
    This function never feeds this component's own solution enthalpy into
    CoolProp: that enthalpy's reference does not match CoolProp's (see the
    module note above h_sol), while temperature, pressure and concentration
    are reference-independent and safe to hand across.
    """
    T_K = state.T + 273.15
    P_Pa = state.P * 1000.0
    if state.X == 0.0:
        s_J = PropsSI("S", "H", state.h * 1000.0, "P", P_Pa, "Water")
    else:
        s_J = PropsSI("S", "T", T_K, "P", P_Pa, f"INCOMP::LiBr[{state.X / 100.0}]")
    return s_J / 1000.0  # J/(kg.K) -> kJ/(kg.K)
