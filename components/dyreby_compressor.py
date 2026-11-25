"""SNL/Dyreby (2014) radial sCO2 compressor: design sizing and steady off-design.

Reference: J. J. Dyreby, *Modeling the Supercritical Carbon Dioxide Brayton
Cycle with Recompression* (2014), sec. 3.2 and Appendix IV.

All quantities are SI: K, Pa, kg/s, m, rpm, J/kg, W.
Requires CoolProp: pip install CoolProp

"""
from __future__ import annotations

from dataclasses import dataclass
import math
from scipy.optimize import brentq
from CoolProp.CoolProp import PropsSI

FLUID = 'CO2'
PHI_DESIGN = 0.02971
PHI_MIN = 0.020
PHI_MAX = 0.050


def _head_star(phi_star: float) -> float:
    """Dyreby/SNL modified head coefficient, Appendix IV."""
    x = phi_star
    return ((((-498626.0*x + 53224.0)*x - 2505.0)*x + 54.6)*x + 0.04049)


def _eta_star(phi_star: float) -> float:
    """Dyreby/SNL modified relative efficiency polynomial, Appendix IV."""
    x = phi_star
    return ((((-1.638e6*x + 182725.0)*x - 8089.0)*x + 168.6)*x - 0.7069)


@dataclass(frozen=True)
class CompressorResult:
    T_out_K: float
    P_out_Pa: float
    h_out_Jkg: float
    eta_is: float
    mdot_kgs: float
    N_rpm: float
    phi: float
    phi_star: float
    psi: float
    power_W: float                   # compressor power consumption, positive
    tip_speed_ms: float
    tip_to_sound_out: float
    surge: bool
    valid: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class DyrebyCompressor:
    diameter_m: float
    N_design_rpm: float
    eta_design: float
    phi_design: float = PHI_DESIGN

    @classmethod
    def from_design(
        cls,
        T_in_K: float,
        P_in_Pa: float,
        P_out_Pa: float,
        mdot_kgs: float,
        eta_design: float = 0.90,
    ) -> "DyrebyCompressor":
        """Size D and N_design for an existing thermodynamic design point.

        The design efficiency scales the off-design efficiency polynomial. It
        does NOT come from the polynomial by itself.
        """
        if not (0.0 < eta_design <= 1.0):
            raise ValueError('eta_design must lie in (0,1].')
        if not (mdot_kgs > 0.0 and P_out_Pa > P_in_Pa > 0.0 and T_in_K > 0.0):
            raise ValueError('Invalid compressor design inputs.')

        h_in = PropsSI('Hmass', 'T', T_in_K, 'P', P_in_Pa, FLUID)
        s_in = PropsSI('Smass', 'T', T_in_K, 'P', P_in_Pa, FLUID)
        rho_in = PropsSI('Dmass', 'T', T_in_K, 'P', P_in_Pa, FLUID)
        h_out_s = PropsSI('Hmass', 'P', P_out_Pa, 'Smass', s_in, FLUID)
        dh_s = h_out_s - h_in
        psi_design = _head_star(PHI_DESIGN)
        if dh_s <= 0.0 or psi_design <= 0.0:
            raise ValueError('Unphysical isentropic work or map coefficient.')

        U_design = math.sqrt(dh_s / psi_design)  # m/s
        D = math.sqrt(mdot_kgs / (PHI_DESIGN * rho_in * U_design))  # m
        N_design = 60.0 * U_design / (math.pi * D)  # rpm
        return cls(D, N_design, eta_design)

    def off_design(
        self,
        T_in_K: float,
        P_in_Pa: float,
        mdot_kgs: float,
        N_rpm: float,
        *,
        strict: bool = True,
    ) -> CompressorResult:
        """Predict the outlet state from inlet state, flow and shaft speed.

        In strict mode, surge and excessive flow yield ValueError. With
        strict=False, phi below the surge boundary is clipped as in Dyreby's
        original routine and flagged as *invalid*. Do not optimize through it.
        """
        if min(T_in_K, P_in_Pa, mdot_kgs, N_rpm) <= 0.0:
            raise ValueError('All off-design inputs must be positive.')
        if self.diameter_m <= 0.0 or self.N_design_rpm <= 0.0:
            raise ValueError('Invalid compressor geometry/speed.')

        h_in = PropsSI('Hmass', 'T', T_in_K, 'P', P_in_Pa, FLUID)
        s_in = PropsSI('Smass', 'T', T_in_K, 'P', P_in_Pa, FLUID)
        rho_in = PropsSI('Dmass', 'T', T_in_K, 'P', P_in_Pa, FLUID)
        U = math.pi * self.diameter_m * N_rpm / 60.0
        phi = mdot_kgs / (rho_in * U * self.diameter_m**2)

        surge = phi < PHI_MIN
        high_flow = phi > PHI_MAX
        if strict and (surge or high_flow):
            raise ValueError(f'Outside SNL compressor flow range: phi={phi:.5f}, '
                             f'valid approximately [{PHI_MIN}, {PHI_MAX}].')
        if high_flow:
            # The fitted polynomial does not supply a valid solution here.
            raise ValueError(f'Flow exceeds SNL map range: phi={phi:.5f}.')

        phi_for_map = max(phi, PHI_MIN)  # Dyreby's below-surge convention
        ratio = N_rpm / self.N_design_rpm
        phi_star = phi_for_map * ratio**0.2
        psi_star = _head_star(phi_star)
        eta_star = _eta_star(phi_star)
        psi = psi_star * ratio ** ((20.0 * phi_star)**3)
        eta_relative = 1.47528 * eta_star * ratio ** ((20.0 * phi_star)**5)
        eta_is = self.eta_design * eta_relative

        if psi <= 0.0 or not (0.0 < eta_is <= 1.0):
            raise ValueError(f'Unphysical SNL map output: psi={psi:.4g}, '
                             f'eta={eta_is:.4g}.')

        dh_s = psi * U**2  # J/kg
        h_out_s = h_in + dh_s
        # Invert the isentropic outlet state (h_out_s, s_in) to find P_out.
        P_out = PropsSI('P', 'Hmass', h_out_s, 'Smass', s_in, FLUID)
        if not P_out > P_in_Pa:
            raise ValueError('Off-design compressor map did not raise pressure.')

        h_out = h_in + dh_s / eta_is
        T_out = PropsSI('T', 'P', P_out, 'Hmass', h_out, FLUID)
        a_out = PropsSI('A', 'P', P_out, 'Hmass', h_out, FLUID)
        tip_to_sound = U / a_out
        warnings = []
        if surge:
            warnings.append('Surge: point is non-operable; head/eta clipped to phi_min.')
        if tip_to_sound >= 1.0:
            warnings.append('Rotor tip speed exceeds outlet sound speed.')
        if abs(ratio - 1.0) > 0.5:
            warnings.append('Far from design speed; SNL empirical fit unverified here.')

        return CompressorResult(
            T_out_K=T_out, P_out_Pa=P_out, h_out_Jkg=h_out,
            eta_is=eta_is, mdot_kgs=mdot_kgs, N_rpm=N_rpm,
            phi=phi, phi_star=phi_star, psi=psi,
            power_W=mdot_kgs * (h_out - h_in), tip_speed_ms=U,
            tip_to_sound_out=tip_to_sound, surge=surge,
            valid=(not surge and tip_to_sound < 1.0),
            warnings=tuple(warnings),
        )
    


@dataclass(frozen=True)
class MultiStageResult:
    T_out_K: float
    P_out_Pa: float
    h_out_Jkg: float
    eta_is: float
    power_W: float
    N_rpm: float
    valid: bool
    stages: tuple[CompressorResult, ...]


@dataclass(frozen=True)
class DyrebyMultiStage:
    stages: tuple[DyrebyCompressor, ...]
    N_design_rpm: float

    @classmethod
    def from_design(
        cls,
        T_in_K,
        P_in_Pa,
        P_out_Pa,
        mdot_kgs,
        n_stages=2,
        eta_design=0.90,
    ):
        if not isinstance(n_stages, int) or n_stages < 1:
            raise ValueError("n_stages must be a positive integer")
        if not 0 < eta_design < 1:
            raise ValueError("eta_design must be between 0 and 1")

        # Initial RPM estimate from the original one-stage model.
        single = DyrebyCompressor.from_design(
            T_in_K, P_in_Pa, P_out_Pa, mdot_kgs, eta_design
        )
        N0 = single.N_design_rpm / n_stages**0.75

        def build(stage_eta):

            def evaluate(N):
                T, P = T_in_K, P_in_Pa
                machines = []
                results = []

                for _ in range(n_stages):
                    rho = PropsSI(
                        'Dmass', 'T', T, 'P', P, FLUID
                    )

                    # Same N for all stages. Each stage has its
                    # own diameter and nominal flow coefficient.
                    D = (
                        60.0 * mdot_kgs
                        / (math.pi * N * PHI_DESIGN * rho)
                    )**(1.0 / 3.0)

                    stage = DyrebyCompressor(
                        diameter_m=D,
                        N_design_rpm=N,
                        eta_design=stage_eta
                    )

                    r = stage.off_design(
                        T_in_K=T,
                        P_in_Pa=P,
                        mdot_kgs=mdot_kgs,
                        N_rpm=N
                    )

                    machines.append(stage)
                    results.append(r)

                    T = r.T_out_K
                    P = r.P_out_Pa

                return tuple(machines), tuple(results)

            def pressure_error(N):
                _, results = evaluate(N)
                return math.log(results[-1].P_out_Pa / P_out_Pa)

            # Find common shaft speed giving target outlet pressure.
            lo = hi = N0
            error0 = pressure_error(N0)

            if abs(error0) <= 1e-10:
                N = N0
            else:
                for _ in range(50):
                    if error0 < 0:
                        hi *= 1.15
                        if pressure_error(hi) > 0:
                            break
                    else:
                        lo /= 1.15
                        if pressure_error(lo) < 0:
                            break
                else:
                    raise RuntimeError(
                        "Could not bracket nominal compressor speed"
                    )

                N = brentq(pressure_error, lo, hi)

            machines, _ = evaluate(N)
            return cls(machines, N)

        if n_stages == 1:
            return build(eta_design)

        # Calibrate stage efficiency so the complete compressor
        # has the requested overall isentropic efficiency.
        def efficiency_error(stage_eta):
            candidate = build(stage_eta)
            r = candidate.off_design(
                T_in_K, P_in_Pa, mdot_kgs,
                candidate.N_design_rpm
            )
            return r.eta_is - eta_design

        stage_eta = brentq(
            efficiency_error, eta_design, 0.99999
        )

        return build(stage_eta)

    def off_design(
        self,
        T_in_K,
        P_in_Pa,
        mdot_kgs,
        N_rpm,
        *,
        strict=True
    ):
        T, P = T_in_K, P_in_Pa

        h_in = PropsSI('Hmass', 'T', T, 'P', P, FLUID)
        s_in = PropsSI('Smass', 'T', T, 'P', P, FLUID)

        results = []

        for stage in self.stages:
            r = stage.off_design(
                T_in_K=T,
                P_in_Pa=P,
                mdot_kgs=mdot_kgs,
                N_rpm=N_rpm,
                strict=strict
            )

            results.append(r)

            # Outlet of this stage = inlet of next stage.
            T = r.T_out_K
            P = r.P_out_Pa

        h_out = results[-1].h_out_Jkg

        # Overall isentropic efficiency, not the stage average.
        h_out_s = PropsSI(
            'Hmass', 'P', P, 'Smass', s_in, FLUID
        )

        eta_total = (h_out_s - h_in) / (h_out - h_in)

        return MultiStageResult(
            T_out_K=T,
            P_out_Pa=P,
            h_out_Jkg=h_out,
            eta_is=eta_total,
            power_W=mdot_kgs * (h_out - h_in),
            N_rpm=N_rpm,
            valid=all(r.valid for r in results),
            stages=tuple(results)
        )

