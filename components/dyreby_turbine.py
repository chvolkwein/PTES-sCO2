"""Dyreby (2014) SNL radial sCO2 turbine: design and off-design.

Based on the alternative turbine model in Appendix V, snl_radial_turbine.f90.

The model uses a fixed effective nozzle area and the SNL polynomial
for turbine efficiency. It assumes unchoked flow and does not enforce
an experimentally validated operating envelope.

Install dependencies with: pip install CoolProp
"""

from dataclasses import dataclass
import math
from typing import Tuple

from CoolProp.CoolProp import PropsSI


FLUID = "CO2"
NU_DESIGN = 0.7476


def _efficiency_factor(nu: float) -> float:
    """Return the SNL efficiency polynomial before Dyreby's [0, 1] clipping."""
    return (
        (((1.0626 * nu - 3.0874) * nu + 1.3668) * nu + 1.3567) * nu
        + 0.179921180
    )


def _validate_positive(**values: float) -> None:
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive; received {value}.")


@dataclass(frozen=True)
class TurbineResult:
    T_out_K: float
    P_out_Pa: float
    h_out_Jkg: float
    eta_is: float
    mdot_kgs: float
    N_rpm: float
    nu: float
    spouting_velocity_ms: float
    tip_speed_ms: float
    tip_to_sound_in: float
    power_W: float
    valid: bool
    warnings: Tuple[str, ...]


@dataclass(frozen=True)
class DyrebyTurbine:
    """Single-stage SNL radial turbine with fixed design-point geometry.

    Attributes:
        diameter_m: Turbine rotor diameter, fixed after sizing.
        nozzle_area_m2: Effective nozzle area, fixed after sizing.
        N_design_rpm: Design shaft speed; off-design speed may differ.
        eta_design: Design isentropic efficiency.
    """

    diameter_m: float
    nozzle_area_m2: float
    N_design_rpm: float
    eta_design: float

    @classmethod
    def from_design(
        cls,
        T_in_K: float,
        P_in_Pa: float,
        P_out_Pa: float,
        mdot_kgs: float,
        N_design_rpm: float,
        eta_design: float = 0.90,
    ) -> "DyrebyTurbine":
        """Size rotor diameter and effective nozzle area at the design point."""
        _validate_positive(
            T_in_K=T_in_K,
            P_in_Pa=P_in_Pa,
            P_out_Pa=P_out_Pa,
            mdot_kgs=mdot_kgs,
            N_design_rpm=N_design_rpm,
            eta_design=eta_design,
        )
        if P_out_Pa >= P_in_Pa:
            raise ValueError("Turbine design outlet pressure must be below inlet pressure.")
        if eta_design > 1.0:
            raise ValueError("eta_design must not exceed 1.")

        h_in = PropsSI("Hmass", "T", T_in_K, "P", P_in_Pa, FLUID)
        s_in = PropsSI("Smass", "T", T_in_K, "P", P_in_Pa, FLUID)
        rho_in = PropsSI("Dmass", "T", T_in_K, "P", P_in_Pa, FLUID)
        h_out_s = PropsSI("Hmass", "P", P_out_Pa, "Smass", s_in, FLUID)

        delta_h_s = h_in - h_out_s
        if not math.isfinite(delta_h_s) or delta_h_s <= 0.0:
            raise ValueError("Design expansion must have a positive isentropic enthalpy drop.")

        C_s = math.sqrt(2.0 * delta_h_s)
        U = NU_DESIGN * C_s
        diameter_m = 60.0 * U / (math.pi * N_design_rpm)
        nozzle_area_m2 = mdot_kgs / (rho_in * C_s)

        return cls(
            diameter_m=diameter_m,
            nozzle_area_m2=nozzle_area_m2,
            N_design_rpm=N_design_rpm,
            eta_design=eta_design,
        )

    def off_design(
        self,
        T_in_K: float,
        P_in_Pa: float,
        P_out_Pa: float,
        N_rpm: float,
        *,
        strict: bool = True,
    ) -> TurbineResult:
        """Calculate allowable mass flow and outlet state for a specified outlet pressure.

        Off-design inputs are T_in, P_in, P_out and N. The design rotor
        diameter and effective nozzle area remain fixed.

        Basic screening is provided, but choking and map validity are NOT
        established by this model.
        """
        _validate_positive(
            T_in_K=T_in_K,
            P_in_Pa=P_in_Pa,
            P_out_Pa=P_out_Pa,
            N_rpm=N_rpm,
        )
        if P_out_Pa >= P_in_Pa:
            raise ValueError("Turbine outlet pressure must be below inlet pressure.")

        h_in = PropsSI("Hmass", "T", T_in_K, "P", P_in_Pa, FLUID)
        s_in = PropsSI("Smass", "T", T_in_K, "P", P_in_Pa, FLUID)
        rho_in = PropsSI("Dmass", "T", T_in_K, "P", P_in_Pa, FLUID)
        a_in = PropsSI("A", "T", T_in_K, "P", P_in_Pa, FLUID)
        h_out_s = PropsSI("Hmass", "P", P_out_Pa, "Smass", s_in, FLUID)

        delta_h_s = h_in - h_out_s
        if not math.isfinite(delta_h_s) or delta_h_s <= 0.0:
            raise ValueError("Expansion must have a positive isentropic enthalpy drop.")

        C_s = math.sqrt(2.0 * delta_h_s)
        U = math.pi * self.diameter_m * N_rpm / 60.0
        nu = U / C_s
        raw_factor = _efficiency_factor(nu)
        factor = max(0.0, min(1.0, raw_factor))
        eta_is = self.eta_design * factor

        warnings = []
        if raw_factor <= 0.0:
            warnings.append("Efficiency correlation returned a nonpositive value.")
            if strict:
                raise ValueError(warnings[-1])
        elif raw_factor > 1.0 + 1e-6:
            warnings.append("Efficiency correlation exceeded 1 and was clipped.")

        tip_to_sound = U / a_in
        if tip_to_sound >= 1.0:
            warnings.append(
                "Rotor tip speed equals or exceeds inlet sound speed "
                "(simplified screening only)."
            )

        mdot_kgs = rho_in * self.nozzle_area_m2 * C_s
        h_out = h_in - eta_is * delta_h_s
        T_out = PropsSI("T", "P", P_out_Pa, "Hmass", h_out, FLUID)
        power_W = mdot_kgs * (h_in - h_out)

        return TurbineResult(
            T_out_K=T_out,
            P_out_Pa=P_out_Pa,
            h_out_Jkg=h_out,
            eta_is=eta_is,
            mdot_kgs=mdot_kgs,
            N_rpm=N_rpm,
            nu=nu,
            spouting_velocity_ms=C_s,
            tip_speed_ms=U,
            tip_to_sound_in=tip_to_sound,
            power_W=power_W,
            valid=(raw_factor > 0.0 and tip_to_sound < 1.0),
            warnings=tuple(warnings),
        )

    def off_design_from_mdot(
        self,
        T_in_K: float,
        P_in_Pa: float,
        mdot_kgs: float,
        N_rpm: float,
        *,
        strict: bool = True,
    ) -> TurbineResult:
        """Invert the SNL flow relation to determine outlet pressure.

        The use of INLET density in Dyreby's SNL variant makes the
        inversion direct: Cs = mdot/(rho_in * nozzle_area).
        This requires a valid CoolProp (h, s) flash.
        """
        _validate_positive(
            T_in_K=T_in_K,
            P_in_Pa=P_in_Pa,
            mdot_kgs=mdot_kgs,
            N_rpm=N_rpm,
        )

        h_in = PropsSI("Hmass", "T", T_in_K, "P", P_in_Pa, FLUID)
        s_in = PropsSI("Smass", "T", T_in_K, "P", P_in_Pa, FLUID)
        rho_in = PropsSI("Dmass", "T", T_in_K, "P", P_in_Pa, FLUID)

        C_s = mdot_kgs / (rho_in * self.nozzle_area_m2)
        h_out_s = h_in - 0.5 * C_s**2
        P_out_Pa = PropsSI("P", "Hmass", h_out_s, "Smass", s_in, FLUID)

        if not math.isfinite(P_out_Pa) or not 0.0 < P_out_Pa < P_in_Pa:
            raise ValueError(
                "No valid expansion outlet pressure was found for the specified mass flow."
            )

        return self.off_design(
            T_in_K=T_in_K,
            P_in_Pa=P_in_Pa,
            P_out_Pa=P_out_Pa,
            N_rpm=N_rpm,
            strict=strict,
        )
