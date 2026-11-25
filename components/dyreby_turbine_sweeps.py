"""Dyreby (2014) SNL radial turbine: design and one-variable off-design sweeps.

Requirements:
    pip install CoolProp numpy matplotlib

Place dyreby_turbine.py in the same folder as this script.
VS Code and Spyder recognize '# %%' as cell separators.

The design geometry (rotor diameter and effective nozzle area) is calculated
once and remains unchanged throughout all off-design sweeps.

The original Dyreby turbine model uses T_in, P_in, P_out, N -> mass flow.
The inverse method uses T_in, P_in, mass flow, N -> P_out.
"""

# %% 1. IMPORTS AND TURBINE DESIGN
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dyreby_turbine import DyrebyTurbine

# Design point: edit these values.
T_IN_DESIGN_C = 400.0           # [degC] turbine inlet temperature
P_IN_DESIGN_MPA = 24.4         # [MPa] turbine inlet pressure
P_OUT_DESIGN_MPA = 8.0         # [MPa] turbine outlet pressure
MDOT_DESIGN_KGS = 100.0        # [kg/s] design mass flow
N_DESIGN_RPM = 3600.0         # [rpm] design shaft speed
ETA_DESIGN = 0.90              # [-] design isentropic efficiency

# Design the turbine only once. Geometry is stored inside turb.
turb = DyrebyTurbine.from_design(
    T_in_K=T_IN_DESIGN_C + 273.15,
    P_in_Pa=P_IN_DESIGN_MPA * 1e6,
    P_out_Pa=P_OUT_DESIGN_MPA * 1e6,
    mdot_kgs=MDOT_DESIGN_KGS,
    N_design_rpm=N_DESIGN_RPM,
    eta_design=ETA_DESIGN,
)

print("\n=== TURBINE DESIGN ===")
print(f"Rotor diameter: {turb.diameter_m:.5f} m")
print(f"Effective nozzle area: {turb.nozzle_area_m2:.7f} m^2")
print(f"Design speed: {turb.N_design_rpm:.0f} rpm")

# Confirm that the off-design method reproduces the design conditions.
nominal = turb.off_design(
    T_in_K=T_IN_DESIGN_C + 273.15,
    P_in_Pa=P_IN_DESIGN_MPA * 1e6,
    P_out_Pa=P_OUT_DESIGN_MPA * 1e6,
    N_rpm=turb.N_design_rpm,
)

print(f"Design expansion ratio: {P_IN_DESIGN_MPA / P_OUT_DESIGN_MPA:.4f}")
print(f"Calculated mass flow: {nominal.mdot_kgs:.3f} kg/s")
print(f"Calculated efficiency: {nominal.eta_is * 100:.2f} %")
print(f"Calculated power: {nominal.power_W / 1e6:.3f} MW")
print(f"Design velocity ratio (U/Cs): {nominal.nu:.4f}")
print(f"Design tip-to-inlet-sound ratio: {nominal.tip_to_sound_in:.3f}")
print(f"Basic checks passed: {nominal.valid}")
if nominal.warnings:
    print("Warnings:", "; ".join(nominal.warnings))


# %% 2. SWEEP CONFIGURATION
# Select ONE sweep: 'T', 'P', 'M', 'N', or 'POUT'.
SWEEP = "T"

# Fixed speed for T, P, M, and POUT sweeps.
# N/N_design = 1.0 means design speed; 0.8 means 80% of design speed.
SPEED_RATIO = 1.0
FIXED_N_RPM = turb.N_design_rpm * SPEED_RATIO

# T sweep: vary inlet temperature; hold inlet pressure and mass flow fixed.
T_MIN_C = T_IN_DESIGN_C - 20.0
T_MAX_C = T_IN_DESIGN_C + 20.0
T_NUM_POINTS = 45
T_FIXED_P_IN_MPA = P_IN_DESIGN_MPA
T_FIXED_MDOT_KGS = MDOT_DESIGN_KGS

# P sweep: vary inlet pressure; hold inlet temperature and mass flow fixed.
P_MIN_MPA = 20.0
P_MAX_MPA = 28.0
P_NUM_POINTS = 45
P_FIXED_T_IN_C = T_IN_DESIGN_C
P_FIXED_MDOT_KGS = MDOT_DESIGN_KGS

# M sweep: vary mass flow; hold inlet temperature and pressure fixed.
M_MIN_KGS = 75.0
M_MAX_KGS = 125.0
M_NUM_POINTS = 45
M_FIXED_T_IN_C = T_IN_DESIGN_C
M_FIXED_P_IN_MPA = P_IN_DESIGN_MPA

# N sweep: vary speed ratio; hold inlet temperature, pressure, and flow fixed.
N_MIN_RATIO = 0.70
N_MAX_RATIO = 1.10
N_NUM_POINTS = 45
N_FIXED_T_IN_C = T_IN_DESIGN_C
N_FIXED_P_IN_MPA = P_IN_DESIGN_MPA
N_FIXED_MDOT_KGS = MDOT_DESIGN_KGS

# POUT sweep: vary outlet pressure; hold inlet temperature and pressure fixed.
# In this mode, Dyreby's original model calculates mass flow as an output.
POUT_MIN_MPA = 6.8
POUT_MAX_MPA = 10.0
POUT_NUM_POINTS = 45
POUT_FIXED_T_IN_C = T_IN_DESIGN_C
POUT_FIXED_P_IN_MPA = P_IN_DESIGN_MPA

OUTPUT_DIR = Path(".")


# %% 3. SWEEP CALCULATION AND PLOTTING

def run_sweep(sweep):
    sweep = sweep.upper()

    if sweep == "T":
        x = np.linspace(T_MIN_C, T_MAX_C, T_NUM_POINTS)
        T_values_C = x
        P_values_MPa = np.full_like(x, T_FIXED_P_IN_MPA)
        m_values_kgs = np.full_like(x, T_FIXED_MDOT_KGS)
        N_values_rpm = np.full_like(x, FIXED_N_RPM)
        Pout_values_MPa = None
        xlabel = "Turbine inlet temperature [degC]"
        nominal_x = T_IN_DESIGN_C
        filename = "temperature"

    elif sweep == "P":
        x = np.linspace(P_MIN_MPA, P_MAX_MPA, P_NUM_POINTS)
        T_values_C = np.full_like(x, P_FIXED_T_IN_C)
        P_values_MPa = x
        m_values_kgs = np.full_like(x, P_FIXED_MDOT_KGS)
        N_values_rpm = np.full_like(x, FIXED_N_RPM)
        Pout_values_MPa = None
        xlabel = "Turbine inlet pressure [MPa]"
        nominal_x = P_IN_DESIGN_MPA
        filename = "inlet_pressure"

    elif sweep == "M":
        x = np.linspace(M_MIN_KGS, M_MAX_KGS, M_NUM_POINTS)
        T_values_C = np.full_like(x, M_FIXED_T_IN_C)
        P_values_MPa = np.full_like(x, M_FIXED_P_IN_MPA)
        m_values_kgs = x
        N_values_rpm = np.full_like(x, FIXED_N_RPM)
        Pout_values_MPa = None
        xlabel = "Turbine mass flow [kg/s]"
        nominal_x = MDOT_DESIGN_KGS
        filename = "mass_flow"

    elif sweep == "N":
        x = np.linspace(N_MIN_RATIO, N_MAX_RATIO, N_NUM_POINTS)
        T_values_C = np.full_like(x, N_FIXED_T_IN_C)
        P_values_MPa = np.full_like(x, N_FIXED_P_IN_MPA)
        m_values_kgs = np.full_like(x, N_FIXED_MDOT_KGS)
        N_values_rpm = x * turb.N_design_rpm
        Pout_values_MPa = None
        xlabel = "Shaft speed / design speed [-]"
        nominal_x = 1.0
        filename = "shaft_speed"

    elif sweep == "POUT":
        x = np.linspace(POUT_MIN_MPA, POUT_MAX_MPA, POUT_NUM_POINTS)
        T_values_C = np.full_like(x, POUT_FIXED_T_IN_C)
        P_values_MPa = np.full_like(x, POUT_FIXED_P_IN_MPA)
        m_values_kgs = np.full_like(x, np.nan)
        N_values_rpm = np.full_like(x, FIXED_N_RPM)
        Pout_values_MPa = x
        xlabel = "Turbine outlet pressure [MPa]"
        nominal_x = P_OUT_DESIGN_MPA
        filename = "outlet_pressure"

    else:
        raise ValueError("SWEEP must be 'T', 'P', 'M', 'N', or 'POUT'.")

    rows = []
    for i, xi in enumerate(x):
        T_C = float(T_values_C[i])
        P_MPa = float(P_values_MPa[i])
        mdot = float(m_values_kgs[i])
        N_rpm = float(N_values_rpm[i])

        row = {
            "x": float(xi),
            "T_in_C": T_C,
            "P_in_MPa": P_MPa,
            "mdot_kg_s": mdot,
            "N_rpm": N_rpm,
            "P_out_MPa": np.nan,
            "expansion_ratio": np.nan,
            "T_out_C": np.nan,
            "eta_percent": np.nan,
            "power_MW": np.nan,
            "velocity_ratio_U_over_Cs": np.nan,
            "tip_to_sound_in": np.nan,
            "valid": False,
            "warnings": "",
        }

        try:
            if sweep == "POUT":
                # Original Dyreby direction: P_out is specified; mdot is calculated.
                result = turb.off_design(
                    T_in_K=T_C + 273.15,
                    P_in_Pa=P_MPa * 1e6,
                    P_out_Pa=float(Pout_values_MPa[i]) * 1e6,
                    N_rpm=N_rpm,
                    strict=True,
                )
            else:
                # Inverse direction: mdot is specified; P_out is calculated.
                result = turb.off_design_from_mdot(
                    T_in_K=T_C + 273.15,
                    P_in_Pa=P_MPa * 1e6,
                    mdot_kgs=mdot,
                    N_rpm=N_rpm,
                    strict=True,
                )

            row.update({
                "mdot_kg_s": float(result.mdot_kgs),
                "P_out_MPa": float(result.P_out_Pa / 1e6),
                "expansion_ratio": float(P_MPa * 1e6 / result.P_out_Pa),
                "T_out_C": float(result.T_out_K - 273.15),
                "eta_percent": float(result.eta_is * 100),
                "power_MW": float(result.power_W / 1e6),
                "velocity_ratio_U_over_Cs": float(result.nu),
                "tip_to_sound_in": float(result.tip_to_sound_in),
                "valid": bool(result.valid),
                "warnings": "; ".join(result.warnings),
            })
        except (ValueError, RuntimeError, OverflowError) as exc:
            row["warnings"] = str(exc)

        rows.append(row)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"dyreby_turbine_sweep_{filename}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    valid_rows = [r for r in rows if r["valid"]]
    invalid_rows = [r for r in rows if not r["valid"]]

    print(f"\n=== {filename.upper().replace('_', ' ')} SWEEP ===")
    if sweep != "N":
        print(f"Fixed speed: {FIXED_N_RPM:.0f} rpm ({100*SPEED_RATIO:.0f}% of design)")
    print(f"Valid points (basic screening only): {len(valid_rows)}/{len(rows)}")
    print(f"CSV: {csv_path.resolve()}")
    if valid_rows:
        print(
            f"Efficiency: {min(r['eta_percent'] for r in valid_rows):.2f}% "
            f"to {max(r['eta_percent'] for r in valid_rows):.2f}%"
        )
        print(
            f"Expansion ratio: {min(r['expansion_ratio'] for r in valid_rows):.3f} "
            f"to {max(r['expansion_ratio'] for r in valid_rows):.3f}"
        )
    for row in invalid_rows[:5]:
        print(f"  Invalid at x={row['x']:.4f}: {row['warnings']}")
    if len(invalid_rows) > 5:
        print(f"  ...and {len(invalid_rows) - 5} additional invalid points (see CSV).")

    x_plot = np.array([r["x"] for r in rows], dtype=float)

    def valid_series(key):
        return np.array(
            [r[key] if r["valid"] else np.nan for r in rows], dtype=float
        )

    quantities = [
        ("eta_percent", "Isentropic efficiency [%]"),
        ("P_out_MPa", "Turbine outlet pressure [MPa]"),
        ("expansion_ratio", "Expansion ratio P_in / P_out [-]"),
        ("mdot_kg_s", "Turbine mass flow [kg/s]"),
        ("power_MW", "Turbine power output [MW]"),
        ("velocity_ratio_U_over_Cs", "Velocity ratio U / Cs [-]"),
        ("tip_to_sound_in", "Tip speed / inlet sound speed [-]"),
    ]

    fig, axes = plt.subplots(len(quantities), 1, figsize=(10, 16), sharex=True)
    for ax, (key, ylabel) in zip(axes, quantities):
        ax.plot(x_plot, valid_series(key), "-o", ms=3)
        if x.min() <= nominal_x <= x.max():
            ax.axvline(
                nominal_x, color="0.55", linestyle=":", linewidth=1,
                label="Design value of swept variable",
            )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    axes[-1].axhline(1.0, color="tab:red", linestyle="--", linewidth=1,
                      label="Simplified tip-to-sound threshold")
    axes[-1].legend(loc="best")
    axes[-1].set_xlabel(xlabel)
    fig.suptitle(
        f"Dyreby SNL turbine: {filename.replace('_', ' ')} sweep",
        fontsize=13,
    )
    fig.tight_layout()
    png_path = OUTPUT_DIR / f"dyreby_turbine_sweep_{filename}.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    print(f"Figure: {png_path.resolve()}")
    plt.show()
    return rows


# %% 4. RUN THE SELECTED SWEEP
sweep_results = run_sweep(SWEEP)
