"""Dyreby (2014) compressor sweeps: inlet temperature, inlet pressure, or mass flow.

Requires dyreby_compressor.py in the same directory.
Install dependencies with: pip install CoolProp numpy matplotlib

In VS Code or Spyder, '# %%' markers divide the script into cells.
Compressor geometry is calculated ONLY ONCE. Each sweep keeps the geometry
and shaft speed fixed; pressure ratio and efficiency are calculated outputs.
"""

# %% 1. IMPORTS AND DESIGN POINT
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from dyreby_compressor import DyrebyCompressor
from dyreby_compressor import DyrebyMultiStage

# === Design point (edit these values) ===
T_DES_C = 673.40389673364 - 273.15  # [degC] inlet temperature
P_DES_MPA = 8.0  # [MPa] inlet pressure
P_OUT_DES_MPA = 26399758.3642865 / 1e6  # [MPa] outlet pressure
MDOT_DES = 100.0  # [kg/s] mass flow rate
ETA_DES = 0.90  # [-] design isentropic efficiency

# comp = DyrebyCompressor.from_design(
#     T_in_K=T_DES_C + 273.15,
#     P_in_Pa=P_DES_MPA * 1e6,
#     P_out_Pa=P_OUT_DES_MPA * 1e6,
#     mdot_kgs=MDOT_DES,
#     eta_design=ETA_DES,
# )

comp = DyrebyMultiStage.from_design(
    T_in_K=T_DES_C + 273.15,
    P_in_Pa=P_DES_MPA * 1e6,
    P_out_Pa=P_OUT_DES_MPA * 1e6,
    mdot_kgs=MDOT_DES,
    n_stages=3,
    eta_design=ETA_DES,
)

# print('\n=== SIZED COMPRESSOR ===')
# print(f'D = {comp.diameter_m:.5f} m')
# print(f'N_design = {comp.N_design_rpm:.0f} rpm')

for i, stage in enumerate(comp.stages, 1):
    print(f'Stage {i} diameter = {stage.diameter_m:.5f} m')

print(f'N_design = {comp.N_design_rpm:.0f} rpm')

# Shaft speed is held constant during each sweep.
# 1.00 = design speed; 0.90 = 90%; 1.10 = 110%.
N_RATIO = 1.0
N_FIXED_RPM = comp.N_design_rpm * N_RATIO

# Check whether off_design reproduces the design point.
try:
    nominal = comp.off_design(
        T_in_K=T_DES_C + 273.15,
        P_in_Pa=P_DES_MPA * 1e6,
        mdot_kgs=MDOT_DES,
        N_rpm=comp.N_design_rpm,
    )
    print(f'Calculated nominal PR = {nominal.P_out_Pa / (P_DES_MPA * 1e6):.4f}')
    print(f'Nominal efficiency = {nominal.eta_is * 100:.2f} %')
    print(f'Nominal power = {nominal.power_W / 1e6:.3f} MW')
    # print(f'Nominal U/a_out = {nominal.tip_to_sound_out:.3f}')
    # if not nominal.valid:
    #     print('WARNING: The design point does not pass the map checks.')
    #     print('Reasons:', '; '.join(nominal.warnings))

    for i, stage_result in enumerate(nominal.stages, 1):
        print(f'\nStage {i}:')
        print(f'  Efficiency = {stage_result.eta_is * 100:.2f} %')
        print(f'  U/a_out = {stage_result.tip_to_sound_out:.3f}')
        print(f'  phi = {stage_result.phi:.5f}')
        print(f'  Valid = {stage_result.valid}')

        if not stage_result.valid:
            print('  Reasons:', '; '.join(stage_result.warnings))

except ValueError as exc:
    print(f'WARNING: Invalid design point for this model: {exc}')


# %% 2. SWEEP SETTINGS
# Choose ONE: 'T', 'P', or 'M'. Running the complete script performs only that sweep.
SWEEP = 'T'

# SWEEP = 'T': vary inlet temperature; hold inlet pressure and mass flow fixed.
T_MIN_C = T_DES_C - 20
T_MAX_C = T_DES_C + 20
T_N_POINTS = 45
T_FIXED_P_MPA = 8.0
T_FIXED_MDOT = MDOT_DES * N_RATIO

# SWEEP = 'P': vary inlet pressure; hold inlet temperature and mass flow fixed.
P_MIN_MPA = 7.6
P_MAX_MPA = 10.0
P_N_POINTS = 45
P_FIXED_T_C = T_DES_C
P_FIXED_MDOT = 100.0

# SWEEP = 'M': vary mass flow; hold inlet temperature and pressure fixed.
M_MIN_KGS = 75.0
M_MAX_KGS = 120.0
M_N_POINTS = 45
M_FIXED_T_C = T_DES_C
M_FIXED_P_MPA = 8.0

# Output directory for CSV and PNG files.
OUTPUT_DIR = Path('.')


# %% 3. FUNCTION TO RUN AND PLOT ONE SWEEP
def run_sweep(sweep_type):
    sweep_type = sweep_type.upper()

    if sweep_type == 'T':
        x = np.linspace(T_MIN_C, T_MAX_C, T_N_POINTS)
        inlet_temperatures_c = x
        inlet_pressures_mpa = np.full_like(x, T_FIXED_P_MPA)
        mass_flows_kgs = np.full_like(x, T_FIXED_MDOT)
        x_label = 'Inlet temperature [°C]'
        x_nominal = T_DES_C
        sweep_name = 'temperature'

    elif sweep_type == 'P':
        x = np.linspace(P_MIN_MPA, P_MAX_MPA, P_N_POINTS)
        inlet_temperatures_c = np.full_like(x, P_FIXED_T_C)
        inlet_pressures_mpa = x
        mass_flows_kgs = np.full_like(x, P_FIXED_MDOT)
        x_label = 'Inlet pressure [MPa]'
        x_nominal = P_DES_MPA
        sweep_name = 'pressure'

    elif sweep_type == 'M':
        x = np.linspace(M_MIN_KGS, M_MAX_KGS, M_N_POINTS)
        inlet_temperatures_c = np.full_like(x, M_FIXED_T_C)
        inlet_pressures_mpa = np.full_like(x, M_FIXED_P_MPA)
        mass_flows_kgs = x
        x_label = 'CO₂ mass flow rate [kg/s]'
        x_nominal = MDOT_DES
        sweep_name = 'mass_flow'

    else:
        raise ValueError("SWEEP must be 'T', 'P', or 'M'.")

    data = []
    for xi, inlet_temp, inlet_pressure, mass_flow in zip(
        x, inlet_temperatures_c, inlet_pressures_mpa, mass_flows_kgs
    ):
        row = {
            'x': float(xi),
            'T_in_C': float(inlet_temp),
            'P_in_MPa': float(inlet_pressure),
            'mdot_kg_s': float(mass_flow),
            'N_rpm': float(N_FIXED_RPM),
            'PR': np.nan,
            'P_out_MPa': np.nan,
            'eta_percent': np.nan,
            'power_MW': np.nan,
            'phi': np.nan,
            'U_over_a_out': np.nan,
            'valid': False,
            'reason': '',
        }
        try:
            output = comp.off_design(
                T_in_K=float(inlet_temp + 273.15),
                P_in_Pa=float(inlet_pressure * 1e6),
                mdot_kgs=float(mass_flow),
                N_rpm=N_FIXED_RPM,
                strict=True,  # Do not force predictions outside the surge/choke map limits.
            )
            row.update({
                'PR': float(output.P_out_Pa / (inlet_pressure * 1e6)),
                'P_out_MPa': float(output.P_out_Pa / 1e6),
                'eta_percent': float(output.eta_is * 100),
                'power_MW': float(output.power_W / 1e6),

                # Maximum flow coefficient across all stages.
                'phi': max(stage.phi for stage in output.stages),

                # Maximum tip-to-sound-speed ratio across all stages.
                'U_over_a_out': max(
                    stage.tip_to_sound_out for stage in output.stages
                ),

                'valid': bool(output.valid),

                'reason': '; '.join(
                    f'Stage {i}: {warning}'
                    for i, stage in enumerate(output.stages, 1)
                    for warning in stage.warnings
                ),
            })
        except (ValueError, RuntimeError) as exc:
            row['reason'] = str(exc)
        data.append(row)

    # Save all rows, including invalid operating points.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f'dyreby_sweep_{sweep_name}.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)

    valid_rows = [row for row in data if row['valid']]
    invalid_rows = [row for row in data if not row['valid']]
    print(f'\n=== {sweep_name.upper()} SWEEP ===')
    print(f'Fixed speed = {N_FIXED_RPM:.0f} rpm ({100 * N_RATIO:.0f}% of design speed)')
    print(f'Valid points: {len(valid_rows)}/{len(data)}')
    print(f'CSV: {csv_path.resolve()}')
    if valid_rows:
        print(
            f'Valid efficiency range: {min(row["eta_percent"] for row in valid_rows):.2f} '
            f'to {max(row["eta_percent"] for row in valid_rows):.2f} %'
        )
        print(
            f'Valid PR range: {min(row["PR"] for row in valid_rows):.3f} '
            f'to {max(row["PR"] for row in valid_rows):.3f}'
        )
    for row in invalid_rows[:5]:
        print(f'  Invalid point x={row["x"]:.3f}: {row["reason"]}')
    if len(invalid_rows) > 5:
        print(f'  ...and {len(invalid_rows) - 5} additional invalid points (see CSV).')

    # Invalid points appear as gaps in the plots.
    x_plot = np.array([row['x'] for row in data])

    def valid_values(key):
        return np.array([
            row[key] if row['valid'] else np.nan
            for row in data
        ], dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(9, 12), sharex=True)
    for ax, key, ylabel in zip(
        axes,
        ('eta_percent', 'PR', 'power_MW', 'U_over_a_out'),
        ('Isentropic efficiency [%]', 'Pressure ratio [-]',
         'Compressor power consumption [MW]', 'Tip speed / sound speed [-]'),
    ):
        ax.plot(x_plot, valid_values(key), '-o', ms=3)
        if x.min() <= x_nominal <= x.max():
            ax.axvline(
                x_nominal, color='0.55', linestyle=':',
                label='Nominal value of swept variable'
            )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    axes[3].axhline(
        1.0, color='tab:red', linestyle='--',
        label='Simplified U/a = 1 threshold'
    )
    axes[3].legend(loc='best')
    axes[-1].set_xlabel(x_label)
    fig.suptitle(f'Dyreby: {sweep_name} sweep (fixed N = {N_FIXED_RPM:.0f} rpm)')
    fig.tight_layout()
    png_path = OUTPUT_DIR / f'dyreby_sweep_{sweep_name}.png'
    fig.savefig(png_path, dpi=180, bbox_inches='tight')
    print(f'Figure: {png_path.resolve()}')
    plt.show()
    return data


# %% 4. RUN ONLY THE SELECTED SWEEP
sweep_results = run_sweep(SWEEP)
