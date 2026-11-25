
"""Dyreby (2014) compressor sweeps: T_in, P_in, or mass flow rate.

Requires dyreby_compressor.py in the same folder.
Installation: pip install CoolProp numpy matplotlib

In VS Code or Spyder, '# %%' markers divide the file into cells.
The compressor geometry is calculated ONLY ONCE. All sweeps
keep D and N constant; PR and efficiency are calculated outputs.
"""

# %% 1. IMPORTS AND NOMINAL DESIGN
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from dyreby_compressor import DyrebyCompressor
from dyreby_compressor import DyrebyMultiStage

# === Design point (modify these inputs) ===
T_DES_C = 358.962991820001 - 273.15 + 100  # [degC] inlet temperature
P_DES_MPA = 8.0  # [MPa] inlet pressure
P_OUT_DES_MPA = 24720984.3784001/1000000  # [MPa] outlet pressure; PR=3.05
MDOT_DES = 100  # [kg/s] mass flow rate
ETA_DES = 0.90  # [-] design isentropic efficiency

comp = DyrebyCompressor.from_design(
    T_in_K=T_DES_C + 273.15,
    P_in_Pa=P_DES_MPA * 1e6,
    P_out_Pa=P_OUT_DES_MPA * 1e6,
    mdot_kgs=MDOT_DES,
    eta_design=ETA_DES,
)

# comp = DyrebyMultiStage.from_design(
#     T_in_K=T_DES_C + 273.15,
#     P_in_Pa=P_DES_MPA * 1e6,
#     P_out_Pa=P_OUT_DES_MPA * 1e6,
#     mdot_kgs=MDOT_DES,
#     n_stages=3,
#     eta_design=ETA_DES
# )

print('\n=== COMPRESSOR DESIGN ===')
print(f'D = {comp.diameter_m:.5f} m')
print(f'N_design = {comp.N_design_rpm:.0f} rpm')

# for i, stage in enumerate(comp.stages, 1):
#     print(f'D stage {i} = {stage.diameter_m:.5f} m')

# print(f'N_design = {comp.N_design_rpm:.0f} rpm')

# Shaft speed is kept constant during each sweep.
# 1.00 = N_design; 0.90 = 90%; 1.10 = 110%.
N_RATIO = 0.75
N_FIXED_RPM = comp.N_design_rpm * N_RATIO

# Verify that the off_design method reproduces the design point.
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
    print(f'Nominal U/a_out = {nominal.tip_to_sound_out:.3f}')
    if not nominal.valid:
        print('WARNING: The nominal point does NOT pass the map validity checks.')
        print('Reasons:', '; '.join(nominal.warnings))

    # for i, r in enumerate(nominal.stages, 1):
    #     print(f'\nStage {i}:')
    #     print(f'  Efficiency = {r.eta_is * 100:.2f} %')
    #     print(f'  U/a_out = {r.tip_to_sound_out:.3f}')
    #     print(f'  phi = {r.phi:.5f}')
    #     print(f'  Valid = {r.valid}')

    #     if not r.valid:
    #         print('  Reasons:', '; '.join(r.warnings))

except ValueError as exc:
    print(f'WARNING: Invalid nominal point for this model: {exc}')


# %% 2. SWEEP CONFIGURATION
# Select ONE: 'T', 'P', or 'M'. Running the entire file executes ONLY that sweep.
SWEEP = 'T'

# SWEEP = 'T': vary T_in while keeping P_in and mdot constant.
T_MIN_C = T_DES_C - 20
T_MAX_C = T_DES_C + 20
T_N_POINTS = 45
T_FIXED_P_MPA = P_DES_MPA
T_FIXED_MDOT = MDOT_DES * N_RATIO 

# SWEEP = 'P': vary P_in while keeping T_in and mdot constant.
P_MIN_MPA = 7.6
P_MAX_MPA = 10.0
P_N_POINTS = 45
P_FIXED_T_C = 35.0
P_FIXED_MDOT = 100.0

# SWEEP = 'M': vary mdot while keeping T_in and P_in constant.
M_MIN_KGS = 75.0
M_MAX_KGS = 120.0
M_N_POINTS = 45
M_FIXED_T_C = 35.0
M_FIXED_P_MPA = 8.0

# Output directory for CSV and PNG files.
OUTPUT_DIR = Path('.')


# %% 3. FUNCTION TO CALCULATE AND PLOT A SWEEP
def run_sweep(sweep_type):
    sweep_type = sweep_type.upper()

    if sweep_type == 'T':
        x = np.linspace(T_MIN_C, T_MAX_C, T_N_POINTS)
        temperatures_c = x
        pressures_mpa = np.full_like(x, T_FIXED_P_MPA)
        mass_flows_kgs = np.full_like(x, T_FIXED_MDOT)
        x_label = 'Inlet temperature [°C]'
        x_nominal = T_DES_C
        name = 'temperature'

    elif sweep_type == 'P':
        x = np.linspace(P_MIN_MPA, P_MAX_MPA, P_N_POINTS)
        temperatures_c = np.full_like(x, P_FIXED_T_C)
        pressures_mpa = x
        mass_flows_kgs = np.full_like(x, P_FIXED_MDOT)
        x_label = 'Inlet pressure [MPa]'
        x_nominal = P_DES_MPA
        name = 'pressure'

    elif sweep_type == 'M':
        x = np.linspace(M_MIN_KGS, M_MAX_KGS, M_N_POINTS)
        temperatures_c = np.full_like(x, M_FIXED_T_C)
        pressures_mpa = np.full_like(x, M_FIXED_P_MPA)
        mass_flows_kgs = x
        x_label = 'CO₂ mass flow rate [kg/s]'
        x_nominal = MDOT_DES
        name = 'mass_flow'

    else:
        raise ValueError("SWEEP must be 'T', 'P', or 'M'.")

    data = []
    for xi, Ti, Pi, mi in zip(
        x, temperatures_c, pressures_mpa, mass_flows_kgs
    ):
        row = {
            'x': float(xi),
            'T_in_C': float(Ti),
            'P_in_MPa': float(Pi),
            'mdot_kg_s': float(mi),
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
            result = comp.off_design(
                T_in_K=float(Ti + 273.15),
                P_in_Pa=float(Pi * 1e6),
                mdot_kgs=float(mi),
                N_rpm=N_FIXED_RPM,
                strict=True,  # Do not extrapolate beyond surge/choke limits.
            )
            row.update({
                'PR': float(result.P_out_Pa / (Pi * 1e6)),
                'P_out_MPa': float(result.P_out_Pa / 1e6),
                'eta_percent': float(result.eta_is * 100),
                'power_MW': float(result.power_W / 1e6),
                'phi': float(result.phi),
                'U_over_a_out': float(result.tip_to_sound_out),
                'valid': bool(result.valid),
                'reason': '; '.join(result.warnings),
            })
        except (ValueError, RuntimeError) as exc:
            row['reason'] = str(exc)
        data.append(row)

    # Save ALL rows, including invalid operating points.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f'dyreby_sweep_{name}.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)

    valid_rows = [row for row in data if row['valid']]
    invalid_rows = [row for row in data if not row['valid']]

    print(f'\n=== {name.upper()} SWEEP ===')
    print(f'Fixed N = {N_FIXED_RPM:.0f} rpm ({100 * N_RATIO:.0f}% of design)')
    print(f'Valid points: {len(valid_rows)}/{len(data)}')
    print(f'CSV: {csv_path.resolve()}')

    if valid_rows:
        print(
            f'Valid efficiency range: '
            f'{min(r["eta_percent"] for r in valid_rows):.2f} '
            f'to {max(r["eta_percent"] for r in valid_rows):.2f} %'
        )
        print(
            f'Valid PR range: '
            f'{min(r["PR"] for r in valid_rows):.3f} '
            f'to {max(r["PR"] for r in valid_rows):.3f}'
        )

    for row in invalid_rows[:5]:
        print(f'  Invalid point x={row["x"]:.3f}: {row["reason"]}')
    if len(invalid_rows) > 5:
        print(
            f'  ...and {len(invalid_rows) - 5} additional invalid points '
            f'(see CSV).'
        )

    # Invalid operating points appear as gaps in the plots.
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
        (
            'Isentropic efficiency [%]',
            'Pressure ratio [-]',
            'Power consumption [MW]',
            'Tip speed / speed of sound [-]',
        ),
    ):
        ax.plot(x_plot, valid_values(key), '-o', ms=3)
        if x.min() <= x_nominal <= x.max():
            ax.axvline(
                x_nominal,
                color='0.55',
                linestyle=':',
                label='Nominal X value',
            )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    axes[3].axhline(
        1.0,
        color='tab:red',
        linestyle='--',
        label='Simplified U/a = 1 limit',
    )
    axes[3].legend(loc='best')
    axes[-1].set_xlabel(x_label)

    fig.suptitle(
        f'Dyreby: {name} sweep (fixed N = {N_FIXED_RPM:.0f} rpm)'
    )
    fig.tight_layout()

    png_path = OUTPUT_DIR / f'dyreby_sweep_{name}.png'
    fig.savefig(png_path, dpi=180, bbox_inches='tight')
    print(f'Figure: {png_path.resolve()}')

    plt.show()
    return data


# %% 4. RUN ONLY THE SELECTED SWEEP
sweep_results = run_sweep(SWEEP)
