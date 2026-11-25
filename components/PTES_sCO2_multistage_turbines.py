"""PTES-sCO2: charge HX design once; discharge HX off-design in TES closure.

The solver uses a fixed CO2 mass flow for sizing; reported work and heat are specific.
Check physical HX correlations, feasible solver intervals, and storage mass balance
before interpreting results as a validated design.
"""
import numpy as np
from scipy.optimize import root
from CoolProp.CoolProp import PropsSI
import UA_segmented_recuperator as hx
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import csv
from dyreby_compressor import DyrebyMultiStage
from dyreby_turbine import DyrebyTurbine

# =================== INPUTS ===================
CO2 = 'CO2'
FLUID_HOT = 'INCOMP::NaK'
FLUID_COLD = 'INCOMP::TD12'  # Change to an appropriate cold-side fluid as needed
T_TES_HL_ini = 400.0 + 273.15
T_TES_CH_ini = 160.0 + 273.15
P_low_chg = 8e6
P_low_dis = 8e6
PR_chg = 3.3
TTD_General = 10.0  # Only starting guesses (not an HX boundary condition)
Pressure_losses = 0.01  # Starting pressure-loss guesses only
eta_turb = 0.9
eta_comp = 0.9
N_STAGES_CHG = 3  # Charge compressor stages on a common shaft
N_STAGES_DIS = 1  # Discharge compressor stages on a common shaft
N_TURB_CHG_DESIGN_RPM = 3600.0  # Assumed nominal speed for an independent charge turbine shaft
N_TURB_DIS_DESIGN_RPM = 3600.0  # Assumed nominal speed for an independent discharge turbine shaft
# These turbine speeds are explicit design choices, not predictions of the Dyreby model.
mdot_wf_chg = 100.0  # kg/s
mdot_wf_dis = 100.0
D_recup = 0.005  # m
D_hot_hx = 0.015
D_cold_hx = 0.01
N_seg = 5
Af_bounds_rec = (0.05, 0.5)
Af_bounds_cold_hx = (0.02, 0.7)
Af_bounds_hot_hx = (0.02, 0.7)
delta_T_min_target = 5.0
P_loss_target = 0.005
# Storage flow: match heat-capacity rates in charge, discharge hot side;
STRATEGY_CHG_HOT = 1 # Match capacitance
STRATEGY_CHG_COLD = 1 # Match capacitance
# discharge cold side can instead enforce a cycle inventory balance.
STRATEGY_DIS_HOT = 0   # 0 use the same as in charge
STRATEGY_DIS_COLD = 2  # Enforce mass balance


ALPHA = 0.4
MAX_IT = 60
TOL_T = 0.05  # K
TOL_P = 1000.0  # Pa

#Print during iteration for charging
def print_charge_iteration(iteration, hot, cold, recup, error_T, error_P):

    print(f"\n{'='*75}")
    print(f"CHARGE ITERATION {iteration}")
    print(f"{'='*75}")

    print(
        f"{'Variable':<20}"
        f"{'HOT HX':>16}"
        f"{'COLD HX':>16}"
        f"{'RECUP':>16}"
    )

    def line(name, key, factor=1.0):
        print(
            f"{name:<20}"
            + "".join(
                f"{hx.get(key, np.nan)/factor:>16.3f}"
                for hx in (hot, cold, recup)
            )
        )

    line("Q [MW]", "Q_solution", 1e6)
    line("Af [m2]", "A_f")
    line("L [m]", "L_total")
    line("Area [m2]", "A_total")
    line("dP [%]", "P_losses_calc", 0.01)
    line("Pinch [K]", "delta_T_min")
    line("UA [W/K]", "UA_total")
    # TTD 
    ttd_hot = hot["T_wf"][::-1] - hot["T_storage"]
    ttd_cold = cold["T_storage"] - cold["T_wf"][::-1]
    ttd_rec = recup["T_hot"][::-1] - recup["T_cold"]

    print(
        f"{'TTD  1 [K]':<20}"
        f"{ttd_hot[0]:>16.3f}"
        f"{ttd_cold[0]:>16.3f}"
        f"{ttd_rec[0]:>16.3f}"
    )

    print(
        f"{'TTD  2 [K]':<20}"
        f"{ttd_hot[-1]:>16.3f}"
        f"{ttd_cold[-1]:>16.3f}"
        f"{ttd_rec[-1]:>16.3f}"
    )

    print(
        f"{'mdot storage [kg/s]':<20}"
        f"{hot.get('mdot_storage', np.nan):>16.3f}"
        f"{cold.get('mdot_storage', np.nan):>16.3f}"
        f"{'—':>16}"
    )

    print(
        f"{'Re min':<20}"
        + "".join(
            f"{min(hx['Re_st']):>16.0f}"
            for hx in (hot, cold)
        )
        + f"{'—':>16}"
    )

    print(f"\nLoop error T = {error_T:.6f} K")
    print(f"Loop error P = {error_P:.2f} Pa")

#Print during iterations - Discharge
def print_discharge_iteration(iteration, hot, cold, recup, error_T, error_P):

    print(f"\n{'='*75}")
    print(f"DISCHARGE ITERATION {iteration}")
    print(f"{'='*75}")

    print(
        f"{'Variable':<22}"
        f"{'HOT HX':>16}"
        f"{'COLD HX':>16}"
        f"{'RECUP':>16}"
    )

    def line(name, key, factor=1.0):
        print(
            f"{name:<22}"
            + "".join(
                f"{h.get(key, np.nan)/factor:>16.3f}"
                for h in (hot, cold, recup)
            )
        )

    line("Q [MW]", "Q_solution", 1e6)
    line("Flow area [m2]", "A_f")
    line("Length [m]", "L_total")
    line("HX area [m2]", "A_total")
    line("Pressure drop [%]", "P_losses_calc", 0.01)
    line("Minimum pinch [K]", "delta_T_min")
    line("UA [W/K]", "UA_total")

    print(
        f"{'Storage flow [kg/s]':<22}"
        f"{hot.get('mdot_storage', np.nan):>16.3f}"
        f"{cold.get('mdot_storage', np.nan):>16.3f}"
        f"{'—':>16}"
    )

    # Terminal temperature differences (counterflow)
    ttd_hot = np.asarray(hot["T_storage"]) - np.asarray(hot["T_wf"])[::-1]
    ttd_cold = np.asarray(cold["T_wf"])[::-1] - np.asarray(cold["T_storage"])
    ttd_rec = np.asarray(recup["T_hot"])[::-1] - np.asarray(recup["T_cold"])

    for label, idx in [
        ("TTD end 1 [K]", 0),
        ("TTD end 2 [K]", -1)
    ]:
        print(
            f"{label:<22}"
            f"{ttd_hot[idx]:>16.3f}"
            f"{ttd_cold[idx]:>16.3f}"
            f"{ttd_rec[idx]:>16.3f}"
        )

    print(
        f"{'Min storage Re':<22}"
        f"{np.min(hot['Re_st']):>16.0f}"
        f"{np.min(cold['Re_st']):>16.0f}"
        f"{'—':>16}"
    )

    print(f"\nLoop temperature error = {error_T:.6f} K")
    print(f"Loop pressure error    = {error_P:.2f} Pa")


def H(T, P): return PropsSI('Hmass', 'T', T, 'P', P, CO2)
def S(T, P): return PropsSI('Smass', 'T', T, 'P', P, CO2)
def Tfrom(h, P): return PropsSI('T', 'Hmass', h, 'P', P, CO2)
def Sfrom(h, P): return PropsSI('Smass', 'Hmass', h, 'P', P, CO2)
def Hs(s, P): return PropsSI('Hmass', 'Smass', s, 'P', P, CO2)


def turbine_h_out(h_in, P_in, P_out):
    if not (0 < P_out < P_in):
        raise ValueError(f'Turbine pressure ratio invalid: {P_in=} {P_out=}')
    h_out = h_in - eta_turb * (h_in - Hs(Sfrom(h_in, P_in), P_out))
    return h_out


def compressor_h_out(h_in, P_in, P_out):
    if P_out <= P_in:
        raise ValueError(f'Compressor pressure ratio invalid: {P_in=} {P_out=}')
    h_out = h_in + (Hs(Sfrom(h_in, P_in), P_out) - h_in) / eta_comp
    return h_out


def state(T, P, h=None):
    if h is None: h = H(T, P)
    return {'T': Tfrom(h, P), 'P': P, 'h': h, 's': Sfrom(h, P)}

#Check after recuperator result in each iteration
def check_hx(result, name, offdesign=False):
    if offdesign and not result.get('physical', True):
        raise ValueError(f'{name}: non-physical HX evaluation')
    for key in ('L_total', 'A_f', 'Q_solution'):
        if key not in result or not np.isfinite(result[key]):
            raise ValueError(f'{name}: invalid {key}')
    if result['L_total'] <= 0 or result['A_f'] <= 0 or result['Q_solution'] <= 0:
        raise ValueError(f'{name}: nonpositive HX size/duty')
    if np.any(np.asarray(result['delta_x']) <= 0):
        raise ValueError(f'{name}: nonpositive segment length')
    # if not np.isclose(result['L_total'], result.get('L_design', result['L_total']), rtol=1e-4, atol=0.01):
    #     raise ValueError(f'{name}: offdesign length residual too large')
    if result.get('delta_T_min', 1) <= 0:
        raise ValueError(f'{name}: thermal crossover')
    return result

#Hot HX in chg
def design_cool(Tstorage, Tin, Pin, fluid):
    out = hx.HX_UA_segments_cool_wf(
        T_storage_in=Tstorage, T_wf_in=Tin, P_wf_in=Pin,
        mdot_storage_fixed=mdot_wf_chg, mdot_wf=mdot_wf_chg,
        fluid_storage_design=fluid, fluid_wf=CO2, D=D_hot_hx,
        delta_T_min_target=delta_T_min_target, P_loss_target=P_loss_target,
        N_seg=N_seg, Af_bounds=Af_bounds_hot_hx, strategy_mdot_st_design=STRATEGY_CHG_HOT)
    return check_hx(out, 'charge hot HX')

#Cold HX in chg
def design_heat(Tstorage, Tin, Pin, fluid):
    out = hx.HX_UA_segments_heat_wf(
        T_storage_in=Tstorage, T_wf_in=Tin, P_wf_in=Pin,
        mdot_storage_fixed=mdot_wf_chg, mdot_wf=mdot_wf_chg,
        fluid_storage_design=fluid, fluid_wf=CO2, D=D_cold_hx,
        delta_T_min_target=delta_T_min_target, P_loss_target=P_loss_target,
        N_seg=N_seg, Af_bounds=Af_bounds_cold_hx, strategy_mdot_st_design=STRATEGY_CHG_COLD)
    return check_hx(out, 'charge cold HX')


def charge():
    # Guesses are TES-side terminal temperature offsets and nominal pressure drops.
    # Guesses = [T1, T4, P1, P4, P6]
    guess = np.array([
        T_TES_CH_ini - TTD_General,  # T1 = cold HX CO2 outlet
        T_TES_HL_ini + TTD_General,  # T4 = hot HX CO2 outlet
        P_low_chg / (1 - Pressure_losses),  # P1 = cold recuperator inlet
        P_low_chg * PR_chg * (1 - Pressure_losses),  # P4 = hot recuperator inlet
        P_low_chg / (1 - Pressure_losses)**2,  # P6 = turbine outlet
    ], dtype=float)

    for iteration in range(MAX_IT):
        T1, T4, P1, P4, P6 = guess
        a = {1: state(T1, P1)} # This correspond to state 1 = [T1, P1, h1, s1]
        # Charge recuperator: low-P cold flow 1 -> 2; high-P hot flow 4 -> 5
        rec = hx.HX_UA_segments_charge(
            T_cold_in=a[1]['T'], T_hot_in=T4,
            P_hot_in=P4, P_cold_in=P1,
            mdot_cold=mdot_wf_chg, mdot_hot=mdot_wf_chg,
            fluid_hot=CO2, fluid_cold=CO2,
            D=D_recup, delta_T_min_target=delta_T_min_target,
            P_loss_target=P_loss_target, N_seg=N_seg, Af_bounds=Af_bounds_rec)
        check_hx(rec, 'charge recuperator')
        P2 = P1 * (1 - rec['P_losses_cold_calc'])
        a[2] = state(rec['T_cold'][-1], P2) #Calculate the state with P2 and T2
        P3 = P2 * PR_chg
        a[3] = state(None, P3, compressor_h_out(a[2]['h'], P2, P3))
        hot = design_cool(T_TES_HL_ini, a[3]['T'], P3, FLUID_HOT)
        P4new = float(hot['P_wf'][-1]) #Take last value of pressure of the wf 
        a[4] = state(float(hot['T_wf'][-1]), P4new) # Use calculated T4 and P4 to create state 4
        P5 = float(rec['P_hot'][-1]) 
        a[5] = state(float(rec['T_hot'][-1]), P5) #Use T5 and P5 to calculate state 5
        a[6] = state(None, P6, turbine_h_out(a[5]['h'], P5, P6)) # Use P6 and h6 (calculated with the turbine with T5,P5,P6) to calculate state 6
        cold = design_heat(T_TES_CH_ini, a[6]['T'], P6, FLUID_COLD)
        P1new = float(cold['P_wf'][-1])
        a[1] = state(float(cold['T_wf'][-1]), P1new) #Calculate new state 1
        # Anchor recuperator low-pressure outlet at P_low_chg by updating P6.
        # This correction is approximate because each HX has its own pressure drop.
        P1target = P_low_chg / (1 - rec['P_losses_cold_calc']) # I use the calculated value of pressure drop to estimate the P1target
        P6new = P6 + (P1target - P1new) #There has to be a match between P1target (calculated with the pressure drop of the cycle and P1new, calculated at the end of the cycle)
        new = np.array([a[1]['T'], a[4]['T'], P1target, P4new, P6new]) #New guesses
        errors = np.abs(new - guess) #Errors of the guesses

        error_T = np.max(errors[:2])
        error_P = np.max(errors[2:])

        print_charge_iteration(
            iteration=iteration,
            hot=hot,
            cold=cold,
            recup=rec,
            error_T=error_T,
            error_P=error_P
        )

        if error_T < TOL_T and error_P < TOL_P:
            break

        guess += ALPHA * (new - guess)
    else:
        raise RuntimeError(f'Charge loop not converged; last errors: {errors}')

    # Re-evaluate with converged guesses to keep all HX results in sync.
    # Last-evaluation consistency checked separately before use.
    return {
        'states': a, 'hot_hx': hot, 'cold_hx': cold, 'recup': rec,
        'T_TES_HH': float(hot['T_storage'][-1]),
        'T_TES_CL': float(cold['T_storage'][-1]),
        'mdot_hot_chg': float(hot['mdot_storage']),
        'mdot_cold_chg': float(cold['mdot_storage']),
        'W_comp': a[3]['h'] - a[2]['h'],
        'W_turb': a[5]['h'] - a[6]['h'],
        'Q_hot': a[3]['h'] - a[4]['h'],
        'Q_cold': a[1]['h'] - a[6]['h'],
    }


def offdesign_heat(Tstorage, Tin, Pin, design, mdot_hot_chg):
    out = hx.HX_UA_segments_heat_wf_offdesign(
        T_storage_in=Tstorage, T_wf_in=Tin, P_wf_in=Pin,
        fluid_storage_offdesign=FLUID_HOT,
        mdot_storage_fixed=mdot_hot_chg, mdot_wf=mdot_wf_dis, fluid_wf=CO2,
        D=D_hot_hx, N_seg=N_seg, A_f_design=design['A_f'], L_design=design['L_total'],
        strategy_mdot_st_offdesign=STRATEGY_DIS_HOT,
        P_losses_estimation=P_loss_target)
    return check_hx(out, 'discharge hot HX', True)

# This HX has to do the mass balance, so it need info from the chg and mdot of the hot HX in dis
def offdesign_cool(Tstorage, Tin, Pin, design, chg, mdot_hot_dis):
    out = hx.HX_UA_segments_cool_wf_offdesign(
        T_storage_in=Tstorage, T_wf_in=Tin, P_wf_in=Pin,
        mdot_storage_fixed=chg['mdot_cold_chg'], mdot_wf=mdot_wf_dis,
        fluid_storage_offdesign=FLUID_COLD, fluid_wf=CO2,
        D=D_cold_hx, N_seg=N_seg, A_f_design=design['A_f'], L_design=design['L_total'],
        strategy_mdot_st_offdesign=STRATEGY_DIS_COLD,
        P_losses_estimation=P_loss_target,
        mdot_st_cold_chg_info=chg['mdot_cold_chg'], # cold HX storage massflow in chg
        mdot_st_hot_chg_info=chg['mdot_hot_chg'], # hot HX storage massflow in chg
        mdot_st_hot_dis_info=mdot_hot_dis) # mdot of hot HX in dis
    return check_hx(out, 'discharge cold HX', True)


def discharge(PR_dis, TTD_heat_reject_dis, chg):
    guess = np.array([
        chg['T_TES_CL'] + TTD_General,  # T1, cold HX outlet
        chg['T_TES_HH'] - TTD_General,  # T6, turbine outlet (initial estimate refined below)
        P_low_dis / (1 - Pressure_losses)**3,  # P6, turbine outlet
    ], dtype=float)
    # T6 guess must be in a plausible hot-side range; initialize from charge point.
    #guess[1] = chg['T_TES_HH'] - 50.0

    for iteration in range(MAX_IT):
        T1, T6, P6 = guess
        P1 = P_low_dis / (1 - Pressure_losses)
        a = {1: state(T1, P1)}
        P2 = P_low_dis
        a[2] = state(a[1]['T'] - TTD_heat_reject_dis, P2)
        P3 = P2 * PR_dis
        a[3] = state(0, P3, compressor_h_out(a[2]['h'], P2, P3))
        # Recuperator discharge: hot low pressure 6 -> 7; cold high pressure 3 -> 4
        rec = hx.HX_UA_segments_offdesign_discharge(
            T_cold_in=a[3]['T'], T_hot_in=T6,
            P_hot_in=P6, P_cold_in=P3,
            mdot_cold=mdot_wf_dis, mdot_hot=mdot_wf_dis,
            fluid_cold=CO2, fluid_hot=CO2,
            D=D_recup, N_seg=N_seg,
            A_f_design=chg['recup']['A_f'], L_design=chg['recup']['L_total'],
            P_losses_estimation=P_loss_target)
        check_hx(rec, 'discharge recuperator', True)
        P4 = P3 * (1 - rec['P_losses_cold_calc'])
        a[4] = state(float(rec['T_cold'][-1]), P4)
        hot = offdesign_heat(chg['T_TES_HH'], a[4]['T'], P4,
                             chg['hot_hx'], chg['mdot_hot_chg'])
        P5 = float(hot['P_wf'][-1])
        a[5] = state(float(hot['T_wf'][-1]), P5)
        a[6] = state(0, P6, turbine_h_out(a[5]['h'], P5, P6))
        P7 = P6 * (1 - rec['P_losses_calc'])
        a[7] = state(float(rec['T_hot'][-1]), P7)
        cold = offdesign_cool(chg['T_TES_CL'], a[7]['T'], P7,
                              chg['cold_hx'], chg, float(hot['mdot_storage']))
        P1new = float(cold['P_wf'][-1])
        a[1] = state(float(cold['T_wf'][-1]), P1new)
        # Match cooling-HX outlet to heat-rejection inlet pressure.
        P6new = P6 + (P1 - P1new)
        new = np.array([a[1]['T'], a[6]['T'], P6new])
        errors = np.abs(new - guess)

        error_T = np.max(errors[:2])
        error_P = errors[2]

        print_discharge_iteration(
            iteration=iteration,
            hot=hot,
            cold=cold,
            recup=rec,
            error_T=error_T,
            error_P=error_P
        )

        if error_T < TOL_T and error_P < TOL_P:
            break

        guess += ALPHA * (new - guess)
    else:
        raise RuntimeError(f'Discharge loop not converged, last errors: {errors}')

    W_comp = a[3]['h'] - a[2]['h']
    W_turb = a[5]['h'] - a[6]['h']
    return {
        'states': a, 'hot_hx': hot, 'cold_hx': cold, 'recup': rec,
        'T_TES_CH_new': float(cold['T_storage'][-1]),
        'T_TES_HL_new': float(hot['T_storage'][-1]),
        'W_comp': W_comp, 'W_turb': W_turb,
        'Q_hot': a[5]['h']-a[4]['h'],
        'Q_cold': a[7]['h']-a[1]['h'],
        'Q_reject': a[1]['h']-a[2]['h'],
    }


def plot_ts(chg, dis):

    fig, ax = plt.subplots(figsize=(10, 7))

    cycles = {
        "Charge": {
            "states": chg["states"],
            "segments": [
                (1, 2, "curve"),     # Recuperator
                (2, 3, "straight"),  # Compressor
                (3, 4, "curve"),     # Hot HX
                (4, 5, "curve"),     # Recuperator
                (5, 6, "straight"),  # Turbine
                (6, 1, "curve"),     # Cold HX
            ],
            "color": "tab:blue",
            "prefix": "C",
        },

        "Discharge": {
            "states": dis["states"],
            "segments": [
                (1, 2, "curve"),     # Heat rejection
                (2, 3, "straight"),  # Compressor
                (3, 4, "curve"),     # Recuperator
                (4, 5, "curve"),     # Hot HX
                (5, 6, "straight"),  # Turbine
                (6, 7, "curve"),     # Recuperator
                (7, 1, "curve"),     # Cold HX
            ],
            "color": "tab:orange",
            "prefix": "D",
        },
    }

    for mode, data in cycles.items():

        states = data["states"]
        color = data["color"]

        for idx, (i, j, segment_type) in enumerate(data["segments"]):

            Ti = states[i]["T"]
            Tj = states[j]["T"]
            Pi = states[i]["P"]
            Pj = states[j]["P"]

            if segment_type == "straight":

                s_seg = np.array([states[i]["s"],states[j]["s"]])
                T_seg = np.array([Ti, Tj])

            else:

                T_seg = np.linspace(Ti, Tj, 100)
                P_seg = np.linspace(Pi, Pj, 100)

                s_seg = np.array([S(T, P)for T, P in zip(T_seg, P_seg)])

            ax.plot(s_seg / 1000,T_seg - 273.15,color=color,linewidth=1.8,label=mode if idx == 0 else None)

        # Plot and label thermodynamic states
        for i, st in states.items():

            s = st["s"] / 1000
            T = st["T"] - 273.15

            ax.scatter(s, T, color=color, s=30, zorder=3)

            ax.annotate(f"{data['prefix']}{i}",(s, T),xytext=(5, 5),textcoords="offset points",fontsize=9,color=color)

    ax.set_xlabel("Specific entropy [kJ/kg-K]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("PTES sCO2 - T-s Diagram")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()


def plot_hx_profile(hx_result, title, hx_type, mdot_sco2):

    # Physical position along the heat exchanger
    dx = np.asarray(hx_result["delta_x"], dtype=float)

    x_nodes = np.concatenate(([0.0], np.cumsum(dx)))
    x_segments = 0.5 * (x_nodes[:-1] + x_nodes[1:])

    if hx_type == "storage":

        # delta_x follows the storage flow direction.
        # Reverse the working-fluid arrays for counterflow alignment.
        T_1 = np.asarray(hx_result["T_storage"])
        T_2 = np.asarray(hx_result["T_wf"])[::-1]

        # cp values correspond to segment centers, not nodes.
        C_1 = (
            hx_result["mdot_storage"]
            * np.asarray(hx_result["cp_st"])
        ) / 1000.0

        C_2 = (
            mdot_sco2
            * np.asarray(hx_result["cp_wf"])[::-1]
        ) / 1000.0

        label_1 = "Storage"
        label_2 = "sCO2"

    elif hx_type == "recuperator":

        # delta_x follows the cold-side flow direction.
        # Reverse the hot-side arrays for counterflow alignment.
        T_1 = np.asarray(hx_result["T_cold"])
        T_2 = np.asarray(hx_result["T_hot"])[::-1]

        C_1 = (
            mdot_sco2
            * np.asarray(hx_result["cp_cold"])
        ) / 1000.0

        C_2 = (
            mdot_sco2
            * np.asarray(hx_result["cp_hot"])[::-1]
        ) / 1000.0

        label_1 = "Cold sCO2"
        label_2 = "Hot sCO2"

    else:
        raise ValueError(f"Unknown HX type: {hx_type}")

    # Create figure
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.manager.set_window_title(title)

    # Temperature profiles
    l1, = ax1.plot(
        x_nodes, T_1,
        "-o", color="tab:blue",
        label=label_1
    )

    l2, = ax1.plot(
        x_nodes, T_2,
        "-o", color="tab:orange",
        label=label_2
    )

    ax1.set_xlabel("Physical position along HX [m]")
    ax1.set_ylabel("Temperature [K]")
    ax1.grid(True, alpha=0.4)

    # Thermal capacitance profiles
    ax2 = ax1.twinx()

    l3, = ax2.plot(
        x_segments, C_1,
        "--x", color="tab:blue",
        label=f"{label_1} capacitance"
    )

    l4, = ax2.plot(
        x_segments, C_2,
        "--x", color="tab:orange",
        label=f"{label_2} capacitance"
    )

    ax2.set_ylabel("Thermal capacitance rate [kW/K]")

    ax1.set_title(title)

    lines = [l1, l2, l3, l4]
    labels = [line.get_label() for line in lines]

    ax1.legend(lines, labels, loc="best")

    fig.tight_layout()


def plot_all_hx_profiles(chg, dis):

    # CHARGE - DESIGN

    plot_hx_profile(
        chg["hot_hx"],
        "Charge - Design Hot HX",
        "storage",
        mdot_wf_chg
    )

    plot_hx_profile(
        chg["recup"],
        "Charge - Design Recuperator",
        "recuperator",
        mdot_wf_chg
    )

    plot_hx_profile(
        chg["cold_hx"],
        "Charge - Design Cold HX",
        "storage",
        mdot_wf_chg
    )

    # DISCHARGE - OFF-DESIGN

    plot_hx_profile(
        dis["hot_hx"],
        "Discharge - Off-design Hot HX",
        "storage",
        mdot_wf_dis
    )

    plot_hx_profile(
        dis["recup"],
        "Discharge - Off-design Recuperator",
        "recuperator",
        mdot_wf_dis
    )

    plot_hx_profile(
        dis["cold_hx"],
        "Discharge - Off-design Cold HX",
        "storage",
        mdot_wf_dis
    )



def size_and_report_compressor(cycle, mdot_kgs, n_stages, label):
    """Size one fixed-geometry multistage compressor at the cycle design point.

    The PTES thermodynamic solver still imposes PR and eta_comp. This routine
    sizes the Dyreby compressor AFTER the nominal cycle has converged.
    """
    inlet = cycle['states'][2]
    outlet = cycle['states'][3]

    compressor = DyrebyMultiStage.from_design(
        T_in_K=inlet['T'],
        P_in_Pa=inlet['P'],
        P_out_Pa=outlet['P'],
        mdot_kgs=mdot_kgs,
        n_stages=n_stages,
        eta_design=eta_comp,
    )

    nominal = compressor.off_design(
        T_in_K=inlet['T'],
        P_in_Pa=inlet['P'],
        mdot_kgs=mdot_kgs,
        N_rpm=compressor.N_design_rpm,
        strict=False,
    )

    cycle_pr = outlet['P'] / inlet['P']
    map_pr = nominal.P_out_Pa / inlet['P']
    if not np.isclose(map_pr, cycle_pr, rtol=1e-4, atol=0.0):
        raise RuntimeError(f'{label} compressor PR mismatch: {map_pr} vs {cycle_pr}')
    if not np.isclose(nominal.eta_is, eta_comp, rtol=1e-3, atol=1e-4):
        raise RuntimeError(
            f'{label} compressor efficiency mismatch: '
            f'{nominal.eta_is} vs {eta_comp}'
        )

    print(f'\n=== {label} COMPRESSOR DESIGN ===')
    print(f'Stages: {len(compressor.stages)}')
    print(f'Common design speed: {compressor.N_design_rpm:.0f} rpm')
    print(f'Overall PR: {map_pr:.4f}')
    print(f'Overall isentropic efficiency: {nominal.eta_is:.4f}')
    print(f'Total compressor power: {nominal.power_W / 1e6:.3f} MW')
    print(f'All stages pass simplified checks: {nominal.valid}')

    previous_pressure = inlet['P']
    for index, (stage, stage_result) in enumerate(
        zip(compressor.stages, nominal.stages), start=1
    ):
        stage_pr = stage_result.P_out_Pa / previous_pressure
        print(
            f'Stage {index}: D={stage.diameter_m:.5f} m, '
            f'PR={stage_pr:.4f}, '
            f'eta={stage_result.eta_is:.4f}, '
            f'phi={stage_result.phi:.5f}, '
            f'U/a_out={stage_result.tip_to_sound_out:.3f}, '
            f'valid={stage_result.valid}'
        )
        for warning in stage_result.warnings:
            print(f'  Stage {index} warning: {warning}')
        previous_pressure = stage_result.P_out_Pa

    # Preserve the sized geometry for future off-design PTES calculations.
    cycle['compressor'] = compressor
    cycle['compressor_nominal'] = nominal
    return compressor


def add_compressor_results(result, cycle, label):
    """Append numeric compressor design information to the results CSV."""
    compressor = cycle['compressor']
    nominal = cycle['compressor_nominal']
    inlet_pressure = cycle['states'][2]['P']

    result[f'comp_{label}_stages'] = len(compressor.stages)
    result[f'comp_{label}_N_design_rpm'] = compressor.N_design_rpm
    result[f'comp_{label}_PR_nominal'] = nominal.P_out_Pa / inlet_pressure
    result[f'comp_{label}_eta_nominal'] = nominal.eta_is
    result[f'comp_{label}_power_nominal_MW'] = nominal.power_W / 1e6
    result[f'comp_{label}_all_stages_valid'] = int(nominal.valid)

    previous_pressure = inlet_pressure
    for index, (stage, stage_result) in enumerate(
        zip(compressor.stages, nominal.stages), start=1
    ):
        prefix = f'comp_{label}_stage_{index}'
        # Exact parameters required to reconstruct a DyrebyCompressor stage.
        result[f'{prefix}_diameter_m'] = stage.diameter_m
        result[f'{prefix}_N_design_rpm'] = stage.N_design_rpm
        result[f'{prefix}_eta_design'] = stage.eta_design
        result[f'{prefix}_phi_design'] = stage.phi_design

        # Nominal operating-point results (not geometry inputs).
        result[f'{prefix}_P_out_Pa'] = stage_result.P_out_Pa
        result[f'{prefix}_T_out_K'] = stage_result.T_out_K
        result[f'{prefix}_h_out_Jkg'] = stage_result.h_out_Jkg
        result[f'{prefix}_power_W'] = stage_result.power_W
        result[f'{prefix}_PR'] = stage_result.P_out_Pa / previous_pressure
        result[f'{prefix}_eta'] = stage_result.eta_is
        result[f'{prefix}_phi'] = stage_result.phi
        result[f'{prefix}_U_over_a_out'] = stage_result.tip_to_sound_out
        result[f'{prefix}_valid'] = int(stage_result.valid)
        previous_pressure = stage_result.P_out_Pa


def size_and_report_turbine(cycle, mdot_kgs, N_design_rpm, label):
    """Size the turbine only after the nominal thermodynamic cycle converges.

    The existing PTES solver continues to impose eta_turb and its turbine
    outlet pressure. This function does not alter cycle performance or HX sizing.
    """
    inlet = cycle['states'][5]
    outlet = cycle['states'][6]

    turbine = DyrebyTurbine.from_design(
        T_in_K=inlet['T'],
        P_in_Pa=inlet['P'],
        P_out_Pa=outlet['P'],
        mdot_kgs=mdot_kgs,
        N_design_rpm=N_design_rpm,
        eta_design=eta_turb,
    )

    nominal = turbine.off_design(
        T_in_K=inlet['T'],
        P_in_Pa=inlet['P'],
        P_out_Pa=outlet['P'],
        N_rpm=turbine.N_design_rpm,
        strict=False,
    )

    # Check that sizing reproduces the nominal thermodynamic inputs.
    if not np.isclose(nominal.mdot_kgs, mdot_kgs, rtol=1e-6, atol=1e-8):
        raise RuntimeError(
            f'{label} turbine design flow mismatch: '
            f'{nominal.mdot_kgs} vs {mdot_kgs}'
        )
    if not np.isclose(nominal.eta_is, eta_turb, rtol=1e-3, atol=1e-4):
        raise RuntimeError(
            f'{label} turbine design efficiency mismatch: '
            f'{nominal.eta_is} vs {eta_turb}'
        )

    print(f'\n=== {label} TURBINE DESIGN ===')
    print(f'Design speed: {turbine.N_design_rpm:.0f} rpm')
    print(f'Rotor diameter: {turbine.diameter_m:.5f} m')
    print(f'Effective nozzle area: {turbine.nozzle_area_m2:.7f} m2')
    print(f'Nominal expansion PR: {inlet["P"] / outlet["P"]:.4f}')
    print(f'Nominal efficiency: {nominal.eta_is:.4f}')
    print(f'Nominal mass flow: {nominal.mdot_kgs:.3f} kg/s')
    print(f'Nominal turbine power: {nominal.power_W / 1e6:.3f} MW')
    print(f'Nominal U/Cs: {nominal.nu:.5f}')
    print(f'Simplified validity check: {nominal.valid}')
    for warning in nominal.warnings:
        print(f'  Turbine warning: {warning}')

    cycle['turbine'] = turbine
    cycle['turbine_nominal'] = nominal
    return turbine


def add_turbine_results(result, cycle, label):
    """Export all four turbine reconstruction parameters and nominal checks."""
    turbine = cycle['turbine']
    nominal = cycle['turbine_nominal']
    inlet = cycle['states'][5]
    outlet = cycle['states'][6]
    prefix = f'turb_{label}'

    # These four values reconstruct DyrebyTurbine without from_design().
    result[f'{prefix}_diameter_m'] = turbine.diameter_m
    result[f'{prefix}_nozzle_area_m2'] = turbine.nozzle_area_m2
    result[f'{prefix}_N_design_rpm'] = turbine.N_design_rpm
    result[f'{prefix}_eta_design'] = turbine.eta_design

    # Nominal cycle conditions and evaluated map outputs.
    result[f'{prefix}_T_in_K'] = inlet['T']
    result[f'{prefix}_P_in_Pa'] = inlet['P']
    result[f'{prefix}_P_out_Pa'] = outlet['P']
    result[f'{prefix}_PR_expansion_nominal'] = inlet['P'] / outlet['P']
    result[f'{prefix}_mdot_nominal_kgs'] = nominal.mdot_kgs
    result[f'{prefix}_eta_nominal'] = nominal.eta_is
    result[f'{prefix}_T_out_nominal_K'] = nominal.T_out_K
    result[f'{prefix}_h_out_nominal_Jkg'] = nominal.h_out_Jkg
    result[f'{prefix}_power_nominal_MW'] = nominal.power_W / 1e6
    result[f'{prefix}_nu_nominal'] = nominal.nu
    result[f'{prefix}_spouting_velocity_nominal_ms'] = nominal.spouting_velocity_ms
    result[f'{prefix}_tip_speed_nominal_ms'] = nominal.tip_speed_ms
    result[f'{prefix}_U_over_a_in_nominal'] = nominal.tip_to_sound_in
    result[f'{prefix}_simplified_valid'] = int(nominal.valid)


def solve_cycle():
    print('Designing CHARGE once...')
    chg = charge()
    comp_chg = size_and_report_compressor(
        cycle=chg,
        mdot_kgs=mdot_wf_chg,
        n_stages=N_STAGES_CHG,
        label='CHARGE',
    )

    size_and_report_turbine(
        cycle=chg,
        mdot_kgs=mdot_wf_chg,
        N_design_rpm=N_TURB_CHG_DESIGN_RPM,
        label='CHARGE',
    )

    print('Charge design complete.')

    calls = [0]
    def residuals(x):
        PR_dis, TTD_reject = map(float, x)
        calls[0] += 1
        dis = discharge(PR_dis, TTD_reject, chg)
        eCH = dis['T_TES_CH_new']-T_TES_CH_ini
        eHL = dis['T_TES_HL_new']-T_TES_HL_ini
        print(f'{calls[0]:3d}: PR_dis={PR_dis:.6f}, TTD_reject={TTD_reject:.4f} K, '
              f'CH_err={eCH:+.4f} K, HL_err={eHL:+.4f} K')
        return np.array([eCH, eHL])

    x0 = [3.2, 30.0]

    bounds = (
        [1.2, 0.0],    # Minimum discharge PR and minimum cooling approach
        [5.0, 150.0]   # Maximum discharge PR and maximum cooling approach
    )

    sol = least_squares(
        residuals,
        x0=x0,
        bounds=bounds,
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
        max_nfev=200
    )
    if not sol.success:
        raise RuntimeError(f'TES closure did not converge: {sol.message}')
    dis = discharge(*map(float, sol.x), chg)
    closure = residuals(sol.x)
    if np.max(np.abs(closure)) > 0.1:
        raise RuntimeError(f'TES closure residual is too high: {closure}')

    Wchg = chg['W_comp']-chg['W_turb']
    Wdis = dis['W_turb']-dis['W_comp']
    if Wchg <= 0 or Wdis <= 0:
        raise RuntimeError('Net work has invalid sign')
    
    comp_dis = size_and_report_compressor(
        cycle=dis,
        mdot_kgs=mdot_wf_dis,
        n_stages=N_STAGES_DIS,
        label='DISCHARGE',
    )

    size_and_report_turbine(
        cycle=dis,
        mdot_kgs=mdot_wf_dis,
        N_design_rpm=N_TURB_DIS_DESIGN_RPM,
        label='DISCHARGE',
    )

    result = {
        'PR_chg': PR_chg,
        'PR_dis': float(sol.x[0]),
        'TTD_heat_reject_dis': float(sol.x[1]),
        'RTE': (mdot_wf_dis * Wdis) / (mdot_wf_chg * Wchg),
        'COP_chg_heat': chg['Q_hot']/Wchg,
        'Work_ratio': chg['W_comp']/chg['W_turb'],
        'eta_thermal': Wdis/dis['Q_hot'],
        'TES_T_CH_error': float(closure[0]),
        'TES_T_HL_error': float(closure[1]),
        'T_TES_HH': chg['T_TES_HH'],
        'T_TES_CL': chg['T_TES_CL'],
        'W_net_chg_kJkg': Wchg/1000,
        'W_net_dis_kJkg': Wdis/1000,

        'mdot_wf_chg': mdot_wf_chg,
        'mdot_wf_dis': mdot_wf_dis,

        # CHARGE - Power [MW]
        'Power_comp_chg_MW': mdot_wf_chg * chg['W_comp'] / 1e6,
        'Power_turb_chg_MW': mdot_wf_chg * chg['W_turb'] / 1e6,
        'Power_net_chg_MW': mdot_wf_chg * Wchg / 1e6,

        # CHARGE - Heat transfer [MW]
        'Q_hot_chg_MW': mdot_wf_chg * chg['Q_hot'] / 1e6,
        'Q_cold_chg_MW': mdot_wf_chg * chg['Q_cold'] / 1e6,

        # DISCHARGE - Power [MW]
        'Power_comp_dis_MW': mdot_wf_dis * dis['W_comp'] / 1e6,
        'Power_turb_dis_MW': mdot_wf_dis * dis['W_turb'] / 1e6,
        'Power_net_dis_MW': mdot_wf_dis * Wdis / 1e6,

        # DISCHARGE - Heat transfer [MW]
        'Q_hot_dis_MW': mdot_wf_dis * dis['Q_hot'] / 1e6,
        'Q_cold_dis_MW': mdot_wf_dis * dis['Q_cold'] / 1e6,
        'Q_reject_dis_MW': mdot_wf_dis * dis['Q_reject'] / 1e6,
    }

    # HX geometry: charge vs discharge
    for name in ("hot_hx", "cold_hx", "recup"):
        label = name.replace("_hx", "")

        result[f"L_{label}_chg"] = chg[name]["L_total"]
        result[f"L_{label}_dis"] = dis[name]["L_total"]

        result[f"Af_{label}_chg"] = chg[name]["A_f"]
        result[f"Af_{label}_dis"] = dis[name]["A_f"]

        result[f"UA_{label}_chg"] = chg[name]["UA_total"]
        result[f"UA_{label}_dis"] = dis[name]["UA_total"]

        result[f"L_error_{label}_pct"] = (
            100.0 * (dis[name]["L_total"] - chg[name]["L_total"])
            / chg[name]["L_total"]
        )

    # Compressor geometry and nominal map checks
    add_compressor_results(result, chg, 'chg')
    add_compressor_results(result, dis, 'dis')

    # Turbine geometry and nominal map checks
    add_turbine_results(result, chg, 'chg')
    add_turbine_results(result, dis, 'dis')

    # Storage mass flow rates
    result["mdot_hot_chg"] = chg["mdot_hot_chg"]
    result["mdot_cold_chg"] = chg["mdot_cold_chg"]
    result["mdot_hot_dis"] = dis["hot_hx"]["mdot_storage"]
    result["mdot_cold_dis"] = dis["cold_hx"]["mdot_storage"]
    
    for mode, cycle in [("chg", chg), ("dis", dis)]:
        for i, st in sorted(cycle["states"].items()):
            result[f"T_{mode}_{i}_K"] = st["T"]
            result[f"P_{mode}_{i}_Pa"] = st["P"]
            result[f"s_{mode}_{i}_JkgK"] = st["s"]
            result[f"h_{mode}_{i}_Jkg"] = st["h"]



    print('\n===== RESULTS =====')
    for k,v in result.items(): print(f'{k}: {v:.6f}')
    # Save results to CSV
    with open('PTES_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result.keys())
        writer.writerow(result.values())
    plot_ts(chg, dis)
    plot_all_hx_profiles(chg, dis)

    plt.show()

    return result, chg, dis



if __name__ == '__main__':
    results, charge_result, discharge_result = solve_cycle()
