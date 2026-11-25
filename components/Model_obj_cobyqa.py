import numpy as np
from CoolProp.CoolProp import PropsSI
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from functools import lru_cache
import UA_segmented
import pandas as pd
from functools import lru_cache

W_FLUID = "CO2"
ST_FLUID = "Dowtherm A"

T_TES_HL_ini = 45 + 273.15
T_TES_CH_ini = 250 + 273.15
P_low_chg = 8e6 # Pa
P_low_dis = 8e6 # Pa
W_mdot_dis = 30000 # W
t_dis = 4 * 3600 # s
eff_comp_chg = 0.9
eff_turb_chg = 0.9
eff_comp_dis = 0.9
eff_turb_dis = 0.9
pressure_losses_chg = 0.01
min_T_pinch_chg = 5.0 # K
m_dot_ref = 110.0 # kg/s
D = 0.01 # m
strategy_mdot_st_design_cool = 1   # 0: fixed mdot storage, 1: variable mdot storage to match capacitance rate
strategy_mdot_st_design_heat = 1 # 0: fixed mdot storage, 1: variable mdot storage to match capacitance rate
strategy_mdot_st_offdesign_cool = 2 # 0: fixed mdot storage. 1: variable mdot storage to match capacitance rate. 2: mdot to achieve mass balance between chg and dis
strategy_mdot_st_offdesign_heat = 1 # 0: fixed mdot storage. 1: variable mdot storage to match capacitance rate. 2: mdot to achieve mass balance between chg and dis
delta_T_min_target = 5.0 #K
P_loss_target = 0.01 # %
N_seg=5
Af_bounds=(0.05, 0.5)



# ============================================================
# Property functions
# ============================================================

def h_TP(T, P):
    return PropsSI("Hmass", "T", T, "P", P, W_FLUID)


def s_TP(T, P):
    return PropsSI("Smass", "T", T, "P", P, W_FLUID)


def h_PS(P, s):
    return PropsSI("Hmass", "P", P, "Smass", s, W_FLUID)


def T_HP(h, P):
    return PropsSI("T", "Hmass", h, "P", P, W_FLUID)


def s_HP(h, P):
    return PropsSI("Smass", "Hmass", h, "P", P, W_FLUID)

class SimulationLogger:
    def __init__(self):
        self.records = []

    def log(self, **kwargs):
        self.records.append(kwargs)

    def save(self, filename="simulation_log.xlsx"):
        pd.DataFrame(self.records).to_excel(
            filename, index=False
        )

import atexit

logger = SimulationLogger()
atexit.register(logger.save)

# ============================================================
# Complete PTES cycle
# ============================================================




def PTES_cycle(
    PR_chg,
    PR_dis,
    TTD_heat_reject_dis,
    T_TES_HL_ini,
    T_TES_CH_ini,
    eta_comp_chg,
    eta_turb_chg,
    eta_comp_dis,
    eta_turb_dis,
    P_low_chg,
    P_low_dis,
    delta_T_min_target,
    P_loss_target,
    strategy_mdot_st_design_cool,
    strategy_mdot_st_design_heat,
    strategy_mdot_st_offdesign_cool,
    strategy_mdot_st_offdesign_heat,
    D,
    N_seg=5,
    Af_bounds=Af_bounds,
    TTD_heat_reject_chg=0.0,
):

    # Arrays use positions 1...5 to match EES
    T_chg = np.full(6, np.nan)
    P_chg = np.full(6, np.nan)
    h_chg = np.full(6, np.nan)
    s_chg = np.full(6, np.nan)

    T_dis = np.full(6, np.nan)
    P_dis = np.full(6, np.nan)
    h_dis = np.full(6, np.nan)
    s_dis = np.full(6, np.nan)

    # ========================================================
    # CHARGE
    # ========================================================


    T1_guess = T_TES_CH_ini - 5.0  # Initial guess for T1
    error = 999
    iteration = 0



    while error > 0.5 and iteration < 100:
        # --------------------------------------------------------
        # State 1
        # After cold HX / before compressor
        # --------------------------------------------------------

        #TTD_Cold_chg_high = TTD_General

        print("Iteration charge:", iteration)

        P_chg[1] = P_low_chg
        T_chg[1] = T1_guess

        h_chg[1] = h_TP(T_chg[1], P_chg[1])
        s_chg[1] = s_TP(T_chg[1], P_chg[1])


        # --------------------------------------------------------
        # State 2
        # Compressor outlet
        # --------------------------------------------------------

        P_chg[2] = P_low_chg * PR_chg

        print("PR_chg =", PR_chg)
        print("P_low_chg =", P_low_chg)
        print("P_chg[2] =", P_chg[2])

        s_chg_s_2 = s_chg[1]

        h_chg_s_2 = h_PS(
            P_chg[2],
            s_chg_s_2
        )

        w_comp_chg_s = h_chg_s_2 - h_chg[1]

        w_comp_chg = w_comp_chg_s / eta_comp_chg

        h_chg[2] = h_chg[1] + w_comp_chg

        T_chg[2] = T_HP(h_chg[2], P_chg[2])
        s_chg[2] = s_HP(h_chg[2], P_chg[2])


        # --------------------------------------------------------
        # State 3
        # Hot HX outlet
        # --------------------------------------------------------

        # TTD_Hot_chg_high = TTD_General
        # TTD_Hot_chg_low = TTD_General

        UA_cool_wf_chg_state3 = UA_segmented.HX_UA_segments_cool_wf(
            T_storage_in= T_TES_HL_ini,
            T_wf_in = T_chg[2],
            P_wf_in = P_chg[2],
            mdot_storage_fixed = None,
            mdot_wf = m_dot_ref,
            fluid_storage = ST_FLUID,
            fluid_wf = W_FLUID,
            D=D,
            delta_T_min_target=delta_T_min_target,
            P_loss_target=P_loss_target,
            N_seg=N_seg,
            Af_bounds=Af_bounds,
            strategy_mdot_st_design=strategy_mdot_st_design_cool,
        )
        # Hot-high TES temperature generated during charge
        T_TES_HH = UA_cool_wf_chg_state3["T_storage"][-1]
        P_chg[3] = UA_cool_wf_chg_state3["P_wf"][-1]
        T_chg[3] = UA_cool_wf_chg_state3["T_wf"][-1]

        h_chg[3] = h_TP(T_chg[3], P_chg[3])
        s_chg[3] = s_TP(T_chg[3], P_chg[3])

        q_hot_chg = h_chg[2] - h_chg[3]


        # --------------------------------------------------------
        # State 4
        # Heat rejection
        # --------------------------------------------------------

        T_chg[4] = T_chg[3] - TTD_heat_reject_chg

        # No pressure loss, no rejection in charge
        P_chg[4] = P_chg[3]

        h_chg[4] = h_TP(T_chg[4], P_chg[4])
        s_chg[4] = s_TP(T_chg[4], P_chg[4])

        q_reject_chg = h_chg[3] - h_chg[4]


        # --------------------------------------------------------
        # State 5
        # Turbine outlet
        # --------------------------------------------------------

        P_chg[5] = P_low_chg / (1 - P_loss_target)

        s_chg_s_5 = s_chg[4]

        h_chg_s_5 = h_PS(
            P_chg[5],
            s_chg_s_5
        )

        w_turb_chg_s = h_chg[4] - h_chg_s_5

        w_turb_chg = w_turb_chg_s * eta_turb_chg

        h_chg[5] = h_chg[4] - w_turb_chg

        T_chg[5] = T_HP(h_chg[5], P_chg[5])
        s_chg[5] = s_HP(h_chg[5], P_chg[5])


        # --------------------------------------------------------
        # Cold HX
        # --------------------------------------------------------

        UA_heat_wf_chg_state6 = UA_segmented.HX_UA_segments_heat_wf(
            T_storage_in= T_TES_CH_ini,
            T_wf_in = T_chg[5],
            P_wf_in = P_chg[5],
            mdot_storage_fixed = None,
            mdot_wf = m_dot_ref,
            fluid_storage = ST_FLUID,
            fluid_wf = W_FLUID,
            D=D,
            delta_T_min_target=delta_T_min_target,
            P_loss_target=P_loss_target,
            N_seg=N_seg,
            Af_bounds=Af_bounds,
            strategy_mdot_st_design=strategy_mdot_st_design_heat,
        )

        T_TES_CL = UA_heat_wf_chg_state6["T_storage"][-1]

        q_cold_chg = h_chg[1] - h_chg[5]

        T1_chg_new = UA_heat_wf_chg_state6["T_wf"][-1]
        error = abs(T1_chg_new - T1_guess)

        print(
            "Charge closure:",
            "T1 guess =", T1_guess,
            "T1 new =", T1_chg_new,
            "error =", abs(T1_chg_new - T1_guess)
        )

        T1_guess = 0.5*T1_guess + 0.5*T1_chg_new

        iteration += 1




    # ========================================================
    # DISCHARGE
    # ========================================================

    error_T = 999
    #error_P = 99999
    iteration = 0
    T1_dis_guess = T_TES_CL + 5.0  # Initial guess for T1 in discharge
    P1_dis_guess = P_loss_target

    #while (error_T > 0.5 or error_P > 10000.0) and iteration < 100:
    while error_T > 0.5  and iteration < 100:
        # --------------------------------------------------------
        # State 1
        # Cold HX outlet / heat-rejection inlet
        # --------------------------------------------------------

        #TTD_Cold_dis_low = TTD_General

        print("Iteration dischrge:", iteration)

        T_dis[1] = T1_dis_guess

        P_dis[1] = P_low_dis / (1 - P_loss_target)

        h_dis[1] = h_TP(T_dis[1], P_dis[1])
        s_dis[1] = s_TP(T_dis[1], P_dis[1])


        # --------------------------------------------------------
        # State 2
        # Heat rejection outlet / compressor inlet
        # --------------------------------------------------------

        T_dis[2] = T_dis[1] - TTD_heat_reject_dis

        P_dis[2] = P_low_dis

        h_dis[2] = h_TP(T_dis[2], P_dis[2])
        s_dis[2] = s_TP(T_dis[2], P_dis[2])

        q_reject_dis = h_dis[1] - h_dis[2]


        # --------------------------------------------------------
        # State 3
        # Compressor outlet
        # --------------------------------------------------------

        P_dis[3] = P_low_dis * PR_dis

        s_dis_s_3 = s_dis[2]

        h_dis_s_3 = h_PS(
            P_dis[3],
            s_dis_s_3
        )

        w_comp_dis_s = h_dis_s_3 - h_dis[2]

        w_comp_dis = w_comp_dis_s / eta_comp_dis

        h_dis[3] = h_dis[2] + w_comp_dis

        T_dis[3] = T_HP(h_dis[3], P_dis[3])
        s_dis[3] = s_HP(h_dis[3], P_dis[3])


        # --------------------------------------------------------
        # State 4
        # Hot HX outlet
        # --------------------------------------------------------

        UA_heat_wf_offdesign_dis_state4 = UA_segmented.HX_UA_segments_heat_wf_offdesign(
        T_storage_in = T_TES_HH,
        T_wf_in = T_dis[3],
        P_wf_in = P_dis[3],
        mdot_storage_fixed = None,
        mdot_wf = m_dot_ref,
        fluid_storage = ST_FLUID,
        fluid_wf = W_FLUID,
        D=D,
        N_seg=N_seg,
        A_f_design = UA_cool_wf_chg_state3["A_f"], #Same heat exchanger as in charge, so same design area (cooling the WF)
        L_design = UA_cool_wf_chg_state3["L_total"], #Same heat exchanger as in charge, so same design area (cooling the WF)
        strategy_mdot_st_offdesign=strategy_mdot_st_offdesign_heat,
        P_losses_estimation = 0.01, #Just to calculate the mass flow that matches the capacitances
        mdot_st_cold_chg_info=None,
        mdot_st_hot_chg_info=None,
        mdot_st_cold_dis_info=None,
        mdot_st_hot_dis_info=None,
    )

        T_TES_HL_new =  UA_heat_wf_offdesign_dis_state4["T_storage"][-1]  # T_dis[3] + TTD_Hot_dis_low

        T_dis[4] = UA_heat_wf_offdesign_dis_state4["T_wf"][-1] #T_TES_HH - TTD_Hot_dis_high

        P_dis[4] = UA_heat_wf_offdesign_dis_state4["P_wf"][-1] #P_dis[3] - P_loss_target

        h_dis[4] = h_TP(T_dis[4], P_dis[4])
        s_dis[4] = s_TP(T_dis[4], P_dis[4])

        q_hot_dis = h_dis[4] - h_dis[3]


        # --------------------------------------------------------
        # State 5
        # Turbine outlet
        # --------------------------------------------------------

        P_dis[5] = P_dis[1] / (1 - P1_dis_guess ) #We will know the pressure after we calculate the pressure losses in the cold HX

        s_dis_s_5 = s_dis[4]

        h_dis_s_5 = h_PS(
            P_dis[5],
            s_dis_s_5
        )

        w_turb_dis_s = h_dis[4] - h_dis_s_5

        w_turb_dis = w_turb_dis_s * eta_turb_dis

        h_dis[5] = h_dis[4] - w_turb_dis

        T_dis[5] = T_HP(h_dis[5], P_dis[5])
        s_dis[5] = s_HP(h_dis[5], P_dis[5])


        # --------------------------------------------------------
        # Cold HX inlet and outlet
        # --------------------------------------------------------

        UA_cool_wf_offdesign_dis_state1 = UA_segmented.HX_UA_segments_cool_wf_offdesign(
        T_storage_in = T_TES_CL,
        T_wf_in = T_dis[5],
        P_wf_in = P_dis[5],
        mdot_storage_fixed = None,
        mdot_wf = m_dot_ref,
        fluid_storage = ST_FLUID,
        fluid_wf = W_FLUID,
        D = D,
        N_seg = N_seg,
        A_f_design = UA_heat_wf_chg_state6["A_f"], #Same heat exchanger as in charge, so same design area (heating the WF)
        L_design = UA_heat_wf_chg_state6["L_total"], #Same heat exchanger as in charge, so same design area (heating the WF)
        strategy_mdot_st_offdesign=strategy_mdot_st_offdesign_cool,
        P_losses_estimation = 0.01,
        mdot_st_cold_chg_info=UA_heat_wf_chg_state6["mdot_storage"],
        mdot_st_hot_chg_info=UA_cool_wf_chg_state3["mdot_storage"],
        mdot_st_cold_dis_info=None,
        mdot_st_hot_dis_info=UA_heat_wf_offdesign_dis_state4["mdot_storage"],
    )


        T_TES_CH_new = UA_cool_wf_offdesign_dis_state1["T_storage"][-1] #T_dis[5] - TTD_Cold_dis_high
        #P1_loss_new = UA_cool_wf_offdesign_dis_state1["P_wf"][-1] #P_dis[5] - P_loss_target
        T1_dis_new = UA_cool_wf_offdesign_dis_state1["T_wf"][-1] #T_TES_CL + TTD_Cold_dis_low

        P1_dis_new = UA_cool_wf_offdesign_dis_state1["P_losses_calc"]

        logger.log(
            component="HX_discharge",
            iteration=iteration,
            PR_chg=PR_chg,
            PR_dis=PR_dis,
            TTD_reject=TTD_heat_reject_dis,
            P_guess=P1_dis_guess,
            P_new=P1_dis_new,
            P_in=P_dis[5],
            P_out=UA_cool_wf_offdesign_dis_state1["P_wf"][-1],
            m_dot_wf=m_dot_ref,
            m_dot_storage=UA_cool_wf_offdesign_dis_state1["mdot_storage"],
            T_in=T_dis[5],
            T_out=T1_dis_new,
            P_loss_fraction=UA_cool_wf_offdesign_dis_state1["P_losses_calc"],
            velocity_max=np.max(UA_cool_wf_offdesign_dis_state1["vel_wf"]),
            density_min=np.min(UA_cool_wf_offdesign_dis_state1["rho_wf"]),
            Re_min=np.min(UA_cool_wf_offdesign_dis_state1["Re_wf"]),
            Re_max=np.max(UA_cool_wf_offdesign_dis_state1["Re_wf"]),
            friction_factor_max=np.max(UA_cool_wf_offdesign_dis_state1["C_f_wf"]),
            delta_p_max_Pa=np.max(UA_cool_wf_offdesign_dis_state1["delta_p"]),
            delta_p_total_Pa=UA_cool_wf_offdesign_dis_state1["delta_p_total"],
            error_P=abs(P1_dis_new-P1_dis_guess)
        )

        print("P1_dis_new", P1_dis_new)
        print("P1_dis_guess", P1_dis_guess)
        P1_dis_guess = P1_dis_new

        q_cold_dis = h_dis[5] - h_dis[1]

        error_T = abs(T1_dis_new - T1_dis_guess)
        #error_P = abs(P1_loss_new - P1_dis_guess)

        T1_dis_guess = 0.5*T1_dis_guess + 0.5*T1_dis_new
        #P1_dis_guess = 0.5*P1_dis_guess + 0.5*P1_loss_new

        iteration += 1


    # ========================================================
    # NET WORK
    # ========================================================

    w_net_chg = w_comp_chg - w_turb_chg

    w_net_dis = w_turb_dis - w_comp_dis


    # ========================================================
    # ENERGY BALANCES
    # ========================================================

    E_balance_chg = (
        w_comp_chg
        + q_cold_chg
        - q_hot_chg
        - q_reject_chg
        - w_turb_chg
    )

    E_balance_dis = (
        w_comp_dis
        + q_hot_dis
        - q_cold_dis
        - q_reject_dis
        - w_turb_dis
    )


    # ========================================================
    # PERFORMANCE
    # ========================================================

    eta_RT = w_net_dis / w_net_chg

    COP_chg_heat = q_hot_chg / w_net_chg

    Work_ratio = w_comp_chg / w_turb_chg

    # ========================================================
    # Return everything
    # ========================================================

    print("RTE", eta_RT)
    print("TES T_HL error:", T_TES_HL_new - T_TES_HL_ini)
    print("TES T_CH error:", T_TES_CH_new - T_TES_CH_ini)
    print("\n" + "="*65)
    print("HEAT EXCHANGER DIAGNOSTICS")
    print("="*65)

    HX_results = {
        "HOT CHARGE": UA_cool_wf_chg_state3,
        "COLD CHARGE": UA_heat_wf_chg_state6,
        "HOT DISCHARGE": UA_heat_wf_offdesign_dis_state4,
        "COLD DISCHARGE": UA_cool_wf_offdesign_dis_state1,
    }

    for name, hx in HX_results.items():

        print(f"\n--- {name} ---")

        print(f"A_f           = {hx.get('A_f', np.nan):.6f}")
        print(f"L_total       = {hx.get('L_total', np.nan):.4f} m")
        print(f"mdot_storage  = {hx['mdot_storage']:.4f} kg/s")

        print(f"T_storage_in  = {hx['T_storage'][0] - 273.15:.2f} °C")
        print(f"T_storage_out = {hx['T_storage'][-1] - 273.15:.2f} °C")
        print(f"T_wf_in       = {hx['T_wf'][0] - 273.15:.2f} °C")
        print(f"T_wf_out      = {hx['T_wf'][-1] - 273.15:.2f} °C")

        print(f"P_losses_calc = {hx.get('P_losses_calc', np.nan):.6f}")
        print("PR_chg:",PR_chg)
        print("PR_dis:",PR_dis)
        print("TTD_heat_reject_dis:",TTD_heat_reject_dis)

        # Terminal Temperature Differences (TTD)
        T_st_in = hx["T_storage"][0]
        T_st_out = hx["T_storage"][-1]
        T_wf_in = hx["T_wf"][0]
        T_wf_out = hx["T_wf"][-1]

        if "CHARGE" in name:
            # Hot Charge: WF heats storage
            if "HOT" in name:
                TTD_high = T_wf_in - T_st_out
                TTD_low = T_wf_out - T_st_in
            else:
                TTD_high = T_wf_out - T_st_in
                TTD_low = T_wf_in - T_st_out
        else:
            # Discharge
            if "HOT" in name:
                TTD_high = T_st_in - T_wf_out
                TTD_low = T_st_out - T_wf_in
            else:
                TTD_high = T_wf_in - T_st_out
                TTD_low = T_wf_out - T_st_in

        print(f"TTD_high       = {TTD_high:.4f} K")
        print(f"TTD_low        = {TTD_low:.4f} K")
        print(f"delta_T_min    = {hx.get('delta_T_min', np.nan):.4f} K")

    return {

        "charge": {
            "T": T_chg,
            "P": P_chg,
            "h": h_chg,
            "s": s_chg,
        },

        "discharge": {
            "T": T_dis,
            "P": P_dis,
            "h": h_dis,
            "s": s_dis,
        },

        "TES": {
            "T_HH": T_TES_HH,
            "T_CL": T_TES_CL,
            "T_HL_new": T_TES_HL_new,
            "T_CH_new": T_TES_CH_new,
        },

        "work": {
            "comp_chg": w_comp_chg,
            "turb_chg": w_turb_chg,
            "net_chg": w_net_chg,

            "comp_dis": w_comp_dis,
            "turb_dis": w_turb_dis,
            "net_dis": w_net_dis,
        },

        "heat": {
            "hot_chg": q_hot_chg,
            "cold_chg": q_cold_chg,
            "reject_chg": q_reject_chg,

            "hot_dis": q_hot_dis,
            "cold_dis": q_cold_dis,
            "reject_dis": q_reject_dis,
        },

        "performance": {
            "RTE": eta_RT,
            "COP": COP_chg_heat,
            "work_ratio": Work_ratio,
        },

        "energy_balance": {
            "charge": E_balance_chg,
            "discharge": E_balance_dis,
        }
    }


@lru_cache(maxsize=128)
def cached_cycle(PR_chg, PR_dis, TTD_reject):
    return PTES_cycle(
        PR_chg=PR_chg,
        PR_dis=PR_dis,
        TTD_heat_reject_dis=TTD_reject,
        T_TES_HL_ini=T_TES_HL_ini,
        T_TES_CH_ini=T_TES_CH_ini,
        eta_comp_chg=eff_comp_chg,
        eta_turb_chg=eff_turb_chg,
        eta_comp_dis=eff_comp_dis,
        eta_turb_dis=eff_turb_dis,
        P_low_chg=P_low_chg,
        P_low_dis=P_low_dis,
        delta_T_min_target=delta_T_min_target,
        P_loss_target=P_loss_target,
        strategy_mdot_st_design_cool=strategy_mdot_st_design_cool,
        strategy_mdot_st_design_heat=strategy_mdot_st_design_heat,
        strategy_mdot_st_offdesign_cool=strategy_mdot_st_offdesign_cool,
        strategy_mdot_st_offdesign_heat=strategy_mdot_st_offdesign_heat,
        D=D,
        N_seg=N_seg,
    )


def objective(x):
    # PR_chg, PR_dis, TTD_reject = x

    # result = PTES_cycle(
    #     PR_chg=PR_chg,
    #     PR_dis=PR_dis,
    #     TTD_heat_reject_dis=TTD_reject,
    #     T_TES_HL_ini=T_TES_HL_ini,
    #     T_TES_CH_ini=T_TES_CH_ini,

    #     eta_comp_chg=eff_comp_chg,
    #     eta_turb_chg=eff_turb_chg,
    #     eta_comp_dis=eff_comp_dis,
    #     eta_turb_dis=eff_turb_dis,

    #     P_low_chg=P_low_chg,
    #     P_low_dis=P_low_dis,

    #     delta_T_min_target=delta_T_min_target,
    #     P_loss_target=P_loss_target,

    #     strategy_mdot_st_design_cool=strategy_mdot_st_design_cool,
    #     strategy_mdot_st_design_heat=strategy_mdot_st_design_heat,
    #     strategy_mdot_st_offdesign_cool=strategy_mdot_st_offdesign_cool,
    #     strategy_mdot_st_offdesign_heat=strategy_mdot_st_offdesign_heat,

    #     D=D,
    #     N_seg=N_seg,
    # )

    result = cached_cycle(*map(float, x))

    return -result["performance"]["RTE"]

def closure_CH(x):
    # PR_chg, PR_dis, TTD_reject = x

    # result = PTES_cycle(
    #     PR_chg=PR_chg,
    #     PR_dis=PR_dis,
    #     TTD_heat_reject_dis=TTD_reject,
    #     T_TES_HL_ini=T_TES_HL_ini,
    #     T_TES_CH_ini=T_TES_CH_ini,

    #     eta_comp_chg=eff_comp_chg,
    #     eta_turb_chg=eff_turb_chg,
    #     eta_comp_dis=eff_comp_dis,
    #     eta_turb_dis=eff_turb_dis,

    #     P_low_chg=P_low_chg,
    #     P_low_dis=P_low_dis,

    #     delta_T_min_target=delta_T_min_target,
    #     P_loss_target=P_loss_target,

    #     strategy_mdot_st_design_cool=strategy_mdot_st_design_cool,
    #     strategy_mdot_st_design_heat=strategy_mdot_st_design_heat,
    #     strategy_mdot_st_offdesign_cool=strategy_mdot_st_offdesign_cool,
    #     strategy_mdot_st_offdesign_heat=strategy_mdot_st_offdesign_heat,

    #     D=D,
    #     N_seg=N_seg,
    # )

    result = cached_cycle(*map(float, x))

    return result["TES"]["T_CH_new"] - T_TES_CH_ini


def closure_HL(x):
    # PR_chg, PR_dis, TTD_reject = x

    # result = PTES_cycle(
    #     PR_chg=PR_chg,
    #     PR_dis=PR_dis,
    #     TTD_heat_reject_dis=TTD_reject,
    #     T_TES_HL_ini=T_TES_HL_ini,
    #     T_TES_CH_ini=T_TES_CH_ini,

    #     eta_comp_chg=eff_comp_chg,
    #     eta_turb_chg=eff_turb_chg,
    #     eta_comp_dis=eff_comp_dis,
    #     eta_turb_dis=eff_turb_dis,

    #     P_low_chg=P_low_chg,
    #     P_low_dis=P_low_dis,

    #     delta_T_min_target=delta_T_min_target,
    #     P_loss_target=P_loss_target,

    #     strategy_mdot_st_design_cool=strategy_mdot_st_design_cool,
    #     strategy_mdot_st_design_heat=strategy_mdot_st_design_heat,
    #     strategy_mdot_st_offdesign_cool=strategy_mdot_st_offdesign_cool,
    #     strategy_mdot_st_offdesign_heat=strategy_mdot_st_offdesign_heat,

    #     D=D,
    #     N_seg=N_seg,
    # )

    result = cached_cycle(*map(float, x))

    return result["TES"]["T_HL_new"] - T_TES_HL_ini

from scipy.optimize import minimize



bounds = [
    (1.3, 5.0),    # PR_chg
    (1.3, 5.0),    # PR_dis
    (0.0, 100.0),  # TTD_heat_reject_dis
]

#x0 = [2.6276161660490334, 1.500052855221119, 42.24388119999081]   
x0 = [2.646592793123182, 1.4604769920214362, 42.28716724950158] 

# constraints = [
#     {"type": "eq", "fun": closure_CH},
#     {"type": "eq", "fun": closure_HL},
# ]

# sol = minimize(
#     objective,
#     x0,
#     method="SLSQP",
#     bounds=bounds,
#     constraints=constraints,
#     options={
#         "eps": 1e-3,
#         "maxiter": 100,
#         "disp": True
#     }

# )


sol = minimize(
    objective,
    x0,
    method="COBYQA",
    bounds=bounds,
    constraints=[
        {"type": "eq", "fun": closure_CH},
        {"type": "eq", "fun": closure_HL}
    ],
    options={
        "maxfev": 5000,
        "feasibility_tol": 0.1,
        "final_tr_radius": 1e-7,
        "disp": True
    }
)

print(cached_cycle.cache_info())

print("Success:", sol.success)
print("Message:", sol.message)

PR_chg_opt, PR_dis_opt, TTD_reject_opt = sol.x

print("PR_chg =", PR_chg_opt)
print("PR_dis =", PR_dis_opt)
print("TTD_reject_dis =", TTD_reject_opt)

print("RTE =", -sol.fun)

print("closure CH =", closure_CH(sol.x))
print("closure HL =", closure_HL(sol.x))

print("\n==============================")
print("CHECK PHYSICAL SIGNS")
print("==============================")

# -----------------------------
# WORK
# -----------------------------
# Calculate optimal cycle

result_opt = cached_cycle(*map(float, sol.x))

results = {
        # Optimization variables
    "PR_chg": sol.x[0],
    "PR_dis": sol.x[1],
    "TTD_reject": sol.x[2],

    # Charge states
    **{f"T_chg_{i}": result_opt["charge"]["T"][i] for i in range(1, 6)},
    **{f"P_chg_{i}": result_opt["charge"]["P"][i] for i in range(1, 6)},
    **{f"h_chg_{i}": result_opt["charge"]["h"][i] for i in range(1, 6)},
    **{f"s_chg_{i}": result_opt["charge"]["s"][i] for i in range(1, 6)},

    # Discharge states
    **{f"T_dis_{i}": result_opt["discharge"]["T"][i] for i in range(1, 6)},
    **{f"P_dis_{i}": result_opt["discharge"]["P"][i] for i in range(1, 6)},
    **{f"h_dis_{i}": result_opt["discharge"]["h"][i] for i in range(1, 6)},
    **{f"s_dis_{i}": result_opt["discharge"]["s"][i] for i in range(1, 6)},

    # TES
    **result_opt["TES"],

    # Work
    **{f"w_{k}": v for k, v in result_opt["work"].items()},

    # Heat
    **{f"q_{k}": v for k, v in result_opt["heat"].items()},

    # Performance
    **result_opt["performance"],

    # Energy balances
    **{f"E_balance_{k}": v for k, v in result_opt["energy_balance"].items()},

    # TES closure errors
    "TES_T_HL_error": result_opt["TES"]["T_HL_new"] - T_TES_HL_ini,
    "TES_T_CH_error": result_opt["TES"]["T_CH_new"] - T_TES_CH_ini,
}


pd.DataFrame([results]).to_excel(
    "PTES_optimized_results.xlsx",
    index=False
)

# result_opt = PTES_cycle(
#     PR_chg=PR_chg_opt,
#     PR_dis=PR_dis_opt,
#     TTD_heat_reject_dis=TTD_reject_opt,
#     T_TES_HL_ini=T_TES_HL_ini,
#     T_TES_CH_ini=T_TES_CH_ini,

#     eta_comp_chg=eff_comp_chg,
#     eta_turb_chg=eff_turb_chg,
#     eta_comp_dis=eff_comp_dis,
#     eta_turb_dis=eff_turb_dis,

#     P_low_chg=P_low_chg,
#     P_low_dis=P_low_dis,

#     delta_T_min_target=delta_T_min_target,
#     P_loss_target=P_loss_target,

#     strategy_mdot_st_design_cool=strategy_mdot_st_design_cool,
#     strategy_mdot_st_design_heat=strategy_mdot_st_design_heat,
#     strategy_mdot_st_offdesign_cool=strategy_mdot_st_offdesign_cool,
#     strategy_mdot_st_offdesign_heat=strategy_mdot_st_offdesign_heat,

#     D=D,
#     N_seg=N_seg,
# )


print("\nWORK [kJ/kg]")
print("W_comp_chg =", result_opt["work"]["comp_chg"] / 1000)
print("W_turb_chg =", result_opt["work"]["turb_chg"] / 1000)
print("W_net_chg  =", result_opt["work"]["net_chg"] / 1000)

print("W_comp_dis =", result_opt["work"]["comp_dis"] / 1000)
print("W_turb_dis =", result_opt["work"]["turb_dis"] / 1000)
print("W_net_dis  =", result_opt["work"]["net_dis"] / 1000)

# -----------------------------
# HEAT
# -----------------------------
print("\nHEAT [kJ/kg]")
print("Q_hot_chg    =", result_opt["heat"]["hot_chg"] / 1000)
print("Q_cold_chg   =", result_opt["heat"]["cold_chg"] / 1000)
print("Q_reject_chg =", result_opt["heat"]["reject_chg"] / 1000)

print("Q_hot_dis    =", result_opt["heat"]["hot_dis"] / 1000)
print("Q_cold_dis   =", result_opt["heat"]["cold_dis"] / 1000)
print("Q_reject_dis =", result_opt["heat"]["reject_dis"] / 1000)

# -----------------------------
# SIGN CHECKS
# -----------------------------
print("\nSIGN CHECKS")
print("W_net_chg  > 0 :", result_opt["work"]["net_chg"] > 0)
print("W_net_dis  > 0 :", result_opt["work"]["net_dis"] > 0)

print("Q_hot_chg  > 0 :", result_opt["heat"]["hot_chg"] > 0)
print("Q_cold_chg > 0 :", result_opt["heat"]["cold_chg"] > 0)

print("Q_hot_dis  > 0 :", result_opt["heat"]["hot_dis"] > 0)
print("Q_cold_dis > 0 :", result_opt["heat"]["cold_dis"] > 0)

print("Q_rej_chg  >= 0:", result_opt["heat"]["reject_chg"] >= 0)
print("Q_rej_dis  >= 0:", result_opt["heat"]["reject_dis"] >= 0)

# ============================================================
# T-s plot - Charge and Discharge
# HX / heat rejection = property curve
# Compressor / Turbine = straight line
# ============================================================

plt.figure(figsize=(8, 6))

for mode in ["charge", "discharge"]:

    T = result_opt[mode]["T"]
    P = result_opt[mode]["P"]
    s = result_opt[mode]["s"]

    # Correct component sequence for each mode
    if mode == "charge":
        segments = [
            (1, 2, "straight"),   # compressor
            (2, 3, "curve"),      # hot HX
            (3, 4, "curve"),      # heat rejection
            (4, 5, "straight"),   # turbine
            (5, 1, "curve"),      # cold HX
        ]

    else:  # discharge
        segments = [
            (1, 2, "curve"),      # heat rejection
            (2, 3, "straight"),   # compressor
            (3, 4, "curve"),      # hot HX
            (4, 5, "straight"),   # turbine
            (5, 1, "curve"),      # cold HX
        ]

    first_segment = True

    for a, b, segment_type in segments:

        if segment_type == "straight":

            # Compressor or turbine: straight line
            plt.plot(
                [s[a] / 1000, s[b] / 1000],
                [T[a] - 273.15, T[b] - 273.15],
                label=mode.capitalize() if first_segment else None
            )

        else:

            # HX / heat rejection: evaluate thermodynamic states
            T_seg = np.linspace(T[a], T[b], 100)
            P_seg = np.linspace(P[a], P[b], 100)

            s_seg = np.array([
                s_TP(Ti, Pi)
                for Ti, Pi in zip(T_seg, P_seg)
            ])

            plt.plot(
                s_seg / 1000,
                T_seg - 273.15,
                label=mode.capitalize() if first_segment else None
            )

        first_segment = False

    # Actual states
    plt.scatter(
        s[1:6] / 1000,
        T[1:6] - 273.15
    )

    # State labels
    for i in range(1, 6):
        plt.annotate(
            f"{mode[0].upper()}{i}",
            (s[i] / 1000, T[i] - 273.15),
            xytext=(5, 5),
            textcoords="offset points"
        )

plt.xlabel("Entropy [kJ/kg-K]")
plt.ylabel("Temperature [°C]")
plt.title("PTES CO₂ - T-s Diagram")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()