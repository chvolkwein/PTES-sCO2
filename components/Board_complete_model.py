import numpy as np
from CoolProp.CoolProp import PropsSI
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

FLUID = "CO2"


# ============================================================
# Property functions
# ============================================================

def h_TP(T, P):
    return PropsSI("Hmass", "T", T, "P", P, FLUID)


def s_TP(T, P):
    return PropsSI("Smass", "T", T, "P", P, FLUID)


def h_PS(P, s):
    return PropsSI("Hmass", "P", P, "Smass", s, FLUID)


def T_HP(h, P):
    return PropsSI("T", "Hmass", h, "P", P, FLUID)


def s_HP(h, P):
    return PropsSI("Smass", "Hmass", h, "P", P, FLUID)


# ============================================================
# Complete PTES cycle
# ============================================================

def PTES_cycle(
    PR_chg,
    PR_dis,
    TTD_heat_reject_dis=15.36460930737223,
    #PR_dis=2.44,
    TTD_General=5.0,
    Pressure_losses=0.01,
    eta_comp=0.9,
    eta_turb=0.9,
    T_TES_HL_ini=45 + 273.15,
    T_TES_CH_ini=110 + 273.15,
    P_low_chg=8e6,
    P_low_dis=8e6,
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

    # --------------------------------------------------------
    # State 1
    # After cold HX / before compressor
    # --------------------------------------------------------

    TTD_Cold_chg_high = TTD_General

    P_chg[1] = P_low_chg
    T_chg[1] = T_TES_CH_ini - TTD_Cold_chg_high

    h_chg[1] = h_TP(T_chg[1], P_chg[1])
    s_chg[1] = s_TP(T_chg[1], P_chg[1])


    # --------------------------------------------------------
    # State 2
    # Compressor outlet
    # --------------------------------------------------------

    P_chg[2] = P_low_chg * PR_chg

    s_chg_s_2 = s_chg[1]

    h_chg_s_2 = h_PS(
        P_chg[2],
        s_chg_s_2
    )

    w_comp_chg_s = h_chg_s_2 - h_chg[1]

    w_comp_chg = w_comp_chg_s / eta_comp

    h_chg[2] = h_chg[1] + w_comp_chg

    T_chg[2] = T_HP(h_chg[2], P_chg[2])
    s_chg[2] = s_HP(h_chg[2], P_chg[2])


    # --------------------------------------------------------
    # State 3
    # Hot HX outlet
    # --------------------------------------------------------

    TTD_Hot_chg_high = TTD_General
    TTD_Hot_chg_low = TTD_General

    # Hot-high TES temperature generated during charge
    T_TES_HH = T_chg[2] - TTD_Hot_chg_high

    P_chg[3] = P_chg[2] * (1 - Pressure_losses)

    T_chg[3] = T_TES_HL_ini + TTD_Hot_chg_low

    h_chg[3] = h_TP(T_chg[3], P_chg[3])
    s_chg[3] = s_TP(T_chg[3], P_chg[3])

    q_hot_chg = h_chg[2] - h_chg[3]


    # --------------------------------------------------------
    # State 4
    # Heat rejection
    # --------------------------------------------------------

    T_chg[4] = T_chg[3] - TTD_heat_reject_chg

    # Same as your EES because pressure-loss term is commented
    P_chg[4] = P_chg[3]

    h_chg[4] = h_TP(T_chg[4], P_chg[4])
    s_chg[4] = s_TP(T_chg[4], P_chg[4])

    q_reject_chg = h_chg[3] - h_chg[4]


    # --------------------------------------------------------
    # State 5
    # Turbine outlet
    # --------------------------------------------------------

    P_chg[5] = P_low_chg / (1 - Pressure_losses)

    s_chg_s_5 = s_chg[4]

    h_chg_s_5 = h_PS(
        P_chg[5],
        s_chg_s_5
    )

    w_turb_chg_s = h_chg[4] - h_chg_s_5

    w_turb_chg = w_turb_chg_s * eta_turb

    h_chg[5] = h_chg[4] - w_turb_chg

    T_chg[5] = T_HP(h_chg[5], P_chg[5])
    s_chg[5] = s_HP(h_chg[5], P_chg[5])


    # --------------------------------------------------------
    # Cold HX
    # --------------------------------------------------------

    TTD_Cold_chg_low = TTD_General

    T_TES_CL = T_chg[5] + TTD_Cold_chg_low

    q_cold_chg = h_chg[1] - h_chg[5]


    # ========================================================
    # DISCHARGE
    # ========================================================

    # --------------------------------------------------------
    # State 1
    # Cold HX outlet / heat-rejection inlet
    # --------------------------------------------------------

    TTD_Cold_dis_low = TTD_General

    T_dis[1] = T_TES_CL + TTD_Cold_dis_low

    P_dis[1] = P_low_dis / (1 - Pressure_losses)

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

    w_comp_dis = w_comp_dis_s / eta_comp

    h_dis[3] = h_dis[2] + w_comp_dis

    T_dis[3] = T_HP(h_dis[3], P_dis[3])
    s_dis[3] = s_HP(h_dis[3], P_dis[3])


    # --------------------------------------------------------
    # State 4
    # Hot HX outlet
    # --------------------------------------------------------

    TTD_Hot_dis_low = TTD_General

    T_TES_HL_new = T_dis[3] + TTD_Hot_dis_low

    TTD_Hot_dis_high = TTD_General

    T_dis[4] = T_TES_HH - TTD_Hot_dis_high

    P_dis[4] = P_dis[3] * (1 - Pressure_losses)

    h_dis[4] = h_TP(T_dis[4], P_dis[4])
    s_dis[4] = s_TP(T_dis[4], P_dis[4])

    q_hot_dis = h_dis[4] - h_dis[3]


    # --------------------------------------------------------
    # State 5
    # Turbine outlet
    # --------------------------------------------------------

    P_dis[5] = P_dis[1] / (1 - Pressure_losses)

    s_dis_s_5 = s_dis[4]

    h_dis_s_5 = h_PS(
        P_dis[5],
        s_dis_s_5
    )

    w_turb_dis_s = h_dis[4] - h_dis_s_5

    w_turb_dis = w_turb_dis_s * eta_turb

    h_dis[5] = h_dis[4] - w_turb_dis

    T_dis[5] = T_HP(h_dis[5], P_dis[5])
    s_dis[5] = s_HP(h_dis[5], P_dis[5])


    # --------------------------------------------------------
    # Cold HX inlet
    # --------------------------------------------------------

    TTD_Cold_dis_high = TTD_General

    T_TES_CH_new = T_dis[5] - TTD_Cold_dis_high

    q_cold_dis = h_dis[5] - h_dis[1]


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


def residuals(x):

    PR_chg = x[0]
    PR_dis = x[1]
    #TTD_heat_reject_dis = x[1]

    result = PTES_cycle(
        PR_chg=PR_chg,
        PR_dis=PR_dis,
        #TTD_heat_reject_dis=TTD_heat_reject_dis,
    )

    T_CH_ini = 110 + 273.15
    T_HL_ini = 45 + 273.15

    T_CH_new = result["TES"]["T_CH_new"]
    T_HL_new = result["TES"]["T_HL_new"]

    res_CH = T_CH_new - T_CH_ini
    res_HL = T_HL_new - T_HL_ini
    print(result["performance"]["RTE"])

    return [
        res_CH,
        res_HL
    ]

x0 = [
    3.0,    # initial guess PR_chg
    3.2,    # initial guess PR_dis
    #15.0    # initial guess TTD_heat_reject_dis [K]
]



solution, info, ier, msg = fsolve(
    residuals,
    x0,
    full_output=True
)

PR_chg_sol = solution[0]
#TTD_reject_dis_sol = solution[1]
PR_dis_sol = solution[1]

print("Converged:", ier == 1)
print(msg)

print("PR_chg =", PR_chg_sol)
#print("TTD_heat_reject_dis =", TTD_reject_dis_sol)
print("PR_dis =", PR_dis_sol)

print("Residuals =", residuals(solution))

