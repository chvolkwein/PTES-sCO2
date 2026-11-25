import numpy as np
from CoolProp.CoolProp import PropsSI
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

FLUID = "CO2"

T_TES_HL_ini = 45 + 273.15
T_TES_CH_ini = 250 + 273.15

TTD_General=40.0


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
    TTD_heat_reject_dis,     
    T_TES_HL_ini,
    T_TES_CH_ini,
    TTD_General=50.0,
    Pressure_losses=0.01,
    eta_comp=0.9,
    eta_turb=0.9,
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


def objective(x):
    PR_chg, PR_dis, TTD_reject = x

    result = PTES_cycle(
        PR_chg=PR_chg,
        PR_dis=PR_dis,
        TTD_heat_reject_dis=TTD_reject,
        T_TES_HL_ini=T_TES_HL_ini,
        T_TES_CH_ini=T_TES_CH_ini
    )

    return -result["performance"]["RTE"]

def closure_CH(x):
    PR_chg, PR_dis, TTD_reject = x

    result = PTES_cycle(
        PR_chg=PR_chg,
        PR_dis=PR_dis,
        TTD_heat_reject_dis=TTD_reject,
        T_TES_HL_ini=T_TES_HL_ini,
        T_TES_CH_ini=T_TES_CH_ini,
    )

    return result["TES"]["T_CH_new"] - T_TES_CH_ini 


def closure_HL(x):
    PR_chg, PR_dis, TTD_reject = x

    result = PTES_cycle(
        PR_chg=PR_chg,
        PR_dis=PR_dis,
        TTD_heat_reject_dis=TTD_reject,
        T_TES_HL_ini=T_TES_HL_ini,
        T_TES_CH_ini=T_TES_CH_ini
    )

    return result["TES"]["T_HL_new"] - (T_TES_HL_ini)

from scipy.optimize import minimize

constraints = [
    {"type": "eq", "fun": closure_CH},
    {"type": "eq", "fun": closure_HL},
]

bounds = [
    (1.1, 5.5),    # PR_chg
    (1.1, 5.5),    # PR_dis
    (0.0, 150.0),  # TTD_heat_reject_dis
]

x0 = [3.0, 2.5, 15.0]

sol = minimize(
    objective,
    x0,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)

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
result_opt = PTES_cycle(
    PR_chg=PR_chg_opt,
    PR_dis=PR_dis_opt,
    TTD_heat_reject_dis=TTD_reject_opt,
    T_TES_HL_ini=T_TES_HL_ini,
    T_TES_CH_ini=T_TES_CH_ini
)

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