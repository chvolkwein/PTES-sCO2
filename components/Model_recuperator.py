
import numpy as np
from scipy.optimize import root
from CoolProp.CoolProp import PropsSI

# ============================================================
# INPUTS
# ============================================================

T_TES_HL_ini = 400 + 273
T_TES_CH_ini = 20 + 273

P_low_chg = 8e6
P_low_dis = 8e6

PR_chg = 3.3

TTD_General = 10
Pressure_losses = 0.01

eta_turb = 0.9
eta_comp = 0.9

eff_chg = 0.99
eff_dis = 0.99

fluid = "CO2"


# ============================================================
# THERMODYNAMIC PROPERTIES
# ============================================================

def H_TP(T, P):
    return PropsSI("H", "T", T, "P", P, fluid)

def S_TP(T, P):
    return PropsSI("S", "T", T, "P", P, fluid)

def H_SP(s, P):
    return PropsSI("H", "S", s, "P", P, fluid)

def T_HP(h, P):
    return PropsSI("T", "H", h, "P", P, fluid)

def S_HP(h, P):
    return PropsSI("S", "H", h, "P", P, fluid)


# ============================================================
# CHARGE
# ============================================================

def charge():

    T, P, h, s = {}, {}, {}, {}

    # State 1 - After cold HX
    P[1] = P_low_chg / (1 - Pressure_losses)
    T[1] = T_TES_CH_ini - TTD_General
    h[1] = H_TP(T[1], P[1])
    s[1] = S_TP(T[1], P[1])

    # State 4 - After hot HX
    T[4] = T_TES_HL_ini + TTD_General
    P[4] = P_low_chg * PR_chg * (1 - Pressure_losses)
    h[4] = H_TP(T[4], P[4])
    s[4] = S_TP(T[4], P[4])

    # Recuperator balance
    Q_max_high = (
        H_TP(T[4], P[4])
        - H_TP(T[1], P[4] * (1 - Pressure_losses))
    )

    Q_max_low = (
        H_TP(T[4], P_low_chg)
        - H_TP(T[1], P[1])
    )

    Q_recup = min(Q_max_high, Q_max_low) * eff_chg

    # State 2 - Recuperator low pressure outlet
    h[2] = h[1] + Q_recup
    P[2] = P_low_chg
    T[2] = T_HP(h[2], P[2])
    s[2] = S_TP(T[2], P[2])

    # State 3 - Compressor outlet
    P[3] = P[2] * PR_chg

    h_3s = H_SP(s[2], P[3])
    W_comp = (h_3s - h[2]) / eta_comp

    h[3] = h[2] + W_comp
    T[3] = T_HP(h[3], P[3])
    s[3] = S_HP(h[3], P[3])

    # Hot HX
    T_TES_HH = T[3] - TTD_General
    Q_hot = h[3] - h[4]

    # State 5 - Recuperator high pressure outlet
    h[5] = h[4] - Q_recup
    P[5] = P[4] * (1 - Pressure_losses)
    T[5] = T_HP(h[5], P[5])
    s[5] = S_TP(T[5], P[5])

    # State 6 - Turbine outlet
    P[6] = P_low_chg / (1 - Pressure_losses)**2

    h_6s = H_SP(s[5], P[6])
    W_turb = (h[5] - h_6s) * eta_turb

    h[6] = h[5] - W_turb
    T[6] = T_HP(h[6], P[6])
    s[6] = S_HP(h[6], P[6])

    # State 7 - After heat rejection
    TTD_heat_reject_chg = 0

    T[7] = T[6] - TTD_heat_reject_chg
    P[7] = P_low_chg / (1 - Pressure_losses)**2
    h[7] = H_TP(T[7], P[7])
    s[7] = S_TP(T[7], P[7])

    Q_reject = h[6] - h[7]

    # Cold HX
    T_TES_CL = T[7] + TTD_General
    Q_cold = h[1] - h[7]

    # Performance
    W_net = W_comp - W_turb

    return {
        "T": T, "P": P, "h": h, "s": s,
        "T_TES_HH": T_TES_HH,
        "T_TES_CL": T_TES_CL,
        "Q_hot": Q_hot,
        "Q_cold": Q_cold,
        "Q_reject": Q_reject,
        "Q_recup": Q_recup,
        "W_comp": W_comp,
        "W_turb": W_turb,
        "W_net": W_net,
        "TTD_recup_high": T[4] - T[2],
        "TTD_recup_low": T[5] - T[1],
    }


# ============================================================
# DISCHARGE
# ============================================================

def discharge(PR_dis, TTD_heat_reject_dis, chg):

    T, P, h, s = {}, {}, {}, {}

    T_TES_HH = chg["T_TES_HH"]
    T_TES_CL = chg["T_TES_CL"]

    # State 1 - Cold HX outlet
    T[1] = T_TES_CL + TTD_General
    P[1] = P_low_dis / (1 - Pressure_losses)
    h[1] = H_TP(T[1], P[1])
    s[1] = S_TP(T[1], P[1])

    # State 2 - Heat rejection outlet
    T[2] = T[1] - TTD_heat_reject_dis
    P[2] = P_low_dis
    h[2] = H_TP(T[2], P[2])
    s[2] = S_TP(T[2], P[2])

    Q_reject = h[1] - h[2]

    # State 3 - Compressor outlet
    P[3] = P_low_dis * PR_dis

    h_3s = H_SP(s[2], P[3])
    W_comp = (h_3s - h[2]) / eta_comp

    h[3] = h[2] + W_comp
    T[3] = T_HP(h[3], P[3])
    s[3] = S_HP(h[3], P[3])

    # State 5 - Hot HX outlet
    T[5] = T_TES_HH - TTD_General
    P[5] = P_low_dis * PR_dis * (1 - Pressure_losses)**2
    h[5] = H_TP(T[5], P[5])
    s[5] = S_TP(T[5], P[5])

    # State 6 - Turbine outlet
    P[6] = P_low_dis / (1 - Pressure_losses)**3

    h_6s = H_SP(s[5], P[6])
    W_turb = (h[5] - h_6s) * eta_turb

    h[6] = h[5] - W_turb
    T[6] = T_HP(h[6], P[6])
    s[6] = S_HP(h[6], P[6])

    # Recuperator pressures
    P[4] = P[3] * (1 - Pressure_losses)
    P[7] = P_low_dis / (1 - Pressure_losses)**2

    # Recuperator balance
    Q_max_high = (
        H_TP(T[6], P[4])
        - H_TP(T[3], P[3])
    )

    Q_max_low = (
        H_TP(T[6], P[6])
        - H_TP(T[3], P[7])
    )

    Q_recup = min(Q_max_high, Q_max_low) * eff_dis

    # State 4 - Recuperator high pressure outlet
    h[4] = h[3] + Q_recup
    T[4] = T_HP(h[4], P[4])
    s[4] = S_TP(T[4], P[4])

    # State 7 - Recuperator low pressure outlet
    h[7] = h[6] - Q_recup
    T[7] = T_HP(h[7], P[7])
    s[7] = S_TP(T[7], P[7])

    # Hot HX
    Q_hot = h[5] - h[4]

    # Cold HX
    Q_cold = h[7] - h[1]

    # TES outlet temperatures
    T_TES_CH_new = T[7] - TTD_General
    T_TES_HL_new = T[4] + TTD_General

    # Performance
    W_net = W_turb - W_comp

    return {
        "T": T, "P": P, "h": h, "s": s,
        "T_TES_CH_new": T_TES_CH_new,
        "T_TES_HL_new": T_TES_HL_new,
        "T_TES_HH": T_TES_HH,
        "T_TES_CL": T_TES_CL,
        "Q_hot": Q_hot,
        "Q_cold": Q_cold,
        "Q_reject": Q_reject,
        "Q_recup": Q_recup,
        "W_comp": W_comp,
        "W_turb": W_turb,
        "W_net": W_net,
        "TTD_recup_high": T[6] - T[4],
        "TTD_recup_low": T[7] - T[3],
    }


# ============================================================
# SOLVE TES TEMPERATURE CLOSURE
# ============================================================

chg = charge()

def residuals(x):

    PR_dis, TTD_heat_reject_dis = x

    dis = discharge(PR_dis, TTD_heat_reject_dis, chg)

    error_CH = dis["T_TES_CH_new"] - T_TES_CH_ini
    error_HL = dis["T_TES_HL_new"] - T_TES_HL_ini

    return [error_CH, error_HL]


x0 = [3.2, 30.0]

sol = root(residuals, x0)

if not sol.success:
    raise RuntimeError(sol.message)

PR_dis, TTD_heat_reject_dis = sol.x

dis = discharge(PR_dis, TTD_heat_reject_dis, chg)


# ============================================================
# PERFORMANCE
# ============================================================

eta_RT = dis["W_net"] / chg["W_net"]

COP_chg_heat = chg["Q_hot"] / chg["W_net"]

Work_ratio = chg["W_comp"] / chg["W_turb"]

eta_thermal = dis["W_net"] / dis["Q_hot"]


# ============================================================
# RESULTS
# ============================================================

results = {
    "PR_chg": PR_chg,
    "PR_dis": PR_dis,
    "TTD_heat_reject_dis": TTD_heat_reject_dis,
    "RTE": eta_RT,
    "COP_chg_heat": COP_chg_heat,
    "Work_ratio": Work_ratio,
    "eta_thermal": eta_thermal,
    "TES_T_CH_error": dis["T_TES_CH_new"] - T_TES_CH_ini,
    "TES_T_HL_error": dis["T_TES_HL_new"] - T_TES_HL_ini,
    "TTD_recup_chg_high": chg["TTD_recup_high"],
    "TTD_recup_chg_low": chg["TTD_recup_low"],
    "TTD_recup_dis_high": dis["TTD_recup_high"],
    "TTD_recup_dis_low": dis["TTD_recup_low"],
    "T_TES_HH": chg["T_TES_HH"],
    "T_TES_HL": dis["T_TES_HL_new"],
    "T_TES_CL": chg["T_TES_CL"],
    "T_TES_CH": dis["T_TES_CH_new"],

    # Charge - Specific energy (kJ/kg)
    "W_comp_chg": chg["W_comp"] / 1000,
    "W_turb_chg": chg["W_turb"] / 1000,
    "W_net_chg": chg["W_net"] / 1000,
    "Q_hot_chg": chg["Q_hot"] / 1000,
    "Q_recup_chg": chg["Q_recup"] / 1000,

    # Discharge - Specific energy (kJ/kg)
    "W_comp_dis": dis["W_comp"] / 1000,
    "W_turb_dis": dis["W_turb"] / 1000,
    "W_net_dis": dis["W_net"] / 1000,
    "Q_hot_dis": dis["Q_hot"] / 1000,
    "Q_recup_dis": dis["Q_recup"] / 1000,
}

for name, value in results.items():
    print(f"{name}: {value:.6f}")
