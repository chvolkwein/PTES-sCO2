"""Simplified sCO2 PTES: ambient HX + recuperator + two-tank hot TES.

Install: pip install CoolProp scipy numpy
Run:     python ptes_ambient_recuperator.py

All temperatures K, pressures Pa, enthalpies J/kg, heat/work J/kg.
Assumes a single CO2 flow rate through each cycle, specified minimum HX terminal TTDs,
and a closed hot TES temperature swing. HX feasibility is checked afterwards.
"""
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from CoolProp.CoolProp import PropsSI

FLUID = "CO2"

@dataclass
class Inputs:
    T_TES_HL: float = 390.0 + 273.15
    T_amb: float = 5.0 + 273.15
    P_low_chg: float = 8.0e6
    P_low_dis: float = 8.0e6
    PR_chg: float = 3.3
    TTD: float = 5.0
    dp: float = 0.01
    eta_comp: float = 0.9
    eta_turb: float = 0.9


def state_TP(T, P):
    return {"T": T, "P": P, "h": PropsSI("H", "T", T, "P", P, FLUID),
            "s": PropsSI("S", "T", T, "P", P, FLUID)}


def state_HP(h, P):
    return {"T": PropsSI("T", "H", h, "P", P, FLUID), "P": P,
            "h": h, "s": PropsSI("S", "H", h, "P", P, FLUID)}


def compressor(inlet, P_out, eta):
    h_s = PropsSI("H", "P", P_out, "S", inlet["s"], FLUID)
    return state_HP(inlet["h"] + (h_s - inlet["h"]) / eta, P_out)


def turbine(inlet, P_out, eta):
    h_s = PropsSI("H", "P", P_out, "S", inlet["s"], FLUID)
    return state_HP(inlet["h"] - eta * (inlet["h"] - h_s), P_out)


def cycles(x, p):
    # Unknowns: high-end recuperator approach in charge,
    # high-end recuperator approach in discharge, PR_dis.
    dT_rec_high_chg, dT_rec_dis, PR_dis = x
    d = p.dp

    # CHARGE: 1 -> 2 -> compressor -> 3 -> hot TES -> 4 ->
    #         recuperator -> 5 -> turbine -> 6 -> ambient HX -> 1
    c = {}
    c[1] = state_TP(p.T_amb - p.TTD, p.P_low_chg / (1-d))
    c[4] = state_TP(p.T_TES_HL + p.TTD, p.P_low_chg*p.PR_chg*(1-d))
    c[2] = state_TP(c[4]["T"] - dT_rec_high_chg, p.P_low_chg)
    c[3] = compressor(c[2], p.P_low_chg*p.PR_chg, p.eta_comp)
    T_TES_HH = c[3]["T"] - p.TTD
    c[5] = state_TP(c[1]["T"] + p.TTD, c[4]["P"]*(1-d))
    c[6] = turbine(c[5], p.P_low_chg/(1-d)**2, p.eta_turb)
    rec_chg = (c[4]["h"]-c[5]["h"]) - (c[2]["h"]-c[1]["h"])

    # DISCHARGE: 1 -> compressor -> 2 -> recuperator -> 3 ->
    #            hot TES -> 4 -> turbine -> 5 -> recuperator -> 6 -> ambient HX -> 1
    v = {}
    v[1] = state_TP(p.T_amb + p.TTD, p.P_low_dis)
    v[2] = compressor(v[1], p.P_low_dis*PR_dis, p.eta_comp)
    v[6] = state_TP(v[2]["T"] + p.TTD, p.P_low_dis/(1-d))
    v[4] = state_TP(T_TES_HH - p.TTD, p.P_low_dis*PR_dis*(1-d)**2)
    v[5] = turbine(v[4], p.P_low_dis/(1-d)**2, p.eta_turb)
    v[3] = state_TP(v[5]["T"] - dT_rec_dis, v[2]["P"]*(1-d))
    rec_dis = (v[3]["h"]-v[2]["h"]) - (v[5]["h"]-v[6]["h"])
    T_TES_HL_new = v[3]["T"] + p.TTD

    wc = c[3]["h"]-c[2]["h"]
    wt = c[5]["h"]-c[6]["h"]
    wc_d = v[2]["h"]-v[1]["h"]
    wt_d = v[4]["h"]-v[5]["h"]
    result = {
        "charge": c, "discharge": v, "PR_dis": PR_dis,
        "TTD_rec_high_chg": dT_rec_high_chg, "TTD_rec_low_chg": p.TTD, "TTD_rec_high_dis": dT_rec_dis,
        "T_TES_HH": T_TES_HH, "T_TES_HL_new": T_TES_HL_new,
        "W_comp_chg": wc, "W_turb_chg": wt,
        "W_comp_dis": wc_d, "W_turb_dis": wt_d,
        "W_net_chg": wc-wt, "W_net_dis": wt_d-wc_d,
        "Q_hot_chg": c[3]["h"]-c[4]["h"],
        "Q_hot_dis": v[4]["h"]-v[3]["h"],
        "Q_ambient_chg": c[1]["h"]-c[6]["h"],  # positive = heat absorbed
        "Q_ambient_dis": v[6]["h"]-v[1]["h"],  # positive = heat rejected
        "recup_chg_residual": rec_chg,
        "recup_dis_residual": rec_dis,
        "TES_T_residual": T_TES_HL_new-p.T_TES_HL,
    }
    return result


def residual(x, p):
    try:
        r = cycles(x, p)
        return np.array([r["recup_chg_residual"]/1000,
                         r["recup_dis_residual"]/1000,
                         r["TES_T_residual"]])
    except (ValueError, OverflowError):
        # Some iterations enter unphysical fluid-property states.
        return np.array([1e6, 1e6, 1e6], dtype=float)


def report(r, p):
    print("\n===== CYCLE RESULTS =====")
    for k in ("PR_dis", "TTD_rec_high_chg", "TTD_rec_low_chg", "TTD_rec_high_dis"):
        print(f"{k:25s}: {r[k]:.6f}")
    for k in ("T_TES_HH", "T_TES_HL_new"):
        print(f"{k:25s}: {r[k]:.3f} K ({r[k]-273.15:.2f} C)")
    for k in ("W_comp_chg", "W_turb_chg", "W_net_chg", "W_comp_dis",
              "W_turb_dis", "W_net_dis", "Q_hot_chg", "Q_hot_dis",
              "Q_ambient_chg", "Q_ambient_dis"):
        print(f"{k:25s}: {r[k]/1000:12.3f} kJ/kg")
    for name in ("charge", "discharge"):
        print(f"\n{name.upper()} STATES")
        print("State      T [C]       P [MPa]       h [kJ/kg]")
        for i, s in sorted(r[name].items()):
            print(f"{i:>3d}      {s['T']-273.15:9.3f}    {s['P']/1e6:9.3f}     {s['h']/1000:11.3f}")
    if r["W_net_chg"] > 0:
        print(f"\nCOP_charge_hot          : {r['Q_hot_chg']/r['W_net_chg']:.5f}")
        print(f"Specific-work ratio (*) : {r['W_net_dis']/r['W_net_chg']:.5f}")
    if r["Q_hot_dis"] > 0:
        print(f"eta_thermal             : {r['W_net_dis']/r['Q_hot_dis']:.5f}")
    print("(*) NOT the true RTE unless total circulating CO2 masses are equal.")
    if r["Q_hot_chg"] > 0 and r["Q_hot_dis"] > 0 and r["W_net_chg"] > 0:
        # If same TES fluid, equal delta h per kg storage, and both cycles
        # transfer all sensible heat between the same tank temperatures:
        # m_CO2_dis/m_CO2_chg = q_hot_chg/q_hot_dis.
        rte = (r["W_net_dis"]/r["W_net_chg"])*(r["Q_hot_chg"]/r["Q_hot_dis"])
        print(f"TES energy-matched RTE   : {rte:.5f} (ideal two-tank TES)")
    print("\n===== CHECKS =====")
    print(f"Recuperator charge residual   : {r['recup_chg_residual']/1000:.5g} kJ/kg")
    print(f"Recuperator discharge residual: {r['recup_dis_residual']/1000:.5g} kJ/kg")
    print(f"TES temperature closure       : {r['TES_T_residual']:.5g} K")
    conditions = {
        "Charge recuperator hot end": r["charge"][4]["T"] - r["charge"][2]["T"] >= p.TTD - 1e-6,
        "Charge recuperator cold end": r["charge"][5]["T"] - r["charge"][1]["T"] >= p.TTD - 1e-6,
        "Discharge recuperator hot end": r["discharge"][5]["T"] - r["discharge"][3]["T"] >= p.TTD - 1e-6,
        "Discharge recuperator cold end": r["discharge"][6]["T"] - r["discharge"][2]["T"] >= p.TTD - 1e-6,
        "Heat absorbed from ambient in charge": r["Q_ambient_chg"] > 0,
        "Heat rejected to ambient in discharge": r["Q_ambient_dis"] > 0,
        "Heat stored in charge": r["Q_hot_chg"] > 0,
        "Heat extracted in discharge": r["Q_hot_dis"] > 0,
        "Positive discharge net work": r["W_net_dis"] > 0,
    }
    for key, passed in conditions.items():
        print(f"{'OK  ' if passed else 'FAIL'} {key}")
    print("Note: terminal temperature checks do not guarantee internal HX pinch feasibility.")



def plot_ts_diagram(r, save_path=None, show=True):
    """T-s chart: property-based HX curves and straight turbomachinery segments."""
    fig, ax = plt.subplots(figsize=(10, 7))

    cycles_to_plot = {
        "Charge": {
            "states": r["charge"],
            "segments": [
                (1, 2, "curve"),     # Recuperator, low-pressure side
                (2, 3, "straight"),  # Compressor
                (3, 4, "curve"),     # Hot TES HX
                (4, 5, "curve"),     # Recuperator, high-pressure side
                (5, 6, "straight"),  # Turbine
                (6, 1, "curve"),     # Ambient HX
            ],
            "color": "tab:blue",
            "prefix": "C",
        },
        "Discharge": {
            "states": r["discharge"],
            "segments": [
                (1, 2, "straight"),  # Compressor
                (2, 3, "curve"),     # Recuperator, high-pressure side
                (3, 4, "curve"),     # Hot TES HX
                (4, 5, "straight"),  # Turbine
                (5, 6, "curve"),     # Recuperator, low-pressure side
                (6, 1, "curve"),     # Ambient heat rejection
            ],
            "color": "tab:orange",
            "prefix": "D",
        },
    }

    for mode, data in cycles_to_plot.items():
        states = data["states"]
        color = data["color"]
        for idx, (i, j, segment_type) in enumerate(data["segments"]):
            if segment_type == "straight":
                s_seg = np.array([states[i]["s"], states[j]["s"]])
                T_seg = np.array([states[i]["T"], states[j]["T"]])
            else:
                T_seg = np.linspace(states[i]["T"], states[j]["T"], 100)
                P_seg = np.linspace(states[i]["P"], states[j]["P"], 100)
                s_seg = np.array([PropsSI("S", "T", float(T), "P", float(P), FLUID)
                                  for T, P in zip(T_seg, P_seg)])
            ax.plot(s_seg / 1000, T_seg - 273.15, color=color,
                    linewidth=1.8, label=mode if idx == 0 else None)

        for i, st in states.items():
            s = st["s"] / 1000
            T = st["T"] - 273.15
            ax.scatter(s, T, color=color, s=30, zorder=3)
            ax.annotate(f"{data['prefix']}{i}", (s, T), xytext=(5, 5),
                        textcoords="offset points", fontsize=9, color=color)

    ax.set_xlabel("Specific entropy [kJ/kg-K]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("PTES sCO₂ - T-s Diagram")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    p = Inputs()
    # Variables = [charge recuperator hot-end TTD (K),
    #              discharge recuperator hot-end TTD (K), PR_dis]
    x0 = np.array([15.0, 100.0, 1.7])
    sol = least_squares(residual, x0, args=(p,),
                        bounds=([p.TTD, p.TTD, 1.05], [300.0, 300.0, 8.0]),
                        x_scale=[10.0, 10.0, 3.0],
                        xtol=1e-11, ftol=1e-11, gtol=1e-11,
                        max_nfev=500)
    r = cycles(sol.x, p)
    print(f"Optimizer: {sol.message}")
    if np.linalg.norm(residual(sol.x, p), ord=np.inf) > 1e-3:
        print("WARNING: The three equations did NOT close; this is not a valid solution.")
    report(r, p)
    plot_ts_diagram(r)

if __name__ == "__main__":
    main()
