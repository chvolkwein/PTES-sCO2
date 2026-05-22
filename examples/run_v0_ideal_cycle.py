from pathlib import Path
import sys

# Allow imports from project root when running:
# python examples/run_v0_ideal_cycle.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from properties import state_from_TP
from components.compressor import Compressor
from components.turbine import Turbine
from components.heat_exchanger import TemperatureTargetHeatExchanger
from cycles.DischargeCycle import DischargeCycle
from cycles.ChargeCycle import ChargeCycle
from plots import plot_ts_diagram


def main():
    TTD = 5.0

    #Charge
    P_low_chg = 8e6
    P_high_chg = P_low_chg*3.06 #11e6
    beta_chg = P_high_chg / P_low_chg

    T_comp_in_chg = 400 + 273.15
    T_turb_in_chg = 30 + 273.15

    P_low_dischg = P_low_chg# 8e6
    P_high_dischg = P_low_dischg*3.26 #11e6
    beta_dischg = P_high_dischg / P_low_dischg

    #T_comp_in_dischg = 35 + 273.15
    #T_turb_in_dischg = 800 + 273.15


    #Charge:
    T_1_chg = T_comp_in_chg
    T_3_chg = T_turb_in_chg
    state_1 = state_from_TP(T=T_1_chg, P=P_low_chg)
    state_3 = state_from_TP(T=T_3_chg, P=P_high_chg)

    compressor_chg = Compressor(eta=0.9, P_out=P_high_chg)
    turbine_chg = Turbine(eta=0.9, P_out=P_low_chg)

    hot_hx_chg = TemperatureTargetHeatExchanger(approach_temp=TTD)
    cold_hx_chg = TemperatureTargetHeatExchanger(approach_temp=TTD)

    cycle_chg = ChargeCycle(
        compressor=compressor_chg,
        turbine=turbine_chg,
        hot_hx=hot_hx_chg,
        cold_hx=cold_hx_chg
    )

    results_chg = cycle_chg.solve(
        state_1=state_1,
        state_3=state_3
    )

    print("Charge specific quantities: ", results_chg["specific_quantities"])
    print("Charge states: ", results_chg["states"])


    #Discharge

    T_low_hotTES = results_chg["TES_temperatures"]["T_low_hotTES_in"]
    T_high_HotTES = results_chg["TES_temperatures"]["T_high_HotTES_out"]

    T_high_ColdTES = results_chg["TES_temperatures"]["T_high_ColdTES_in"]
    T_low_ColdTES = results_chg["TES_temperatures"]["T_low_ColdTES_out"]

    T_1_dischg = T_low_ColdTES + TTD
    T_3_dischg = T_high_HotTES - TTD

    state_1_dischg = state_from_TP(T=T_1_dischg, P=P_low_dischg)
    state_3_dischg = state_from_TP(T=T_3_dischg, P=P_high_dischg)

    compressor_dischg = Compressor(eta=0.9, P_out=P_high_dischg)
    turbine_dischg = Turbine(eta=0.9, P_out=P_low_dischg)

    # hot_hx = TemperatureTargetHeatExchanger(approach_temp=TTD)
    # cold_hx = TemperatureTargetHeatExchanger(approach_temp=TTD)

    cycle_dischg = DischargeCycle(
        compressor=compressor_dischg,
        turbine=turbine_dischg,
    )

    results_dischg = cycle_dischg.solve(
        state_1=state_1_dischg,
        state_3=state_3_dischg
        #m_dot=m_dot
    )

    print("Discharge specific quantities: ", results_dischg["specific_quantities"])
    print("Discharge states: ", results_dischg["states"])

    roundtrip_efficiency = results_dischg["specific_quantities"]["w_net_per_kg"] / results_chg["specific_quantities"]["w_net_per_kg"]
    print(f"Roundtrip efficiency: {roundtrip_efficiency:.2%}")

    thermal_efficiency = results_dischg["specific_quantities"]["w_net_per_kg"] / abs(results_dischg["specific_quantities"]["q_hot_per_kg"])
    print(f"Thermal efficiency: {thermal_efficiency:.2%}")

    COP_charge_heat = abs(results_chg["specific_quantities"]["q_hot_per_kg"]) / results_chg["specific_quantities"]["w_net_per_kg"] 
    print(f"COP of charge cycle (heat): {COP_charge_heat:.2f}")

    COP_charge_cool = abs(results_chg["specific_quantities"]["q_cold_per_kg"]) / results_chg["specific_quantities"]["w_net_per_kg"]
    print(f"COP of charge cycle (cool): {COP_charge_cool:.2f}")

    work_ratio = results_chg["specific_quantities"]["w_comp_per_kg"] / results_chg["specific_quantities"]["w_turb_per_kg"]
    print(f"Work ratio: {work_ratio:.2f}")


    plot_ts_diagram(results_chg, results_dischg)

if __name__ == "__main__":
    main()