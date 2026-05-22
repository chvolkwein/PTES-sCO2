from properties import state_from_TP
from components.compressor import Compressor
from components.turbine import Turbine
from components.heat_exchanger import TemperatureTargetHeatExchanger
from cycles.DischargeCycle import DischargeCycle
from cycles.ChargeCycle import ChargeCycle
from plots import plot_ts_diagram
from components.storage import Tank
from components.storage import TwoTankTES
from properties import EESStorageFluid


TTD = 5.0
#STORAGE_FLUID = "Therminol VP-1"
STORAGE_FLUID = "Therminol VP-1"

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

m_dot_wf_chg = 1
duration_chg = 3600 # seconds, i.e., 1 hour of charge/discharge
m_dot_wf_dischg = 1
duration_dischg = 3600 # seconds, i.e., 1 hour of charge/discharge
P_atm = 101325

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

#Storage

TES_fluid = EESStorageFluid(STORAGE_FLUID)

#Individual Tanks
TES_hot_high = Tank(T_high_HotTES, P_atm)
TES_hot_low = Tank(T_low_hotTES, P_atm)
TES_cold_high = Tank(T_high_ColdTES, P_atm)
TES_cold_low = Tank(T_low_ColdTES, P_atm)

#Two Tank System
#Hot TES System
Hot_TES_system = TwoTankTES(TES_hot_high,TES_hot_low, TES_fluid)

#Cold TES system
Cold_TES_system = TwoTankTES(TES_cold_high,TES_cold_low, TES_fluid)

q_hot_wf = results_chg["specific_quantities"]["q_hot_per_kg"]
q_cold_wf = results_chg["specific_quantities"]["q_cold_per_kg"]

Q_hot_tes = -q_hot_wf * m_dot_wf_chg * duration_chg
Q_cold_tes = -q_cold_wf * m_dot_wf_chg * duration_chg

moved_hot_mass = Hot_TES_system.exchange_heat(Q_hot_tes)
moved_cold_mass = Cold_TES_system.exchange_heat(Q_cold_tes)

print(f"Mass moved from Low TES to High TES in Hot System: {moved_hot_mass:.2f} kg")
print(f"Mass moved from High TES to Low TES in Cold System: {moved_cold_mass:.2f} kg")




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

#Storage in discharge

q_hot_wf_dischg = results_dischg["specific_quantities"]["q_hot_per_kg"]
q_cold_wf_dischg = results_dischg["specific_quantities"]["q_cold_per_kg"]

Q_hot_tes_dischg = -q_hot_wf_dischg * m_dot_wf_dischg * duration_dischg
Q_cold_tes_dischg  = -q_cold_wf_dischg * m_dot_wf_dischg * duration_dischg

moved_hot_mass_dischg = Hot_TES_system.exchange_heat(Q_hot_tes_dischg)
moved_cold_mass_dischg = Cold_TES_system.exchange_heat(Q_cold_tes_dischg)

print(f"Mass moved from Low TES to High TES in Hot System: {moved_hot_mass_dischg:.2f} kg")
print(f"Mass moved from High TES to Low TES in Cold System: {moved_cold_mass_dischg:.2f} kg")