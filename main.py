from properties import state_from_TP
from components.compressor import Compressor
from components.turbine import Turbine
from components.heat_exchanger import TemperatureTargetHeatExchanger
from cycles.DischargeCycle import DischargeCycle
from cycles.ChargeCycle import ChargeCycle

TTD = 0.0

#Charge
P_low_chg = 8e6
P_high_chg = 11e6
beta_dischg = P_high_chg / P_low_chg

T_comp_in_chg = 750 + 273.15
T_turb_in_chg = 35 + 273.15

P_low_dischg = 8e6
P_high_dischg = 11e6
beta_dischg = P_high_dischg / P_low_dischg

#T_comp_in_dischg = 35 + 273.15
#T_turb_in_dischg = 800 + 273.15


#Charge:
T_1_chg = T_comp_in_chg
T_3_chg = T_turb_in_chg
state_1 = state_from_TP(T=T_1_chg, P=P_low_dischg)
state_3 = state_from_TP(T=T_3_chg, P=P_high_chg)

compressor = Compressor(eta=0.85, P_out=P_high_chg)
turbine = Turbine(eta=0.82, P_out=P_low_chg)

hot_hx = TemperatureTargetHeatExchanger(approach_temp=TTD)
cold_hx = TemperatureTargetHeatExchanger(approach_temp=TTD)

cycle = ChargeCycle(
    compressor=compressor,
    turbine=turbine,
    hot_hx=hot_hx,
    cold_hx=cold_hx
)

results = cycle.solve(
    state_1=state_1,
    state_3=state_3
)

print(results["specific_quantities"])


#Discharge

T_low_hotTES = results["TES_temperatures"]["T_low_hotTES_in"]
T_high_HotTES = results["TES_temperatures"]["T_high_HotTES_out"]

T_high_ColdTES = results["TES_temperatures"]["T_high_ColdTES_in"]
T_low_ColdTES = results["TES_temperatures"]["T_low_ColdTES_out"]

T_1_dischg = T_low_ColdTES + TTD
T_3_dischg = T_high_HotTES - TTD

state_1_dischg = state_from_TP(T=T_1_dischg, P=P_low_dischg)
state_3_dischg = state_from_TP(T=T_3_dischg, P=P_high_dischg)

compressor_dischg = Compressor(eta=0.85, P_out=P_high_dischg)
turbine_dischg = Turbine(eta=0.82, P_out=P_low_dischg)

# hot_hx = TemperatureTargetHeatExchanger(approach_temp=TTD)
# cold_hx = TemperatureTargetHeatExchanger(approach_temp=TTD)

cycle = DischargeCycle(
    compressor=compressor_dischg,
    turbine=turbine_dischg,
)

results = cycle.solve(
    state_1=state_1_dischg,
    state_3=state_3_dischg
    #m_dot=m_dot
)

print(results["specific_quantities"])