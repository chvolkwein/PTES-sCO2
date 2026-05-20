from properties import state_from_TP
from components.compressor import Compressor
from components.turbine import Turbine
from components.heat_exchanger import TemperatureTargetHeatExchanger
from cycles.DischargeCycle import DischargeCycle

P_low = 8e6
P_high = 11e6


T_hot_storage = 800 + 273.15
T_cold_storage = 35 + 273.15
T_1 = T_cold_storage
#m_dot = 5.0

state_1 = state_from_TP(T=T_1, P=P_low)

compressor = Compressor(eta=0.85, P_out=P_high)
turbine = Turbine(eta=0.82, P_out=P_low)

hot_hx = TemperatureTargetHeatExchanger(approach_temp=0.0)
cold_hx = TemperatureTargetHeatExchanger(approach_temp=0.0)

cycle = DischargeCycle(
    compressor=compressor,
    turbine=turbine,
    hot_hx=hot_hx,
    cold_hx=cold_hx
)

results = cycle.solve(
    state_1=state_1,
    T_hot_storage=T_hot_storage,
    T_cold_storage=T_cold_storage,
    #m_dot=m_dot
)

print(results["specific_quantities"])