from dataclasses import dataclass
from state import State
from properties import state_from_TP

#If q_dot_to_wf is positive, then the heat exchanger is adding heat to the working fluid. 

class TemperatureTargetHeatExchanger:
    def __init__(self, approach_temp=0.0, pressure_drop=0.0):
        self.approach_temp = approach_temp
        self.pressure_drop = pressure_drop

    def solve_heating(self, wf_in, T_storage_in):
        """
        Storage fluid heats the working fluid.
        """
        P_out = wf_in.P - self.pressure_drop
        T_out = T_storage_in - self.approach_temp

        wf_out = state_from_TP(T=T_out, P=P_out)

        q_to_wf_per_m_dot = wf_out.h - wf_in.h

        return wf_out, q_to_wf_per_m_dot

    def solve_cooling(self, wf_in, T_storage_in):
        """
        Working fluid rejects heat to the storage fluid.
        """
        P_out = wf_in.P - self.pressure_drop
        T_out = T_storage_in + self.approach_temp

        wf_out = state_from_TP(T=T_out, P=P_out)

        q_to_wf_per_m_dot = wf_out.h - wf_in.h

        return wf_out, q_to_wf_per_m_dot