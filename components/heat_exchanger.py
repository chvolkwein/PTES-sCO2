from dataclasses import dataclass
from state import State
from properties import state_from_TP

#If q_dot_to_wf is positive, then the heat exchanger is adding heat to the working fluid. 

class TemperatureTargetHeatExchanger:
    def __init__(self, approach_temp=0.0):
        self.approach_temp = approach_temp

    def solve_heating_chg(self, wf_in, wf_out):
        """
        Storage fluid heats the working fluid.
        """

        T_low_ColdTES_out = wf_in.T + self.approach_temp
        T_high_ColdTES_in = wf_out.T + self.approach_temp


        q_to_wf_per_m_dot = wf_out.h - wf_in.h

        return q_to_wf_per_m_dot, T_high_ColdTES_in, T_low_ColdTES_out

    def solve_cooling_chg(self, wf_in, wf_out):
        """
        Working fluid rejects heat to the storage fluid.
        """

        T_high_HotTES_out = wf_in.T - self.approach_temp
        T_low_hotTES_in = wf_out.T - self.approach_temp

        q_to_wf_per_m_dot = wf_out.h - wf_in.h

        return q_to_wf_per_m_dot, T_low_hotTES_in, T_high_HotTES_out