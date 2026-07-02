from components.heat_exchanger import TemperatureTargetHeatExchanger
from state import State
from properties import state_from_TP


class DischargeCycle:
    def __init__(self, compressor, turbine):
        self.compressor = compressor
        self.turbine = turbine

    def solve(self, state_1, state_3,TTD_reject=0.0 ):#, m_dot):

        #Include heat rejection
        T_new = state_1.T - TTD_reject
        P_new = state_1.P
        state_1_new = state_from_TP(T=T_new, P=P_new)

        state_2 = self.compressor.solve(state_1_new)

        state_4 = self.turbine.solve(state_3)

        W_comp_per_mdot = state_2.h - state_1_new.h
        W_turb_per_mdot = state_3.h - state_4.h
        W_net_per_mdot = W_turb_per_mdot - W_comp_per_mdot
        Q_hot_per_mdot = state_3.h - state_2.h 
        Q_cold_per_mdot = state_4.h - state_1.h
        Q_sink_per_mdot = state_1.h - state_1_new.h

        return {
            "states": {
                "1": state_1_new,
                "2": state_2,
                "3": state_3,
                "4": state_4,
            },
            "specific_quantities": {
                "q_hot_per_kg": Q_hot_per_mdot,
                "q_cold_per_kg": -Q_cold_per_mdot,
                "w_comp_per_kg": W_comp_per_mdot,
                "w_turb_per_kg": W_turb_per_mdot,
                "w_net_per_kg": W_net_per_mdot,
            }#,
            #"rates": {
            #    "Q_dot_hot": m_dot * q_hot_per_mdot,
            #    "Q_dot_cold": m_dot * q_cold_per_mdot,
            #    "W_dot_comp": m_dot * W_comp_per_mdot,
            #    "W_dot_turb": m_dot * W_turb_per_mdot,
            #    "W_dot_net": m_dot * W_net_per_mdot,
            #}
        }

