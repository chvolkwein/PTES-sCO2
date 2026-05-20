from components.heat_exchanger import TemperatureTargetHeatExchanger
from state import State


class DischargeCycle:
    def __init__(self, compressor, turbine, hot_hx, cold_hx):
        self.compressor = compressor
        self.turbine = turbine
        self.hot_hx = hot_hx
        self.cold_hx = cold_hx

    def solve(self, state_1, T_hot_storage, T_cold_storage):#, m_dot):
        state_2 = self.compressor.solve(state_1)

        state_3, q_hot_per_mdot = self.hot_hx.solve_heating(
            wf_in=state_2,
            T_storage_in=T_hot_storage
        )

        state_4 = self.turbine.solve(state_3)

        state_5, q_cold_per_mdot = self.cold_hx.solve_cooling(
            wf_in=state_4,
            T_storage_in=T_cold_storage
        )

        W_comp_per_mdot = state_2.h - state_1.h
        W_turb_per_mdot = state_3.h - state_4.h
        W_net_per_mdot = W_turb_per_mdot - W_comp_per_mdot

        return {
            "states": {
                "1": state_1,
                "2": state_2,
                "3": state_3,
                "4": state_4,
                "5": state_5,
            },
            "specific_quantities": {
                "q_hot_per_kg": q_hot_per_mdot,
                "q_cold_per_kg": q_cold_per_mdot,
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

