from properties import state_from_TP


class ChargeCycle:
    """
    Charge cycle:

        state_1
            |
            v
        Compressor
            |
            v
        state_2
            |
            v
        Hot TES HX
            |
            v
        state_2_hotHX_out
            |
            v
        Heat rejection
            |
            v
        state_3
            |
            v
        Turbine
            |
            v
        state_4
            |
            v
        Cold TES HX
            |
            v
        state_1_calc

    During charge:

        Hot TES:
            T_HL --> HX --> T_HH

        Cold TES:
            T_CH --> HX --> T_CL
    """

    def __init__(self, compressor, turbine, hot_hx, cold_hx):
        self.compressor = compressor
        self.turbine = turbine
        self.hot_hx = hot_hx
        self.cold_hx = cold_hx

    def solve(
        self,
        state_1,
        T_HL,
        T_CH,
        mdot_wf,
        mdot_hot_storage,
        mdot_cold_storage,
        fluid_storage,
        fluid_wf="CO2",
        deltaT_reject=0.0,
    ):

        # ============================================================
        # 1 -> 2 : Compressor
        # ============================================================

        state_2 = self.compressor.solve(state_1)

        W_comp_per_mdot = state_2.h - state_1.h


        # ============================================================
        # 2 -> hot HX outlet
        #
        # Working fluid rejects heat to the HOT storage fluid.
        #
        # Storage:
        #     T_HL --> T_HH
        # ============================================================

        hot_hx_results = self.hot_hx.solve_cooling_wf(
            wf_in=state_2,
            fluid_storage=fluid_storage,
            fluid_wf=fluid_wf,
            mdot_storage=mdot_hot_storage,
            mdot_in_wf=mdot_wf,
            T_in_storage=T_HL,
        )

        state_hot_hx_out = hot_hx_results["wf_out"]

        T_HH = hot_hx_results["T_storage_out"]

        # Heat added to working fluid.
        # Negative here because working fluid is cooling.
        q_hot_per_kg = (
            state_hot_hx_out.h
            - state_2.h
        )


        # ============================================================
        # Heat rejection
        #
        # For now this is simply imposed as a WF temperature drop.
        # Later this can be replaced by an actual heat exchanger model.
        # ============================================================

        T_3 = state_hot_hx_out.T - deltaT_reject
        P_3 = state_hot_hx_out.P

        if T_3 <= 0.0:
            raise ValueError(
                "Invalid working-fluid temperature after heat rejection."
            )

        state_3 = state_from_TP(
            T=T_3,
            P=P_3,
        )

        q_reject_per_kg = (
            state_3.h
            - state_hot_hx_out.h
        )


        # ============================================================
        # 3 -> 4 : Turbine
        # ============================================================

        state_4 = self.turbine.solve(state_3)

        W_turb_per_mdot = (
            state_3.h
            - state_4.h
        )


        # ============================================================
        # 4 -> 1_calc : Cold TES HX
        #
        # Cold storage fluid heats the working fluid.
        #
        # Storage:
        #     T_CH --> T_CL
        # ============================================================

        cold_hx_results = self.cold_hx.solve_heating_wf(
            wf_in=state_4,
            fluid_storage=fluid_storage,
            fluid_wf=fluid_wf,
            mdot_storage=mdot_cold_storage,
            mdot_in_wf=mdot_wf,
            T_in_storage=T_CH,
        )

        state_1_calc = cold_hx_results["wf_out"]

        T_CL = cold_hx_results["T_storage_out"]

        # Heat added to working fluid.
        # Positive here because WF is being heated.
        q_cold_per_kg = (
            state_1_calc.h
            - state_4.h
        )


        # ============================================================
        # Net specific work
        # ============================================================

        W_net_per_mdot = (
            W_comp_per_mdot
            - W_turb_per_mdot
        )


        # ============================================================
        # Cycle closure errors
        #
        # These should NOT be forced to zero here.
        # Main / optimizer will use them as residuals.
        # ============================================================

        closure_h = (
            state_1_calc.h
            - state_1.h
        )

        closure_T = (
            state_1_calc.T
            - state_1.T
        )


        # ============================================================
        # Return results
        # ============================================================

        return {

            "states": {
                "1": state_1,
                "2": state_2,

                # Physical state immediately after hot TES HX
                "hot_hx_out": state_hot_hx_out,

                # Turbine inlet after heat rejection
                "3": state_3,

                "4": state_4,

                # State obtained after closing the cycle through cold HX
                "1_calc": state_1_calc,
            },


            "specific_quantities": {

                # Sign convention:
                # q > 0 : heat added to working fluid
                # q < 0 : heat removed from working fluid

                "q_hot_per_kg": q_hot_per_kg,
                "q_reject_per_kg": q_reject_per_kg,
                "q_cold_per_kg": q_cold_per_kg,

                "w_comp_per_kg": W_comp_per_mdot,
                "w_turb_per_kg": W_turb_per_mdot,
                "w_net_per_kg": W_net_per_mdot,
            },


            "TES_temperatures": {

                # Hot TES
                "T_HL": T_HL,
                "T_HH": T_HH,

                # Cold TES
                "T_CH": T_CH,
                "T_CL": T_CL,
            },


            "heat_exchangers": {
                "hot": hot_hx_results,
                "cold": cold_hx_results,
            },


            "closure": {
                "delta_h": closure_h,
                "delta_T": closure_T,
            },
        }