from properties import state_from_TP


class DischargeCycle:
    """
    Discharge cycle:

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
        state_cold_hx_out
            |
            v
        Heat rejection
            |
            v
        state_1_calc

    During discharge:

        Hot TES:
            T_HH --> HX --> T_HL

        Cold TES:
            T_CL --> HX --> T_CH
    """

    def __init__(self, compressor, turbine, hot_hx, cold_hx):
        self.compressor = compressor
        self.turbine = turbine
        self.hot_hx = hot_hx
        self.cold_hx = cold_hx

    def solve(
        self,
        state_1,
        T_HH,
        T_CL,
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

        W_comp_per_mdot = (
            state_2.h
            - state_1.h
        )


        # ============================================================
        # 2 -> 3 : Hot TES HX
        #
        # Hot storage fluid heats the working fluid.
        #
        # Storage:
        #     T_HH --> T_HL
        # ============================================================

        hot_hx_results = self.hot_hx.solve_heating_wf(
            wf_in=state_2,
            fluid_storage=fluid_storage,
            fluid_wf=fluid_wf,
            mdot_storage=mdot_hot_storage,
            mdot_in_wf=mdot_wf,
            T_in_storage=T_HH,
        )

        state_3 = hot_hx_results["wf_out"]

        T_HL = hot_hx_results["T_storage_out"]

        # Sign convention:
        # q > 0 means heat added to working fluid.
        q_hot_per_kg = (
            state_3.h
            - state_2.h
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
        # 4 -> cold HX outlet : Cold TES HX
        #
        # Working fluid rejects heat to the cold storage fluid.
        #
        # Storage:
        #     T_CL --> T_CH
        # ============================================================

        cold_hx_results = self.cold_hx.solve_cooling_wf(
            wf_in=state_4,
            fluid_storage=fluid_storage,
            fluid_wf=fluid_wf,
            mdot_storage=mdot_cold_storage,
            mdot_in_wf=mdot_wf,
            T_in_storage=T_CL,
        )

        state_cold_hx_out = cold_hx_results["wf_out"]

        T_CH = cold_hx_results["T_storage_out"]

        # Negative because heat leaves the working fluid.
        q_cold_per_kg = (
            state_cold_hx_out.h
            - state_4.h
        )


        # ============================================================
        # Heat rejection
        #
        # For now modeled as an imposed working-fluid temperature drop.
        # Later this can be replaced by a physical sink HX.
        # ============================================================

        T_1_calc = (
            state_cold_hx_out.T
            - deltaT_reject
        )

        P_1_calc = state_cold_hx_out.P

        if T_1_calc <= 0.0:
            raise ValueError(
                "Invalid working-fluid temperature after heat rejection."
            )

        state_1_calc = state_from_TP(
            T=T_1_calc,
            P=P_1_calc,
        )

        q_reject_per_kg = (
            state_1_calc.h
            - state_cold_hx_out.h
        )


        # ============================================================
        # Net specific work
        #
        # Positive W_net = net power produced during discharge.
        # ============================================================

        W_net_per_mdot = (
            W_turb_per_mdot
            - W_comp_per_mdot
        )


        # ============================================================
        # Cycle closure errors
        #
        # Main / optimizer should drive these toward zero.
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
                "3": state_3,
                "4": state_4,

                # Physical state after cold TES HX,
                # before external heat rejection
                "cold_hx_out": state_cold_hx_out,

                # State obtained after completing the full cycle
                "1_calc": state_1_calc,
            },


            "specific_quantities": {

                # Sign convention:
                #
                # q > 0 : heat added to working fluid
                # q < 0 : heat removed from working fluid

                "q_hot_per_kg": q_hot_per_kg,
                "q_cold_per_kg": q_cold_per_kg,
                "q_reject_per_kg": q_reject_per_kg,

                "w_comp_per_kg": W_comp_per_mdot,
                "w_turb_per_kg": W_turb_per_mdot,
                "w_net_per_kg": W_net_per_mdot,
            },


            "TES_temperatures": {

                # Hot TES
                "T_HH": T_HH,
                "T_HL": T_HL,

                # Cold TES
                "T_CL": T_CL,
                "T_CH": T_CH,
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