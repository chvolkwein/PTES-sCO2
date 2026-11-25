from dataclasses import dataclass

from state import State

from properties import state_from_TP
from properties import state_from_PH
from properties import Incompressible

from eeslib import fluid_properties as fluid_properties

import numpy as np
from scipy.optimize import brentq


esol_properties = Incompressible()


# ============================================================
# If q_dot_to_wf is positive:
#
# heat is being added to the working fluid.
# ============================================================


class TemperatureTargetHeatExchanger:

    def __init__(
        self,
        approach_temp,
    ):

        self.approach_temp = (
            approach_temp
        )


    def solve_heating_chg(
        self,
        wf_in,
        wf_out,
    ):
        """
        Storage fluid heats the working fluid.
        """

        T_low_ColdTES_out = (
            wf_in.T
            + self.approach_temp
        )

        T_high_ColdTES_in = (
            wf_out.T
            + self.approach_temp
        )


        q_to_wf_per_m_dot = (
            wf_out.h
            - wf_in.h
        )


        return (
            q_to_wf_per_m_dot,
            T_high_ColdTES_in,
            T_low_ColdTES_out,
        )


    def solve_cooling_chg(
        self,
        wf_in,
        wf_out,
    ):
        """
        Working fluid rejects heat
        to the storage fluid.
        """

        T_high_HotTES_out = (
            wf_in.T
            - self.approach_temp
        )

        T_low_hotTES_in = (
            wf_out.T
            - self.approach_temp
        )


        q_to_wf_per_m_dot = (
            wf_out.h
            - wf_in.h
        )


        return (
            q_to_wf_per_m_dot,
            T_low_hotTES_in,
            T_high_HotTES_out,
        )


# ============================================================
# UA HEAT EXCHANGER
# ============================================================


class UAHX:

    def __init__(
        self,
        eff,
        R_f,
        R_cond,
        N,
    ):

        if not (
            0.0
            < eff
            < 1.0
        ):

            raise ValueError(
                "HX effectiveness must satisfy "
                "0 < eff < 1."
            )


        if N <= 0:

            raise ValueError(
                "Number of HX segments N "
                "must be positive."
            )


        self.eff = eff

        self.R_f = R_f

        self.R_cond = R_cond

        self.N = N

        self._qmax_cache = {}


    @staticmethod
    def _temperature_from_storage_enthalpy(h):

        """Inverse of the Dowtherm A enthalpy correlation used by the model."""

        x = h / 1.0e6

        return (
            -27.06574572 * x**6
            + 151.21088901 * x**5
            - 340.13167310 * x**4
            + 411.02036213 * x**3
            - 388.55934518 * x**2
            + 645.42026919 * x
            + 288.26451925
        )


    def _minimum_pinch(
        self,
        Q,
        wf_in,
        fluid_storage,
        fluid_wf,
        mdot_storage,
        mdot_in_wf,
        T_in_storage,
        storage_heats_wf,
    ):

        """Return the minimum hot-minus-cold temperature difference."""

        fluid_storage_minT = 288.1
        q_i = Q / self.N

        h_storage = np.zeros(self.N + 1)
        h_wf = np.zeros(self.N + 1)
        T_storage = np.zeros(self.N + 1)
        T_wf = np.zeros(self.N + 1)

        h_storage[0] = esol_properties.enthalpy(
            fluid_storage,
            T=T_in_storage,
            T_ref=fluid_storage_minT,
        )
        h_wf[0] = fluid_properties.enthalpy(
            fluid_wf,
            T=wf_in.T,
            P=wf_in.P,
        )
        T_storage[0] = T_in_storage
        T_wf[0] = wf_in.T

        for i in range(self.N):

            if storage_heats_wf:
                h_storage[i + 1] = h_storage[i] - q_i / mdot_storage
                h_wf[i + 1] = h_wf[i] + q_i / mdot_in_wf
            else:
                h_storage[i + 1] = h_storage[i] + q_i / mdot_storage
                h_wf[i + 1] = h_wf[i] - q_i / mdot_in_wf

            T_storage[i + 1] = self._temperature_from_storage_enthalpy(
                h_storage[i + 1]
            )
            T_wf[i + 1] = fluid_properties.temperature(
                fluid_wf,
                h=h_wf[i + 1],
                P=wf_in.P,
            )

        if storage_heats_wf:
            T_h = T_storage
            T_c_spatial = T_wf[::-1]
        else:
            T_h = T_wf
            T_c_spatial = T_storage[::-1]

        delta_T = T_h - T_c_spatial

        if not np.all(np.isfinite(delta_T)):
            raise ValueError(
                "Non-finite temperature difference while finding Q_max_real."
            )

        return float(np.min(delta_T))


    def _real_maximum_duty(
        self,
        Q_endpoint_max,
        wf_in,
        fluid_storage,
        fluid_wf,
        mdot_storage,
        mdot_in_wf,
        T_in_storage,
        storage_heats_wf,
    ):

        """Find the largest duty that avoids temperature crossover."""

        cache_key = (
            float(Q_endpoint_max),
            float(wf_in.T),
            float(wf_in.P),
            float(mdot_storage),
            float(mdot_in_wf),
            float(T_in_storage),
            bool(storage_heats_wf),
        )

        if cache_key in self._qmax_cache:
            return self._qmax_cache[cache_key]

        def pinch(Q_trial):
            return self._minimum_pinch(
                Q=Q_trial,
                wf_in=wf_in,
                fluid_storage=fluid_storage,
                fluid_wf=fluid_wf,
                mdot_storage=mdot_storage,
                mdot_in_wf=mdot_in_wf,
                T_in_storage=T_in_storage,
                storage_heats_wf=storage_heats_wf,
            )

        pinch_at_zero = pinch(0.0)

        if pinch_at_zero <= 0.0:
            raise ValueError(
                "HX inlet temperatures do not provide a positive driving force."
            )

        pinch_at_endpoint_max = pinch(Q_endpoint_max)

        if pinch_at_endpoint_max >= 0.0:
            Q_max_real = Q_endpoint_max

        else:
            Q_max_real = brentq(
                pinch,
                0.0,
                Q_endpoint_max,
                xtol=max(1.0e-6, 1.0e-10 * Q_endpoint_max),
                rtol=1.0e-10,
                maxiter=50,
            )

        self._qmax_cache[cache_key] = Q_max_real

        return Q_max_real


    def _solve_with_fixed_ua(
        self,
        solver,
        UA_target,
        **solver_inputs,
    ):

        """Solve duty directly so the discretized UA matches a fixed design."""

        if not np.isfinite(UA_target) or UA_target <= 0.0:
            raise ValueError("UA_target must be finite and positive.")

        reference_result = solver(**solver_inputs)
        Q_max_real = reference_result["Q_max_real"]
        Q_reference = reference_result["Q"]
        result_cache = {
            float(Q_reference): reference_result,
        }

        def evaluate(Q):
            cache_key = float(Q)

            if cache_key not in result_cache:
                result_cache[cache_key] = solver(
                    Q_target=cache_key,
                    **solver_inputs,
                )

            return result_cache[cache_key]

        def ua_residual(Q):
            return evaluate(Q)["UA_total"] - UA_target

        residual_reference = (
            reference_result["UA_total"] - UA_target
        )

        if residual_reference >= 0.0:
            Q_above = Q_reference
            Q_below = None
            Q_trial = Q_reference

            for _ in range(60):
                Q_trial *= 0.5

                try:
                    residual_trial = ua_residual(Q_trial)
                except ValueError:
                    continue

                if residual_trial <= 0.0:
                    Q_below = Q_trial
                    break

            if Q_below is None:
                raise ValueError(
                    "Fixed UA is below the minimum represented by the HX model."
                )

        else:
            Q_below = Q_reference
            Q_limit = Q_max_real
            Q_above = None

            for _ in range(80):
                Q_trial = 0.5 * (Q_below + Q_limit)

                try:
                    residual_trial = ua_residual(Q_trial)
                except ValueError:
                    Q_limit = Q_trial
                    continue

                if residual_trial >= 0.0:
                    Q_above = Q_trial
                    break

                Q_below = Q_trial

        if Q_above is None:
            raise ValueError(
                "Fixed UA cannot be reached before the segment pinch limit."
            )

        Q = brentq(
            ua_residual,
            Q_below,
            Q_above,
            xtol=max(1.0e-6, 1.0e-12 * Q_max_real),
            rtol=1.0e-12,
            maxiter=80,
        )

        result = evaluate(Q)
        result["effectiveness"] = Q / Q_max_real
        result["UA_target"] = UA_target

        return result


    def solve_heating_wf_fixed_ua(
        self,
        UA_target,
        **solver_inputs,
    ):

        return self._solve_with_fixed_ua(
            solver=self.solve_heating_wf,
            UA_target=UA_target,
            **solver_inputs,
        )


    def solve_cooling_wf_fixed_ua(
        self,
        UA_target,
        **solver_inputs,
    ):

        return self._solve_with_fixed_ua(
            solver=self.solve_cooling_wf,
            UA_target=UA_target,
            **solver_inputs,
        )


    # ========================================================
    # STORAGE HEATS WORKING FLUID
    #
    # HOT  = storage
    # COLD = working fluid
    #
    # Used for:
    #
    # charge cold HX
    # discharge hot HX
    # ========================================================

    def solve_heating_wf(
        self,
        wf_in,
        fluid_storage,
        fluid_wf,
        mdot_storage,
        mdot_in_wf,
        T_in_storage,
        Q_target=None,
    ):

        """
        Storage fluid heats the working fluid.

        Hot stream:
            storage

        Cold stream:
            working fluid
        """


        # ----------------------------------------------------
        # Basic inputs
        # ----------------------------------------------------

        fluid_storage_minT = 288.1

        T_in_wf = wf_in.T

        P_in_wf = wf_in.P


        T_h_in = T_in_storage

        T_c_in = T_in_wf


        # ----------------------------------------------------
        # Basic physical checks
        # ----------------------------------------------------

        if mdot_storage <= 0.0:

            raise ValueError(
                "mdot_storage must be positive."
            )


        if mdot_in_wf <= 0.0:

            raise ValueError(
                "mdot_in_wf must be positive."
            )


        if T_h_in <= T_c_in:

            raise ValueError(
                "Invalid HX inlet temperatures "
                "in solve_heating_wf: "
                f"T_h_in = {T_h_in:.6f} K, "
                f"T_c_in = {T_c_in:.6f} K"
            )


        # ----------------------------------------------------
        # Global maximum possible heat transfer
        # ----------------------------------------------------

        Q_storage_max = (

            mdot_storage

            * (

                esol_properties.enthalpy(
                    fluid_storage,
                    T=T_h_in,
                    T_ref=fluid_storage_minT,
                )

                -

                esol_properties.enthalpy(
                    fluid_storage,
                    T=T_c_in,
                    T_ref=fluid_storage_minT,
                )
            )
        )


        Q_wf_max = (

            mdot_in_wf

            * (

                fluid_properties.enthalpy(
                    fluid_wf,
                    T=T_h_in,
                    P=P_in_wf,
                )

                -

                fluid_properties.enthalpy(
                    fluid_wf,
                    T=T_c_in,
                    P=P_in_wf,
                )
            )
        )


        Q_max = min(
            Q_storage_max,
            Q_wf_max,
        )


        if (
            not np.isfinite(Q_max)
            or Q_max <= 0.0
        ):

            raise ValueError(
                "Invalid global Q_max "
                "in solve_heating_wf: "
                f"Q_storage_max = "
                f"{Q_storage_max:.6f} W, "
                f"Q_wf_max = "
                f"{Q_wf_max:.6f} W, "
                f"Q_max = {Q_max:.6f} W"
            )


        # ----------------------------------------------------
        # Design duty
        # ----------------------------------------------------

        Q_max_endpoint = Q_max

        Q_max_real = self._real_maximum_duty(
            Q_endpoint_max=Q_max_endpoint,
            wf_in=wf_in,
            fluid_storage=fluid_storage,
            fluid_wf=fluid_wf,
            mdot_storage=mdot_storage,
            mdot_in_wf=mdot_in_wf,
            T_in_storage=T_in_storage,
            storage_heats_wf=True,
        )

        if Q_target is None:
            Q_design = self.eff * Q_max_real
        else:
            Q_design = float(Q_target)

            if (
                not np.isfinite(Q_design)
                or Q_design <= 0.0
                or Q_design >= Q_max_real
            ):
                raise ValueError(
                    "Q_target must satisfy 0 < Q_target < Q_max_real."
                )


        q_i = (
            Q_design
            / self.N
        )


        # ====================================================
        # ARRAYS
        # ====================================================

        h_h = np.zeros(
            self.N + 1
        )

        h_c = np.zeros(
            self.N + 1
        )

        T_h = np.zeros(
            self.N + 1
        )

        T_c = np.zeros(
            self.N + 1
        )


        # ----------------------------------------------------
        # Inlet enthalpies
        # ----------------------------------------------------

        h_h_in = (
            esol_properties.enthalpy(
                fluid_storage,
                T=T_h_in,
                T_ref=fluid_storage_minT,
            )
        )


        h_c_in = (
            fluid_properties.enthalpy(
                fluid_wf,
                T=T_c_in,
                P=P_in_wf,
            )
        )


        h_h[0] = h_h_in

        h_c[0] = h_c_in

        T_h[0] = T_h_in

        T_c[0] = T_c_in


        # ====================================================
        # STORAGE T(h)
        # ====================================================

        def T_from_h_storage(
            h,
        ):

            return self._temperature_from_storage_enthalpy(h)


        # ====================================================
        # BUILD STREAM PROFILES
        #
        # q_i is equal for every segment in DESIGN.
        # ====================================================

        for i in range(
            self.N
        ):

            h_h[i + 1] = (

                h_h[i]

                - q_i
                / mdot_storage
            )


            h_c[i + 1] = (

                h_c[i]

                + q_i
                / mdot_in_wf
            )


            T_h[i + 1] = (
                T_from_h_storage(
                    h_h[i + 1]
                )
            )


            T_c[i + 1] = (
                fluid_properties.temperature(
                    fluid_wf,
                    h=h_c[i + 1],
                    P=P_in_wf,
                )
            )


        # ====================================================
        # COUNTERFLOW SPATIAL ARRAYS
        #
        # physical coordinate:
        #
        # node 0 ---------------------------- node N
        #
        # hot:
        #        --------------------------->
        #
        # cold:
        #        <---------------------------
        #
        # T_h:
        # node 0 ---------------------------- node N
        #
        # T_c_spatial:
        # node 0 ---------------------------- node N
        # ====================================================

        T_c_spatial = (
            T_c[::-1]
        )

        h_c_spatial = (
            h_c[::-1]
        )


        # ====================================================
        # SEGMENT CAPACITANCE RATES
        # ====================================================

        C_h_seg = np.zeros(
            self.N
        )

        C_c_seg = np.zeros(
            self.N
        )

        C_min_seg = np.zeros(
            self.N
        )

        C_max_seg = np.zeros(
            self.N
        )

        C_r_seg = np.zeros(
            self.N
        )


        for i in range(
            self.N
        ):

            deltaT_h = (
                T_h[i]
                - T_h[i + 1]
            )


            deltaT_c = (
                T_c_spatial[i]
                - T_c_spatial[i + 1]
            )


            if (
                deltaT_h <= 0.0
                or deltaT_c <= 0.0
            ):

                raise ValueError(
                    f"Invalid temperature change "
                    f"in segment {i}: "
                    f"deltaT_h = "
                    f"{deltaT_h:.6f} K, "
                    f"deltaT_c = "
                    f"{deltaT_c:.6f} K"
                )


            C_h_seg[i] = (

                mdot_storage

                * (
                    h_h[i]
                    - h_h[i + 1]
                )

                / deltaT_h
            )


            C_c_seg[i] = (

                mdot_in_wf

                * (
                    h_c_spatial[i]
                    - h_c_spatial[i + 1]
                )

                / deltaT_c
            )


            if (
                C_h_seg[i] <= 0.0
                or C_c_seg[i] <= 0.0
            ):

                raise ValueError(
                    f"Invalid capacitance rate "
                    f"in segment {i}: "
                    f"C_h = "
                    f"{C_h_seg[i]:.6f}, "
                    f"C_c = "
                    f"{C_c_seg[i]:.6f}"
                )


            C_min_seg[i] = min(
                C_h_seg[i],
                C_c_seg[i],
            )


            C_max_seg[i] = max(
                C_h_seg[i],
                C_c_seg[i],
            )


            C_r_seg[i] = (

                C_min_seg[i]

                / C_max_seg[i]
            )


        # ====================================================
        # EFFECTIVENESS -> NTU -> UA
        # ====================================================

        epsilon_seg = np.zeros(
            self.N
        )

        NTU_seg = np.zeros(
            self.N
        )

        UA_seg = np.zeros(
            self.N
        )


        for i in range(
            self.N
        ):

            # ------------------------------------------------
            # Local counterflow inlet temperature difference
            #
            # Hot inlet:
            #     T_h[i]
            #
            # Cold inlet:
            #     T_c_spatial[i+1]
            # ------------------------------------------------

            deltaT_in_i = (

                T_h[i]

                - T_c_spatial[i + 1]
            )


            if deltaT_in_i <= 0.0:

                raise ValueError(
                    f"Temperature crossover "
                    f"in segment {i}: "
                    f"Th_in = "
                    f"{T_h[i]:.6f} K, "
                    f"Tc_in = "
                    f"{T_c_spatial[i+1]:.6f} K, "
                    f"deltaT = "
                    f"{deltaT_in_i:.6f} K"
                )


            Q_max_i = (

                C_min_seg[i]

                * deltaT_in_i
            )


            if (
                not np.isfinite(Q_max_i)
                or Q_max_i <= 0.0
            ):

                raise ValueError(
                    f"Non-positive Q_max "
                    f"in segment {i}: "
                    f"Q_max_i = "
                    f"{Q_max_i:.6f}"
                )


            epsilon_seg[i] = (

                q_i

                / Q_max_i
            )


            if (
                not np.isfinite(
                    epsilon_seg[i]
                )

                or not (
                    0.0
                    < epsilon_seg[i]
                    < 1.0
                )
            ):

                raise ValueError(
                    f"Invalid effectiveness "
                    f"in segment {i}: "
                    f"epsilon = "
                    f"{epsilon_seg[i]}, "
                    f"Q_i = "
                    f"{q_i:.6f}, "
                    f"Q_max_i = "
                    f"{Q_max_i:.6f}"
                )


            # ------------------------------------------------
            # Counterflow epsilon-NTU inversion
            # ------------------------------------------------

            if np.isclose(
                C_r_seg[i],
                1.0,
            ):

                NTU_seg[i] = (

                    epsilon_seg[i]

                    / (
                        1.0
                        - epsilon_seg[i]
                    )
                )


            else:

                log_argument = (

                    (
                        1.0
                        - epsilon_seg[i]
                        * C_r_seg[i]
                    )

                    /

                    (
                        1.0
                        - epsilon_seg[i]
                    )
                )


                if (
                    not np.isfinite(
                        log_argument
                    )

                    or log_argument <= 0.0
                ):

                    raise ValueError(
                        f"Invalid NTU log argument "
                        f"in segment {i}: "
                        f"log_argument = "
                        f"{log_argument}"
                    )


                NTU_seg[i] = (

                    1.0

                    / (
                        1.0
                        - C_r_seg[i]
                    )

                    * np.log(
                        log_argument
                    )
                )


            if (
                not np.isfinite(
                    NTU_seg[i]
                )

                or NTU_seg[i] <= 0.0
            ):

                raise ValueError(
                    f"Invalid NTU in segment {i}: "
                    f"NTU = {NTU_seg[i]}"
                )


            UA_seg[i] = (

                NTU_seg[i]

                * C_min_seg[i]
            )


            if (
                not np.isfinite(
                    UA_seg[i]
                )

                or UA_seg[i] <= 0.0
            ):

                raise ValueError(
                    f"Invalid UA in segment {i}: "
                    f"UA = {UA_seg[i]}"
                )


        # ====================================================
        # TOTAL UA
        # ====================================================

        UA_total = np.sum(
            UA_seg
        )


        if (
            not np.isfinite(
                UA_total
            )

            or UA_total <= 0.0
        ):

            raise ValueError(
                f"Invalid total UA: "
                f"{UA_total}"
            )


        # ====================================================
        # OUTLETS
        # ====================================================

        wf_out = state_from_PH(
            P_in_wf,
            h_c[-1],
        )


        T_storage_out = (
            T_h[-1]
        )


        # ====================================================
        # RETURN
        # ====================================================

        return {

            "wf_out":
                wf_out,

            "T_storage_out":
                T_storage_out,

            "Q":
                Q_design,

            "Q_max":
                Q_max_real,

            "Q_max_real":
                Q_max_real,

            "Q_max_endpoint":
                Q_max_endpoint,

            "pinch_min":
                float(np.min(T_h - T_c_spatial)),

            "delta_T_hot_end":
                float(T_h[0] - T_c_spatial[0]),

            "delta_T_cold_end":
                float(T_h[-1] - T_c_spatial[-1]),

            "UA_total":
                UA_total,

            "UA_seg":
                UA_seg,

            "epsilon_seg":
                epsilon_seg,

            "NTU_seg":
                NTU_seg,

            "C_h_seg":
                C_h_seg,

            "C_c_seg":
                C_c_seg,

            "C_min_seg":
                C_min_seg,

            "C_max_seg":
                C_max_seg,

            "C_r_seg":
                C_r_seg,

            "h_h":
                h_h,

            "h_c":
                h_c_spatial,

            "T_h":
                T_h,

            "T_c":
                T_c_spatial,
        }


    # ========================================================
    # WORKING FLUID HEATS STORAGE
    #
    # HOT  = working fluid
    # COLD = storage
    #
    # Used for:
    #
    # charge hot HX
    # discharge cold HX
    # ========================================================

    def solve_cooling_wf(
        self,
        wf_in,
        fluid_storage,
        fluid_wf,
        mdot_storage,
        mdot_in_wf,
        T_in_storage,
        Q_target=None,
    ):

        """
        Working fluid heats the storage fluid.

        Hot stream:
            working fluid

        Cold stream:
            storage
        """


        # ----------------------------------------------------
        # Inputs
        # ----------------------------------------------------

        fluid_storage_minT = 288.1

        T_in_wf = wf_in.T

        P_in_wf = wf_in.P


        T_h_in = T_in_wf

        T_c_in = T_in_storage


        # ----------------------------------------------------
        # Basic physical checks
        # ----------------------------------------------------

        if mdot_storage <= 0.0:

            raise ValueError(
                "mdot_storage must be positive."
            )


        if mdot_in_wf <= 0.0:

            raise ValueError(
                "mdot_in_wf must be positive."
            )


        if T_h_in <= T_c_in:

            raise ValueError(
                "Invalid HX inlet temperatures "
                "in solve_cooling_wf: "
                f"T_h_in = {T_h_in:.6f} K, "
                f"T_c_in = {T_c_in:.6f} K"
            )


        # ----------------------------------------------------
        # Global maximum heat transfer
        # ----------------------------------------------------

        Q_storage_max = (

            mdot_storage

            * (

                esol_properties.enthalpy(
                    fluid_storage,
                    T=T_h_in,
                    T_ref=fluid_storage_minT,
                )

                -

                esol_properties.enthalpy(
                    fluid_storage,
                    T=T_c_in,
                    T_ref=fluid_storage_minT,
                )
            )
        )


        Q_wf_max = (

            mdot_in_wf

            * (

                fluid_properties.enthalpy(
                    fluid_wf,
                    T=T_h_in,
                    P=P_in_wf,
                )

                -

                fluid_properties.enthalpy(
                    fluid_wf,
                    T=T_c_in,
                    P=P_in_wf,
                )
            )
        )


        Q_max = min(
            Q_storage_max,
            Q_wf_max,
        )


        if (
            not np.isfinite(Q_max)
            or Q_max <= 0.0
        ):

            raise ValueError(
                "Invalid global Q_max "
                "in solve_cooling_wf: "
                f"Q_storage_max = "
                f"{Q_storage_max:.6f} W, "
                f"Q_wf_max = "
                f"{Q_wf_max:.6f} W, "
                f"Q_max = {Q_max:.6f} W"
            )


        # ----------------------------------------------------
        # Design duty
        # ----------------------------------------------------

        Q_max_endpoint = Q_max

        Q_max_real = self._real_maximum_duty(
            Q_endpoint_max=Q_max_endpoint,
            wf_in=wf_in,
            fluid_storage=fluid_storage,
            fluid_wf=fluid_wf,
            mdot_storage=mdot_storage,
            mdot_in_wf=mdot_in_wf,
            T_in_storage=T_in_storage,
            storage_heats_wf=False,
        )

        if Q_target is None:
            Q_design = self.eff * Q_max_real
        else:
            Q_design = float(Q_target)

            if (
                not np.isfinite(Q_design)
                or Q_design <= 0.0
                or Q_design >= Q_max_real
            ):
                raise ValueError(
                    "Q_target must satisfy 0 < Q_target < Q_max_real."
                )


        q_i = (
            Q_design
            / self.N
        )


        # ====================================================
        # ARRAYS
        # ====================================================

        h_h = np.zeros(
            self.N + 1
        )

        h_c = np.zeros(
            self.N + 1
        )

        T_h = np.zeros(
            self.N + 1
        )

        T_c = np.zeros(
            self.N + 1
        )


        # ----------------------------------------------------
        # Inlet enthalpies
        # ----------------------------------------------------

        h_h_in = (
            fluid_properties.enthalpy(
                fluid_wf,
                T=T_h_in,
                P=P_in_wf,
            )
        )


        h_c_in = (
            esol_properties.enthalpy(
                fluid_storage,
                T=T_c_in,
                T_ref=fluid_storage_minT,
            )
        )


        h_h[0] = h_h_in

        h_c[0] = h_c_in

        T_h[0] = T_h_in

        T_c[0] = T_c_in


        # ====================================================
        # STORAGE T(h)
        # ====================================================

        def T_from_h_storage(
            h,
        ):

            return self._temperature_from_storage_enthalpy(h)


        # ====================================================
        # BUILD STREAM PROFILES
        # ====================================================

        for i in range(
            self.N
        ):

            h_h[i + 1] = (

                h_h[i]

                - q_i
                / mdot_in_wf
            )


            h_c[i + 1] = (

                h_c[i]

                + q_i
                / mdot_storage
            )


            T_h[i + 1] = (

                fluid_properties.temperature(
                    fluid_wf,
                    h=h_h[i + 1],
                    P=P_in_wf,
                )
            )


            T_c[i + 1] = (
                T_from_h_storage(
                    h_c[i + 1]
                )
            )


        # ====================================================
        # COUNTERFLOW SPATIAL ARRAYS
        # ====================================================

        T_c_spatial = (
            T_c[::-1]
        )

        h_c_spatial = (
            h_c[::-1]
        )


        # ====================================================
        # SEGMENT CAPACITANCE RATES
        # ====================================================

        C_h_seg = np.zeros(
            self.N
        )

        C_c_seg = np.zeros(
            self.N
        )

        C_min_seg = np.zeros(
            self.N
        )

        C_max_seg = np.zeros(
            self.N
        )

        C_r_seg = np.zeros(
            self.N
        )


        for i in range(
            self.N
        ):

            deltaT_h = (
                T_h[i]
                - T_h[i + 1]
            )


            deltaT_c = (
                T_c_spatial[i]
                - T_c_spatial[i + 1]
            )


            if (
                deltaT_h <= 0.0
                or deltaT_c <= 0.0
            ):

                raise ValueError(
                    f"Invalid temperature change "
                    f"in segment {i}: "
                    f"deltaT_h = "
                    f"{deltaT_h:.6f} K, "
                    f"deltaT_c = "
                    f"{deltaT_c:.6f} K"
                )


            C_h_seg[i] = (

                mdot_in_wf

                * (
                    h_h[i]
                    - h_h[i + 1]
                )

                / deltaT_h
            )


            C_c_seg[i] = (

                mdot_storage

                * (
                    h_c_spatial[i]
                    - h_c_spatial[i + 1]
                )

                / deltaT_c
            )


            if (
                C_h_seg[i] <= 0.0
                or C_c_seg[i] <= 0.0
            ):

                raise ValueError(
                    f"Invalid capacitance rate "
                    f"in segment {i}: "
                    f"C_h = "
                    f"{C_h_seg[i]:.6f}, "
                    f"C_c = "
                    f"{C_c_seg[i]:.6f}"
                )


            C_min_seg[i] = min(
                C_h_seg[i],
                C_c_seg[i],
            )


            C_max_seg[i] = max(
                C_h_seg[i],
                C_c_seg[i],
            )


            C_r_seg[i] = (

                C_min_seg[i]

                / C_max_seg[i]
            )


        # ====================================================
        # EFFECTIVENESS -> NTU -> UA
        # ====================================================

        epsilon_seg = np.zeros(
            self.N
        )

        NTU_seg = np.zeros(
            self.N
        )

        UA_seg = np.zeros(
            self.N
        )


        for i in range(
            self.N
        ):

            deltaT_in_i = (

                T_h[i]

                - T_c_spatial[i + 1]
            )


            if deltaT_in_i <= 0.0:

                raise ValueError(
                    f"Temperature crossover "
                    f"in segment {i}: "
                    f"Th_in = "
                    f"{T_h[i]:.6f} K, "
                    f"Tc_in = "
                    f"{T_c_spatial[i+1]:.6f} K, "
                    f"deltaT = "
                    f"{deltaT_in_i:.6f} K"
                )


            Q_max_i = (

                C_min_seg[i]

                * deltaT_in_i
            )


            if (
                not np.isfinite(Q_max_i)
                or Q_max_i <= 0.0
            ):

                raise ValueError(
                    f"Non-positive Q_max "
                    f"in segment {i}: "
                    f"Q_max_i = "
                    f"{Q_max_i:.6f}"
                )


            epsilon_seg[i] = (

                q_i

                / Q_max_i
            )


            if (
                not np.isfinite(
                    epsilon_seg[i]
                )

                or not (
                    0.0
                    < epsilon_seg[i]
                    < 1.0
                )
            ):

                raise ValueError(
                    f"Invalid effectiveness "
                    f"in segment {i}: "
                    f"epsilon = "
                    f"{epsilon_seg[i]}, "
                    f"Q_i = "
                    f"{q_i:.6f}, "
                    f"Q_max_i = "
                    f"{Q_max_i:.6f}"
                )


            # ------------------------------------------------
            # Counterflow epsilon-NTU inversion
            # ------------------------------------------------

            if np.isclose(
                C_r_seg[i],
                1.0,
            ):

                NTU_seg[i] = (

                    epsilon_seg[i]

                    / (
                        1.0
                        - epsilon_seg[i]
                    )
                )


            else:

                log_argument = (

                    (
                        1.0
                        - epsilon_seg[i]
                        * C_r_seg[i]
                    )

                    /

                    (
                        1.0
                        - epsilon_seg[i]
                    )
                )


                if (
                    not np.isfinite(
                        log_argument
                    )

                    or log_argument <= 0.0
                ):

                    raise ValueError(
                        f"Invalid NTU log argument "
                        f"in segment {i}: "
                        f"log_argument = "
                        f"{log_argument}"
                    )


                NTU_seg[i] = (

                    1.0

                    / (
                        1.0
                        - C_r_seg[i]
                    )

                    * np.log(
                        log_argument
                    )
                )


            if (
                not np.isfinite(
                    NTU_seg[i]
                )

                or NTU_seg[i] <= 0.0
            ):

                raise ValueError(
                    f"Invalid NTU in segment {i}: "
                    f"NTU = {NTU_seg[i]}"
                )


            UA_seg[i] = (

                NTU_seg[i]

                * C_min_seg[i]
            )


            if (
                not np.isfinite(
                    UA_seg[i]
                )

                or UA_seg[i] <= 0.0
            ):

                raise ValueError(
                    f"Invalid UA in segment {i}: "
                    f"UA = {UA_seg[i]}"
                )


        # ====================================================
        # TOTAL UA
        # ====================================================

        UA_total = np.sum(
            UA_seg
        )


        if (
            not np.isfinite(
                UA_total
            )

            or UA_total <= 0.0
        ):

            raise ValueError(
                f"Invalid total UA: "
                f"{UA_total}"
            )


        # ====================================================
        # OUTLETS
        # ====================================================

        wf_out = state_from_PH(
            P_in_wf,
            h_h[-1],
        )


        T_storage_out = (
            T_c[-1]
        )


        # ====================================================
        # RETURN
        # ====================================================

        return {

            "wf_out":
                wf_out,

            "T_storage_out":
                T_storage_out,

            "Q":
                Q_design,

            "Q_max":
                Q_max_real,

            "Q_max_real":
                Q_max_real,

            "Q_max_endpoint":
                Q_max_endpoint,

            "pinch_min":
                float(np.min(T_h - T_c_spatial)),

            "delta_T_hot_end":
                float(T_h[0] - T_c_spatial[0]),

            "delta_T_cold_end":
                float(T_h[-1] - T_c_spatial[-1]),

            "UA_total":
                UA_total,

            "UA_seg":
                UA_seg,

            "epsilon_seg":
                epsilon_seg,

            "NTU_seg":
                NTU_seg,

            "C_h_seg":
                C_h_seg,

            "C_c_seg":
                C_c_seg,

            "C_min_seg":
                C_min_seg,

            "C_max_seg":
                C_max_seg,

            "C_r_seg":
                C_r_seg,

            "h_h":
                h_h,

            "h_c":
                h_c_spatial,

            "T_h":
                T_h,

            "T_c":
                T_c_spatial,
        }
