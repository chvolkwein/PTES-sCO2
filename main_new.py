import numpy as np
from scipy.optimize import least_squares

from properties import state_from_TP

from components.compressor import Compressor
from components.turbine import Turbine
from components.heat_exchanger import UAHX


# ============================================================
# HELPERS
# ============================================================

def C_to_K(T_C):
    return T_C + 273.15


def K_to_C(T_K):
    return T_K - 273.15


# ============================================================
# FIXED PARAMETERS
# ============================================================

eta_comp = 0.9
eta_turb = 0.9

fluid_storage = "Dowtherm A"
fluid_wf = "CO2"

mdot_wf = 1.0


# ============================================================
# HX PARAMETERS
# ============================================================

eff_hx = 0.98
N_HX = 20

R_f = 0.0
R_cond = 0.0

solver_max_iterations = 100


def hx_terminal_delta_T_values(
    chg,
    dis,
):

    return np.array([
        chg["hot_hx"]["delta_T_hot_end"],
        chg["hot_hx"]["delta_T_cold_end"],
        chg["cold_hx"]["delta_T_hot_end"],
        chg["cold_hx"]["delta_T_cold_end"],
        dis["hot_hx"]["delta_T_hot_end"],
        dis["hot_hx"]["delta_T_cold_end"],
        dis["cold_hx"]["delta_T_hot_end"],
        dis["cold_hx"]["delta_T_cold_end"],
    ], dtype=float)


# ============================================================
# USER-SPECIFIED TES INITIAL CONDITIONS
#
# These are physical boundary conditions and are never changed
# by the solver. Charge starts from the two source tanks.
# ============================================================

T_HL_initial = C_to_K(45.0)    # hot TES source tank
T_CH_initial = C_to_K(400.6)   # cold TES source tank


# ============================================================
# NOMINAL VALUES / INITIAL GUESSES
# ============================================================

T1_chg_guess = C_to_K(400.0)
T3_chg_guess = C_to_K(45.0)
T1_dischg_guess = C_to_K(34.0)

P_low_chg_nom = 8.0e6
beta_chg_nom = 3

P_low_dischg_nom = 8.0e6
beta_dischg_nom = 3.2


r_hot_chg_nom = 0.545
r_hot_dischg_nom = 0.531630

# Equal-SOC storage control uses one common cold-to-hot flow
# proportion. The design solver determines that proportion.
cold_to_hot_storage_flow_ratio_nom = 1.453146


# ============================================================
# CHARGE
# ============================================================

def run_charge(
    P_low_chg,
    beta_chg,
    T1_chg,
    T3_chg,
    T_HL,
    T_CH,
    r_hot,
    r_cold,
):

    # --------------------------------------------------------
    # Pressures
    # --------------------------------------------------------

    P_high_chg = (
        P_low_chg
        * beta_chg
    )


    # --------------------------------------------------------
    # Storage mass flows
    # --------------------------------------------------------

    mdot_hot_storage = (
        r_hot
        * mdot_wf
    )

    mdot_cold_storage = (
        r_cold
        * mdot_wf
    )


    # --------------------------------------------------------
    # Machines
    #
    # IMPORTANT:
    # created INSIDE the function because pressure ratio varies
    # --------------------------------------------------------

    compressor = Compressor(
        eta=eta_comp,
        P_out=P_high_chg,
    )

    turbine = Turbine(
        eta=eta_turb,
        P_out=P_low_chg,
    )


    # ========================================================
    # STATE 1
    #
    # Compressor inlet temperature is DESIGN INPUT
    # ========================================================

    state_1 = state_from_TP(
        T=T1_chg,
        P=P_low_chg,
    )


    # ========================================================
    # COMPRESSOR
    # ========================================================

    state_2 = compressor.solve(
        state_1
    )


    # ========================================================
    # HOT HX - CHARGE
    #
    # WF -> hot TES
    #
    # T_HL -> T_HH
    # ========================================================

    hot_hx = UAHX(
        eff=eff_hx,
        R_f=R_f,
        R_cond=R_cond,
        N=N_HX,
    )


    hot_hx_results = (
        hot_hx.solve_cooling_wf(

            wf_in=state_2,

            fluid_storage=fluid_storage,
            fluid_wf=fluid_wf,

            mdot_storage=mdot_hot_storage,
            mdot_in_wf=mdot_wf,

            T_in_storage=T_HL,
        )
    )


    state_2b = (
        hot_hx_results["wf_out"]
    )

    T_HH = (
        hot_hx_results[
            "T_storage_out"
        ]
    )


    # ========================================================
    # STATE 3
    #
    # Turbine inlet after charge heat rejection.
    # ========================================================

    state_3 = state_from_TP(
        T=T3_chg,
        P=P_high_chg,
    )

    q_reject = (
        state_3.h
        - state_2b.h
    )

    # ========================================================
    # TURBINE
    # ========================================================

    state_4 = turbine.solve(
        state_3
    )


    # ========================================================
    # COLD HX - CHARGE
    #
    # cold TES -> WF
    #
    # T_CH -> T_CL
    # ========================================================

    cold_hx = UAHX(
        eff=eff_hx,
        R_f=R_f,
        R_cond=R_cond,
        N=N_HX,
    )


    cold_hx_results = (
        cold_hx.solve_heating_wf(

            wf_in=state_4,

            fluid_storage=fluid_storage,
            fluid_wf=fluid_wf,

            mdot_storage=mdot_cold_storage,
            mdot_in_wf=mdot_wf,

            T_in_storage=T_CH,
        )
    )


    state_1_calc = (
        cold_hx_results["wf_out"]
    )

    T_CL = (
        cold_hx_results[
            "T_storage_out"
        ]
    )


    # ========================================================
    # CHARGE CYCLE CLOSURE
    # ========================================================

    closure_h = (
        state_1_calc.h
        - state_1.h
    )


    # ========================================================
    # EFFECTIVE CAPACITANCE RATES
    # ========================================================

    Q_hot = (
        hot_hx_results["Q"]
    )

    dT_wf_hot = (
        state_2.T
        - state_2b.T
    )

    dT_storage_hot = (
        T_HH
        - T_HL
    )


    if (
        Q_hot <= 0.0
        or dT_wf_hot <= 0.0
        or dT_storage_hot <= 0.0
    ):

        raise ValueError(
            "Invalid hot-HX capacitance-rate calculation."
        )


    C_wf_hot = (
        Q_hot
        / dT_wf_hot
    )

    C_storage_hot = (
        Q_hot
        / dT_storage_hot
    )


    Q_cold = (
        cold_hx_results["Q"]
    )

    dT_wf_cold = (
        state_1_calc.T
        - state_4.T
    )

    dT_storage_cold = (
        T_CH
        - T_CL
    )


    if (
        Q_cold <= 0.0
        or dT_wf_cold <= 0.0
        or dT_storage_cold <= 0.0
    ):

        raise ValueError(
            "Invalid cold-HX capacitance-rate calculation."
        )


    C_wf_cold = (
        Q_cold
        / dT_wf_cold
    )

    C_storage_cold = (
        Q_cold
        / dT_storage_cold
    )


    # ========================================================
    # WORK
    # ========================================================

    w_comp = (
        state_2.h
        - state_1.h
    )

    w_turb = (
        state_3.h
        - state_4.h
    )

    w_net = (
        w_comp
        - w_turb
    )


    # ========================================================
    # HEAT TO WF
    # ========================================================

    q_hot = (
        state_2b.h
        - state_2.h
    )

    q_cold = (
        state_1_calc.h
        - state_4.h
    )


    return {

        "states": {
            "1": state_1,
            "2": state_2,
            "2b": state_2b,
            "3": state_3,
            "4": state_4,
            "1_calc": state_1_calc,
        },

        "TES": {
            "T_HL": T_HL,
            "T_HH": T_HH,
            "T_CH": T_CH,
            "T_CL": T_CL,
        },

        "pressures": {
            "P_low": P_low_chg,
            "P_high": P_high_chg,
            "beta": beta_chg,
        },

        "mass_flow_ratios": {
            "r_hot": r_hot,
            "r_cold": r_cold,
        },

        "hot_hx": hot_hx_results,
        "cold_hx": cold_hx_results,

        "capacitance": {
            "C_wf_hot": C_wf_hot,
            "C_storage_hot": C_storage_hot,
            "C_wf_cold": C_wf_cold,
            "C_storage_cold": C_storage_cold,
        },

        "closure_h": closure_h,

        "q_hot": q_hot,
        "q_cold": q_cold,
        "q_reject": q_reject,

        "w_comp": w_comp,
        "w_turb": w_turb,
        "w_net": w_net,
    }


# ============================================================
# DISCHARGE
#
# The charge-phase hot and cold HX UA values are reused here.
# Discharge effectiveness and duty are off-design results.
# ============================================================

def run_discharge(
    P_low_dischg,
    beta_dischg,
    T1_dischg,
    T_HH,
    T_CL,
    r_hot,
    r_cold,
    UA_hot_design,
    UA_cold_design,
):

    # --------------------------------------------------------
    # Pressures
    # --------------------------------------------------------

    P_high_dischg = (
        P_low_dischg
        * beta_dischg
    )


    # --------------------------------------------------------
    # Storage mass flows
    # --------------------------------------------------------

    mdot_hot_storage = (
        r_hot
        * mdot_wf
    )

    mdot_cold_storage = (
        r_cold
        * mdot_wf
    )


    # --------------------------------------------------------
    # Machines
    # --------------------------------------------------------

    compressor = Compressor(
        eta=eta_comp,
        P_out=P_high_dischg,
    )

    turbine = Turbine(
        eta=eta_turb,
        P_out=P_low_dischg,
    )


    # ========================================================
    # STATE 1
    # ========================================================

    state_1 = state_from_TP(
        T=T1_dischg,
        P=P_low_dischg,
    )


    # ========================================================
    # COMPRESSOR
    # ========================================================

    state_2 = compressor.solve(
        state_1
    )


    # ========================================================
    # HOT HX - DISCHARGE
    #
    # hot TES -> WF
    #
    # T_HH -> T_HL
    # ========================================================

    hot_hx = UAHX(
        eff=eff_hx,
        R_f=R_f,
        R_cond=R_cond,
        N=N_HX,
    )


    hot_hx_results = (
        hot_hx.solve_heating_wf_fixed_ua(

            UA_target=UA_hot_design,

            wf_in=state_2,

            fluid_storage=fluid_storage,
            fluid_wf=fluid_wf,

            mdot_storage=mdot_hot_storage,
            mdot_in_wf=mdot_wf,

            T_in_storage=T_HH,
        )
    )


    state_3 = (
        hot_hx_results["wf_out"]
    )

    T_HL = (
        hot_hx_results[
            "T_storage_out"
        ]
    )


    # ========================================================
    # TURBINE
    # ========================================================

    state_4 = turbine.solve(
        state_3
    )


    # ========================================================
    # COLD HX - DISCHARGE
    #
    # WF -> cold TES
    #
    # T_CL -> T_CH
    # ========================================================

    cold_hx = UAHX(
        eff=eff_hx,
        R_f=R_f,
        R_cond=R_cond,
        N=N_HX,
    )


    cold_hx_results = (
        cold_hx.solve_cooling_wf_fixed_ua(

            UA_target=UA_cold_design,

            wf_in=state_4,

            fluid_storage=fluid_storage,
            fluid_wf=fluid_wf,

            mdot_storage=mdot_cold_storage,
            mdot_in_wf=mdot_wf,

            T_in_storage=T_CL,
        )
    )


    state_4b = (
        cold_hx_results["wf_out"]
    )

    T_CH = (
        cold_hx_results[
            "T_storage_out"
        ]
    )


    # ========================================================
    # DISCHARGE REJECTION
    #
    # Cooler returns state 4b -> state 1.
    # ========================================================

    q_reject = (
        state_1.h
        - state_4b.h
    )


    # ========================================================
    # DISCHARGE CAPACITANCE RATES
    # ========================================================

    Q_hot = (
        hot_hx_results["Q"]
    )

    dT_wf_hot = (
        state_3.T
        - state_2.T
    )

    dT_storage_hot = (
        T_HH
        - T_HL
    )


    if (
        Q_hot <= 0.0
        or dT_wf_hot <= 0.0
        or dT_storage_hot <= 0.0
    ):

        raise ValueError(
            "Invalid discharge hot-HX capacitance calculation."
        )


    C_wf_hot = (
        Q_hot
        / dT_wf_hot
    )

    C_storage_hot = (
        Q_hot
        / dT_storage_hot
    )


    Q_cold = (
        cold_hx_results["Q"]
    )

    dT_wf_cold = (
        state_4.T
        - state_4b.T
    )

    dT_storage_cold = (
        T_CH
        - T_CL
    )


    if (
        Q_cold <= 0.0
        or dT_wf_cold <= 0.0
        or dT_storage_cold <= 0.0
    ):

        raise ValueError(
            "Invalid discharge cold-HX capacitance calculation."
        )


    C_wf_cold = (
        Q_cold
        / dT_wf_cold
    )

    C_storage_cold = (
        Q_cold
        / dT_storage_cold
    )


    # ========================================================
    # WORK
    # ========================================================

    w_comp = (
        state_2.h
        - state_1.h
    )

    w_turb = (
        state_3.h
        - state_4.h
    )

    w_net = (
        w_turb
        - w_comp
    )


    # ========================================================
    # HEAT TO WF
    # ========================================================

    q_hot = (
        state_3.h
        - state_2.h
    )

    q_cold = (
        state_4b.h
        - state_4.h
    )


    return {

        "states": {
            "1": state_1,
            "2": state_2,
            "3": state_3,
            "4": state_4,
            "4b": state_4b,
        },

        "TES": {
            "T_HH": T_HH,
            "T_HL": T_HL,
            "T_CL": T_CL,
            "T_CH": T_CH,
        },

        "pressures": {
            "P_low": P_low_dischg,
            "P_high": P_high_dischg,
            "beta": beta_dischg,
        },

        "hot_hx": hot_hx_results,
        "cold_hx": cold_hx_results,

        "capacitance": {
            "C_wf_hot": C_wf_hot,
            "C_storage_hot": C_storage_hot,
            "C_wf_cold": C_wf_cold,
            "C_storage_cold": C_storage_cold,
        },

        "q_hot": q_hot,
        "q_cold": q_cold,
        "q_reject": q_reject,

        "w_comp": w_comp,
        "w_turb": w_turb,
        "w_net": w_net,
    }


# ============================================================
# DESIGN VARIABLES
#
# x[0] = T1_chg
# x[1] = T3_chg
# x[2] = T1_dischg
# x[3] = beta_chg
# x[4] = beta_dischg
# x[5] = r_hot_chg
# x[6] = r_hot_dischg
# x[7] = cold_to_hot_storage_flow_ratio
# ============================================================

x_nom = np.array([

    T1_chg_guess,
    T3_chg_guess,
    T1_dischg_guess,
    beta_chg_nom,
    beta_dischg_nom,
    r_hot_chg_nom,
    r_hot_dischg_nom,
    cold_to_hot_storage_flow_ratio_nom,

], dtype=float)


# ============================================================
# BOUNDS
# ============================================================

lower_bounds = np.array([

    288.1,          # T1_chg min
    288.1,          # T3_chg min
    288.1,          # T1_dischg min
    1.5,       # beta_chg min
    1.5,       # beta_dischg min
    0.05,      # r_hot_chg min
    0.05,      # r_hot_dischg min
    0.05,      # cold-to-hot storage flow ratio min

], dtype=float)


upper_bounds = np.array([

    T_CH_initial - 0.01, # T1_chg max
    T_CH_initial - 0.01, # T3_chg max
    C_to_K(300.0),       # T1_dischg max
    4.0,          # beta_chg max
    4.0,          # beta_dischg max
    5.0,           # r_hot_chg max
    5.0,           # r_hot_dischg max
    5.0,           # cold-to-hot storage flow ratio max

], dtype=float)


# ============================================================
# VARIABLE SCALING
# ============================================================

x_scale_solver = np.array([

    100.0,     # T1_chg
    100.0,     # T3_chg
    100.0,     # T1_dischg
    1.0,       # beta_chg
    1.0,       # beta_dischg
    1.0,       # r_hot_chg
    1.0,       # r_hot_dischg
    1.0,       # cold-to-hot storage flow ratio

], dtype=float)


# ============================================================
# RESIDUAL SCALING
# ============================================================

H_scale = 1.0e5
T_scale = 10.0
physical_residual_tolerance = 2.0e-3

# ============================================================
# EVALUATE COMPLETE DESIGN
# ============================================================

def evaluate_design(x):

    (
        T1_chg,
        T3_chg,
        T1_dischg,
        beta_chg,
        beta_dischg,
        r_hot_chg,
        r_hot_dischg,
        cold_to_hot_storage_flow_ratio,

    ) = x

    P_low_chg = P_low_chg_nom
    P_low_dischg = P_low_dischg_nom

    r_cold_chg = (
        cold_to_hot_storage_flow_ratio
        * r_hot_chg
    )

    r_cold_dischg = (
        cold_to_hot_storage_flow_ratio
        * r_hot_dischg
    )


    if T_CH_initial <= T1_chg:

        raise ValueError(
            "T_CH_initial must be greater than T1_chg."
        )


    # ========================================================
    # CHARGE
    # ========================================================

    chg = run_charge(

        P_low_chg=P_low_chg,

        beta_chg=beta_chg,

        T1_chg=T1_chg,

        T3_chg=T3_chg,

        T_HL=T_HL_initial,

        T_CH=T_CH_initial,

        r_hot=r_hot_chg,

        r_cold=r_cold_chg,
    )


    T_HH = (
        chg["TES"]["T_HH"]
    )

    T_CL = (
        chg["TES"]["T_CL"]
    )


    if T_HH <= T_HL_initial:

        raise ValueError(
            "Hot TES does not charge."
        )


    if T_CL >= T_CH_initial:

        raise ValueError(
            "Cold TES does not charge."
        )


    # ========================================================
    # DISCHARGE
    # ========================================================

    dis = run_discharge(

        P_low_dischg=P_low_dischg,

        beta_dischg=beta_dischg,

        T1_dischg=T1_dischg,

        T_HH=T_HH,

        T_CL=T_CL,

        r_hot=r_hot_dischg,

        r_cold=r_cold_dischg,

        UA_hot_design=chg["hot_hx"]["UA_total"],

        UA_cold_design=chg["cold_hx"]["UA_total"],
    )


    # ========================================================
    # PHYSICAL RESIDUALS
    # ========================================================

    r_physical = np.array([

        # ----------------------------------------------------
        # 1. Charge WF cycle closes
        # ----------------------------------------------------

        chg["closure_h"]
        / H_scale,


        # ----------------------------------------------------
        # 2. Hot TES closes
        # ----------------------------------------------------

        (
            dis["TES"]["T_HL"]
            - T_HL_initial
        )
        / T_scale,


        # ----------------------------------------------------
        # 3. Cold TES closes
        # ----------------------------------------------------

        (
            dis["TES"]["T_CH"]
            - T_CH_initial
        )
        / T_scale,


    ], dtype=float)


    if not np.all(
        np.isfinite(
            r_physical
        )
    ):

        raise ValueError(
            "Non-finite physical residual."
        )


    return (
        r_physical,
        chg,
        dis,
    )


# ============================================================
# PERIODIC DESIGN OPTIMIZATION
#
# The objective minimizes only the three thermodynamic closure
# residuals. Terminal HX temperature differences are diagnostics.
# ============================================================

evaluation_cache = {}


def cached_design_evaluation(x):

    x = np.asarray(x, dtype=float)
    cache_key = np.ascontiguousarray(x).tobytes()

    if cache_key in evaluation_cache:

        return evaluation_cache[cache_key]


    try:

        result = evaluate_design(x)
        error = None

    except Exception as exc:

        result = None
        error = exc


    if len(evaluation_cache) >= 512:

        evaluation_cache.clear()


    evaluation_cache[cache_key] = (
        result,
        error,
    )

    return result, error


def closure_residuals(x):

    result, error = cached_design_evaluation(x)

    if error is not None:

        return 1.0e6 * np.ones(5)


    physical, chg, dis = result

    rejection_violations = np.minimum(
        np.array([
            -chg["q_reject"] / H_scale,
            -dis["q_reject"] / H_scale,
        ]),
        0.0,
    )

    return np.concatenate([
        physical,
        rejection_violations,
    ])


def rejection_constraints(x):

    result, error = cached_design_evaluation(
        x
    )

    if error is not None:

        return -100.0 * np.ones(2)


    _, chg, dis = result

    return np.array([
        -chg["q_reject"] / H_scale,
        -dis["q_reject"] / H_scale,
    ])


# ============================================================
# INITIAL DESIGN
# ============================================================

print(
    "\nInitial design:"
)


initial_result, initial_error = cached_design_evaluation(
    x_nom
)


if initial_error is not None:

    raise ValueError(
        "Initial design is invalid."
    ) from initial_error


(
    initial_physical,
    initial_chg,
    initial_dis,

) = initial_result


print(
    "Initial physical residuals:",
    initial_physical
)


solution = least_squares(
    closure_residuals,
    x_nom,
    jac="2-point",
    bounds=(lower_bounds, upper_bounds),
    x_scale=x_scale_solver,
    max_nfev=solver_max_iterations,
    ftol=1.0e-10,
    xtol=1.0e-10,
    gtol=1.0e-8,
    diff_step=1.0e-4,
    verbose=2,
)


solution_x = solution.x


# ============================================================
# EXTRACT SOLUTION
# ============================================================

(
    T1_chg_sol,
    T3_chg_sol,
    T1_dischg_sol,
    beta_chg_sol,
    beta_dischg_sol,
    r_hot_chg_sol,
    r_hot_dischg_sol,
    cold_to_hot_storage_flow_ratio_sol,

) = solution_x

P_low_chg_sol = P_low_chg_nom
P_low_dischg_sol = P_low_dischg_nom

r_cold_chg_sol = (
    cold_to_hot_storage_flow_ratio_sol
    * r_hot_chg_sol
)

r_cold_dischg_sol = (
    cold_to_hot_storage_flow_ratio_sol
    * r_hot_dischg_sol
)


P_high_chg_sol = (
    P_low_chg_sol
    * beta_chg_sol
)


P_high_dischg_sol = (
    P_low_dischg_sol
    * beta_dischg_sol
)


# ============================================================
# FINAL EVALUATION
# ============================================================

(
    final_physical_residuals,
    chg_sol,
    dis_sol,

) = evaluate_design(
    solution_x
)


T_HH_sol = (
    chg_sol["TES"]["T_HH"]
)

T_CL_sol = (
    chg_sol["TES"]["T_CL"]
)


# ============================================================
# MASS FLOWS
# ============================================================

mdot_hot_storage_chg_sol = (
    r_hot_chg_sol
    * mdot_wf
)

mdot_cold_storage_chg_sol = (
    r_cold_chg_sol
    * mdot_wf
)

mdot_hot_storage_dischg_sol = (
    r_hot_dischg_sol
    * mdot_wf
)

mdot_cold_storage_dischg_sol = (
    r_cold_dischg_sol
    * mdot_wf
)


duration_ratio_hot = (
    r_hot_chg_sol
    / r_hot_dischg_sol
)

duration_ratio_cold = (
    r_cold_chg_sol
    / r_cold_dischg_sol
)

duration_ratio_sol = 0.5 * (
    duration_ratio_hot
    + duration_ratio_cold
)


# ============================================================
# HX UA
# ============================================================

UA_hot_chg = (
    chg_sol["hot_hx"]["UA_total"]
)

UA_cold_chg = (
    chg_sol["cold_hx"]["UA_total"]
)

UA_hot_dis = (
    dis_sol["hot_hx"]["UA_total"]
)

UA_cold_dis = (
    dis_sol["cold_hx"]["UA_total"]
)


# ============================================================
# CAPACITANCE RATIOS
# ============================================================

Crate_hot_chg = (
    chg_sol["capacitance"]["C_storage_hot"]
    /
    chg_sol["capacitance"]["C_wf_hot"]
)

Crate_cold_chg = (
    chg_sol["capacitance"]["C_storage_cold"]
    /
    chg_sol["capacitance"]["C_wf_cold"]
)

Crate_hot_dis = (
    dis_sol["capacitance"]["C_storage_hot"]
    /
    dis_sol["capacitance"]["C_wf_hot"]
)

Crate_cold_dis = (
    dis_sol["capacitance"]["C_storage_cold"]
    /
    dis_sol["capacitance"]["C_wf_cold"]
)


terminal_delta_T_sol = hx_terminal_delta_T_values(
    chg_sol,
    dis_sol,
)

final_closure_residuals = final_physical_residuals

final_rejection_constraints = rejection_constraints(
    solution.x
)

max_closure_residual = np.max(
    np.abs(final_closure_residuals)
)

minimum_rejection_margin = np.min(
    final_rejection_constraints
)


# ============================================================
# PERFORMANCE
# ============================================================

periodicity_achieved = (
    np.all(np.isfinite(final_physical_residuals))
    and np.max(np.abs(final_physical_residuals))
    < physical_residual_tolerance
)

w_chg = (
    chg_sol["w_net"]
)

w_dis = (
    dis_sol["w_net"]
)


if w_chg > 0.0 and periodicity_achieved:

    specific_work_ratio = (
        w_dis
        / w_chg
    )

    roundtrip_efficiency = (
        specific_work_ratio
        * duration_ratio_sol
    )

else:

    specific_work_ratio = np.nan
    roundtrip_efficiency = np.nan


# ============================================================
# ENERGY CLOSURE
# ============================================================

charge_energy_error = (

    chg_sol["q_hot"]

    + chg_sol["q_cold"]

    + chg_sol["q_reject"]

    + chg_sol["w_net"]
)


discharge_energy_error = (

    dis_sol["q_hot"]

    + dis_sol["q_cold"]

    + dis_sol["q_reject"]

    - dis_sol["w_net"]
)


# ============================================================
# PHYSICAL SUCCESS
# ============================================================

max_physical_residual = np.max(
    np.abs(
        final_physical_residuals
    )
)


physical_success = (
    periodicity_achieved
    and max_closure_residual
    < physical_residual_tolerance
    and minimum_rejection_margin
    >= -physical_residual_tolerance
)


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n============================================================"
)

print(
    "PTES DESIGN SOLUTION"
)

print(
    "============================================================"
)


# ============================================================
# PRESSURES
# ============================================================

print(
    "\nCHARGE PRESSURES"
)

print(
    f"P_low_chg  = "
    f"{P_low_chg_sol/1e6:.6f} MPa (specified)"
)

print(
    f"P_high_chg = "
    f"{P_high_chg_sol/1e6:.6f} MPa"
)

print(
    f"beta_chg   = "
    f"{beta_chg_sol:.6f}"
)


print(
    "\nDISCHARGE PRESSURES"
)

print(
    f"P_low_dis  = "
    f"{P_low_dischg_sol/1e6:.6f} MPa (specified)"
)

print(
    f"P_high_dis = "
    f"{P_high_dischg_sol/1e6:.6f} MPa"
)

print(
    f"beta_dis   = "
    f"{beta_dischg_sol:.6f} (solved)"
)


# ============================================================
# DESIGN TEMPERATURES
# ============================================================

print(
    "\nCHARGE DESIGN TEMPERATURES"
)

print(
    f"T compressor inlet = "
    f"{K_to_C(T1_chg_sol):.3f} °C (solved)"
)

print(
    f"T turbine inlet    = "
    f"{K_to_C(T3_chg_sol):.3f} °C (solved)"
)


# ============================================================
# MASS FLOW
# ============================================================

print(
    "\nMASS FLOW RATIOS"
)

print(
    f"r_hot charge     = "
    f"{r_hot_chg_sol:.6f}"
)

print(
    f"r_cold charge    = "
    f"{r_cold_chg_sol:.6f}"
)

print(
    f"r_hot discharge  = "
    f"{r_hot_dischg_sol:.6f}"
)

print(
    f"r_cold discharge = "
    f"{r_cold_dischg_sol:.6f}"
)

print(
    f"cold/hot flow ratio = "
    f"{cold_to_hot_storage_flow_ratio_sol:.6f}"
)

print(
    f"mdot hot storage charge     = "
    f"{mdot_hot_storage_chg_sol:.6f} kg/s"
)

print(
    f"mdot cold storage charge    = "
    f"{mdot_cold_storage_chg_sol:.6f} kg/s"
)

print(
    f"mdot hot storage discharge  = "
    f"{mdot_hot_storage_dischg_sol:.6f} kg/s"
)

print(
    f"mdot cold storage discharge = "
    f"{mdot_cold_storage_dischg_sol:.6f} kg/s"
)

print(
    f"duration ratio hot  = "
    f"{duration_ratio_hot:.9f}"
)

print(
    f"duration ratio cold = "
    f"{duration_ratio_cold:.9f}"
)


# ============================================================
# TES
# ============================================================

print(
    "\nTES TEMPERATURES"
)

print(
    f"T_HL = "
    f"{K_to_C(T_HL_initial):.3f} °C (specified)"
)

print(
    f"T_HH = "
    f"{K_to_C(T_HH_sol):.3f} °C"
)

print(
    f"T_CL = "
    f"{K_to_C(T_CL_sol):.3f} °C"
)

print(
    f"T_CH = "
    f"{K_to_C(T_CH_initial):.3f} °C (specified)"
)


print(
    f"T1 discharge = "
    f"{K_to_C(T1_dischg_sol):.3f} °C (solved)"
)


# ============================================================
# CLOSURE
# ============================================================

print(
    "\nCYCLE CLOSURE"
)

print(
    f"Charge delta h = "
    f"{chg_sol['closure_h']:.6e} J/kg"
)

print(
    f"Hot TES delta T = "
    f"{dis_sol['TES']['T_HL'] - T_HL_initial:.6e} K"
)

print(
    f"Cold TES delta T = "
    f"{dis_sol['TES']['T_CH'] - T_CH_initial:.6e} K"
)


# ============================================================
# CAPACITANCE-RATE DIAGNOSTICS
# ============================================================

print(
    "\nCAPACITANCE-RATE RATIOS "
    "(REPORTED, NOT CONSTRAINED)"
)

print(
    f"Hot charge  Cstorage/Cwf = "
    f"{Crate_hot_chg:.6f}"
)

print(
    f"Cold charge Cstorage/Cwf = "
    f"{Crate_cold_chg:.6f}"
)

print(
    f"Hot dis     Cstorage/Cwf = "
    f"{Crate_hot_dis:.6f}"
)

print(
    f"Cold dis    Cstorage/Cwf = "
    f"{Crate_cold_dis:.6f}"
)


# ============================================================
# HX TERMINAL TEMPERATURE DIFFERENCES
# ============================================================

print(
    "\nHX TERMINAL TEMPERATURE DIFFERENCES"
)

print(
    "No terminal delta-T constraint is imposed."
)

print(
    f"Hot HX charge      hot/cold ends = "
    f"{terminal_delta_T_sol[0]:.3f} / "
    f"{terminal_delta_T_sol[1]:.3f} K"
)

print(
    f"Cold HX charge     hot/cold ends = "
    f"{terminal_delta_T_sol[2]:.3f} / "
    f"{terminal_delta_T_sol[3]:.3f} K"
)

print(
    f"Hot HX discharge   hot/cold ends = "
    f"{terminal_delta_T_sol[4]:.3f} / "
    f"{terminal_delta_T_sol[5]:.3f} K"
)

print(
    f"Cold HX discharge  hot/cold ends = "
    f"{terminal_delta_T_sol[6]:.3f} / "
    f"{terminal_delta_T_sol[7]:.3f} K"
)


# ============================================================
# HX UA
# ============================================================

print(
    "\nHX DESIGNS"
)

print(
    f"Hot HX charge      = "
    f"{UA_hot_chg:.3f} W/K "
    f"(effectiveness target = {eff_hx:.6f})"
)

print(
    f"Cold HX charge     = "
    f"{UA_cold_chg:.3f} W/K "
    f"(effectiveness target = {eff_hx:.6f})"
)

print(
    f"Hot HX discharge   = "
    f"{UA_hot_dis:.3f} W/K "
    f"(effectiveness = "
    f"{dis_sol['hot_hx']['effectiveness']:.6f})"
)

print(
    f"Cold HX discharge  = "
    f"{UA_cold_dis:.3f} W/K "
    f"(effectiveness = "
    f"{dis_sol['cold_hx']['effectiveness']:.6f})"
)


# ============================================================
# PERFORMANCE
# ============================================================

print(
    "\nPERFORMANCE"
)

print(
    f"Charge net work    = "
    f"{w_chg/1000:.3f} kJ/kg"
)

print(
    f"Discharge net work = "
    f"{w_dis/1000:.3f} kJ/kg"
)

if periodicity_achieved:

    print(
        f"Specific work ratio = "
        f"{specific_work_ratio:.2%}"
    )

    print(
        f"Round-trip efficiency = "
        f"{roundtrip_efficiency:.2%}"
    )

else:

    print(
        "Round-trip efficiency = unavailable "
        "(cycle is not periodic)"
    )


# ============================================================
# ENERGY
# ============================================================

print(
    "\nENERGY CLOSURE"
)

print(
    f"Charge error = "
    f"{charge_energy_error:.6e} J/kg"
)

print(
    f"Discharge error = "
    f"{discharge_energy_error:.6e} J/kg"
)

print(
    f"Charge heat rejection = "
    f"{-chg_sol['q_reject']/1000:.3f} kJ/kg"
)

print(
    f"Discharge heat rejection = "
    f"{-dis_sol['q_reject']/1000:.3f} kJ/kg"
)


# ============================================================
# SOLVER
# ============================================================

print(
    "\nSOLVER"
)

print(
    "numerical success:",
    solution.success
)

print(
    "message:",
    solution.message
)

print(
    "objective:",
    2.0 * solution.cost
)

print(
    "function evaluations:",
    solution.nfev
)

print(
    "optimality:",
    solution.optimality
)

print(
    "max closure residual:",
    max_closure_residual
)

print(
    "minimum rejection margin:",
    minimum_rejection_margin
)

print(
    "physical success:",
    physical_success
)

print(
    "max physical residual:",
    max_physical_residual
)

print(
    "physical residuals:",
    final_physical_residuals
)
