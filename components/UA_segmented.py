
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from properties import Incompressible 
from eeslib import fluid_properties as fluid_properties
import numpy as np

from scipy.optimize import brentq
#from scipy.optimize import fsolve
from scipy.optimize import least_squares

esol_properties = Incompressible()

#################################
#### Design heating ############
################################

def HX_UA_segments_heat_wf(
    T_storage_in,
    T_wf_in,
    P_wf_in,
    mdot_storage_fixed,
    mdot_wf,
    fluid_storage,
    fluid_wf,
    D,
    delta_T_min_target,
    P_loss_target,
    N_seg=20,
    Af_bounds=(0.02, 0.5),
    strategy_mdot_st_design=1,
):
    fluid_storage_minT = 288.1 # K, min temperature of Dowtherm A

    def UA_segments(A_f, Q_tot):
        """Heating the wf"""

        # A_f is the area of the storage fluid and the sco2 fluid

        def sam_prop_dowthermA_conductivity(T):
            """Returns the properties of Dowtherm A at a given temperature T in K."""
            # Thermal conductivity [W/m-K]
            k = max(0.1419 - 0.00016 * (T - 273.15), 0.001) #T in K
            return k
        
        def T_from_h_storage(h):
            """Returns the temperature of Dowtherm A at a given enthalpy."""
            x = h / 1e6

            T = (
                -27.06574572*x**6 + 151.21088901*x**5-340.13167310*x**4+411.02036213*x**3-388.55934518*x**2+645.42026919*x+288.26451925
            )

            return T
        

        q_i = Q_tot / N_seg

        ###########################
        #Calculate temperatures
        ###########################

        h_storage = np.zeros(N_seg + 1)
        h_wf = np.zeros(N_seg + 1)
        T_storage = np.zeros(N_seg + 1)
        T_wf = np.zeros(N_seg + 1)

        h_storage_in = esol_properties.enthalpy(fluid_storage, T=T_storage_in, T_ref=fluid_storage_minT)
        h_wf_in = fluid_properties.enthalpy(fluid_wf, T=T_wf_in, P=P_wf_in)

        h_storage[0] = h_storage_in
        h_wf[0] = h_wf_in
        T_storage[0] = T_storage_in
        T_wf[0] = T_wf_in

        ####################################################
        ###############  To match capacitance rate #########
        ####################################################

        mdot_storage = mdot_storage_fixed

        if strategy_mdot_st_design == 1: 
            h_wf_out_caprate = h_wf[0] + Q_tot/(mdot_wf)
            T_wf_out_caprate = fluid_properties.temperature(fluid_wf, h=h_wf_out_caprate, P=P_wf_in*(1-P_loss_target))
            cp_wf_avg_caprate = abs((h_wf_out_caprate - h_wf_in)/(T_wf_out_caprate - T_wf_in))
            C_wf_caprate = mdot_wf * cp_wf_avg_caprate
            deltaT_st_caprate = Q_tot / C_wf_caprate

            T_st_out_caprate = T_storage_in - deltaT_st_caprate 
            h_st_in_caprate = esol_properties.enthalpy(fluid_storage, T=T_storage_in, T_ref=fluid_storage_minT)
            h_st_out_caprate = esol_properties.enthalpy(fluid_storage, T=T_st_out_caprate, T_ref=fluid_storage_minT)

            cp_st_avg_caprate = abs((h_st_out_caprate - h_st_in_caprate)/ (T_st_out_caprate - T_storage_in))
            mdot_st_caprate = C_wf_caprate / cp_st_avg_caprate
            mdot_storage = mdot_st_caprate


        # I can't calculate yet the temperature of the working fluid bc pressure varies.
        for i in range(N_seg):
            h_storage[i+1] = h_storage[i] - (q_i / (mdot_storage))
            h_wf[i+1] = h_wf[i] + (q_i / (mdot_wf))

            T_storage[i+1] = T_from_h_storage(h_storage[i+1])
        

        #sco2 fluid
        P_wf = np.full(N_seg + 1, P_wf_in) #Assign inlet Pressure to all the array. 
        vel_wf = np.zeros(N_seg)
        rho_wf = np.zeros(N_seg)
        mu_wf = np.zeros(N_seg)
        Pr_wf = np.zeros(N_seg) 
        k_wf = np.zeros(N_seg)
        cp_wf = np.zeros(N_seg)

        #Storage 
        vel_st = np.zeros(N_seg)
        rho_st = np.zeros(N_seg)
        mu_st = np.zeros(N_seg)
        Pr_st = np.zeros(N_seg)
        k_st = np.zeros(N_seg)
        cp_st = np.zeros(N_seg)

        ################
        #Segments
        ################

        #Working fluid 
        C_f_wf = np.zeros(N_seg)
        Re_wf = np.zeros(N_seg)
        Nu_wf = np.zeros(N_seg)
        h_conv_wf = np.zeros(N_seg)

        #Storage  
        C_f_st = np.zeros(N_seg)
        Re_st = np.zeros(N_seg)
        Nu_st = np.zeros(N_seg)
        h_conv_st = np.zeros(N_seg)

        #Shared
        delta_p = np.zeros(N_seg)
        U_seg = np.zeros(N_seg)
        A_seg = np.zeros(N_seg)
        delta_x = np.zeros(N_seg)


        ##Determine the properties 

        max_iter = 1000
        for iteration in range(max_iter):

            #There is no deltaP in the storage side. So Just wf depends on the pressure drop.
            #Here we use the first guess of pressure drop
            for i in range(N_seg+1):
                T_wf[i] = fluid_properties.temperature(fluid_wf, h = h_wf[i],P=P_wf[i] )


            # Storage: physical/spatial direction
            T_st_seg = 0.5 * (T_storage[:-1] + T_storage[1:])

            # WF: its own physical flow direction
            T_wf_seg = 0.5 * (T_wf[:-1] + T_wf[1:])
            P_wf_seg = 0.5 * (P_wf[:-1] + P_wf[1:])

            #Now we reverse the cold fluid arrays to match the spatial coordinate of the hot fluid
            


            #Now we evaluate the properties and calculate the actual pressure drop
            for i in range(N_seg):
                rho_st[i] = esol_properties.density(fluid = fluid_storage, T=T_st_seg[i])
                mu_st[i] = esol_properties.viscosity(fluid = fluid_storage, T=T_st_seg[i])
                vel_st[i] = mdot_storage / (rho_st[i] * A_f)
                cp_st[i] = esol_properties.specheat(fluid = fluid_storage, T=T_st_seg[i])
                k_st[i] = sam_prop_dowthermA_conductivity(T=T_st_seg[i])
                Pr_st[i] = cp_st[i]*mu_st[i]/k_st[i]
                Re_st[i] = (rho_st[i] * vel_st[i] * D) / mu_st[i]
                C_f_st[i] = (1.58*np.log(Re_st[i]) - 3.28)**(-2) #Check this! Fanning friction factor
                Nu_st[i] = (0.5*C_f_st[i]*Pr_st[i]*(Re_st[i]-1000))/(1 + 12.7*(((C_f_st[i])/2)**(1/2))*(Pr_st[i]**(2/3) - 1)) 
                h_conv_st[i] = (Nu_st[i]*k_st[i])/D

                #Cold wf
                rho_wf[i] = fluid_properties.density(fluid = fluid_wf, T=T_wf_seg[i], P=P_wf_seg[i]) 
                vel_wf[i] = mdot_wf / (rho_wf[i] * A_f)
                mu_wf[i] = fluid_properties.viscosity(fluid = fluid_wf, T=T_wf_seg[i], P=P_wf_seg[i])
                Pr_wf[i] = fluid_properties.prandtl(fluid = fluid_wf, T=T_wf_seg[i], P=P_wf_seg[i])
                cp_wf[i] = fluid_properties.specheat(fluid=fluid_wf,  T=T_wf_seg[i], P=P_wf_seg[i] )
                Re_wf[i] = (rho_wf[i] * vel_wf[i] * D) / mu_wf[i]
                C_f_wf[i] = (1.58*np.log(Re_wf[i]) - 3.28)**(-2) #Check this! Fanning friction factor
                Nu_wf[i] = (0.5*C_f_wf[i]*Pr_wf[i]*(Re_wf[i]-1000))/(1 + 12.7*(((C_f_wf[i])/2)**(1/2))*(Pr_wf[i]**(2/3) - 1))
                k_wf[i] = fluid_properties.conductivity(fluid = fluid_wf, T=T_wf_seg[i], P=P_wf_seg[i])
                h_conv_wf[i] = (Nu_wf[i]*k_wf[i])/D


            ##Put both fluids in same physical coordinate

            hconv_wf_spatial = h_conv_wf[::-1]
            T_wf_seg_spatial = T_wf_seg[::-1]
            P_wf_spatial = P_wf_seg[::-1]
            delta_T_seg = T_st_seg - T_wf_seg_spatial

            #Guard 1
            if np.any(delta_T_seg <= 0):
                return {
                    "physical": False,
                    "delta_T_min": np.min(delta_T_seg)
                }

            for i in range(N_seg):
                U_seg[i] = 1/(1/h_conv_st[i] + 1/hconv_wf_spatial[i])
                A_seg[i] = q_i/(U_seg[i]*delta_T_seg[i])
                delta_x[i] = A_seg[i]*D/(4*A_f)
            
            dx_wf = delta_x[::-1]
            P_wf_new = np.zeros(N_seg + 1)
            P_wf_new[0] = P_wf_in

            for i in range(N_seg):
                delta_p[i] = 4*C_f_wf[i]*(dx_wf[i]/D)*(rho_wf[i]*vel_wf[i]**2)/2
                P_wf_new[i+1] = P_wf_new[i] - delta_p[i]

            # GUARD 2
            if np.any(P_wf_new <= 0):
                return {
                    "physical": False,
                    "delta_T_min": np.min(delta_T_seg)
    }

            #Converge
            if np.allclose(P_wf_new, P_wf, rtol=1e-3, atol=1000.0):
                P_wf = P_wf_new
                break
            
            P_wf = 0.5 * P_wf + 0.5 * P_wf_new

        else:
            raise RuntimeError("HX pressure iteration did not converge")
        
        T_wf_spatial_nodes = T_wf[::-1]
        delta_T_nodes = T_storage - T_wf_spatial_nodes
        delta_T_min = np.min(delta_T_nodes)

        delta_p_total = P_wf_in - P_wf[-1]
        P_losses_calc = delta_p_total / P_wf_in
        L_total = np.sum(delta_x)
        A_total = np.sum(A_seg)
        UA_seg = U_seg * A_seg
        UA_total = np.sum(UA_seg)
        # print(
        #     "Af =", A_f,
        #     "Q =", Q_tot/1e6,
        #     "min dT =", np.min(delta_T_seg),
        #     "min U =", np.min(U_seg),
        #     "min dx =", np.min(delta_x),
        #     "min Re_st =", np.min(Re_st),
        #     "min Re_wf =", np.min(Re_wf),
        # )

        return {
        "P_wf": P_wf,
        "T_storage": T_storage,
        "T_wf": T_wf,
        "cp_wf": cp_wf,
        "cp_st": cp_st,
        "rho_wf": rho_wf,
        "Re_wf": Re_wf,
        "Re_st": Re_st,
        "Nu_wf": Nu_wf,
        "h_conv_wf": h_conv_wf,
        "h_conv_st": h_conv_st,
        "k_wf": k_wf,
        "Pr_wf": Pr_wf,
        "T_wf_seg": T_wf_seg,
        "P_wf_seg": P_wf_seg,

        "delta_T_seg": delta_T_seg,
        "delta_T_min": delta_T_min,

        "U_seg": U_seg,
        "A_seg": A_seg,
        "delta_x": delta_x,

        "delta_p": delta_p,
        "delta_p_total": delta_p_total,
        "P_losses_calc": P_losses_calc,

        "L_total": L_total,
        "A_total": A_total,

        "UA_seg": UA_seg,
        "UA_total": UA_total,
        "physical": True,

        "iterations": iteration + 1,
        "mdot_storage": mdot_storage
        
    }

    ###########################################################################
    #Maximum possible Q. This is used to set the higher bound for the iterator#
    ###########################################################################

    if strategy_mdot_st_design == 1:
        Q_wf_max = mdot_wf*(fluid_properties.enthalpy(fluid = fluid_wf, T=T_storage_in, P=P_wf_in) - fluid_properties.enthalpy(fluid = fluid_wf, T=T_wf_in, P=P_wf_in))
        Q_max = Q_wf_max
    else:
        Q_storage_max = mdot_storage_fixed*(esol_properties.enthalpy(fluid = fluid_storage, T=T_storage_in, T_ref=fluid_storage_minT) - esol_properties.enthalpy(fluid = fluid_storage, T=T_wf_in, T_ref=fluid_storage_minT))
        Q_wf_max = mdot_wf*(fluid_properties.enthalpy(fluid = fluid_wf, T=T_storage_in, P=P_wf_in) - fluid_properties.enthalpy(fluid = fluid_wf, T=T_wf_in, P=P_wf_in))
        Q_max = min(Q_storage_max, Q_wf_max)


    

    def find_Q_for_Af(A_f):

        def residual_Q(Q_tot):
            # print("    Q trial:", Q_tot / 1e6, "MW")

            hx_result = UA_segments(
                A_f=A_f,
                Q_tot=Q_tot
            )

            if not hx_result["physical"]:
                raise ValueError("Non-physical HX evaluation")

            return hx_result["delta_T_min"] - delta_T_min_target

        return brentq(
            residual_Q,
            0.4*Q_max,  #Minimum Q
            0.999999999999*Q_max
            
        )

    # ============================================================
    # Find Af from pressure-loss target
    # ============================================================

    def residual_Af(A_f):

        # print("\nAf trial:", A_f)

        Q_solution = find_Q_for_Af(A_f)

        hx = UA_segments(
            A_f=A_f,
            Q_tot=Q_solution
        )

        # print(
        # "P loss:",
        # hx["P_losses_calc"],
        # "target:",
        # P_loss_target
        # )

        return hx["P_losses_calc"] - P_loss_target


    # Find flow area that satisfies pressure-drop constraint
    A_f_solution = brentq(
        residual_Af,
        Af_bounds[0],
        Af_bounds[1]
    )

    
    # ============================================================
    # Final converged solution
    # ============================================================

    # Find corresponding heat duty
    Q_solution = find_Q_for_Af(A_f_solution)

    # Final HX evaluation
    result = UA_segments(
        A_f=A_f_solution,
        Q_tot=Q_solution
    )

    result["Q_solution"] = Q_solution
    result["Q_max"] = Q_max
    result["A_f"] = A_f_solution

    result["delta_T_min_target"] = delta_T_min_target

    result["P_loss_target"] = P_loss_target
    #result["mdot_storage"] = mdot_storage
    result["mdot_wf"] = mdot_wf
    result["N_seg"] = N_seg


    return result


################################
#### Off-design heating ########
################################

def HX_UA_segments_heat_wf_offdesign(
    T_storage_in,
    T_wf_in,
    P_wf_in,
    mdot_storage_fixed,
    mdot_wf,
    fluid_storage,
    fluid_wf,
    D,
    N_seg,
    A_f_design, 
    L_design,
    strategy_mdot_st_offdesign=0,
    P_losses_estimation = 0.01,
    mdot_st_cold_chg_info=None,
    mdot_st_hot_chg_info=None,
    mdot_st_cold_dis_info=None,
    mdot_st_hot_dis_info=None,
):
    fluid_storage_minT = 288.1 # K, min temperature of Dowtherm A

    # A_seg_discharge = A_seg_charge[::-1]
    # delta_x_discharge = delta_x_charge[::-1


    def sam_prop_dowthermA_conductivity(T):
        """Returns the properties of Dowtherm A at a given temperature T in K."""
        # Thermal conductivity [W/m-K]
        k = max(0.1419 - 0.00016 * (T - 273.15), 0.001) #T in K
        return k
    
    def T_from_h_storage(h):
        """Returns the temperature of Dowtherm A at a given enthalpy."""
        x = h / 1e6

        T = (
            -27.06574572*x**6 + 151.21088901*x**5-340.13167310*x**4+411.02036213*x**3-388.55934518*x**2+645.42026919*x+288.26451925
        )

        return T
    
    # ============================================================
    # Evaluate HX for a given total Q
    # ============================================================

    def evaluate_Q(Q_tot):

        q_i = Q_tot / N_seg

        # ========================================================
        # Enthalpy / temperature profiles
        # ========================================================

        h_storage = np.zeros(N_seg + 1)
        h_wf = np.zeros(N_seg + 1)

        T_storage = np.zeros(N_seg + 1)
        T_wf = np.zeros(N_seg + 1)


        h_storage[0] = esol_properties.enthalpy(fluid_storage,T=T_storage_in,T_ref=fluid_storage_minT)
        h_wf[0] = fluid_properties.enthalpy(fluid_wf,T=T_wf_in,P=P_wf_in)

        T_storage[0] = T_storage_in
        T_wf[0] = T_wf_in

        if strategy_mdot_st_offdesign == 0:
            mdot_storage = mdot_storage_fixed


        if strategy_mdot_st_offdesign == 1: 
            h_wf_out_caprate = h_wf[0] + Q_tot/(mdot_wf)
            # Approximate outlet pressure used only for capacitance-rate matching.
            # Actual off-design pressure loss is calculated later by the HX model.
            T_wf_out_caprate = fluid_properties.temperature(fluid_wf, h=h_wf_out_caprate, P=P_wf_in*(1-P_losses_estimation))
            cp_wf_avg_caprate = abs((h_wf_out_caprate - h_wf[0])/(T_wf_out_caprate - T_wf_in))
            C_wf_caprate = mdot_wf * cp_wf_avg_caprate
            deltaT_st_caprate = Q_tot / C_wf_caprate

            T_st_out_caprate = T_storage_in - deltaT_st_caprate 
            h_st_in_caprate = esol_properties.enthalpy(fluid_storage, T=T_storage_in, T_ref=fluid_storage_minT)
            h_st_out_caprate = esol_properties.enthalpy(fluid_storage, T=T_st_out_caprate, T_ref=fluid_storage_minT)

            cp_st_avg_caprate = abs((h_st_out_caprate - h_st_in_caprate)/ (T_st_out_caprate - T_storage_in))
            mdot_st_caprate = C_wf_caprate / cp_st_avg_caprate
            mdot_storage = mdot_st_caprate


        # Mass-balance strategy:
        # solve for mdot_st_cold_dis from the other three storage mass flows
        if strategy_mdot_st_offdesign == 2:
            mdot_storage = mdot_st_cold_chg_info*mdot_st_hot_dis_info/mdot_st_hot_chg_info


        for i in range(N_seg):

            # Hot storage loses heat
            h_storage[i+1] = (h_storage[i]- q_i/mdot_storage)

            # Cold WF gains heat
            h_wf[i+1] = (h_wf[i]+ q_i/mdot_wf)

            T_storage[i+1] = T_from_h_storage(h_storage[i+1])


        # ========================================================
        # Arrays
        # ========================================================

        P_wf = np.full(N_seg + 1,P_wf_in)

        # WF
        vel_wf = np.zeros(N_seg)
        rho_wf = np.zeros(N_seg)
        mu_wf = np.zeros(N_seg)
        Pr_wf = np.zeros(N_seg)
        k_wf = np.zeros(N_seg)
        cp_wf = np.zeros(N_seg)

        C_f_wf = np.zeros(N_seg)
        Re_wf = np.zeros(N_seg)
        Nu_wf = np.zeros(N_seg)
        h_conv_wf = np.zeros(N_seg)

        # Storage
        vel_st = np.zeros(N_seg)
        rho_st = np.zeros(N_seg)
        mu_st = np.zeros(N_seg)
        Pr_st = np.zeros(N_seg)
        k_st = np.zeros(N_seg)
        cp_st = np.zeros(N_seg)

        C_f_st = np.zeros(N_seg)
        Re_st = np.zeros(N_seg)
        Nu_st = np.zeros(N_seg)
        h_conv_st = np.zeros(N_seg)

        # Shared
        U_seg = np.zeros(N_seg)

        C_hot = np.zeros(N_seg)
        C_cold = np.zeros(N_seg)
        C_min = np.zeros(N_seg)
        C_max = np.zeros(N_seg)
        C_r = np.zeros(N_seg)

        epsilon_seg = np.zeros(N_seg)
        NTU_seg = np.zeros(N_seg)
        UA_seg = np.zeros(N_seg)

        A_seg = np.zeros(N_seg)
        delta_x = np.zeros(N_seg)

        delta_p = np.zeros(N_seg)


        # ========================================================
        # Pressure iteration
        # ========================================================

        max_iter = 1000

        for iteration in range(max_iter):
            # ----------------------------------------------------
            # WF temperature from h and P
            # ----------------------------------------------------

            for i in range(N_seg + 1):

                T_wf[i] = fluid_properties.temperature(fluid_wf,h=h_wf[i],P=P_wf[i])

            # ----------------------------------------------------
            # Segment-average states
            # ----------------------------------------------------

            T_st_seg = 0.5 * (T_storage[:-1]+ T_storage[1:])
            T_wf_seg = 0.5 * (T_wf[:-1]+ T_wf[1:])
            P_wf_seg = 0.5 * (P_wf[:-1]+ P_wf[1:])


            # ----------------------------------------------------
            # Local heat transfer coefficients
            # ----------------------------------------------------

            for i in range(N_seg):

                # ===========================
                # STORAGE
                # ===========================

                rho_st[i] = esol_properties.density(fluid=fluid_storage,T=T_st_seg[i])
                mu_st[i] = esol_properties.viscosity(fluid=fluid_storage,T=T_st_seg[i])
                vel_st[i] = (mdot_storage/(rho_st[i] * A_f_design))
                cp_st[i] = esol_properties.specheat(fluid=fluid_storage,T=T_st_seg[i])
                k_st[i] = sam_prop_dowthermA_conductivity(T=T_st_seg[i])
                Pr_st[i] = (cp_st[i]* mu_st[i]/ k_st[i])
                Re_st[i] = (rho_st[i]* vel_st[i]* D/ mu_st[i])
                C_f_st[i] = (1.58*np.log(Re_st[i])- 3.28)**(-2)
                Nu_st[i] = (0.5* C_f_st[i]* Pr_st[i]* (Re_st[i] - 1000)/(1+ 12.7* np.sqrt(C_f_st[i]/2)* (Pr_st[i]**(2/3) - 1)))
                h_conv_st[i] = (Nu_st[i]* k_st[i]/ D)

                # ===========================
                # WF
                # ===========================

                rho_wf[i] = fluid_properties.density(fluid=fluid_wf,T=T_wf_seg[i],P=P_wf_seg[i])
                vel_wf[i] = (mdot_wf/(rho_wf[i] * A_f_design))
                mu_wf[i] = fluid_properties.viscosity(fluid=fluid_wf,T=T_wf_seg[i],P=P_wf_seg[i])
                Pr_wf[i] = fluid_properties.prandtl(fluid=fluid_wf,T=T_wf_seg[i],P=P_wf_seg[i])
                cp_wf[i] = fluid_properties.specheat(fluid=fluid_wf,T=T_wf_seg[i],P=P_wf_seg[i])
                k_wf[i] = fluid_properties.conductivity(fluid=fluid_wf,T=T_wf_seg[i],P=P_wf_seg[i])
                Re_wf[i] = (rho_wf[i]* vel_wf[i]* D/ mu_wf[i])
                C_f_wf[i] = (1.58*np.log(Re_wf[i])- 3.28)**(-2)
                Nu_wf[i] = (0.5* C_f_wf[i]* Pr_wf[i]* (Re_wf[i] - 1000)/(1+ 12.7* np.sqrt(C_f_wf[i]/2)* (Pr_wf[i]**(2/3) - 1)))
                h_conv_wf[i] = (Nu_wf[i]* k_wf[i]/ D)

            # ====================================================
            # Put WF into storage spatial coordinate
            # ====================================================

            T_wf_spatial = T_wf[::-1]
            h_wf_spatial = h_wf[::-1]

            hconv_wf_spatial = h_conv_wf[::-1]


            # ====================================================
            # epsilon-NTU for every sub-HX
            # ====================================================

            physical = True

            for i in range(N_seg):

                # ------------------------------------------------
                # Average heat-capacity rates
                #
                # Same approach as the book:
                #
                # C = mdot * Delta h / Delta T
                # ------------------------------------------------

                dT_hot = (T_storage[i]- T_storage[i+1])
                dT_cold = (T_wf_spatial[i]- T_wf_spatial[i+1])
                C_hot[i] = (mdot_storage* (h_storage[i]- h_storage[i+1])/ dT_hot)
                C_cold[i] = (mdot_wf* (h_wf_spatial[i]- h_wf_spatial[i+1])/ dT_cold)
                C_min[i] = min(C_hot[i],C_cold[i])
                C_max[i] = max(C_hot[i],C_cold[i])
                C_r[i] = (C_min[i]/ C_max[i])


                # ------------------------------------------------
                # Counterflow inlet temperature difference
                #
                # Storage enters segment from left:
                #       T_storage[i]
                #
                # WF enters segment from right:
                #       T_wf_spatial[i+1]
                # ------------------------------------------------

                delta_T_in = (T_storage[i]- T_wf_spatial[i+1])

                if delta_T_in <= 0:
                    print(f"INVALID delta_T: Q={Q_tot/1e6:.4f} MW, segment={i}, delta_T={delta_T_in:.6f} K", flush=True)
                    physical = False
                    break

                q_max_i = (C_min[i]* delta_T_in)
                epsilon_seg[i] = (q_i/ q_max_i)

                # ------------------------------------------------
                # epsilon must be between 0 and 1
                # ------------------------------------------------

                if (epsilon_seg[i] <= 0 or epsilon_seg[i] >= 1):
                    print(f"INVALID epsilon: Q={Q_tot/1e6:.4f} MW, segment={i}, epsilon={epsilon_seg[i]:.6f}", flush=True)
                    physical = False
                    break


                # ------------------------------------------------
                # Invert counterflow effectiveness-NTU relation
                # ------------------------------------------------

                if abs(1.0 - C_r[i]) < 1e-8:
                    NTU_seg[i] = (epsilon_seg[i]/(1.0 - epsilon_seg[i]))

                else:
                    NTU_seg[i] = (np.log((1.0- epsilon_seg[i]*C_r[i])/(1.0- epsilon_seg[i]))/(1.0 - C_r[i]))

                # ------------------------------------------------
                # Conductance required by sub-HX
                # ------------------------------------------------

                UA_seg[i] = (NTU_seg[i]* C_min[i])

                # ------------------------------------------------
                # Local U
                # ------------------------------------------------

                U_seg[i] = (1/(1/h_conv_st[i]+1/hconv_wf_spatial[i]))

                # ------------------------------------------------
                # Physical area and length required
                # ------------------------------------------------

                A_seg[i] = (UA_seg[i]/ U_seg[i])
                delta_x[i] = (A_seg[i]* D/(4*A_f_design))

            # ====================================================
            # Non-physical trial Q
            # ====================================================

            if not physical:

                return {
                "physical": False,
                "L_total": np.nan,
                "Q_total": Q_tot
            }


            # ====================================================
            # Pressure drop
            # ====================================================

            # delta_x is in STORAGE spatial direction.
            # WF travels in opposite direction.

            dx_wf = delta_x[::-1]
            P_wf_new = np.zeros(N_seg + 1)
            P_wf_new[0] = P_wf_in

            for i in range(N_seg):

                delta_p[i] = (4* C_f_wf[i]* (dx_wf[i]/D)* (rho_wf[i]* vel_wf[i]**2)/ 2)
                P_wf_new[i+1] = (P_wf_new[i]- delta_p[i])

            # ====================================================
            # Pressure convergence
            # ====================================================

            if np.allclose(P_wf_new,P_wf,rtol=1e-3, atol=1000.0):
                P_wf = P_wf_new
                break


            P_wf = (0.5*P_wf+ 0.5*P_wf_new)

        else:
            raise RuntimeError("HX pressure iteration did not converge")


        # ========================================================
        # Final results
        # ========================================================

        A_total = np.sum(A_seg)
        L_total = np.sum(delta_x)
        UA_total = np.sum(UA_seg)

        delta_p_total = (P_wf_in- P_wf[-1])
        P_losses_calc = (delta_p_total/ P_wf_in)
        T_wf_spatial = T_wf[::-1]
        delta_T_nodes = (T_storage- T_wf_spatial)
        delta_T_min = np.min(delta_T_nodes)

        return {

            "physical": True,
            "Q_total": Q_tot,
            "P_wf": P_wf,
            "T_storage": T_storage,
            "T_wf": T_wf,
            "T_st_seg": T_st_seg,
            "T_wf_seg": T_wf_seg,
            "cp_st": cp_st,
            "cp_wf": cp_wf,
            "Re_st": Re_st,
            "Re_wf": Re_wf,
            "C_hot": C_hot,
            "C_cold": C_cold,
            "C_min": C_min,
            "C_r": C_r,

            "epsilon_seg": epsilon_seg,
            "NTU_seg": NTU_seg,

            "h_conv_st": h_conv_st,
            "h_conv_wf": h_conv_wf,

            "U_seg": U_seg,
            "UA_seg": UA_seg,
            "UA_total": UA_total,

            "A_seg": A_seg,
            "A_total": A_total,

            "delta_x": delta_x,
            "L_total": L_total,

            "delta_T_nodes": delta_T_nodes,
            "delta_T_min": delta_T_min,

            "delta_p": delta_p,
            "delta_p_total": delta_p_total,
            "P_losses_calc": P_losses_calc,
            "mdot_storage": mdot_storage,

            "iterations": iteration + 1,
        }


    # ============================================================
    # Maximum possible Q
    # ============================================================

    if strategy_mdot_st_offdesign == 0:
        mdot_storage = mdot_storage_fixed

        Q_storage_max = (mdot_storage*(
                esol_properties.enthalpy(fluid=fluid_storage,T=T_storage_in,T_ref=fluid_storage_minT) - 
                esol_properties.enthalpy(fluid=fluid_storage,T=T_wf_in,T_ref=fluid_storage_minT)
            )
        )
        Q_wf_max = (mdot_wf*(
                fluid_properties.enthalpy(fluid=fluid_wf,T=T_storage_in,P=P_wf_in) -
                fluid_properties.enthalpy(fluid=fluid_wf,T=T_wf_in,P=P_wf_in)
            )
        )

        Q_max = min(Q_storage_max,Q_wf_max)

    if strategy_mdot_st_offdesign == 1:
        Q_wf_max = mdot_wf*(fluid_properties.enthalpy(fluid = fluid_wf, T=T_storage_in, P=P_wf_in) - fluid_properties.enthalpy(fluid = fluid_wf, T=T_wf_in, P=P_wf_in))
        Q_max = Q_wf_max

    if strategy_mdot_st_offdesign == 2:
        mdot_storage = mdot_st_cold_chg_info*mdot_st_hot_dis_info/mdot_st_hot_chg_info

        Q_storage_max = (mdot_storage*(
                esol_properties.enthalpy(fluid=fluid_storage,T=T_storage_in,T_ref=fluid_storage_minT) - 
                esol_properties.enthalpy(fluid=fluid_storage,T=T_wf_in,T_ref=fluid_storage_minT)
            )
        )
        Q_wf_max = (mdot_wf*(
                fluid_properties.enthalpy(fluid=fluid_wf,T=T_storage_in,P=P_wf_in) -
                fluid_properties.enthalpy(fluid=fluid_wf,T=T_wf_in,P=P_wf_in)
            )
        )

        Q_max = min(Q_storage_max,Q_wf_max)


    # ============================================================
    # Solve Q such that required length = physical HX length
    # ============================================================

    def residual_Q(Q_tot):

        hx = evaluate_Q(Q_tot)

        if not hx["physical"]:
            raise ValueError("Non-physical HX evaluation")
    
        return (hx["L_total"]- L_design)
    
    # Q_low = 0.1 * Q_max
    # Q_high =  1.01*Q_max

    # hx_low = evaluate_Q(Q_low)
    # hx_high = evaluate_Q(Q_high)

    # print("\nOFFDESIGN HEAT BRACKET")
    # print("Qmax =", Q_max/1e6, "MW")

    # print(
    #     "LOW:",
    #     Q_low/1e6, "MW",
    #     "physical =", hx_low["physical"],
    #     "L =", hx_low.get("L_total"),
    #     "residual =", residual_Q(Q_low)
    # )

    # print(
    #     "HIGH:",
    #     Q_high/1e6, "MW",
    #     "physical =", hx_high["physical"],
    #     "L =", hx_high.get("L_total"),
    #     "residual =", residual_Q(Q_high)
    # )


    #Q_solution = brentq(residual_Q, 0.1*Q_max, 1.000000000001*Q_max)
    sol = least_squares(
    lambda x: residual_Q(x[0]),
    x0=[0.8 * Q_max],
    bounds=([0.1 * Q_max], [ Q_max]),
    x_scale=[Q_max]
    )

    # print("\n========== Q SWEEP ==========")
    # print(f"L_design = {L_design:.6f} m")
    # print(f"Q_max = {Q_max/1e6:.6f} MW")

    # for fraction in [
    #     0.90, 0.95, 0.98, 0.99,
    #     0.995, 0.998, 0.999,
    #     0.9995, 0.9999, 0.99999, 0.999999
    # ]:
    #     Q_test = fraction * Q_max
    #     hx = evaluate_Q(Q_test)

    #     if hx["physical"]:
    #         print(
    #             f"Q/Qmax={fraction:.6f} | "
    #             f"L={hx['L_total']:.6f} m | "
    #             f"L_error={hx['L_total']-L_design:+.6f} m | "
    #             f"pinch={hx['delta_T_min']:.6f} K | "
    #             f"epsilon_max={np.max(hx['epsilon_seg']):.8f}"
    #         )
    #     else:
    #         print(f"Q/Qmax={fraction:.6f} | NON-PHYSICAL")

    print("LS success =", sol.success)
    print("LS message =", sol.message)
    print("LS Q =", sol.x[0]/1e6, "MW")
    print("LS residual =", sol.fun[0], "m")
    print("LS optimality =", sol.optimality)
    print("Q/Q_max =", sol.x[0] / Q_max)
    print("Active mask =", sol.active_mask)
    # if abs(sol.fun[0]) > 0.01:
    #     raise ValueError(
    #         f"HX off-design did not converge: "
    #         f"L_error={sol.fun[0]:.4f} m, "
    #         f"Q/Q_max={sol.x[0]/Q_max:.6f}"
    #     )

    Q_solution = sol.x[0]

    # ============================================================
    # Final solution
    # ============================================================

    result = evaluate_Q(Q_solution)

    print("L_design =", L_design)
    print("L_calculated =", result["L_total"])
    print("L_error =", result["L_total"] - L_design)

    result["Q_solution"] = Q_solution
    result["Q_max"] = Q_max

    result["A_f"] = A_f_design

    result["L_design"] = L_design

    result["mdot_wf"] = mdot_wf

    result["N_seg"] = N_seg

    return result


############################################
########Design cooling #####################
############################################

def HX_UA_segments_cool_wf(
    T_storage_in,
    T_wf_in,
    P_wf_in,
    mdot_storage_fixed,
    mdot_wf,
    fluid_storage,
    fluid_wf,
    D,
    delta_T_min_target,
    P_loss_target,
    N_seg=20,
    Af_bounds=(0.1, 0.5),
    strategy_mdot_st_design=1,
):
    fluid_storage_minT = 288.1 # K, min temperature of Dowtherm A

    def UA_segments(A_f, Q_tot):
        """Cooling the wf"""

        # A_f is the area of the storage fluid and the sco2 fluid

        def sam_prop_dowthermA_conductivity(T):
            """Returns the properties of Dowtherm A at a given temperature T in K."""
            # Thermal conductivity [W/m-K]
            k = max(0.1419 - 0.00016 * (T - 273.15), 0.001) #T in K
            return k
        
        def T_from_h_storage(h):
            """Returns the temperature of Dowtherm A at a given enthalpy."""
            x = h / 1e6

            T = (
                -27.06574572*x**6 + 151.21088901*x**5-340.13167310*x**4+411.02036213*x**3-388.55934518*x**2+645.42026919*x+288.26451925
            )

            return T
        

        q_i = Q_tot / N_seg

        ###########################
        #Calculate temperatures
        ###########################

        h_storage = np.zeros(N_seg + 1)
        h_wf = np.zeros(N_seg + 1)
        T_storage = np.zeros(N_seg + 1)
        T_wf = np.zeros(N_seg + 1)

        h_storage_in = esol_properties.enthalpy(fluid_storage, T=T_storage_in, T_ref=fluid_storage_minT)
        h_wf_in = fluid_properties.enthalpy(fluid_wf, T=T_wf_in, P=P_wf_in)

        h_storage[0] = h_storage_in
        h_wf[0] = h_wf_in
        T_storage[0] = T_storage_in
        T_wf[0] = T_wf_in


        ####################################################
        ###############  To match capacitance rate #########
        ####################################################

        mdot_storage = mdot_storage_fixed

        if strategy_mdot_st_design == 1: 
            h_wf_out_caprate = h_wf[0] - Q_tot/(mdot_wf)
            T_wf_out_caprate = fluid_properties.temperature(fluid_wf, h=h_wf_out_caprate, P=P_wf_in*(1-P_loss_target))
            cp_wf_avg_caprate = abs((h_wf_in - h_wf_out_caprate)/(T_wf_in - T_wf_out_caprate))
            C_wf_caprate = mdot_wf * cp_wf_avg_caprate
            deltaT_st_caprate = Q_tot / C_wf_caprate

            T_st_out_caprate = T_storage_in + deltaT_st_caprate 
            h_st_in_caprate = esol_properties.enthalpy(fluid_storage, T=T_storage_in, T_ref=fluid_storage_minT)
            h_st_out_caprate = esol_properties.enthalpy(fluid_storage, T=T_st_out_caprate, T_ref=fluid_storage_minT)

            cp_st_avg_caprate = abs((h_st_in_caprate - h_st_out_caprate)/ (T_storage_in - T_st_out_caprate))
            mdot_st_caprate = C_wf_caprate / cp_st_avg_caprate
            mdot_storage = mdot_st_caprate



        # I can't calculate yet the temperature of the working fluid bc pressure varies.
        for i in range(N_seg):
            h_storage[i+1] = h_storage[i] + (q_i / (mdot_storage))
            h_wf[i+1] = h_wf[i] - (q_i / (mdot_wf))

            T_storage[i+1] = T_from_h_storage(h_storage[i+1])
        

        #sco2 fluid
        P_wf = np.full(N_seg + 1, P_wf_in) #Assign inlet Pressure to all the array. 
        vel_wf = np.zeros(N_seg)
        rho_wf = np.zeros(N_seg)
        mu_wf = np.zeros(N_seg)
        Pr_wf = np.zeros(N_seg) 
        k_wf = np.zeros(N_seg)
        cp_wf = np.zeros(N_seg)

        #Storage 
        vel_st = np.zeros(N_seg)
        rho_st = np.zeros(N_seg)
        mu_st = np.zeros(N_seg)
        Pr_st = np.zeros(N_seg)
        k_st = np.zeros(N_seg)
        cp_st = np.zeros(N_seg)

        ################
        #Segments
        ################

        #Working fluid 
        C_f_wf = np.zeros(N_seg)
        Re_wf = np.zeros(N_seg)
        Nu_wf = np.zeros(N_seg)
        h_conv_wf = np.zeros(N_seg)

        #Storage  
        C_f_st = np.zeros(N_seg)
        Re_st = np.zeros(N_seg)
        Nu_st = np.zeros(N_seg)
        h_conv_st = np.zeros(N_seg)

        #Shared
        delta_p = np.zeros(N_seg)
        U_seg = np.zeros(N_seg)
        A_seg = np.zeros(N_seg)
        delta_x = np.zeros(N_seg)


        ##Determine the properties 

        max_iter = 1000
        for iteration in range(max_iter):

            #There is no deltaP in the storage side. So Just wf depends on the pressure drop.
            #Here we use the first guess of pressure drop
            for i in range(N_seg+1):
                T_wf[i] = fluid_properties.temperature(fluid_wf, h = h_wf[i],P=P_wf[i] )


            # Storage: physical/spatial direction
            T_st_seg = 0.5 * (T_storage[:-1] + T_storage[1:])

            # WF: its own physical flow direction
            T_wf_seg = 0.5 * (T_wf[:-1] + T_wf[1:])
            P_wf_seg = 0.5 * (P_wf[:-1] + P_wf[1:])

            #Now we reverse the cold fluid arrays to match the spatial coordinate of the hot fluid
            


            #Now we evaluate the properties and calculate the actual pressure drop
            for i in range(N_seg):
                rho_st[i] = esol_properties.density(fluid = fluid_storage, T=T_st_seg[i])
                mu_st[i] = esol_properties.viscosity(fluid = fluid_storage, T=T_st_seg[i])
                vel_st[i] = mdot_storage / (rho_st[i] * A_f)
                cp_st[i] = esol_properties.specheat(fluid = fluid_storage, T=T_st_seg[i])
                k_st[i] = sam_prop_dowthermA_conductivity(T=T_st_seg[i])
                Pr_st[i] = cp_st[i]*mu_st[i]/k_st[i]
                Re_st[i] = (rho_st[i] * vel_st[i] * D) / mu_st[i]
                C_f_st[i] = (1.58*np.log(Re_st[i]) - 3.28)**(-2) #Check this! Fanning friction factor
                Nu_st[i] = (0.5*C_f_st[i]*Pr_st[i]*(Re_st[i]-1000))/(1 + 12.7*(((C_f_st[i])/2)**(1/2))*(Pr_st[i]**(2/3) - 1)) 
                h_conv_st[i] = (Nu_st[i]*k_st[i])/D

                #Cold wf
                rho_wf[i] = fluid_properties.density(fluid = fluid_wf, T=T_wf_seg[i], P=P_wf_seg[i]) 
                vel_wf[i] = mdot_wf / (rho_wf[i] * A_f)
                mu_wf[i] = fluid_properties.viscosity(fluid = fluid_wf, T=T_wf_seg[i], P=P_wf_seg[i])
                Pr_wf[i] = fluid_properties.prandtl(fluid = fluid_wf, T=T_wf_seg[i], P=P_wf_seg[i])
                cp_wf[i] = fluid_properties.specheat(fluid=fluid_wf,  T=T_wf_seg[i], P=P_wf_seg[i] )
                Re_wf[i] = (rho_wf[i] * vel_wf[i] * D) / mu_wf[i]
                C_f_wf[i] = (1.58*np.log(Re_wf[i]) - 3.28)**(-2) #Check this! Fanning friction factor
                Nu_wf[i] = (0.5*C_f_wf[i]*Pr_wf[i]*(Re_wf[i]-1000))/(1 + 12.7*(((C_f_wf[i])/2)**(1/2))*(Pr_wf[i]**(2/3) - 1))
                k_wf[i] = fluid_properties.conductivity(fluid = fluid_wf, T=T_wf_seg[i], P=P_wf_seg[i])
                h_conv_wf[i] = (Nu_wf[i]*k_wf[i])/D


            ##Put both fluids in same physical coordinate

            hconv_wf_spatial = h_conv_wf[::-1]
            T_wf_seg_spatial = T_wf_seg[::-1]
            P_wf_spatial = P_wf_seg[::-1]
            delta_T_seg = T_wf_seg_spatial - T_st_seg

            for i in range(N_seg):
                U_seg[i] = 1/(1/h_conv_st[i] + 1/hconv_wf_spatial[i])
                A_seg[i] = q_i/(U_seg[i]*delta_T_seg[i])
                delta_x[i] = A_seg[i]*D/(4*A_f)
            
            dx_wf = delta_x[::-1]
            P_wf_new = np.zeros(N_seg + 1)
            P_wf_new[0] = P_wf_in

            for i in range(N_seg):
                delta_p[i] = 4*C_f_wf[i]*(dx_wf[i]/D)*(rho_wf[i]*vel_wf[i]**2)/2
                P_wf_new[i+1] = P_wf_new[i] - delta_p[i]

            #Converge
            if np.allclose(P_wf_new, P_wf, rtol=1e-3, atol=1000.0):
                P_wf = P_wf_new
                break
            
            P_wf = 0.5 * P_wf + 0.5 * P_wf_new

        else:
            raise RuntimeError("HX pressure iteration did not converge")
        
        T_wf_spatial_nodes = T_wf[::-1]
        delta_T_nodes = T_wf_spatial_nodes - T_storage
        delta_T_min = np.min(delta_T_nodes)

        delta_p_total = P_wf_in - P_wf[-1]
        P_losses_calc = delta_p_total / P_wf_in
        L_total = np.sum(delta_x)
        A_total = np.sum(A_seg)
        UA_seg = U_seg * A_seg
        UA_total = np.sum(UA_seg)

        return {
        "P_wf": P_wf,
        "T_storage": T_storage,
        "T_wf": T_wf,
        "cp_wf": cp_wf,
        "cp_st": cp_st,
        "Re_st": Re_st,
        "rho_wf": rho_wf,
        "Re_wf": Re_wf,
        "Nu_wf": Nu_wf,
        "C_f_wf": C_f_wf,
        "h_conv_wf": h_conv_wf,
        "h_conv_st": h_conv_st,
        "k_wf": k_wf,
        "Pr_wf": Pr_wf,
        "T_wf_seg": T_wf_seg,
        "P_wf_seg": P_wf_seg,

        "delta_T_seg": delta_T_seg,
        "delta_T_min": delta_T_min,

        "U_seg": U_seg,
        "A_seg": A_seg,
        "delta_x": delta_x,

        "delta_p": delta_p,
        "delta_p_total": delta_p_total,
        "P_losses_calc": P_losses_calc,

        "L_total": L_total,
        "A_total": A_total,

        "UA_seg": UA_seg,
        "UA_total": UA_total,

        "iterations": iteration + 1,
        "mdot_storage": mdot_storage
    }

    ###########################################################################
    #Maximum possible Q. This is used to set the higher bound for the iterator#
    ###########################################################################

    if strategy_mdot_st_design == 1:
        Q_wf_max = mdot_wf*(
        fluid_properties.enthalpy(fluid = fluid_wf, T=T_wf_in, P=P_wf_in) - 
        fluid_properties.enthalpy(fluid = fluid_wf, T=T_storage_in, P=P_wf_in))
        Q_max = Q_wf_max
    else: 
        Q_storage_max = mdot_storage_fixed*(
            esol_properties.enthalpy(fluid = fluid_storage, T=T_wf_in, T_ref=fluid_storage_minT) - 
            esol_properties.enthalpy(fluid = fluid_storage, T=T_storage_in, T_ref=fluid_storage_minT))
        Q_wf_max = mdot_wf*(
            fluid_properties.enthalpy(fluid = fluid_wf, T=T_wf_in, P=P_wf_in) - 
            fluid_properties.enthalpy(fluid = fluid_wf, T=T_storage_in, P=P_wf_in))
        Q_max = min(Q_storage_max, Q_wf_max)

    def find_Q_for_Af(A_f):

        def residual_Q(Q_tot):

            #fsolve works with arrays, so we better extract the item of the array to avoid warnings and future failure. 
            Q_tot = Q_tot.item()

            hx_result = UA_segments(
                A_f=A_f,
                Q_tot=Q_tot
            )

            return hx_result["delta_T_min"] - delta_T_min_target

        Q_guess = 0.8 * Q_max

        Q_solution = least_squares(
            residual_Q,
            x0=[Q_guess],
            bounds=(
                [0.2 * Q_max],
                #[1.001 * Q_max]
                [0.999999999999999999999999 * Q_max]
            )
        ).x[0]


        # Q_solution = fsolve(
        #     residual_Q,
        #     Q_guess
        # )[0]

        return Q_solution

    # ============================================================
    # Find Af from pressure-loss target
    # ============================================================

    def residual_Af(A_f):

        Q_solution = find_Q_for_Af(A_f)

        hx = UA_segments(
            A_f=A_f,
            Q_tot=Q_solution
        )

        return hx["P_losses_calc"] - P_loss_target


    # Find flow area that satisfies pressure-drop constraint
    A_f_solution = brentq(
        residual_Af,
        Af_bounds[0],
        Af_bounds[1]
    )

    
    # ============================================================
    # Final converged solution
    # ============================================================

    # Find corresponding heat duty
    Q_solution = find_Q_for_Af(A_f_solution)

    # Final HX evaluation
    result = UA_segments(
        A_f=A_f_solution,
        Q_tot=Q_solution
    )

    result["Q_solution"] = Q_solution
    result["A_f"] = A_f_solution

    result["delta_T_min_target"] = delta_T_min_target

    result["P_loss_target"] = P_loss_target

    #result["mdot_storage"] = mdot_storage
    result["mdot_wf"] = mdot_wf
    result["N_seg"] = N_seg
    result["Q_max"] = Q_max


    return result


############################
#### Off-design cooling ####
############################

def HX_UA_segments_cool_wf_offdesign(
    T_storage_in,
    T_wf_in,
    P_wf_in,
    mdot_storage_fixed,
    mdot_wf,
    fluid_storage,
    fluid_wf,
    D,
    N_seg,
    A_f_design, 
    L_design,
    strategy_mdot_st_offdesign=0,
    P_losses_estimation = 0.01,
    mdot_st_cold_chg_info=None,
    mdot_st_hot_chg_info=None,
    mdot_st_cold_dis_info=None,
    mdot_st_hot_dis_info=None,
):
    fluid_storage_minT = 288.1 # K, min temperature of Dowtherm A

    # A_seg_discharge = A_seg_charge[::-1]
    # delta_x_discharge = delta_x_charge[::-1]


    def sam_prop_dowthermA_conductivity(T):
        """Returns the properties of Dowtherm A at a given temperature T in K."""
        # Thermal conductivity [W/m-K]
        k = max(0.1419 - 0.00016 * (T - 273.15), 0.001) #T in K
        return k
    
    def T_from_h_storage(h):
        """Returns the temperature of Dowtherm A at a given enthalpy."""
        x = h / 1e6

        T = (
            -27.06574572*x**6 + 151.21088901*x**5-340.13167310*x**4+411.02036213*x**3-388.55934518*x**2+645.42026919*x+288.26451925
        )

        return T
    
    # ============================================================
    # Evaluate HX for a given total Q
    # ============================================================

    def evaluate_Q(Q_tot):

        q_i = Q_tot / N_seg

        # ========================================================
        # Enthalpy / temperature profiles
        # ========================================================

        h_storage = np.zeros(N_seg + 1)
        h_wf = np.zeros(N_seg + 1)

        T_storage = np.zeros(N_seg + 1)
        T_wf = np.zeros(N_seg + 1)


        h_storage[0] = esol_properties.enthalpy(fluid_storage,T=T_storage_in,T_ref=fluid_storage_minT)
        h_wf[0] = fluid_properties.enthalpy(fluid_wf,T=T_wf_in,P=P_wf_in)

        T_storage[0] = T_storage_in
        T_wf[0] = T_wf_in

        if strategy_mdot_st_offdesign == 0:
            mdot_storage = mdot_storage_fixed

        if strategy_mdot_st_offdesign == 1: 
            h_wf_out_caprate = h_wf[0] - Q_tot/(mdot_wf)
            # Approximate outlet pressure used only for capacitance-rate matching.
            # Actual off-design pressure loss is calculated later by the HX model.
            T_wf_out_caprate = fluid_properties.temperature(fluid_wf, h=h_wf_out_caprate, P=P_wf_in*(1-P_losses_estimation))
            cp_wf_avg_caprate = abs((h_wf[0] - h_wf_out_caprate)/(T_wf_in - T_wf_out_caprate))
            C_wf_caprate = mdot_wf * cp_wf_avg_caprate
            deltaT_st_caprate = Q_tot / C_wf_caprate

            T_st_out_caprate = T_storage_in + deltaT_st_caprate 
            h_st_in_caprate = esol_properties.enthalpy(fluid_storage, T=T_storage_in, T_ref=fluid_storage_minT)
            h_st_out_caprate = esol_properties.enthalpy(fluid_storage, T=T_st_out_caprate, T_ref=fluid_storage_minT)

            cp_st_avg_caprate = abs((h_st_out_caprate - h_st_in_caprate)/ (T_st_out_caprate - T_storage_in))
            mdot_st_caprate = C_wf_caprate / cp_st_avg_caprate
            mdot_storage = mdot_st_caprate


        # Mass-balance strategy:
        # solve for mdot_st_cold_dis from the other three storage mass flows
        if strategy_mdot_st_offdesign == 2:
            mdot_storage = mdot_st_cold_chg_info*mdot_st_hot_dis_info/mdot_st_hot_chg_info


        for i in range(N_seg):

            # Hot storage gains heat
            h_storage[i+1] = (h_storage[i]+ q_i/mdot_storage)

            # Cold WF loses heat
            h_wf[i+1] = (h_wf[i]- q_i/mdot_wf)

            T_storage[i+1] = T_from_h_storage(h_storage[i+1])


        # ========================================================
        # Arrays
        # ========================================================

        P_wf = np.full(N_seg + 1,P_wf_in)

        # WF
        vel_wf = np.zeros(N_seg)
        rho_wf = np.zeros(N_seg)
        mu_wf = np.zeros(N_seg)
        Pr_wf = np.zeros(N_seg)
        k_wf = np.zeros(N_seg)
        cp_wf = np.zeros(N_seg)

        C_f_wf = np.zeros(N_seg)
        Re_wf = np.zeros(N_seg)
        Nu_wf = np.zeros(N_seg)
        h_conv_wf = np.zeros(N_seg)

        # Storage
        vel_st = np.zeros(N_seg)
        rho_st = np.zeros(N_seg)
        mu_st = np.zeros(N_seg)
        Pr_st = np.zeros(N_seg)
        k_st = np.zeros(N_seg)
        cp_st = np.zeros(N_seg)

        C_f_st = np.zeros(N_seg)
        Re_st = np.zeros(N_seg)
        Nu_st = np.zeros(N_seg)
        h_conv_st = np.zeros(N_seg)

        # Shared
        U_seg = np.zeros(N_seg)

        C_hot = np.zeros(N_seg)
        C_cold = np.zeros(N_seg)
        C_min = np.zeros(N_seg)
        C_max = np.zeros(N_seg)
        C_r = np.zeros(N_seg)

        epsilon_seg = np.zeros(N_seg)
        NTU_seg = np.zeros(N_seg)
        UA_seg = np.zeros(N_seg)

        A_seg = np.zeros(N_seg)
        delta_x = np.zeros(N_seg)

        delta_p = np.zeros(N_seg)


        # ========================================================
        # Pressure iteration
        # ========================================================

        max_iter = 1000

        for iteration in range(max_iter):
            # ----------------------------------------------------
            # WF temperature from h and P
            # ----------------------------------------------------

            for i in range(N_seg + 1):

                T_wf[i] = fluid_properties.temperature(fluid_wf,h=h_wf[i],P=P_wf[i])

            # ----------------------------------------------------
            # Segment-average states
            # ----------------------------------------------------

            T_st_seg = 0.5 * (T_storage[:-1]+ T_storage[1:])
            T_wf_seg = 0.5 * (T_wf[:-1]+ T_wf[1:])
            P_wf_seg = 0.5 * (P_wf[:-1]+ P_wf[1:])


            # ----------------------------------------------------
            # Local heat transfer coefficients
            # ----------------------------------------------------

            for i in range(N_seg):

                # ===========================
                # STORAGE
                # ===========================

                rho_st[i] = esol_properties.density(fluid=fluid_storage,T=T_st_seg[i])
                mu_st[i] = esol_properties.viscosity(fluid=fluid_storage,T=T_st_seg[i])
                vel_st[i] = (mdot_storage/(rho_st[i] * A_f_design))
                cp_st[i] = esol_properties.specheat(fluid=fluid_storage,T=T_st_seg[i])
                k_st[i] = sam_prop_dowthermA_conductivity(T=T_st_seg[i])
                Pr_st[i] = (cp_st[i]* mu_st[i]/ k_st[i])
                Re_st[i] = (rho_st[i]* vel_st[i]* D/ mu_st[i])
                C_f_st[i] = (1.58*np.log(Re_st[i])- 3.28)**(-2)
                Nu_st[i] = (0.5* C_f_st[i]* Pr_st[i]* (Re_st[i] - 1000)/(1+ 12.7* np.sqrt(C_f_st[i]/2)* (Pr_st[i]**(2/3) - 1)))
                h_conv_st[i] = (Nu_st[i]* k_st[i]/ D)

                # ===========================
                # WF
                # ===========================

                rho_wf[i] = fluid_properties.density(fluid=fluid_wf,T=T_wf_seg[i],P=P_wf_seg[i])
                vel_wf[i] = (mdot_wf/(rho_wf[i] * A_f_design))
                mu_wf[i] = fluid_properties.viscosity(fluid=fluid_wf,T=T_wf_seg[i],P=P_wf_seg[i])
                Pr_wf[i] = fluid_properties.prandtl(fluid=fluid_wf,T=T_wf_seg[i],P=P_wf_seg[i])
                cp_wf[i] = fluid_properties.specheat(fluid=fluid_wf,T=T_wf_seg[i],P=P_wf_seg[i])
                k_wf[i] = fluid_properties.conductivity(fluid=fluid_wf,T=T_wf_seg[i],P=P_wf_seg[i])
                Re_wf[i] = (rho_wf[i]* vel_wf[i]* D/ mu_wf[i])
                C_f_wf[i] = (1.58*np.log(Re_wf[i])- 3.28)**(-2)
                Nu_wf[i] = (0.5* C_f_wf[i]* Pr_wf[i]* (Re_wf[i] - 1000)/(1+ 12.7* np.sqrt(C_f_wf[i]/2)* (Pr_wf[i]**(2/3) - 1)))
                h_conv_wf[i] = (Nu_wf[i]* k_wf[i]/ D)

            # ====================================================
            # Put WF into storage spatial coordinate
            # ====================================================

            T_wf_spatial = T_wf[::-1]
            h_wf_spatial = h_wf[::-1]

            hconv_wf_spatial = h_conv_wf[::-1]


            # ====================================================
            # epsilon-NTU for every sub-HX
            # ====================================================

            physical = True

            for i in range(N_seg):

                # ------------------------------------------------
                # Average heat-capacity rates
                #
                # Same approach as the book:
                #
                # C = mdot * Delta h / Delta T
                # ------------------------------------------------

                dT_hot = (T_wf_spatial[i+1]- T_wf_spatial[i])
                dT_cold =  (T_storage[i+1] - T_storage[i])
                C_hot[i] = (mdot_wf* (h_wf_spatial[i+1]- h_wf_spatial[i]) / dT_hot)
                C_cold[i] = (mdot_storage * (h_storage[i+1]- h_storage[i]) / dT_cold)
                C_min[i] = min(C_hot[i],C_cold[i])
                C_max[i] = max(C_hot[i],C_cold[i])
                C_r[i] = (C_min[i]/ C_max[i])


                # ------------------------------------------------
                # Counterflow inlet temperature difference
                #
                # Storage enters segment from left:
                #       T_storage[i]
                #
                # WF enters segment from right:
                #       T_wf_spatial[i+1]
                # ------------------------------------------------

                delta_T_in = (T_wf_spatial[i+1] - T_storage[i])

                if delta_T_in <= 0:
                    print(f"INVALID delta_T: Q={Q_tot/1e6:.4f} MW, segment={i}, delta_T={delta_T_in:.6f} K", flush=True)
                    physical = False
                    break

                q_max_i = (C_min[i]* delta_T_in)
                epsilon_seg[i] = (q_i/ q_max_i)

                # ------------------------------------------------
                # epsilon must be between 0 and 1
                # ------------------------------------------------

                if (epsilon_seg[i] <= 0 or epsilon_seg[i] >= 1):
                    print(f"INVALID epsilon: Q={Q_tot/1e6:.4f} MW, segment={i}, epsilon={epsilon_seg[i]:.6f}", flush=True)
                    physical = False
                    break


                # ------------------------------------------------
                # Invert counterflow effectiveness-NTU relation
                # ------------------------------------------------

                if abs(1.0 - C_r[i]) < 1e-8:
                    NTU_seg[i] = (epsilon_seg[i]/(1.0 - epsilon_seg[i]))

                else:
                    NTU_seg[i] = (np.log((1.0- epsilon_seg[i]*C_r[i])/(1.0- epsilon_seg[i]))/(1.0 - C_r[i]))

                # ------------------------------------------------
                # Conductance required by sub-HX
                # ------------------------------------------------

                UA_seg[i] = (NTU_seg[i]* C_min[i])

                # ------------------------------------------------
                # Local U
                # ------------------------------------------------

                U_seg[i] = (1/(1/h_conv_st[i]+1/hconv_wf_spatial[i]))

                # ------------------------------------------------
                # Physical area and length required
                # ------------------------------------------------

                A_seg[i] = (UA_seg[i]/ U_seg[i])
                delta_x[i] = (A_seg[i]* D/(4*A_f_design))

            # ====================================================
            # Non-physical trial Q
            # ====================================================

            if not physical:

                return {
                "physical": False,
                "L_total": np.nan,
                "Q_total": Q_tot
            }


            # ====================================================
            # Pressure drop
            # ====================================================

            # delta_x is in STORAGE spatial direction.
            # WF travels in opposite direction.

            dx_wf = delta_x[::-1]
            P_wf_new = np.zeros(N_seg + 1)
            P_wf_new[0] = P_wf_in

            for i in range(N_seg):

                delta_p[i] = (4* C_f_wf[i]* (dx_wf[i]/D)* (rho_wf[i]* vel_wf[i]**2)/ 2)
                P_wf_new[i+1] = (P_wf_new[i]- delta_p[i])

            # ====================================================
            # Pressure convergence
            # ====================================================

            if np.allclose(P_wf_new,P_wf,rtol=1e-3, atol=1000.0):
                P_wf = P_wf_new
                break


            P_wf = (0.5*P_wf+ 0.5*P_wf_new)

        else:
            raise RuntimeError("HX pressure iteration did not converge")


        # ========================================================
        # Final results
        # ========================================================

        A_total = np.sum(A_seg)
        L_total = np.sum(delta_x)
        UA_total = np.sum(UA_seg)

        delta_p_total = (P_wf_in- P_wf[-1])
        P_losses_calc = (delta_p_total/ P_wf_in)
        T_wf_spatial = T_wf[::-1]
        delta_T_nodes = (T_wf_spatial - T_storage)
        delta_T_min = np.min(delta_T_nodes)

        return {

            "physical": True,
            "Q_total": Q_tot,
            "P_wf": P_wf,
            "T_storage": T_storage,
            "T_wf": T_wf,
            "T_st_seg": T_st_seg,
            "T_wf_seg": T_wf_seg,
            "h_conv_st": h_conv_st,
            "h_conv_wf": h_conv_wf,
            "cp_st": cp_st,
            "cp_wf": cp_wf,
            "Re_st": Re_st,
            "Re_wf": Re_wf,
            "C_hot": C_hot,
            "C_cold": C_cold,
            "C_min": C_min,
            "C_r": C_r,
            "vel_wf": vel_wf,
            "rho_wf": rho_wf,
            "C_f_wf": C_f_wf,

            "epsilon_seg": epsilon_seg,
            "NTU_seg": NTU_seg,

            "U_seg": U_seg,
            "UA_seg": UA_seg,
            "UA_total": UA_total,

            "A_seg": A_seg,
            "A_total": A_total,

            "delta_x": delta_x,
            "L_total": L_total,

            "delta_T_nodes": delta_T_nodes,
            "delta_T_min": delta_T_min,

            "delta_p": delta_p,
            "delta_p_total": delta_p_total,
            "P_losses_calc": P_losses_calc,
            "mdot_storage": mdot_storage,

            "iterations": iteration + 1,
        }


    # ============================================================
    # Maximum possible Q
    # ============================================================

    if strategy_mdot_st_offdesign == 0:
        mdot_storage = mdot_storage_fixed

        Q_storage_max = (mdot_storage*(
                esol_properties.enthalpy(fluid=fluid_storage,T=T_wf_in,T_ref=fluid_storage_minT) - 
                esol_properties.enthalpy(fluid=fluid_storage,T=T_storage_in,T_ref=fluid_storage_minT)
            )
        )
        Q_wf_max = (mdot_wf*(
                fluid_properties.enthalpy(fluid=fluid_wf,T=T_wf_in,P=P_wf_in) -
                fluid_properties.enthalpy(fluid=fluid_wf,T=T_storage_in,P=P_wf_in)
            )
        )

        Q_max = min(Q_storage_max,Q_wf_max)


    if strategy_mdot_st_offdesign == 1:
        Q_wf_max = (mdot_wf*(
                fluid_properties.enthalpy(fluid=fluid_wf,T=T_wf_in,P=P_wf_in) -
                fluid_properties.enthalpy(fluid=fluid_wf,T=T_storage_in,P=P_wf_in)
            )
        )
        Q_max = Q_wf_max

    if strategy_mdot_st_offdesign == 2:
        mdot_storage = mdot_st_cold_chg_info*mdot_st_hot_dis_info/mdot_st_hot_chg_info
        Q_storage_max = (mdot_storage*(
                esol_properties.enthalpy(fluid=fluid_storage,T=T_wf_in,T_ref=fluid_storage_minT) - 
                esol_properties.enthalpy(fluid=fluid_storage,T=T_storage_in,T_ref=fluid_storage_minT)
            )
        )
        Q_wf_max = (mdot_wf*(
                fluid_properties.enthalpy(fluid=fluid_wf,T=T_wf_in,P=P_wf_in) -
                fluid_properties.enthalpy(fluid=fluid_wf,T=T_storage_in,P=P_wf_in)
            )
        )

        Q_max = min(Q_storage_max,Q_wf_max)


    # ============================================================
    # Solve Q such that required length = physical HX length
    # ============================================================

    def residual_Q(Q_tot):

        hx = evaluate_Q(Q_tot)
        if not hx["physical"]:
            raise ValueError("Non-physical HX evaluation")


        return (hx["L_total"]- L_design)
    
    


    #Q_solution = fsolve(residual_Q, 0.5*Q_max)[0]
    #Q_solution = brentq(residual_Q, 0.05*Q_max, 0.999*Q_max)

    sol = least_squares(
    lambda x: residual_Q(x[0]),
    x0=[0.8 * Q_max],
    bounds=([0.1 * Q_max], [0.999999999999999999999999 * Q_max]),
    x_scale=[Q_max]
)
    print("LS success =", sol.success)
    print("LS message =", sol.message)
    print("LS Q =", sol.x[0]/1e6, "MW")
    print("LS residual =", sol.fun[0], "m")
    print("LS optimality =", sol.optimality)
    print("Q/Q_max =", sol.x[0] / Q_max)
    print("Active mask =", sol.active_mask)

    # if abs(sol.fun[0]) > 0.01:
    #     raise ValueError(
    #         f"HX off-design did not converge: "
    #         f"L_error={sol.fun[0]:.4f} m, "
    #         f"Q/Q_max={sol.x[0]/Q_max:.6f}"
    #     )

    Q_solution = sol.x[0]

    # ============================================================
    # Final solution
    # ============================================================

    result = evaluate_Q(Q_solution)

    result["Q_solution"] = Q_solution
    result["A_f"] = A_f_design
    result["L_design"] = L_design
    result["mdot_wf"] = mdot_wf
    result["N_seg"] = N_seg
    result["Q_max"] = Q_max

    return result