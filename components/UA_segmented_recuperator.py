
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

#########################################################
#### Design cooling wf (charge hot HX) ############
########################################################

def HX_UA_segments_cool_wf(
    T_storage_in,
    T_wf_in,
    P_wf_in,
    mdot_storage_fixed,
    mdot_wf,
    fluid_storage_design,
    fluid_wf,
    D,
    delta_T_min_target,
    P_loss_target,
    N_seg,
    Af_bounds,
    strategy_mdot_st_design,
):
    fluid_storage_minT = 288.1 # K, min temperature of Dowtherm A
    fluid_storage_minT_salts = 533.2 # K, min temperature of  Salt (60% NaNO3, 40% KNO3)

    def P_reference(fluid_storage_design):
        if fluid_storage_design == "Water":
            T_max = 170 + 273.15 #Max temperature of the storage, we need to pressurize tank to maintain liquid water
            P_sat = fluid_properties.pressure(fluid_storage_design, T = T_max, Q = 0)
            P_reference = P_sat
        else:
            P_atm = 101325 #Pa
            P_reference = P_atm
        return P_reference
    
    P_ref = P_reference(fluid_storage_design)
    
    def heat_transfer_correlations(Re, Pr):

        def turbulent(Re, Pr):
            Cf = (1.58*np.log(Re) - 3.28)**(-2)
            Nu = (0.5*Cf*Pr*(Re-1000)) / (
                1 + 12.7*np.sqrt(Cf/2)*(Pr**(2/3)-1)
            )
            return Nu, Cf

        # Laminar
        if Re < 2300:
            Nu = 3.66
            Cf = 16/Re

        # Transition
        elif Re < 3000:
            w = (Re-2300)/(3000-2300)

            Nu_3000, _ = turbulent(3000, Pr)
            Nu = (1-w)*3.66 + w*Nu_3000

            Cf_lam = 16/2300
            _, Cf_turb = turbulent(3000, Pr)
            Cf = (1-w)*Cf_lam + w*Cf_turb

        # Turbulent
        else:
            Nu, Cf = turbulent(Re, Pr)

        return Nu, Cf

    def UA_segments(A_f, Q_tot):
        """Cooling the wf"""

        # A_f is the area of the storage fluid and the sco2 fluid

        # def sam_prop_conductivity(fluid_storage_design, T):
        #     if fluid_storage_design == "Dowtherm A":
        #         k = max(0.1419 - 0.00016 * (T - 273.15), 0.001) #T in K
        #     if fluid_storage_design == "Salt (60 NaNO3, 40 KNO3)":
        #         k = -1E-11*T*T*T + 3E-08*T*T + 0.0002*T + 0.3922 #  T in K
        #     if fluid_storage_design == "Therminol Oil":
        #         k = max(0.1381 - 0.00008708 * T - 0.0000001729 * T*T,.001)
        #     return k
        
        # def T_from_h_storage(fluid_storage_design, h):

        #     if fluid_storage_design == "Dowtherm A":
        #         """Returns the temperature of Dowtherm A at a given enthalpy."""
        #         x = h / 1e6
        #         T = (-27.06574572*x**6 + 151.21088901*x**5-340.13167310*x**4+411.02036213*x**3-388.55934518*x**2+645.42026919*x+288.26451925)

        #     if fluid_storage_design == "Salt (60 NaNO3, 40 KNO3)":
        #         """Returns the properties of Salt (60% NaNO3, 40% KNO3) at a given enthalpy in J/kg, T in K"""
        #         x = (h - 320000) / 100000
        #         T = 744.933727 + 65.4220756*x - 0.275421663*x**2
        #     return T
        
        

        q_i = Q_tot / N_seg

        ###########################
        #Calculate temperatures
        ###########################

        h_storage = np.zeros(N_seg + 1)
        h_wf = np.zeros(N_seg + 1)
        T_storage = np.zeros(N_seg + 1)
        T_wf = np.zeros(N_seg + 1)

        #h_storage_in = esol_properties.enthalpy(fluid_storage_design, T=T_storage_in, T_ref=fluid_storage_minT_salts)
        h_storage_in = fluid_properties.enthalpy(fluid_storage_design, T=T_storage_in, P=P_ref)
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
            # h_st_in_caprate = esol_properties.enthalpy(fluid_storage_design, T=T_storage_in, T_ref=fluid_storage_minT_salts)
            # h_st_out_caprate = esol_properties.enthalpy(fluid_storage_design, T=T_st_out_caprate, T_ref=fluid_storage_minT_salts)
            h_st_in_caprate = fluid_properties.enthalpy(fluid_storage_design, T=T_storage_in, P=P_ref)
            h_st_out_caprate = fluid_properties.enthalpy(fluid_storage_design, T=T_st_out_caprate, P=P_ref)

            cp_st_avg_caprate = abs((h_st_in_caprate - h_st_out_caprate)/ (T_storage_in - T_st_out_caprate))
            mdot_st_caprate = C_wf_caprate / cp_st_avg_caprate
            mdot_storage = mdot_st_caprate



        # I can't calculate yet the temperature of the working fluid bc pressure varies.
        for i in range(N_seg):
            h_storage[i+1] = h_storage[i] + (q_i / (mdot_storage))
            h_wf[i+1] = h_wf[i] - (q_i / (mdot_wf))

            #T_storage[i+1] = T_from_h_storage_salts(h_storage[i+1])
            T_storage[i+1] = fluid_properties.temperature(fluid_storage_design, h=h_storage[i+1], P=P_ref)
        

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
                # rho_st[i] = esol_properties.density(fluid = fluid_storage_design, T=T_st_seg[i])
                # mu_st[i] = esol_properties.viscosity(fluid = fluid_storage_design, T=T_st_seg[i])
                rho_st[i] = fluid_properties.density(fluid = fluid_storage_design, T=T_st_seg[i], P = P_ref)
                mu_st[i] = fluid_properties.viscosity(fluid = fluid_storage_design, T=T_st_seg[i], P = P_ref)
                vel_st[i] = mdot_storage / (rho_st[i] * A_f)
                #cp_st[i] = esol_properties.specheat(fluid = fluid_storage_design, T=T_st_seg[i])
                cp_st[i] = fluid_properties.specheat(fluid = fluid_storage_design, T=T_st_seg[i], P = P_ref)
                #k_st[i] = sam_prop_solarsalt60_40_conductivity(T=T_st_seg[i])
                k_st[i] = fluid_properties.conductivity(fluid = fluid_storage_design, T=T_st_seg[i], P = P_ref)
                Pr_st[i] = cp_st[i]*mu_st[i]/k_st[i]
                Re_st[i] = (rho_st[i] * vel_st[i] * D) / mu_st[i]
                #C_f_st[i] = (1.58*np.log(Re_st[i]) - 3.28)**(-2) #Check this! Fanning friction factor
                #Nu_st[i] = (0.5*C_f_st[i]*Pr_st[i]*(Re_st[i]-1000+1000))/(1 + 12.7*(((C_f_st[i])/2)**(1/2))*(Pr_st[i]**(2/3) - 1)) 
                Nu_st[i], C_f_st[i] = heat_transfer_correlations(Re_st[i], Pr_st[i])
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
            if np.allclose(P_wf_new, P_wf, rtol=1e-6, atol=1000.0):
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

    P_ref = P_reference(fluid_storage_design)

    if strategy_mdot_st_design == 1:
        Q_wf_max = mdot_wf*(
        fluid_properties.enthalpy(fluid = fluid_wf, T=T_wf_in, P=P_wf_in) - 
        fluid_properties.enthalpy(fluid = fluid_wf, T=T_storage_in, P=P_wf_in))
        Q_max = Q_wf_max
    else: 
        # Q_storage_max = mdot_storage_fixed*(
        #     esol_properties.enthalpy(fluid = fluid_storage_design, T=T_wf_in, T_ref=fluid_storage_minT_salts) - 
        #     esol_properties.enthalpy(fluid = fluid_storage_design, T=T_storage_in, T_ref=fluid_storage_minT_salts))
        Q_storage_max = mdot_storage_fixed*(
            fluid_properties.enthalpy(fluid = fluid_storage_design, T=T_wf_in, P=P_ref) - 
            fluid_properties.enthalpy(fluid = fluid_storage_design, T=T_storage_in, P=P_ref))
        
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

        Q_guess = 0.7 * Q_max

        Q_solution = least_squares(
            residual_Q,
            x0=[Q_guess],
            bounds=(
                [0.3 * Q_max],
                [0.999 * Q_max]
            )
        ).x[0]

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


#########################################################
#### Design heating wf (charge Cold HX) ############
########################################################

def HX_UA_segments_heat_wf(
    T_storage_in,
    T_wf_in,
    P_wf_in,
    mdot_storage_fixed,
    mdot_wf,
    fluid_storage_design,
    fluid_wf,
    D,
    delta_T_min_target,
    P_loss_target,
    N_seg,
    Af_bounds=(0.02, 0.5),
    strategy_mdot_st_design=1,
):
    #fluid_storage_minT = 288.1 # K, min temperature of Dowtherm A

    def P_reference(fluid_storage_design):
            if fluid_storage_design == "Water":
                T_max = 170 + 273.15 #Max temperature of the storage, we need to pressurize tank to maintain liquid water
                P_sat = fluid_properties.pressure(fluid_storage_design, T = T_max, Q = 0)
                P_reference = P_sat
            else:
                P_atm = 101325 #Pa
                P_reference = P_atm
            return P_reference
        
    P_ref = P_reference(fluid_storage_design)
    
    def heat_transfer_correlations(Re, Pr):

        def turbulent(Re, Pr):
            Cf = (1.58*np.log(Re) - 3.28)**(-2)
            Nu = (0.5*Cf*Pr*(Re-1000)) / (
                1 + 12.7*np.sqrt(Cf/2)*(Pr**(2/3)-1)
            )
            return Nu, Cf

        # Laminar
        if Re < 2300:
            Nu = 3.66
            Cf = 16/Re

        # Transition
        elif Re < 3000:
            w = (Re-2300)/(3000-2300)

            Nu_3000, _ = turbulent(3000, Pr)
            Nu = (1-w)*3.66 + w*Nu_3000

            Cf_lam = 16/2300
            _, Cf_turb = turbulent(3000, Pr)
            Cf = (1-w)*Cf_lam + w*Cf_turb

        # Turbulent
        else:
            Nu, Cf = turbulent(Re, Pr)

        return Nu, Cf

    def UA_segments(A_f, Q_tot):
        """Heating the wf"""

        # A_f is the area of the storage fluid and the sco2 fluid

        # def sam_prop_dowthermA_conductivity(T):
        #     """Returns the properties of Dowtherm A at a given temperature T in K."""
        #     # Thermal conductivity [W/m-K]
        #     k = max(0.1419 - 0.00016 * (T - 273.15), 0.001) #T in K
        #     return k
        
        # def T_from_h_storage(h):
        #     """Returns the temperature of Dowtherm A at a given enthalpy."""
        #     x = h / 1e6

        #     T = (
        #         -27.06574572*x**6 + 151.21088901*x**5-340.13167310*x**4+411.02036213*x**3-388.55934518*x**2+645.42026919*x+288.26451925
        #     )

            # return T

        

        q_i = Q_tot / N_seg

        ###########################
        #Calculate temperatures
        ###########################

        h_storage = np.zeros(N_seg + 1)
        h_wf = np.zeros(N_seg + 1)
        T_storage = np.zeros(N_seg + 1)
        T_wf = np.zeros(N_seg + 1)

        h_storage_in = fluid_properties.enthalpy(fluid_storage_design, T=T_storage_in, P=P_ref)
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
            h_st_in_caprate = fluid_properties.enthalpy(fluid_storage_design, T=T_storage_in, P=P_ref)
            h_st_out_caprate = fluid_properties.enthalpy(fluid_storage_design, T=T_st_out_caprate, P=P_ref)

            cp_st_avg_caprate = abs((h_st_out_caprate - h_st_in_caprate)/ (T_st_out_caprate - T_storage_in))
            mdot_st_caprate = C_wf_caprate / cp_st_avg_caprate
            mdot_storage = mdot_st_caprate


        # I can't calculate yet the temperature of the working fluid bc pressure varies.
        for i in range(N_seg):
            h_storage[i+1] = h_storage[i] - (q_i / (mdot_storage))
            h_wf[i+1] = h_wf[i] + (q_i / (mdot_wf))

            T_storage[i+1] = fluid_properties.temperature(fluid_storage_design, h=h_storage[i+1], P=P_ref)
        

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
                rho_st[i] = fluid_properties.density(fluid = fluid_storage_design, T=T_st_seg[i], P = P_ref)
                mu_st[i] = fluid_properties.viscosity(fluid = fluid_storage_design, T=T_st_seg[i], P = P_ref)
                vel_st[i] = mdot_storage / (rho_st[i] * A_f)
                cp_st[i] = fluid_properties.specheat(fluid = fluid_storage_design, T=T_st_seg[i], P = P_ref)
                k_st[i] = fluid_properties.conductivity(fluid = fluid_storage_design, T=T_st_seg[i], P = P_ref)
                Pr_st[i] = cp_st[i]*mu_st[i]/k_st[i]
                Re_st[i] = (rho_st[i] * vel_st[i] * D) / mu_st[i]
                # C_f_st[i] = (1.58*np.log(Re_st[i]) - 3.28)**(-2) #Check this! Fanning friction factor
                # Nu_st[i] = (0.5*C_f_st[i]*Pr_st[i]*(Re_st[i]-1000))/(1 + 12.7*(((C_f_st[i])/2)**(1/2))*(Pr_st[i]**(2/3) - 1)) 
                Nu_st[i], C_f_st[i] = heat_transfer_correlations(Re_st[i], Pr_st[i])
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
        delta_T_nodes = T_storage - T_wf_spatial_nodes
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
        "rho_wf": rho_wf,
        "rho_st": rho_st,
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
        "mu_st": mu_st,
        "mu_wf": mu_wf,

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
        Q_wf_max = mdot_wf*(fluid_properties.enthalpy(fluid = fluid_wf, T=T_storage_in, P=P_wf_in) - fluid_properties.enthalpy(fluid = fluid_wf, T=T_wf_in, P=P_wf_in))
        Q_max = Q_wf_max
    else:
        Q_storage_max = mdot_storage_fixed*(
            fluid_properties.enthalpy(fluid = fluid_storage_design, T=T_storage_in, P=P_ref) - 
            fluid_properties.enthalpy(fluid = fluid_storage_design, T=T_wf_in, P=P_ref))
        Q_wf_max = mdot_wf*(fluid_properties.enthalpy(fluid = fluid_wf, T=T_storage_in, P=P_wf_in) - fluid_properties.enthalpy(fluid = fluid_wf, T=T_wf_in, P=P_wf_in))
        Q_max = min(Q_storage_max, Q_wf_max)


    

    def find_Q_for_Af(A_f):

        def residual_Q(Q_tot):

            Q_tot = Q_tot.item()

            hx_result = UA_segments(
                A_f=A_f,
                Q_tot=Q_tot
            )

            return hx_result["delta_T_min"] - delta_T_min_target

        # return brentq(
        #     residual_Q,
        #     1e2,  #Minimum Q
        #     Q_max
        # )

        Q_guess = 0.7 * Q_max

        Q_solution = least_squares(
            residual_Q,
            x0=[Q_guess],
            bounds=(
                [0.3 * Q_max],
                [0.999 * Q_max]
            )
        ).x[0]

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
    result["Q_max"] = Q_max
    result["A_f"] = A_f_solution

    result["delta_T_min_target"] = delta_T_min_target

    result["P_loss_target"] = P_loss_target
    #result["mdot_storage"] = mdot_storage
    result["mdot_wf"] = mdot_wf
    result["N_seg"] = N_seg


    return result


#########################################################
#### Design charge recuperator ############
########################################################

def HX_UA_segments_charge(
    T_cold_in,
    T_hot_in,
    P_hot_in,
    P_cold_in,
    mdot_cold,
    mdot_hot,
    fluid_hot,
    fluid_cold,
    D,
    delta_T_min_target,
    P_loss_target,
    N_seg,
    Af_bounds,
):

    def UA_segments(A_f, Q_tot):
        """charge"""

        # A_f is the area of the cold fluid and the sco2 fluid
     

        q_i = Q_tot / N_seg

        ###########################
        #Calculate temperatures
        ###########################

        h_cold = np.zeros(N_seg + 1)
        h_hot = np.zeros(N_seg + 1)
        T_cold = np.zeros(N_seg + 1)
        T_hot = np.zeros(N_seg + 1)

        h_cold_in = fluid_properties.enthalpy(fluid_cold, T=T_cold_in, P=P_cold_in)
        h_hot_in = fluid_properties.enthalpy(fluid_hot, T=T_hot_in, P=P_hot_in)

        h_cold[0] = h_cold_in
        h_hot[0] = h_hot_in
        T_cold[0] = T_cold_in
        T_hot[0] = T_hot_in




        # I can't calculate yet the temperature of the working fluid bc pressure varies.
        for i in range(N_seg):
            h_cold[i+1] = h_cold[i] + (q_i / (mdot_cold))
            h_hot[i+1] = h_hot[i] - (q_i / (mdot_hot))
        

        #sco2 fluid
        P_hot = np.full(N_seg + 1, P_hot_in) #Assign inlet Pressure to all the array. 
        vel_hot = np.zeros(N_seg)
        rho_hot = np.zeros(N_seg)
        mu_hot = np.zeros(N_seg)
        Pr_hot = np.zeros(N_seg) 
        k_hot = np.zeros(N_seg)
        cp_hot = np.zeros(N_seg)

        #cold 
        P_cold = np.full(N_seg + 1, P_cold_in) #Assign inlet Pressure to all the array. 
        vel_cold= np.zeros(N_seg)
        rho_cold= np.zeros(N_seg)
        mu_cold= np.zeros(N_seg)
        Pr_cold= np.zeros(N_seg)
        k_cold= np.zeros(N_seg)
        cp_cold= np.zeros(N_seg)

        ################
        #Segments
        ################

        #Working fluid 
        C_f_hot = np.zeros(N_seg)
        Re_hot = np.zeros(N_seg)
        Nu_hot = np.zeros(N_seg)
        h_conv_hot = np.zeros(N_seg)

        #cold  
        C_f_cold= np.zeros(N_seg)
        Re_cold= np.zeros(N_seg)
        Nu_cold= np.zeros(N_seg)
        h_conv_cold= np.zeros(N_seg)

        delta_p = np.zeros(N_seg)
        #Shared
        delta_p_cold = np.zeros(N_seg)
        U_seg = np.zeros(N_seg)
        A_seg = np.zeros(N_seg)
        delta_x = np.zeros(N_seg)


        ##Determine the properties 

        max_iter = 1000
        for iteration in range(max_iter):

            #There is no deltaP in the cold side. So Just hot depends on the pressure drop.
            #Here we use the first guess of pressure drop
            for i in range(N_seg+1):
                T_hot[i] = fluid_properties.temperature(fluid_hot, h = h_hot[i],P=P_hot[i] )
                T_cold[i] = fluid_properties.temperature(fluid_cold, h = h_cold[i],P=P_cold[i] )


            # cold: physical/spatial direction
            T_cold_seg = 0.5 * (T_cold[:-1] + T_cold[1:])

            # hot: its own physical flow direction
            T_hot_seg = 0.5 * (T_hot[:-1] + T_hot[1:])
            P_hot_seg = 0.5 * (P_hot[:-1] + P_hot[1:])

            P_cold_seg = 0.5 * (P_cold[:-1] + P_cold[1:])

            #Now we reverse the cold fluid arrays to match the spatial coordinate of the hot fluid
            


            #Now we evaluate the properties and calculate the actual pressure drop
            for i in range(N_seg):
                rho_cold[i] = fluid_properties.density(fluid = fluid_cold, T=T_cold_seg[i],P=P_cold_seg[i])
                mu_cold[i] = fluid_properties.viscosity(fluid = fluid_cold, T=T_cold_seg[i],P=P_cold_seg[i])
                vel_cold[i] = mdot_cold / (rho_cold[i] * A_f)
                cp_cold[i] = fluid_properties.specheat(fluid = fluid_cold, T=T_cold_seg[i],P=P_cold_seg[i])
                k_cold[i] = fluid_properties.conductivity(fluid = fluid_cold, T=T_cold_seg[i], P=P_cold_seg[i])
                Pr_cold[i] = cp_cold[i]*mu_cold[i]/k_cold[i]
                Re_cold[i] = (rho_cold[i] * vel_cold[i] * D) / mu_cold[i]
                C_f_cold[i] = (1.58*np.log(Re_cold[i]) - 3.28)**(-2) #Check this! Fanning friction factor
                Nu_cold[i] = (0.5*C_f_cold[i]*Pr_cold[i]*(Re_cold[i]-1000))/(1 + 12.7*(((C_f_cold[i])/2)**(1/2))*(Pr_cold[i]**(2/3) - 1)) 
                h_conv_cold[i] = (Nu_cold[i]*k_cold[i])/D

                #Cold hot
                rho_hot[i] = fluid_properties.density(fluid = fluid_hot, T=T_hot_seg[i], P=P_hot_seg[i]) 
                vel_hot[i] = mdot_hot / (rho_hot[i] * A_f)
                mu_hot[i] = fluid_properties.viscosity(fluid = fluid_hot, T=T_hot_seg[i], P=P_hot_seg[i])
                Pr_hot[i] = fluid_properties.prandtl(fluid = fluid_hot, T=T_hot_seg[i], P=P_hot_seg[i])
                cp_hot[i] = fluid_properties.specheat(fluid=fluid_hot,  T=T_hot_seg[i], P=P_hot_seg[i] )
                Re_hot[i] = (rho_hot[i] * vel_hot[i] * D) / mu_hot[i]
                C_f_hot[i] = (1.58*np.log(Re_hot[i]) - 3.28)**(-2) #Check this! Fanning friction factor
                Nu_hot[i] = (0.5*C_f_hot[i]*Pr_hot[i]*(Re_hot[i]-1000))/(1 + 12.7*(((C_f_hot[i])/2)**(1/2))*(Pr_hot[i]**(2/3) - 1))
                k_hot[i] = fluid_properties.conductivity(fluid = fluid_hot, T=T_hot_seg[i], P=P_hot_seg[i])
                h_conv_hot[i] = (Nu_hot[i]*k_hot[i])/D


            ##Put both fluids in same physical coordinate

            hconv_hot_spatial = h_conv_hot[::-1]
            T_hot_seg_spatial = T_hot_seg[::-1]
            P_hot_spatial = P_hot_seg[::-1]
            delta_T_seg = T_hot_seg_spatial - T_cold_seg

            for i in range(N_seg):
                U_seg[i] = 1/(1/h_conv_cold[i] + 1/hconv_hot_spatial[i])
                A_seg[i] = q_i/(U_seg[i]*delta_T_seg[i])
                delta_x[i] = A_seg[i]*D/(4*A_f)
            
            dx_hot = delta_x[::-1]
            P_hot_new = np.zeros(N_seg + 1)
            P_hot_new[0] = P_hot_in

            dx_cold = delta_x
            P_cold_new = np.zeros(N_seg + 1)
            P_cold_new[0] = P_cold_in

            for i in range(N_seg):
                delta_p[i] = 4*C_f_hot[i]*(dx_hot[i]/D)*(rho_hot[i]*vel_hot[i]**2)/2
                P_hot_new[i+1] = P_hot_new[i] - delta_p[i]

            for i in range(N_seg):
                delta_p_cold[i] = 4*C_f_cold[i]*(dx_cold[i]/D)*(rho_cold[i]*vel_cold[i]**2)/2
                P_cold_new[i+1] = P_cold_new[i] - delta_p_cold[i]

            if np.any(P_cold_new <= 0):
                print("INVALID COLD PRESSURE")
                print("A_f =", A_f)
                print("Q =", Q_tot / 1e6, "MW")
                print("P_cold_new =", P_cold_new / 1e6, "MPa")
                print("L =", np.sum(delta_x), "m")
                print("Re cold min =", np.min(Re_cold))
                print("Re hot min =", np.min(Re_hot))
                raise ValueError("Negative cold-side pressure")

            #Converge
            if np.allclose(P_hot_new, P_hot, rtol=1e-6, atol=1000.0) and np.allclose(P_cold_new, P_cold, rtol=1e-6, atol=1000.0) :
                P_hot = P_hot_new
                P_cold = P_cold_new
                break
            
            P_hot = 0.5 * P_hot + 0.5 * P_hot_new
            P_cold = 0.5 * P_cold + 0.5 * P_cold_new

        else:
            raise RuntimeError("HX pressure iteration did not converge")
        
        T_hot_spatial_nodes = T_hot[::-1]
        delta_T_nodes = T_hot_spatial_nodes - T_cold
        delta_T_min = np.min(delta_T_nodes)

        delta_p_total = P_hot_in - P_hot[-1]
        P_losses_calc = delta_p_total / P_hot_in
        delta_p_cold_total = P_cold_in - P_cold[-1]
        P_losses_cold_calc = delta_p_cold_total/P_cold_in

        L_total = np.sum(delta_x)
        A_total = np.sum(A_seg)
        UA_seg = U_seg * A_seg
        UA_total = np.sum(UA_seg)

        return {
        "P_hot": P_hot,
        "P_cold": P_cold,
        "T_cold": T_cold,
        "T_hot": T_hot,
        "cp_hot": cp_hot,
        "cp_cold": cp_cold,
        "Re_cold": Re_cold,
        "rho_hot": rho_hot,
        "rho_cold": rho_cold,
        "Re_hot": Re_hot,
        "Nu_hot": Nu_hot,
        "h_conv_hot": h_conv_hot,
        "h_conv_cold": h_conv_cold,
        "k_hot": k_hot,
        "Pr_hot": Pr_hot,
        "T_hot_seg": T_hot_seg,
        "P_hot_seg": P_hot_seg,

        "delta_T_seg": delta_T_seg,
        "delta_T_min": delta_T_min,

        "U_seg": U_seg,
        "A_seg": A_seg,
        "delta_x": delta_x,

        "delta_p": delta_p,
        "delta_p_total": delta_p_total,
        "P_losses_calc": P_losses_calc,
        "P_losses_cold_calc":P_losses_cold_calc,

        "L_total": L_total,
        "A_total": A_total,

        "UA_seg": UA_seg,
        "UA_total": UA_total,

        "iterations": iteration + 1,
        "mdot_cold": mdot_cold
    }

    ###########################################################################
    #Maximum possible Q. This is used to set the higher bound for the iterator#
    ###########################################################################

 
    Q_cold_max = mdot_cold*(
        fluid_properties.enthalpy(fluid = fluid_cold, T=T_hot_in, P=P_cold_in) - 
        fluid_properties.enthalpy(fluid = fluid_cold, T=T_cold_in, P=P_cold_in))
    Q_hot_max = mdot_hot*(
        fluid_properties.enthalpy(fluid = fluid_hot, T=T_hot_in, P=P_hot_in) - 
        fluid_properties.enthalpy(fluid = fluid_hot, T=T_cold_in, P=P_hot_in))
    Q_max = min(Q_cold_max, Q_hot_max)

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
                [0.01 * Q_max],
                [0.999 * Q_max]
            )
        ).x[0]

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

    #result["mdot_cold"] = mdot_cold
    result["mdot_hot"] = mdot_hot
    result["N_seg"] = N_seg
    result["Q_max"] = Q_max


    return result


#########################################################
#### OffDesign heating wf (discharge Hot HX) ############
########################################################

def HX_UA_segments_heat_wf_offdesign(
    T_storage_in,
    T_wf_in,
    P_wf_in,
    fluid_storage_offdesign,
    mdot_wf,
    mdot_storage_fixed,
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
    #fluid_storage_minT = 288.1 # K, min temperature of Dowtherm A

    # A_seg_discharge = A_seg_charge[::-1]
    # delta_x_discharge = delta_x_charge[::-1
    def P_reference(fluid_storage_offdesign):
        if fluid_storage_offdesign == "Water":
            T_max = 170 + 273.15 #Max temperature of the storage, we need to pressurize tank to maintain liquid water
            P_sat = fluid_properties.pressure(fluid_storage_offdesign, T = T_max, Q = 0)
            P_reference = P_sat
        else:
            P_atm = 101325 #Pa
            P_reference = P_atm
        return P_reference
    
    P_ref = P_reference(fluid_storage_offdesign)

    def heat_transfer_correlations(Re, Pr):

        def turbulent(Re, Pr):
            Cf = (1.58*np.log(Re) - 3.28)**(-2)
            Nu = (0.5*Cf*Pr*(Re-1000)) / (
                1 + 12.7*np.sqrt(Cf/2)*(Pr**(2/3)-1)
            )
            return Nu, Cf

        # Laminar
        if Re < 2300:
            Nu = 3.66
            Cf = 16/Re

        # Transition
        elif Re < 3000:
            w = (Re-2300)/(3000-2300)

            Nu_3000, _ = turbulent(3000, Pr)
            Nu = (1-w)*3.66 + w*Nu_3000

            Cf_lam = 16/2300
            _, Cf_turb = turbulent(3000, Pr)
            Cf = (1-w)*Cf_lam + w*Cf_turb

        # Turbulent
        else:
            Nu, Cf = turbulent(Re, Pr)

        return Nu, Cf


    # def sam_prop_dowthermA_conductivity(T):
    #     """Returns the properties of Dowtherm A at a given temperature T in K."""
    #     # Thermal conductivity [W/m-K]
    #     k = max(0.1419 - 0.00016 * (T - 273.15), 0.001) #T in K
    #     return k
    
    # def T_from_h_storage(h):
    #     """Returns the temperature of Dowtherm A at a given enthalpy."""
    #     x = h / 1e6

    #     T = (
    #         -27.06574572*x**6 + 151.21088901*x**5-340.13167310*x**4+411.02036213*x**3-388.55934518*x**2+645.42026919*x+288.26451925
    #     )

    #     return T
    
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


        h_storage[0] = fluid_properties.enthalpy(fluid_storage_offdesign,T=T_storage_in,P=P_ref)
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
            h_st_in_caprate = fluid_properties.enthalpy(fluid_storage_offdesign, T=T_storage_in, P=P_ref)
            h_st_out_caprate = fluid_properties.enthalpy(fluid_storage_offdesign, T=T_st_out_caprate, P=P_ref)

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

            T_storage[i+1] = fluid_properties.temperature(fluid_storage_offdesign, h = h_storage[i+1], P=P_ref)


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

                rho_st[i] = fluid_properties.density(fluid=fluid_storage_offdesign,T=T_st_seg[i], P = P_ref)
                mu_st[i] = fluid_properties.viscosity(fluid=fluid_storage_offdesign,T=T_st_seg[i], P = P_ref)
                vel_st[i] = (mdot_storage/(rho_st[i] * A_f_design))
                cp_st[i] = fluid_properties.specheat(fluid=fluid_storage_offdesign,T=T_st_seg[i], P = P_ref)
                k_st[i] = fluid_properties.conductivity(fluid=fluid_storage_offdesign,T=T_st_seg[i], P = P_ref)
                Pr_st[i] = (cp_st[i]* mu_st[i]/ k_st[i])
                Re_st[i] = (rho_st[i]* vel_st[i]* D/ mu_st[i])
                # C_f_st[i] = (1.58*np.log(Re_st[i])- 3.28)**(-2)
                # Nu_st[i] = (0.5* C_f_st[i]* Pr_st[i]* (Re_st[i] - 1000)/(1+ 12.7* np.sqrt(C_f_st[i]/2)* (Pr_st[i]**(2/3) - 1)))
                Nu_st[i], C_f_st[i] = heat_transfer_correlations(Re_st[i], Pr_st[i])
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
                    physical = False
                    break

                q_max_i = (C_min[i]* delta_T_in)
                epsilon_seg[i] = (q_i/ q_max_i)

                # ------------------------------------------------
                # epsilon must be between 0 and 1
                # ------------------------------------------------

                if (epsilon_seg[i] <= 0 or epsilon_seg[i] >= 1):
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

            # if not physical:

            #     return {
            #     "physical": False,
            #     "L_total": np.nan,
            #     "Q_total": Q_tot
            # }


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
            "rho_st": rho_st,
            "rho_wf": rho_wf,
            "mu_st": mu_st,
            "mu_wf": mu_wf,

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
                fluid_properties.enthalpy(fluid=fluid_storage_offdesign,T=T_storage_in,P=P_ref) - 
                fluid_properties.enthalpy(fluid=fluid_storage_offdesign,T=T_wf_in,P=P_ref)
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
                fluid_properties.enthalpy(fluid=fluid_storage_offdesign,T=T_storage_in,P=P_ref) - 
                fluid_properties.enthalpy(fluid=fluid_storage_offdesign,T=T_wf_in,P=P_ref)
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

        # if not hx["physical"]:
        #     return 1e6
    
        return (hx["L_total"]- L_design)


    Q_solution = brentq(residual_Q, 0.1*Q_max, 0.999*Q_max)

    # ============================================================
    # Final solution
    # ============================================================

    result = evaluate_Q(Q_solution)

    result["Q_solution"] = Q_solution
    result["Q_max"] = Q_max

    result["A_f"] = A_f_design

    result["L_design"] = L_design

    result["mdot_wf"] = mdot_wf

    result["N_seg"] = N_seg

    return result


###############################################
#######OffDesign cool wf (discharge cold HX)

def HX_UA_segments_cool_wf_offdesign(
    T_storage_in,
    T_wf_in,
    P_wf_in,
    mdot_storage_fixed,
    mdot_wf,
    fluid_storage_offdesign,
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
    fluid_storage_minT_salts = 533.2 # K, min temperature of  Salt (60% NaNO3, 40% KNO3)

    def P_reference(fluid_storage_offdesign):
        if fluid_storage_offdesign == "Water":
            T_max = 170 + 273.15 #Max temperature of the storage, we need to pressurize tank to maintain liquid water
            P_sat = fluid_properties.pressure(fluid_storage_offdesign, T = T_max, Q = 0)
            P_reference = P_sat
        else:
            P_atm = 101325 #Pa
            P_reference = P_atm
        return P_reference
    
    P_ref = P_reference(fluid_storage_offdesign)

    def heat_transfer_correlations(Re, Pr):

        def turbulent(Re, Pr):
            Cf = (1.58*np.log(Re) - 3.28)**(-2)
            Nu = (0.5*Cf*Pr*(Re-1000)) / (
                1 + 12.7*np.sqrt(Cf/2)*(Pr**(2/3)-1)
            )
            return Nu, Cf

        # Laminar
        if Re < 2300:
            Nu = 3.66
            Cf = 16/Re

        # Transition
        elif Re < 3000:
            w = (Re-2300)/(3000-2300)

            Nu_3000, _ = turbulent(3000, Pr)
            Nu = (1-w)*3.66 + w*Nu_3000

            Cf_lam = 16/2300
            _, Cf_turb = turbulent(3000, Pr)
            Cf = (1-w)*Cf_lam + w*Cf_turb

        # Turbulent
        else:
            Nu, Cf = turbulent(Re, Pr)

        return Nu, Cf

    # A_seg_discharge = A_seg_charge[::-1]
    # delta_x_discharge = delta_x_charge[::-1]


    # def sam_prop_dowthermA_conductivity(T):
    #         """Returns the properties of Dowtherm A at a given temperature T in K."""
    #         # Thermal conductivity [W/m-K]
    #         k = max(0.1419 - 0.00016 * (T - 273.15), 0.001) #T in K
    #         return k
        
    # def sam_prop_solarsalt60_40_conductivity(T):
    #     """Returns the properties of Salt (60% NaNO3, 40% KNO3) at a given temperature T in K."""
    #     # Thermal conductivity [W/m-K]
    #     k = -1E-11*T*T*T + 3E-08*T*T + 0.0002*T + 0.3922 #  T in K
    #     return k
    
    # def T_from_h_storage(h):
    #     """Returns the temperature of Dowtherm A at a given enthalpy."""
    #     x = h / 1e6

    #     T = (
    #         -27.06574572*x**6 + 151.21088901*x**5-340.13167310*x**4+411.02036213*x**3-388.55934518*x**2+645.42026919*x+288.26451925
    #     )

    #     return T
    
    # def T_from_h_storage_salts(h):
    #     """Returns the properties of Salt (60% NaNO3, 40% KNO3) at a given enthalpy in J/kg, T in K"""
    #     x = (h - 320000) / 100000
    #     return 744.933727 + 65.4220756*x - 0.275421663*x**2
    
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

        P_atm = 101325 #Pa

        # h_storage[0] = esol_properties.enthalpy(fluid_storage_offdesign,T=T_storage_in,T_ref=fluid_storage_minT_salts)
        h_storage[0] = fluid_properties.enthalpy(fluid_storage_offdesign,T=T_storage_in,P=P_ref)
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
            # h_st_in_caprate = esol_properties.enthalpy(fluid_storage_offdesign, T=T_storage_in, T_ref=fluid_storage_minT)
            # h_st_out_caprate = esol_properties.enthalpy(fluid_storage_offdesign, T=T_st_out_caprate, T_ref=fluid_storage_minT)
            h_st_in_caprate = fluid_properties.enthalpy(fluid_storage_offdesign, T=T_storage_in, P=P_ref)
            h_st_out_caprate = fluid_properties.enthalpy(fluid_storage_offdesign, T=T_st_out_caprate, P=P_ref)

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

            #T_storage[i+1] = T_from_h_storage(h_storage[i+1])
            T_storage[i+1] = fluid_properties.temperature(fluid_storage_offdesign, h = h_storage[i+1], P=P_ref)


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

                rho_st[i] = fluid_properties.density(fluid=fluid_storage_offdesign,T=T_st_seg[i], P = P_ref)
                mu_st[i] = fluid_properties.viscosity(fluid=fluid_storage_offdesign,T=T_st_seg[i], P = P_ref)
                vel_st[i] = (mdot_storage/(rho_st[i] * A_f_design))
                cp_st[i] = fluid_properties.specheat(fluid=fluid_storage_offdesign,T=T_st_seg[i], P = P_ref)
                #k_st[i] = sam_prop_dowthermA_conductivity(T=T_st_seg[i])
                k_st[i] = fluid_properties.conductivity(fluid=fluid_storage_offdesign,T=T_st_seg[i], P = P_ref)
                Pr_st[i] = (cp_st[i]* mu_st[i]/ k_st[i])
                Re_st[i] = (rho_st[i]* vel_st[i]* D/ mu_st[i])
                # C_f_st[i] = (1.58*np.log(Re_st[i])- 3.28)**(-2)
                # Nu_st[i] = (0.5* C_f_st[i]* Pr_st[i]* (Re_st[i] - 1000)/(1+ 12.7* np.sqrt(C_f_st[i]/2)* (Pr_st[i]**(2/3) - 1)))
                Nu_st[i], C_f_st[i] = heat_transfer_correlations(Re_st[i], Pr_st[i])
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
                    physical = False
                    break

                q_max_i = (C_min[i]* delta_T_in)
                epsilon_seg[i] = (q_i/ q_max_i)

                # ------------------------------------------------
                # epsilon must be between 0 and 1
                # ------------------------------------------------

                if (epsilon_seg[i] <= 0 or epsilon_seg[i] >= 1):
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

            # if not physical:

            #     return {
            #     "physical": False,
            #     "L_total": np.nan,
            #     "Q_total": Q_tot
            # }


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

            if np.allclose(P_wf_new,P_wf,rtol=1e-6,atol=1000.0):
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
                fluid_properties.enthalpy(fluid=fluid_storage_offdesign,T=T_wf_in,P=P_ref) - 
                fluid_properties.enthalpy(fluid=fluid_storage_offdesign,T=T_storage_in,P=P_ref)
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
                fluid_properties.enthalpy(fluid=fluid_storage_offdesign,T=T_wf_in,P=P_ref) - 
                fluid_properties.enthalpy(fluid=fluid_storage_offdesign,T=T_storage_in,P=P_ref)
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


        return (hx["L_total"]- L_design)
    
    


    #Q_solution = fsolve(residual_Q, 0.5*Q_max)[0]
    #Q_solution = brentq(residual_Q, 0.05*Q_max, 0.999*Q_max)
    Q_solution = least_squares(
    lambda x: [residual_Q(x[0])],
                x0=[0.8 * Q_max],
                bounds=([0.2 * Q_max], [0.999 * Q_max])
                    ).x[0]

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


##############################################
#########Recuperator discharge ##############
###########################################

def HX_UA_segments_offdesign_discharge(
    T_cold_in,
    T_hot_in,
    P_hot_in,
    P_cold_in,
    mdot_cold,
    mdot_hot,
    fluid_cold,
    fluid_hot,
    D,
    N_seg,
    A_f_design, 
    L_design,
    P_losses_estimation = 0.01,
):


    
    # ============================================================
    # Evaluate HX for a given total Q
    # ============================================================

    def evaluate_Q(Q_tot):

        q_i = Q_tot / N_seg

        # ========================================================
        # Enthalpy / temperature profiles
        # ========================================================

        h_cold = np.zeros(N_seg + 1)
        h_hot = np.zeros(N_seg + 1)

        T_cold = np.zeros(N_seg + 1)
        T_hot = np.zeros(N_seg + 1)


        h_cold[0] = fluid_properties.enthalpy(fluid_cold,T=T_cold_in,P=P_cold_in)
        h_hot[0] = fluid_properties.enthalpy(fluid_hot,T=T_hot_in,P=P_hot_in)

        T_cold[0] = T_cold_in
        T_hot[0] = T_hot_in


        for i in range(N_seg):

            # Hot cold gains heat
            h_cold[i+1] = (h_cold[i]+ q_i/mdot_cold)

            # Cold hot loses heat
            h_hot[i+1] = (h_hot[i]- q_i/mdot_hot)


        # ========================================================
        # Arrays
        # ========================================================

        P_hot = np.full(N_seg + 1,P_hot_in)

        # hot
        vel_hot = np.zeros(N_seg)
        rho_hot = np.zeros(N_seg)
        mu_hot = np.zeros(N_seg)
        Pr_hot = np.zeros(N_seg)
        k_hot = np.zeros(N_seg)
        cp_hot = np.zeros(N_seg)

        C_f_hot = np.zeros(N_seg)
        Re_hot = np.zeros(N_seg)
        Nu_hot = np.zeros(N_seg)
        h_conv_hot = np.zeros(N_seg)

        # cold
        P_cold = np.full(N_seg + 1,P_cold_in)

        vel_cold= np.zeros(N_seg)
        rho_cold= np.zeros(N_seg)
        mu_cold= np.zeros(N_seg)
        Pr_cold= np.zeros(N_seg)
        k_cold= np.zeros(N_seg)
        cp_cold= np.zeros(N_seg)

        C_f_cold= np.zeros(N_seg)
        Re_cold= np.zeros(N_seg)
        Nu_cold= np.zeros(N_seg)
        h_conv_cold= np.zeros(N_seg)

        delta_p = np.zeros(N_seg)
        delta_p_cold = np.zeros(N_seg)
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

        


        # ========================================================
        # Pressure iteration
        # ========================================================

        max_iter = 1000

        for iteration in range(max_iter):
            # ----------------------------------------------------
            # hot temperature from h and P
            # ----------------------------------------------------

            for i in range(N_seg + 1):

                T_hot[i] = fluid_properties.temperature(fluid_hot,h=h_hot[i],P=P_hot[i])
                T_cold[i] = fluid_properties.temperature(fluid_cold,h=h_cold[i],P=P_cold[i])

            # ----------------------------------------------------
            # Segment-average states
            # ----------------------------------------------------

            T_cold_seg = 0.5 * (T_cold[:-1]+ T_cold[1:])
            T_hot_seg = 0.5 * (T_hot[:-1]+ T_hot[1:])
            P_hot_seg = 0.5 * (P_hot[:-1]+ P_hot[1:])
            P_cold_seg = 0.5 * (P_cold[:-1]+ P_cold[1:])


            # ----------------------------------------------------
            # Local heat transfer coefficients
            # ----------------------------------------------------

            for i in range(N_seg):

                # ===========================
                # cold
                # ===========================

                rho_cold[i] = fluid_properties.density(fluid=fluid_cold,T=T_cold_seg[i],P=P_cold_seg[i])
                mu_cold[i] = fluid_properties.viscosity(fluid=fluid_cold,T=T_cold_seg[i],P=P_cold_seg[i])
                vel_cold[i] = (mdot_cold/(rho_cold[i] * A_f_design))
                cp_cold[i] = fluid_properties.specheat(fluid=fluid_cold,T=T_cold_seg[i],P=P_cold_seg[i])
                k_cold[i] = fluid_properties.conductivity(fluid=fluid_cold,T=T_cold_seg[i],P=P_cold_seg[i])
                Pr_cold[i] = (cp_cold[i]* mu_cold[i]/ k_cold[i])
                Re_cold[i] = (rho_cold[i]* vel_cold[i]* D/ mu_cold[i])
                C_f_cold[i] = (1.58*np.log(Re_cold[i])- 3.28)**(-2)
                Nu_cold[i] = (0.5* C_f_cold[i]* Pr_cold[i]* (Re_cold[i] - 1000)/(1+ 12.7* np.sqrt(C_f_cold[i]/2)* (Pr_cold[i]**(2/3) - 1)))
                h_conv_cold[i] = (Nu_cold[i]* k_cold[i]/ D)

                # ===========================
                # hot
                # ===========================

                rho_hot[i] = fluid_properties.density(fluid=fluid_hot,T=T_hot_seg[i],P=P_hot_seg[i])
                vel_hot[i] = (mdot_hot/(rho_hot[i] * A_f_design))
                mu_hot[i] = fluid_properties.viscosity(fluid=fluid_hot,T=T_hot_seg[i],P=P_hot_seg[i])
                Pr_hot[i] = fluid_properties.prandtl(fluid=fluid_hot,T=T_hot_seg[i],P=P_hot_seg[i])
                cp_hot[i] = fluid_properties.specheat(fluid=fluid_hot,T=T_hot_seg[i],P=P_hot_seg[i])
                k_hot[i] = fluid_properties.conductivity(fluid=fluid_hot,T=T_hot_seg[i],P=P_hot_seg[i])
                Re_hot[i] = (rho_hot[i]* vel_hot[i]* D/ mu_hot[i])
                C_f_hot[i] = (1.58*np.log(Re_hot[i])- 3.28)**(-2)
                Nu_hot[i] = (0.5* C_f_hot[i]* Pr_hot[i]* (Re_hot[i] - 1000)/(1+ 12.7* np.sqrt(C_f_hot[i]/2)* (Pr_hot[i]**(2/3) - 1)))
                h_conv_hot[i] = (Nu_hot[i]* k_hot[i]/ D)

            # ====================================================
            # Put hot into cold spatial coordinate
            # ====================================================

            T_hot_spatial = T_hot[::-1]
            h_hot_spatial = h_hot[::-1]

            hconv_hot_spatial = h_conv_hot[::-1]


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

                dT_hot = (T_hot_spatial[i+1]- T_hot_spatial[i])
                dT_cold =  (T_cold[i+1] - T_cold[i])
                C_hot[i] = (mdot_hot* (h_hot_spatial[i+1]- h_hot_spatial[i]) / dT_hot)
                C_cold[i] = (mdot_cold * (h_cold[i+1]- h_cold[i]) / dT_cold)
                C_min[i] = min(C_hot[i],C_cold[i])
                C_max[i] = max(C_hot[i],C_cold[i])
                C_r[i] = (C_min[i]/ C_max[i])


                # ------------------------------------------------
                # Counterflow inlet temperature difference
                #
                # cold enters segment from left:
                #       T_cold[i]
                #
                # hot enters segment from right:
                #       T_hot_spatial[i+1]
                # ------------------------------------------------

                delta_T_in = (T_hot_spatial[i+1] - T_cold[i])

                # if delta_T_in <= 0:
                #     physical = False
                #     break

                q_max_i = (C_min[i]* delta_T_in)
                epsilon_seg[i] = (q_i/ q_max_i)

                # ------------------------------------------------
                # epsilon must be between 0 and 1
                # ------------------------------------------------

                # if (epsilon_seg[i] <= 0 or epsilon_seg[i] >= 1):
                #     physical = False
                #     break


                # ------------------------------------------------
                # Invert counterflow effectiveness-NTU relation
                # ------------------------------------------------

                if abs(1.0 - C_r[i]) < 1e-8:
                    NTU_seg[i] = (epsilon_seg[i]/(1.0 - epsilon_seg[i]))

                else:
                    NTU_seg[i] = (np.log((1.0- epsilon_seg[i]*C_r[i])/(1.0- epsilon_seg[i]))/(1.0 - C_r[i]))

                
                # if not 0 < epsilon_seg[i] < 1:
                #     print("\nINVALID EPSILON")
                #     print("Iteration:", iteration)
                #     print("Segment:", i)
                #     print("Q_tot [MW]:", Q_tot / 1e6)
                #     print("epsilon:", epsilon_seg[i])
                #     print("C_r:", C_r[i])
                #     print("C_min:", C_min[i])
                #     print("C_max:", C_max[i])
                #     print("delta_T_in:", delta_T_in)
                #     print("q_i:", q_i)
                #     print("q_max_i:", q_max_i)
                #     print("T_hot:", T_hot)
                #     print("T_cold:", T_cold)
                #     print("P_hot:", P_hot)
                #     print("P_cold:", P_cold)
                #     breakpoint()

                # ------------------------------------------------
                # Conductance required by sub-HX
                # ------------------------------------------------

                UA_seg[i] = (NTU_seg[i]* C_min[i])

                # ------------------------------------------------
                # Local U
                # ------------------------------------------------

                U_seg[i] = (1/(1/h_conv_cold[i]+1/hconv_hot_spatial[i]))

                # ------------------------------------------------
                # Physical area and length required
                # ------------------------------------------------

                A_seg[i] = (UA_seg[i]/ U_seg[i])
                delta_x[i] = (A_seg[i]* D/(4*A_f_design))

            # ====================================================
            # Non-physical trial Q
            # ====================================================

            # if not physical:

            #     return {
            #     "physical": False,
            #     "L_total": np.nan,
            #     "Q_total": Q_tot
            # }


            # ====================================================
            # Pressure drop
            # ====================================================

            # delta_x is in cold spatial direction.
            # hot travels in opposite direction.
            delta_x_pressure = delta_x * L_design / np.sum(delta_x)
            dx_hot = delta_x_pressure[::-1]
            dx_cold = delta_x_pressure

            #dx_hot = delta_x[::-1]
            P_hot_new = np.zeros(N_seg + 1)
            P_hot_new[0] = P_hot_in

            #dx_cold = delta_x
            P_cold_new = np.zeros(N_seg + 1)
            P_cold_new[0] = P_cold_in

            for i in range(N_seg):

                delta_p[i] = (4* C_f_hot[i]* (dx_hot[i]/D)* (rho_hot[i]* vel_hot[i]**2)/ 2)
                P_hot_new[i+1] = (P_hot_new[i]- delta_p[i])

            for i in range(N_seg):

                delta_p_cold[i] = (4* C_f_cold[i]* (dx_cold[i]/D)* (rho_cold[i]* vel_cold[i]**2)/ 2)
                P_cold_new[i+1] = (P_cold_new[i]- delta_p_cold[i])

            # ====================================================
            # Pressure convergence
            # ====================================================

            print(
                    f"Iter: {iteration:3d} | "
                    f"Q: {Q_tot/1e6:.4f} MW | "
                    f"L: {np.sum(delta_x):.4f} m | "
                    f"Pinch: {np.min(T_hot[::-1] - T_cold):.5f} K | "
                    f"Max epsilon: {np.max(epsilon_seg):.6f}"
                )

            if np.allclose(P_hot_new,P_hot,rtol=1e-6,atol=1000.0) and np.allclose(P_cold_new,P_cold,rtol=1e-6,atol=1000.0):
                P_hot = P_hot_new
                P_cold = P_cold_new
                break


            P_hot = (0.8*P_hot+ 0.2*P_hot_new)
            P_cold = (0.8*P_cold+ 0.2*P_cold_new)

        else:
            raise RuntimeError("HX pressure iteration did not converge")


        # ========================================================
        # Final results
        # ========================================================

        A_total = np.sum(A_seg)
        L_total = np.sum(delta_x)
        UA_total = np.sum(UA_seg)

        delta_p_total = (P_hot_in- P_hot[-1])
        P_losses_calc = (delta_p_total/ P_hot_in)
        delta_p_cold_total = P_cold_in - P_cold[-1]
        P_losses_cold_calc = (delta_p_cold_total/ P_cold_in)

        T_hot_spatial = T_hot[::-1]
        delta_T_nodes = (T_hot_spatial - T_cold)
        delta_T_min = np.min(delta_T_nodes)

        return {

            "physical": True,
            "Q_total": Q_tot,
            "P_hot": P_hot,
            "P_cold": P_cold,
            "T_cold": T_cold,
            "T_hot": T_hot,
            "T_cold_seg": T_cold_seg,
            "T_hot_seg": T_hot_seg,
            "h_conv_cold": h_conv_cold,
            "h_conv_hot": h_conv_hot,
            "cp_cold": cp_cold,
            "cp_hot": cp_hot,
            "Re_cold": Re_cold,
            "Re_hot": Re_hot,
            "C_hot": C_hot,
            "C_cold": C_cold,
            "C_min": C_min,
            "C_r": C_r,
            "rho_cold": rho_cold,
            "rho_hot": rho_hot,

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
            "P_losses_cold_calc": P_losses_cold_calc,
            "mdot_cold": mdot_cold,

            "iterations": iteration + 1,
        }


    # ============================================================
    # Maximum possible Q
    # ============================================================


    mdot_cold = mdot_cold

    Q_cold_max = (mdot_cold*(
            fluid_properties.enthalpy(fluid=fluid_cold,T=T_hot_in,P=P_cold_in) - 
            fluid_properties.enthalpy(fluid=fluid_cold,T=T_cold_in,P=P_cold_in)
        )
    )
    Q_hot_max = (mdot_hot*(
            fluid_properties.enthalpy(fluid=fluid_hot,T=T_hot_in,P=P_hot_in) -
            fluid_properties.enthalpy(fluid=fluid_hot,T=T_cold_in,P=P_hot_in)
        )
    )

    Q_max = min(Q_cold_max,Q_hot_max)



    # ============================================================
    # Solve Q such that required length = physical HX length
    # ============================================================

    def residual_Q(Q_tot):

        hx = evaluate_Q(Q_tot)

        return (hx["L_total"]- L_design)
    
    


    #Q_solution = fsolve(residual_Q, 0.5*Q_max)[0]
    #Q_solution = brentq(residual_Q, 0.3*Q_max, 0.999*Q_max)
    sol = least_squares(
    lambda x: [residual_Q(float(x[0]))],
    x0=[0.5 * Q_max],
    bounds=([0.3 * Q_max], [0.9948*Q_max]),
    method="trf"
    )

    Q_solution = sol.x[0]

    # ============================================================
    # Final solution
    # ============================================================

    result = evaluate_Q(Q_solution)

    result["Q_solution"] = Q_solution
    result["A_f"] = A_f_design
    result["L_design"] = L_design
    result["mdot_hot"] = mdot_hot
    result["N_seg"] = N_seg
    result["Q_max"] = Q_max

    return result