
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from state import State
from properties import state_from_TP
from properties import Incompressible 
from eeslib import fluid_properties as fluid_properties
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

esol_properties = Incompressible()


T_storage_in = 658 #658 #650.0 # K, temperature of storage fluid entering the heat exchanger
T_wf_in = 305 #290 #300.0 # K, temperature of working fluid entering the heat exchanger
N_seg = 100

D= 0.005

delta_T_min_target=5.0,
P_loss_target=0.01


mdot_storage = 96.
mdot_wf = 110.
fluid_storage = "Dowtherm A"
fluid_wf = 'CO2'
P_wf_in = 8e6*3 # Pa, pressure of working fluid entering the heat exchanger
Af_bounds=(0.02, 0.5)

def HX_UA_segments_heat_wf(
    T_storage_in,
    T_wf_in,
    P_wf_in,
    mdot_storage,
    mdot_wf,
    fluid_storage,
    fluid_wf,
    D,
    delta_T_min_target,
    P_loss_target,
    N_seg,
    Af_bounds=(0.02, 0.5),
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
            if np.allclose(P_wf_new, P_wf, rtol=1e-6, atol=1.0):
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
        "Re_wf": Re_wf,
        "Nu_wf": Nu_wf,
        "h_conv_wf": h_conv_wf,
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

        "iterations": iteration + 1
    }

    ###########################################################################
    #Maximum possible Q. This is used to set the higher bound for the iterator#
    ###########################################################################

    Q_storage_max = mdot_storage*(esol_properties.enthalpy(fluid = fluid_storage, T=T_storage_in, T_ref=fluid_storage_minT) - esol_properties.enthalpy(fluid = fluid_storage, T=T_wf_in, T_ref=fluid_storage_minT))
    Q_wf_max = mdot_wf*(fluid_properties.enthalpy(fluid = fluid_wf, T=T_storage_in, P=P_wf_in) - fluid_properties.enthalpy(fluid = fluid_wf, T=T_wf_in, P=P_wf_in))
    Q_max = min(Q_storage_max, Q_wf_max)

    def find_Q_for_Af(A_f):

        def residual_Q(Q_tot):

            hx_result = UA_segments(
                A_f=A_f,
                Q_tot=Q_tot
            )

            return hx_result["delta_T_min"] - delta_T_min_target

        return brentq(
            residual_Q,
            1e2,  #Minimum Q
            Q_max
        )

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
    result["mdot_storage"] = mdot_storage
    result["mdot_wf"] = mdot_wf
    result["N_seg"] = N_seg


    return result

result = HX_UA_segments_heat_wf(
    T_storage_in=T_storage_in,
    T_wf_in=T_wf_in,
    P_wf_in=P_wf_in,

    mdot_storage=mdot_storage,
    mdot_wf=mdot_wf,

    fluid_storage="Dowtherm A",
    fluid_wf="CO2",

    D=D,

    delta_T_min_target=delta_T_min_target,
    P_loss_target=P_loss_target,

    N_seg=N_seg,
    Af_bounds=(0.02, 0.5),
)

print("Required A_f =", result["A_f"], "m2")
print("Q solution =", result["Q_solution"] / 1e6, "MW")
print("P loss =", result["P_losses_calc"] * 100, "%")
print("Delta P =", result["delta_p_total"] / 1e3, "kPa")
print("Delta T min =", result["delta_T_min"], "K")
print("L =", result["L_total"], "m")
print("A =", result["A_total"], "m2")

print("UA =", result["UA_total"] / 1e6, "MW/K")
print("Iterations =", result["iterations"])

print("T hot out =", result["T_storage"][-1], "K")
print("T cold out =", result["T_wf"][-1], "K")
print("TTD high T side =", result["T_storage"][0] - result["T_wf"][-1], "K")
print("TTD low T side =", result["T_storage"][-1] - result["T_wf"][0], "K")

print("\n--- WF segment properties ---")

for i in range(N_seg):
    print(
        f"Seg {i:2d}: "
        f"T={result['T_wf_seg'][i]:8.2f} K | "
        f"P={result['P_wf_seg'][i]/1e6:7.3f} MPa | "
        f"rho={result['rho_wf'][i]:8.2f} kg/m3 | "
        f"cp={result['cp_wf'][i]:8.1f} J/kgK | "
        f"Re={result['Re_wf'][i]:10.2e} | "
        f"Nu={result['Nu_wf'][i]:8.2f} | "
        f"h={result['h_conv_wf'][i]:8.1f} W/m2K | "
        f"U={result['U_seg'][i]:8.1f} W/m2K | "
        f"dT={result['delta_T_seg'][i]:7.2f} K"
    )

##########################
## Plot Temperatures######
##########################

# Real physical coordinate along the HX
x_nodes = np.concatenate((
[0.0],
np.cumsum(result["delta_x"])
))

# Real segment-center positions
x_seg = 0.5 * (x_nodes[:-1] + x_nodes[1:])


# Temperatures in the SAME PHYSICAL DIRECTION
T_storage_spatial = result["T_storage"]
T_wf_spatial = result["T_wf"][::-1]

# cp is defined at segment centers and must also be reversed
cp_wf_spatial = result["cp_wf"][::-1]


fig, ax1 = plt.subplots(figsize=(9, 5))

# Temperatures
ax1.plot(x_nodes, T_storage_spatial, marker="o", label="Storage")
ax1.plot(x_nodes, T_wf_spatial, marker="o", label="sCO2")

ax1.set_xlabel("Physical position along HX [m]")
ax1.set_ylabel("Temperature [K]")
ax1.grid(True)
ax1.legend(loc="upper left")


# Second y-axis for cp
ax2 = ax1.twinx()

ax2.plot(
x_seg,
cp_wf_spatial / 1000,
linestyle="--",
marker="x",
label="sCO2 cp"
)

ax2.set_ylabel("sCO2 cp [kJ/kg-K]")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.show()


##############################
# Plot heat capacitance rates
##############################

# Heat capacitance rates [W/K]
C_st = result["mdot_storage"]  * result["cp_st"]
C_wf = result["mdot_wf"] * result["cp_wf"]

# WF must be reversed so both fluids are plotted
# in the same physical direction along the HX
C_wf_spatial = C_wf[::-1]

# IMPORTANT:
# Your segments do NOT have equal delta_x, so don't use linspace.
# Get the real physical positions from delta_x.

plt.figure(figsize=(9, 5))

plt.plot(
x_seg,
C_st / 1000,
marker="o",
label="Storage fluid"
)

plt.plot(
x_seg,
C_wf_spatial / 1000,
marker="o",
label="sCO2"
)

plt.xlabel("Physical position along HX [m]")
plt.ylabel("Heat capacitance rate, C = mdot cp [kW/K]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("UA_seg =", result["UA_seg"] / 1e6, "MW/K")

def HX_UA_segments_heat_wf_fixedA(
    T_storage_in,
    T_wf_in,
    P_wf_in,
    mdot_storage,
    mdot_wf,
    fluid_storage,
    fluid_wf,
    D,
    N_seg,
    A_f_design, 
    A_seg_design,
    delta_x_design,
):
    fluid_storage_minT = 288.1 # K, min temperature of Dowtherm A

    # A_seg_discharge = A_seg_charge[::-1]
    # delta_x_discharge = delta_x_charge[::-1]

    A_seg_design = np.asarray(A_seg_design, dtype=float) # Convert to float. Points to same location
    delta_x_design = np.asarray(delta_x_design, dtype=float)


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
    

    h_storage_in = esol_properties.enthalpy(fluid_storage, T=T_storage_in, T_ref=fluid_storage_minT)
    h_wf_in = fluid_properties.enthalpy(fluid_wf, T=T_wf_in, P=P_wf_in)

    def evaluate_h_wf_out(h_wf_out_guess):
        ###########################
        #Arrays
        ###########################

        h_storage = np.zeros(N_seg + 1)
        h_wf_spatial = np.zeros(N_seg + 1)
        T_storage = np.zeros(N_seg + 1)
        T_wf_spatial = np.zeros(N_seg + 1)

        P_wf_spatial = np.full(N_seg + 1, P_wf_in) #Assign inlet Pressure to all the array. 
        #sco2 fluid
        
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
        q_seg = np.zeros(N_seg) #New in this function

        T_st_seg = np.zeros(N_seg)
        T_wf_seg_spatial = np.zeros(N_seg)
        P_wf_seg_spatial = np.zeros(N_seg)
        delta_T_seg = np.zeros(N_seg)



        ##Determine the properties 

        max_iter = 1000
        for iteration in range(max_iter):
            #Boundary conditions
            h_storage[0] = h_storage_in
            T_storage[0] = T_storage_in

            # IMPORTANT:
            # WF array follows SPATIAL direction, not WF flow.
            #
            # x = 0:
            # storage inlet / WF outlet
            #
            # x = L:
            # storage outlet / WF inlet

            h_wf_spatial[0] = h_wf_out_guess #This correspond to the exit of the wf in the HX
            T_wf_spatial[0] = fluid_properties.temperature(fluid_wf, h=h_wf_spatial[0], P=P_wf_spatial[0])


            for i in range(N_seg):
                # Use current-node properties to get first q guess
                T_st_i = T_from_h_storage(h_storage[i])
                T_wf_i = fluid_properties.temperature(fluid_wf,h=h_wf_spatial[i],P=P_wf_spatial[i])

                # Storage properties at current node
                rho_st_i = esol_properties.density(fluid=fluid_storage,T=T_st_i)
                mu_st_i = esol_properties.viscosity(fluid=fluid_storage,T=T_st_i)
                vel_st_i = (mdot_storage/ (rho_st_i * A_f_design))
                cp_st_i = esol_properties.specheat(fluid=fluid_storage,T=T_st_i)
                k_st_i = sam_prop_dowthermA_conductivity(T=T_st_i)
                Pr_st_i = (cp_st_i* mu_st_i/ k_st_i)
                Re_st_i = (rho_st_i* vel_st_i* D/ mu_st_i)
                C_f_st_i = (1.58*np.log(Re_st_i)- 3.28)**(-2)
                Nu_st_i = (0.5* C_f_st_i* Pr_st_i* (Re_st_i - 1000)) / (1+ 12.7* np.sqrt(C_f_st_i/2)* (Pr_st_i**(2/3) - 1))
                h_conv_st_i = (Nu_st_i* k_st_i/ D)

                # WF properties at current node
                rho_wf_i = fluid_properties.density(fluid=fluid_wf,T=T_wf_i,P=P_wf_spatial[i])
                mu_wf_i = fluid_properties.viscosity(fluid=fluid_wf,T=T_wf_i,P=P_wf_spatial[i])
                vel_wf_i = (mdot_wf/ (rho_wf_i * A_f_design))
                Pr_wf_i = fluid_properties.prandtl(fluid=fluid_wf,T=T_wf_i,P=P_wf_spatial[i])
                k_wf_i = fluid_properties.conductivity(fluid=fluid_wf,T=T_wf_i,P=P_wf_spatial[i])
                Re_wf_i = (rho_wf_i* vel_wf_i* D/ mu_wf_i)
                C_f_wf_i = (1.58*np.log(Re_wf_i)- 3.28)**(-2)
                Nu_wf_i = (0.5* C_f_wf_i* Pr_wf_i* (Re_wf_i - 1000)) / (1+ 12.7* np.sqrt(C_f_wf_i/2)* (Pr_wf_i**(2/3) - 1))
                h_conv_wf_i = (Nu_wf_i* k_wf_i/ D)

                # First estimate of U and q
                U_predict = 1 / (1/h_conv_st_i+ 1/h_conv_wf_i)
                delta_T_predict = (T_st_i - T_wf_i)
                q_predict = (U_predict* A_seg_design[i]* delta_T_predict)

                # Predict next enthalpies
                h_storage_predict = (h_storage[i]- q_predict/mdot_storage)
                h_wf_predict = (h_wf_spatial[i]- q_predict/mdot_wf) #Remember it is against physical flow

                # Segment midpoint states

                h_st_mid = 0.5 * (h_storage[i]+ h_storage_predict)
                h_wf_mid = 0.5 * (h_wf_spatial[i]+ h_wf_predict)
                P_wf_mid = 0.5 * (P_wf_spatial[i]+ P_wf_spatial[i+1])
                T_st_mid = T_from_h_storage(h_st_mid)
                T_wf_mid = fluid_properties.temperature(fluid_wf,h=h_wf_mid,P=P_wf_mid)

                # Actual properties at segment midpoint

                # Storage
                rho_st[i] = esol_properties.density(fluid=fluid_storage,T=T_st_mid)
                mu_st[i] = esol_properties.viscosity(fluid=fluid_storage,T=T_st_mid)
                vel_st[i] = (mdot_storage/ (rho_st[i] * A_f_design))
                cp_st[i] = esol_properties.specheat(fluid=fluid_storage,T=T_st_mid)
                k_st[i] = sam_prop_dowthermA_conductivity(T=T_st_mid)
                Pr_st[i] = (cp_st[i]* mu_st[i]/ k_st[i])
                Re_st[i] = (rho_st[i]* vel_st[i]* D/ mu_st[i])
                C_f_st[i] = (1.58*np.log(Re_st[i])- 3.28)**(-2)
                Nu_st[i] = (0.5* C_f_st[i]* Pr_st[i]* (Re_st[i] - 1000)) / (1+ 12.7* np.sqrt(C_f_st[i]/2)* (Pr_st[i]**(2/3) - 1))
                h_conv_st[i] = (Nu_st[i]* k_st[i]/ D)

                # WF
                rho_wf[i] = fluid_properties.density(fluid=fluid_wf,T=T_wf_mid,P=P_wf_mid)
                mu_wf[i] = fluid_properties.viscosity(fluid=fluid_wf,T=T_wf_mid,P=P_wf_mid)
                vel_wf[i] = (mdot_wf/ (rho_wf[i] * A_f_design))
                Pr_wf[i] = fluid_properties.prandtl(fluid=fluid_wf,T=T_wf_mid,P=P_wf_mid)
                cp_wf[i] = fluid_properties.specheat(fluid=fluid_wf,T=T_wf_mid,P=P_wf_mid)
                k_wf[i] = fluid_properties.conductivity(fluid=fluid_wf,T=T_wf_mid,P=P_wf_mid)
                Re_wf[i] = (rho_wf[i]* vel_wf[i]* D/ mu_wf[i])
                C_f_wf[i] = (1.58*np.log(Re_wf[i])- 3.28)**(-2)
                Nu_wf[i] = (0.5* C_f_wf[i]* Pr_wf[i]* (Re_wf[i] - 1000)) / (1+ 12.7* np.sqrt(C_f_wf[i]/2)* (Pr_wf[i]**(2/3) - 1))
                h_conv_wf[i] = (Nu_wf[i]* k_wf[i]/ D)

                # U and heat transfer in physical segment

                U_seg[i] = 1 / (1/h_conv_st[i]+ 1/h_conv_wf[i])
                delta_T_seg[i] = (T_st_mid- T_wf_mid)
                q_seg[i] = (U_seg[i]* A_seg_design[i]* delta_T_seg[i])

                # Final enthalpies at next spatial node
                h_storage[i+1] = (h_storage[i]- q_seg[i]/mdot_storage)
                h_wf_spatial[i+1] = (h_wf_spatial[i]- q_seg[i]/mdot_wf)

                # Temperatures at next spatial node

                T_storage[i+1] = T_from_h_storage(h_storage[i+1])
                T_wf_spatial[i+1] = fluid_properties.temperature(fluid_wf,h=h_wf_spatial[i+1],P=P_wf_spatial[i+1])

                # Store segment midpoint values
                T_st_seg[i] = T_st_mid
                T_wf_seg_spatial[i] = T_wf_mid
                P_wf_seg_spatial[i] = P_wf_mid

            # Calculate pressure drop
            # WF flows physically from x=L to x=0.
            #
            # Therefore:
            #
            # P_wf_spatial[-1] = WF inlet pressure
            # P_wf_spatial[0]  = WF outlet pressure

            P_wf_new = np.zeros(N_seg + 1)

            P_wf_new[-1] = P_wf_in


            for i in range(N_seg - 1, -1, -1): #Start in the last position and stop in the first 

                delta_p[i] = (4* C_f_wf[i]* (delta_x_design[i]/D)* (rho_wf[i] * vel_wf[i]**2)/ 2)
                P_wf_new[i] = (P_wf_new[i+1]- delta_p[i])
            
            # Pressure convergence
            if np.allclose(P_wf_new,P_wf_spatial,rtol=1e-6,atol=1.0):
                P_wf_spatial = P_wf_new
                break


            P_wf_spatial = (0.5 * P_wf_spatial+ 0.5 * P_wf_new)

        else:
            raise RuntimeError("HX pressure iteration did not converge")
        
        # Final residual
        # ========================================================

        # At x=L we have reached the real WF inlet.
        #
        # Therefore the calculated enthalpy must equal the known
        # WF inlet enthalpy.

        residual_h = (h_wf_spatial[-1]- h_wf_in)

         # Totals

        Q_total = np.sum(q_seg)
        UA_seg = (U_seg* A_seg_design)
        UA_total = np.sum(UA_seg)
        delta_p_total = (P_wf_in- P_wf_spatial[0])
        P_losses_calc = (delta_p_total/ P_wf_in)


        # ========================================================
        # Return
        # ========================================================

        return {

            "residual_h": residual_h,

            "h_storage": h_storage,
            "h_wf_spatial": h_wf_spatial,

            "T_storage": T_storage,
            "T_wf_spatial": T_wf_spatial,

            "P_wf_spatial": P_wf_spatial,

            "T_st_seg": T_st_seg,
            "T_wf_seg_spatial": T_wf_seg_spatial,
            "P_wf_seg_spatial": P_wf_seg_spatial,

            "cp_st": cp_st,
            "cp_wf": cp_wf,

            "rho_wf": rho_wf,
            "Re_wf": Re_wf,
            "Nu_wf": Nu_wf,
            "h_conv_wf": h_conv_wf,

            "rho_st": rho_st,
            "Re_st": Re_st,
            "Nu_st": Nu_st,
            "h_conv_st": h_conv_st,

            "delta_T_seg": delta_T_seg,

            "U_seg": U_seg,

            "q_seg": q_seg,
            "Q_total": Q_total,

            # Fixed geometry from design

            "A_seg": A_seg_design,
            "delta_x": delta_x_design,
            "A_f": A_f_design,

            "A_total": np.sum(A_seg_design),
            "L_total": np.sum(delta_x_design),

            "UA_seg": UA_seg,
            "UA_total": UA_total,

            "delta_p": delta_p,
            "delta_p_total": delta_p_total,
            "P_losses_calc": P_losses_calc,

            "iterations": iteration + 1,
        }
    
    # Maximum physically possible heat transfer
    # Used only to construct an upper bound for h_wf_out

    Q_storage_max = (mdot_storage* (esol_properties.enthalpy(fluid=fluid_storage,T=T_storage_in,T_ref=fluid_storage_minT)-esol_properties.enthalpy(fluid=fluid_storage,T=T_wf_in,T_ref=fluid_storage_minT)))
    Q_wf_max = (mdot_wf* (fluid_properties.enthalpy(fluid=fluid_wf,T=T_storage_in,P=P_wf_in)-h_wf_in))
    Q_max = min(Q_storage_max,Q_wf_max)


    # ============================================================
    # Bounds for h_wf_out
    # ============================================================

    # Lower bound:
    # zero heat transfer -> WF outlet enthalpy = WF inlet enthalpy

    T_wf_out_low = T_storage_in - 50.0
    T_wf_out_high = T_storage_in - 1.0

    h_wf_out_low = fluid_properties.enthalpy(fluid_wf,T=T_wf_out_low,P=P_wf_in )
    h_wf_out_high = fluid_properties.enthalpy(fluid_wf,T=T_wf_out_high,P=P_wf_in)

    # ============================================================
    # Residual for outer enthalpy iterator
    # ============================================================

    def residual_h_wf_out(h_wf_out_guess):
        hx_result = evaluate_h_wf_out(h_wf_out_guess)
        return hx_result["residual_h"]

    # ============================================================
    # Find WF outlet enthalpy
    # ============================================================

    h_wf_out_solution = brentq(residual_h_wf_out,h_wf_out_low,h_wf_out_high)


    # ============================================================
    # Final converged HX
    # ============================================================

    result = evaluate_h_wf_out(h_wf_out_solution)
    result["h_wf_out"] = h_wf_out_solution
    result["T_wf_out"] = (result["T_wf_spatial"][0])
    result["Q_solution"] = (result["Q_total"])
    result["mdot_storage"] = mdot_storage
    result["mdot_wf"] = mdot_wf
    result["N_seg"] = N_seg

    return result
    


offdesign = HX_UA_segments_heat_wf_fixedA(

    T_storage_in=T_storage_in,
    T_wf_in=T_wf_in,
    P_wf_in=P_wf_in,

    mdot_storage=mdot_storage,
    mdot_wf=mdot_wf,

    fluid_storage="Dowtherm A",
    fluid_wf="CO2",

    D=0.005,

    N_seg=N_seg,

    A_f_design=result["A_f"],
    A_seg_design=result["A_seg"],
    delta_x_design=result["delta_x"],
)



print("\n================ OFF-DESIGN SUMMARY ================\n")

print(f"WF outlet enthalpy = {offdesign['h_wf_out']/1e3:.2f} kJ/kg")
print(f"WF outlet temperature = {offdesign['T_wf_out']:.2f} K")

print(f"Residual h = {offdesign['residual_h']:.6f} J/kg")

print(f"Q total = {offdesign['Q_total']/1e6:.3f} MW")
print(f"UA total = {offdesign['UA_total']/1e6:.4f} MW/K")

print(f"Storage inlet T = {offdesign['T_storage'][0]:.2f} K")
print(f"Storage outlet T = {offdesign['T_storage'][-1]:.2f} K")

print(f"WF outlet T = {offdesign['T_wf_spatial'][0]:.2f} K")
print(f"WF inlet T calculated = {offdesign['T_wf_spatial'][-1]:.2f} K")

print(f"WF outlet P = {offdesign['P_wf_spatial'][0]/1e6:.4f} MPa")
print(f"WF inlet P = {offdesign['P_wf_spatial'][-1]/1e6:.4f} MPa")
print(f"Pressure drop = {offdesign['delta_p_total']/1e3:.2f} kPa")
print(f"Pressure loss = {100*offdesign['P_losses_calc']:.4f} %")

print(f"Minimum segment dT = {np.min(offdesign['delta_T_seg']):.4f} K")

print("\n================ SEGMENTS ================\n")

for i in range(offdesign["N_seg"]):
    print(
        f"Seg {i:2d} | "
        f"q = {offdesign['q_seg'][i]/1e6:8.3f} MW | "
        f"U = {offdesign['U_seg'][i]:8.1f} W/m2K | "
        f"dT = {offdesign['delta_T_seg'][i]:8.3f} K | "
        f"T_st = {offdesign['T_st_seg'][i]:8.2f} K | "
        f"T_wf = {offdesign['T_wf_seg_spatial'][i]:8.2f} K | "
        f"P_wf = {offdesign['P_wf_seg_spatial'][i]/1e6:7.3f} MPa | "
        f"A = {offdesign['A_seg'][i]:8.3f} m2"
    )

print("\n================ DESIGN vs FIXED-A ================\n")

print(f"Q design     = {result['Q_solution']/1e6:.3f} MW")
print(f"Q fixed-A    = {offdesign['Q_total']/1e6:.3f} MW")

print(f"T_st,out design  = {result['T_storage'][-1]:.2f} K")
print(f"T_st,out fixed-A = {offdesign['T_storage'][-1]:.2f} K")

print(f"T_wf,out design  = {result['T_wf'][-1]:.2f} K")
print(f"T_wf,out fixed-A = {offdesign['T_wf_spatial'][0]:.2f} K")

print("\n===== DESIGN vs FIXED-A SEGMENTS =====")

for i in range(N_seg):
    print(
        f"Seg {i:2d} | "
        f"q design = {result['Q_solution']/N_seg/1e6:7.3f} MW | "
        f"q fixed = {offdesign['q_seg'][i]/1e6:7.3f} MW | "
        f"dT design = {result['delta_T_seg'][i]:8.3f} K | "
        f"dT fixed = {offdesign['delta_T_seg'][i]:8.3f} K | "
        f"U design = {result['U_seg'][i]:8.1f} | "
        f"U fixed = {offdesign['U_seg'][i]:8.1f}"
    )
    