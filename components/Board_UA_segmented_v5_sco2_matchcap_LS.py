
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
from scipy.optimize import fsolve
from scipy.optimize import least_squares

esol_properties = Incompressible()
""" Notes
I set low pressure losses goal  = 0.1% becuase now I would have to calculate pressure losses for cold and hot side. I don't want to 
do the solver more complex, so I tried lower deltaP and then check that the pressure of the other side is near 1%. That
happens at 0.1% in the high pressure side. 
Both sides have the same Af. 


"""

T_cold_in = 160 + 273 #350 #658 #650.0 # K, temperature of cold fluid entering the heat exchanger
T_hot_in = 400.23 + 273 #658 #290 #300.0 # K, temperature of working fluid entering the heat exchanger
N_seg =5

D=0.005
delta_T_min_target=5.0
P_loss_target=0.001

mdot_cold = 100.
mdot_hot = 100.
fluid_hot = 'CO2'
fluid_cold = 'CO2'
P_hot_in = 8e6*3.2 # Pa, pressure of working fluid entering the heat exchanger
P_cold_in = 8e6
Af_bounds=(0.04, 0.8)


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
        fluid_properties.enthalpy(fluid = fluid_hot, T=T_hot_in, P=P_cold_in) - 
        fluid_properties.enthalpy(fluid = fluid_cold, T=T_cold_in, P=P_cold_in))
    Q_hot_max = mdot_hot*(
        fluid_properties.enthalpy(fluid = fluid_hot, T=T_hot_in, P=P_hot_in) - 
        fluid_properties.enthalpy(fluid = fluid_cold, T=T_cold_in, P=P_hot_in))
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



result = HX_UA_segments_charge(
    T_cold_in=T_cold_in,
    T_hot_in=T_hot_in,
    P_hot_in=P_hot_in,
    P_cold_in=P_cold_in,

    mdot_cold=mdot_cold,
    mdot_hot=mdot_hot,

    fluid_cold="CO2",
    fluid_hot="CO2",

    D=D,

    delta_T_min_target=delta_T_min_target,
    P_loss_target=P_loss_target,

    N_seg=N_seg,
    Af_bounds=Af_bounds,
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

print("T cold out =", result["T_cold"][-1], "K")
print("T hot out =", result["T_hot"][-1], "K")
print("TTD cold-in side =",result["T_hot"][-1] - result["T_cold"][0],"K")
print("TTD hot-in side =",result["T_hot"][0] - result["T_cold"][-1],"K")
print("\n--- hot segment properties ---")

for i in range(len(result["T_hot_seg"])):
    print(
        f"Seg {i:2d}: "
        f"T={result['T_hot_seg'][i]:8.2f} K | "
        f"P={result['P_hot_seg'][i]/1e6:7.3f} MPa | "
        f"rho={result['rho_hot'][i]:8.2f} kg/m3 | "
        f"cp={result['cp_hot'][i]:8.1f} J/kgK | "
        f"Re={result['Re_hot'][i]:10.2e} | "
        f"Nu={result['Nu_hot'][i]:8.2f} | "
        f"h={result['h_conv_hot'][i]:8.1f} W/m2K | "
        f"U={result['U_seg'][i]:8.1f} W/m2K | "
        f"dT={result['delta_T_seg'][i]:7.2f} K"
    )

print("\n--- Reynolds numbers ---")

for i in range(N_seg):
    print(
        f"Seg {i:2d}: "
        f"Re_cold = {result['Re_cold'][i]:10.2e} | "
        f"Re_hot = {result['Re_hot'][i]:10.2e}"
    )

print("\n--- DESIGN SEGMENTS ---")

for i in range(N_seg):
    print(
        f"Seg {i:2d}: "
        f"A = {result['A_seg'][i]:8.2f} m2 | "
        f"dT = {result['delta_T_seg'][i]:8.2f} K | "
        f"U = {result['U_seg'][i]:8.1f} W/m2K | "
        f"q = {result['Q_solution']/N_seg/1e6:7.3f} MW"
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
T_cold_spatial = result["T_cold"]
T_hot_spatial = result["T_hot"][::-1]

# cp is defined at segment centers and must also be reversed
cp_hot_spatial = result["cp_hot"][::-1]
cp_cold_spatial = result["cp_cold"]


fig, ax1 = plt.subplots(figsize=(9, 5))

# Temperatures
ax1.plot(x_nodes, T_cold_spatial, marker="o", label="cold")
ax1.plot(x_nodes, T_hot_spatial, marker="o", label="sCO2")

ax1.set_xlabel("Physical position along HX [m]")
ax1.set_ylabel("Temperature [K]")
ax1.grid(True)
ax1.legend(loc="upper left")


# Second y-axis for cp
ax2 = ax1.twinx()

ax2.plot(
x_seg,
cp_hot_spatial / 1000,
linestyle="--",
marker="x",
label="sCO2 cp"
)

ax2.plot(
    x_seg,
    cp_cold_spatial / 1000,
    linestyle="--",
    marker="x",
    label="cold cp"
)

ax2.set_ylabel("sCO2 cp [kJ/kg-K]")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.show()


##############################
# Plot heat capacitance rates
##############################

# Heat capacitance rates [W/K]

C_cold= result["mdot_cold"] * result["cp_cold"]
C_hot = result["mdot_hot"] * result["cp_hot"]

# hot must be reversed so both fluids are plotted
# in the same physical direction along the HX
C_hot_spatial = C_hot[::-1]

# IMPORTANT:
# Your segments do NOT have equal delta_x, so don't use linspace.
# Get the real physical positions from delta_x.

plt.figure(figsize=(9, 5))

plt.plot(
x_seg,
C_cold/ 1000,
marker="o",
label="cold fluid"
)

plt.plot(
x_seg,
C_hot_spatial / 1000,
marker="o",
label="sCO2"
)

plt.xlabel("Physical position along HX [m]")
plt.ylabel("Heat capacitance rate, C = mdot cp [kW/K]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

##########################################################################


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

                U_seg[i] = (1/(1/h_conv_cold[i]+1/hconv_hot_spatial[i]))

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

            # delta_x is in cold spatial direction.
            # hot travels in opposite direction.

            dx_hot = delta_x[::-1]
            P_hot_new = np.zeros(N_seg + 1)
            P_hot_new[0] = P_hot_in

            dx_cold = delta_x
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

            if np.allclose(P_hot_new,P_hot,rtol=1e-6,atol=1000.0) and np.allclose(P_cold_new,P_cold,rtol=1e-6,atol=1000.0):
                P_hot = P_hot_new
                P_cold = P_cold_new
                break


            P_hot = (0.5*P_hot+ 0.5*P_hot_new)
            P_cold = (0.5*P_cold+ 0.5*P_cold_new)

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
    #Q_solution = brentq(residual_Q, 0.05*Q_max, 0.999*Q_max)
    sol = least_squares(
    lambda x: [residual_Q(float(x[0]))],
    x0=[0.7 * Q_max],
    bounds=([0.3 * Q_max], [Q_max]),
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
    



offdesign = HX_UA_segments_offdesign_discharge(

    T_cold_in=T_cold_in,
    T_hot_in=T_hot_in,
    P_hot_in=P_hot_in,
    P_cold_in=P_cold_in,

    mdot_cold=mdot_cold,
    mdot_hot=mdot_hot,

    fluid_cold="CO2",
    fluid_hot="CO2",

    D=D,
    N_seg=N_seg,
    P_losses_estimation=P_loss_target,

    A_f_design=result["A_f"],
    L_design=result["L_total"]
)


print("\n================ DESIGN vs OFF-DESIGN ================\n")

print(f"Q design        = {result['Q_solution']/1e6:.3f} MW")
print(f"Q off-design    = {offdesign['Q_solution']/1e6:.3f} MW")

print()

print(f"L design        = {result['L_total']:.6f} m")
print(f"L off-design    = {offdesign['L_total']:.6f} m")

print()

print(f"A design        = {result['A_total']:.3f} m2")
print(f"A off-design    = {offdesign['A_total']:.3f} m2")

print()

print(f"DeltaT min design      = {result['delta_T_min']:.4f} K")
print(f"DeltaT min off-design  = {offdesign['delta_T_min']:.4f} K")

print()

print(f"Pressure loss hot design      = {100*result['P_losses_calc']:.4f} %")
print(f"Pressure loss hot off-design  = {100*offdesign['P_losses_calc']:.4f} %")

print()


print(f"Pressure loss cold design      = {100*result['P_losses_cold_calc']:.4f} %")
print(f"Pressure loss cold off-design  = {100*offdesign['P_losses_cold_calc']:.4f} %")

print()

print(f"T_cold,out design      = {result['T_cold'][-1]:.2f} K")
print(f"T_cold,out off-design  = {offdesign['T_cold'][-1]:.2f} K")

print(f"T_hot,out design           = {result['T_hot'][-1]:.2f} K")
print(f"T_hot,out off-design       = {offdesign['T_hot'][-1]:.2f} K")

print("\n================ epsilon / NTU =================\n")

for i in range(N_seg):

    print(
        f"Seg {i:2d} | "
        f"eps={offdesign['epsilon_seg'][i]:7.4f} | "
        f"NTU={offdesign['NTU_seg'][i]:8.3f} | "
        f"UA={offdesign['UA_seg'][i]/1e3:9.2f} kW/K | "
        f"dx={offdesign['delta_x'][i]:8.4f} m"
    )


# ============================================================
# OFF-DESIGN TEMPERATURE PROFILE
# ============================================================

# Physical x-coordinate
x_nodes_off = np.concatenate((
    [0.0],
    np.cumsum(offdesign["delta_x"])
))

# Both fluids in the SAME physical direction
T_cold_off = offdesign["T_cold"]
T_hot_off_spatial = offdesign["T_hot"][::-1]

plt.figure(figsize=(9, 5))

plt.plot(
    x_nodes_off,
    T_cold_off,
    marker="o",
    label="cold"
)

plt.plot(
    x_nodes_off,
    T_hot_off_spatial,
    marker="o",
    label="sCO2"
)

plt.xlabel("Physical position along HX [m]")
plt.ylabel("Temperature [K]")
plt.title("Off-design temperature profiles")

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# ============================================================
# OFF-DESIGN EFFECTIVENESS
# ============================================================

x_seg_off = 0.5 * (
    x_nodes_off[:-1]
    + x_nodes_off[1:]
)

plt.figure(figsize=(9, 5))

plt.plot(
    x_seg_off,
    offdesign["epsilon_seg"],
    marker="o"
)

plt.xlabel("Physical position along HX [m]")
plt.ylabel("Effectiveness, ε [-]")
plt.title("Off-design segment effectiveness")

plt.grid(True)

plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))

plt.plot(
    x_seg_off,
    offdesign["epsilon_seg"],
    marker="o",
    label="Segment effectiveness"
)

plt.axhline(
    y=1.0,
    linestyle="--",
    label="ε = 1"
)

plt.xlabel("Physical position along HX [m]")
plt.ylabel("Effectiveness, ε [-]")
plt.title("Off-design segment effectiveness")

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()



# ============================================================
# DESIGN vs OFF-DESIGN TEMPERATURE PROFILES
# ============================================================

# ----------------------------
# DESIGN coordinates
# ----------------------------

x_nodes_design = np.concatenate((
    [0.0],
    np.cumsum(result["delta_x"])
))

T_cold_design = result["T_cold"]
T_hot_design_spatial = result["T_hot"][::-1]


# ----------------------------
# OFF-DESIGN coordinates
# ----------------------------

x_nodes_off = np.concatenate((
    [0.0],
    np.cumsum(offdesign["delta_x"])
))

T_cold_off = offdesign["T_cold"]
T_hot_off_spatial = offdesign["T_hot"][::-1]



# ----------------------------
# Plot
# ----------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    x_nodes_design,
    T_cold_design,
    marker="o",
    label="cold - Design"
)

plt.plot(
    x_nodes_design,
    T_hot_design_spatial,
    marker="o",
    label="hot - Design"
)

plt.plot(
    x_nodes_off,
    T_cold_off,
    marker="s",
    linestyle="--",
    label="cold - Off-design"
)

plt.plot(
    x_nodes_off,
    T_hot_off_spatial,
    marker="s",
    linestyle="--",
    label="hot - Off-design"
)

plt.xlabel("Physical position along HX [m]")
plt.ylabel("Temperature [K]")
plt.title("Design vs Off-design Temperature Profiles")

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# DESIGN
x_nodes_design = np.concatenate((
    [0.0],
    np.cumsum(result["delta_x"])
))

x_seg_design = 0.5 * (
    x_nodes_design[:-1] + x_nodes_design[1:]
)

# OFF-DESIGN
x_nodes_off = np.concatenate((
    [0.0],
    np.cumsum(offdesign["delta_x"])
))

x_seg_off = 0.5 * (
    x_nodes_off[:-1] + x_nodes_off[1:]
)

plt.figure(figsize=(9,5))

plt.plot(
    x_seg_design,
    result["U_seg"],
    marker="o",
    label="Design"
)

plt.plot(
    x_seg_off,
    offdesign["U_seg"],
    marker="s",
    linestyle="--",
    label="Off-design"
)

plt.xlabel("Physical position along HX [m]")
plt.ylabel("U [W/m²-K]")
plt.title("Design vs Off-design Overall Heat Transfer Coefficient")

plt.grid()
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# DESIGN vs OFF-DESIGN REYNOLDS NUMBERS
# ============================================================

plt.figure(figsize=(9, 5))

# Cold side
plt.plot(x_seg_design, result["Re_cold"],
         marker="o", label="Re cold - Design")

plt.plot(x_seg_off, offdesign["Re_cold"],
         marker="s", linestyle="--",
         label="Re cold - Off-design")

# Hot side (reverse to match physical coordinate)
plt.plot(x_seg_design, result["Re_hot"][::-1],
         marker="o", label="Re hot - Design")

plt.plot(x_seg_off, offdesign["Re_hot"][::-1],
         marker="s", linestyle="--",
         label="Re hot - Off-design")

plt.xlabel("Physical position along HX [m]")
plt.ylabel("Re [-]")
plt.title("Design vs Off-design Reynolds Numbers")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()


# DESIGN
plt.plot(
    x_seg_design,
    result["h_conv_cold"],
    marker="o",
    label="h cold - Design"
)

plt.plot(
    x_seg_design,
    result["h_conv_hot"][::-1],
    marker="o",
    label="h hot - Design"
)

# OFF-DESIGN
plt.plot(
    x_seg_off,
    offdesign["h_conv_cold"],
    marker="s",
    linestyle="--",
    label="h cold - Off-design"
)

plt.plot(
    x_seg_off,
    offdesign["h_conv_hot"][::-1],
    marker="s",
    linestyle="--",
    label="h hot - Off-design"
)

plt.xlabel("Physical position along HX [m]")
plt.ylabel("h [W/m²-K]")
plt.title("Design vs Off-design Convective Heat Transfer Coefficients")

plt.grid()
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# DESIGN vs OFF-DESIGN DENSITIES
# ============================================================

plt.figure(figsize=(9, 5))

# Cold side
plt.plot(x_seg_design, result["rho_cold"],
         marker="o", label="rho cold - Design")

plt.plot(x_seg_off, offdesign["rho_cold"],
         marker="s", linestyle="--",
         label="rho cold - Off-design")

# Hot side (reverse to match physical coordinate)
plt.plot(x_seg_design, result["rho_hot"][::-1],
         marker="o", label="rho hot - Design")

plt.plot(x_seg_off, offdesign["rho_hot"][::-1],
         marker="s", linestyle="--",
         label="rho hot - Off-design")

plt.xlabel("Physical position along HX [m]")
plt.ylabel("Density [kg/m³]")
plt.title("Design vs Off-design Density Profiles")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
