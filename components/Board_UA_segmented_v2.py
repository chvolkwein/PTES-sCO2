
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


T_in_storage = 813 #658 #650.0 # K, temperature of storage fluid entering the heat exchanger
T_in_wf = 305 #290 #300.0 # K, temperature of working fluid entering the heat exchanger
N = 20


fluid_storage_minT = 288.1 # K, min temperature of Dowtherm A
T_h_in = T_in_storage
T_c_in = T_in_wf
mdot_storage = 96.
mdot_in_wf = 150.
fluid_storage = "Dowtherm A"
fluid_wf = 'CO2'
P_in_wf = 8e6*3 # Pa, pressure of working fluid entering the heat exchanger

def UA_segments(A_f, Q_tot, mdot_storage, mdot_in_wf, fluid_storage, fluid_wf, T_h_in, T_c_in, P_in_wf, N_seg,  P_losses, D, R_f, k_wall):
    """Heating the wf"""

    # A_f is the area of the storage fluid and the sco2 fluid

    def sam_prop_dowthermA_conductivity(T):
        """Returns the properties of Dowtherm A at a given temperature T in K."""
        # Thermal conductivity [W/m-K]
        k = max(0.1419 - 0.00016 * (T - 273.15), 0.001) #T in K
        return k
    
    # def sam_prop_dowthermA_viscosity(T):
    #     """Returns the properties of Dowtherm A at a given temperature T in K."""
    #     # Absolute viscosity [Pa-s]
    #     mu = 0.786512 * max((T - 273.15), 20) ** -1.44263 #T in K
    #     return mu
    
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

    h_h = np.zeros(N_seg + 1)
    h_c = np.zeros(N_seg + 1)
    T_h = np.zeros(N_seg + 1)
    T_c = np.zeros(N_seg + 1)

    h_h_in = esol_properties.enthalpy(fluid_storage, T=T_h_in, T_ref=fluid_storage_minT)
    h_c_in = fluid_properties.enthalpy(fluid_wf, T=T_c_in, P=P_in_wf)

    h_h[0] = h_h_in
    h_c[0] = h_c_in
    T_h[0] = T_h_in
    T_c[0] = T_c_in

    # I can't calculate yet the temperature of the working fluid bc pressure varies.
    for i in range(N_seg):
        h_h[i+1] = h_h[i] - (q_i / (mdot_storage))
        h_c[i+1] = h_c[i] + (q_i / (mdot_in_wf))

        T_h[i+1] = T_from_h_storage(h_h[i+1])
    

    #sco2 fluid
    P_wf = np.full(N_seg + 1, P_in_wf) #Assign inlet Pressure to all the array. 
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
            T_c[i] = fluid_properties.temperature(fluid_wf, h = h_c[i],P=P_wf[i] )


        # Storage: physical/spatial direction
        T_st_seg = 0.5 * (T_h[:-1] + T_h[1:])

        # WF: its own physical flow direction
        T_wf_seg = 0.5 * (T_c[:-1] + T_c[1:])
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
            vel_wf[i] = mdot_in_wf / (rho_wf[i] * A_f)
            mu_wf[i] = fluid_properties.viscosity(fluid = fluid_wf, T=T_wf_seg[i], P=P_wf_seg[i])
            Pr_wf[i] = fluid_properties.prandtl(fluid = fluid_wf, T=T_wf_seg[i], P=P_wf_seg[i])
            cp_wf[i] = fluid_properties.specheat(fluid=fluid_wf,  T=T_wf_seg[i], P=P_wf_seg[i] )
            Re_wf[i] = (rho_wf[i] * vel_wf[i] * D) / mu_wf[i]
            C_f_wf[i] = (1.58*np.log(Re_wf[i]) - 3.28)**(-2) #Check this! Fanning friction factor
            Nu_wf[i] = (0.5*C_f_wf[i]*Pr_wf[i]*(Re_wf[i]-1000))/(1 + 12.7*(((C_f_wf[i])/2)**(1/2))*(Pr_wf[i]**(2/3) - 1))
            k_wf[i] = fluid_properties.conductivity(fluid = fluid_wf, T=T_wf_seg[i], P=P_wf_seg[i])
            h_conv_wf[i] = (Nu_wf[i]*k_wf[i])/D

        hconv_wf_spatial = h_conv_wf[::-1]
        T_c_seg_spatial = T_wf_seg[::-1]
        P_wf_spatial = P_wf_seg[::-1]
        delta_T_seg = T_st_seg - T_c_seg_spatial

        for i in range(N_seg):
            U_seg[i] = 1/(1/h_conv_st[i] + 1/hconv_wf_spatial[i])
            A_seg[i] = q_i/(U_seg[i]*delta_T_seg[i])
            delta_x[i] = A_seg[i]*D/(4*A_f)
        
        dx_wf = delta_x[::-1]
        P_wf_new = np.zeros(N_seg + 1)
        P_wf_new[0] = P_in_wf

        for i in range(N_seg):
            delta_p[i] = 4*C_f_wf[i]*(dx_wf[i]/D)*(rho_wf[i]*vel_wf[i]**2)/2
            P_wf_new[i+1] = P_wf_new[i] - delta_p[i]

        #Converge
        if np.allclose(P_wf_new, P_wf, rtol=1e-6, atol=1.0):
            P_wf = P_wf_new
            break
        
        P_wf = 0.5 * P_wf + 0.5 * P_wf_new

    else:
        print("Warning: pressure iteration did not converge")
    
    T_c_spatial_nodes = T_c[::-1]
    delta_T_nodes = T_h - T_c_spatial_nodes
    delta_T_min = np.min(delta_T_nodes)

    delta_p_total = P_in_wf - P_wf[-1]
    P_losses_calc = delta_p_total / P_in_wf
    L_total = np.sum(delta_x)
    A_total = np.sum(A_seg)
    UA_seg = U_seg * A_seg
    UA_total = np.sum(UA_seg)

    return {
    "P_wf": P_wf,
    "T_h": T_h,
    "T_c": T_c,
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
    "U_seg": U_seg,
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


Q_storage_max = mdot_storage*(esol_properties.enthalpy(fluid = fluid_storage, T=T_h_in, T_ref=fluid_storage_minT) - esol_properties.enthalpy(fluid = fluid_storage, T=T_c_in, T_ref=fluid_storage_minT))
Q_wf_max = mdot_in_wf*(fluid_properties.enthalpy(fluid = fluid_wf, T=T_h_in, P=P_in_wf) - fluid_properties.enthalpy(fluid = fluid_wf, T=T_c_in, P=P_in_wf))
Q_max = min(Q_storage_max, Q_wf_max)

# result = UA_segments(
#     A_f=0.05,                # m2, guess inicial
#     Q_tot=1e6,               # W, empieza bajo para evitar crossing
#     mdot_storage=100,        # kg/s
#     mdot_in_wf=100,          # kg/s
#     fluid_storage=fluid_storage,
#     fluid_wf=fluid_wf,
#     T_h_in=673.15,           # K
#     T_c_in=320.0,            # K
#     P_in_wf=10e6,            # Pa
#     N_seg=20,
#     P_losses=0.01,
#     D=0.005,                 # m
#     R_f=0.0,
#     k_wall=15.0
# )

# print("Iterations =", result["iterations"])

# print("P loss =", result["P_losses_calc"] * 100, "%")
# print("Delta P =", result["delta_p_total"] / 1e3, "kPa")

# print("L =", result["L_total"], "m")
# print("A =", result["A_total"], "m2")
# print("UA =", result["UA_total"] / 1e6, "MW/K")

# print("Delta T min =", result["delta_T_min"], "K")

# print("P_wf =", result["P_wf"])
# print("T_h =", result["T_h"])
# print("T_c =", result["T_c"])
# print("delta_x =", result["delta_x"])

# print("Any NaN U? ", np.any(np.isnan(result["U_seg"])))
# print("Any negative A? ", np.any(result["A_seg"] <= 0))
# print("Any negative dx?", np.any(result["delta_x"] <= 0))
# print("Pressure always decreases?", np.all(np.diff(result["P_wf"]) < 0))
# print("Hot fluid cools?", np.all(np.diff(result["T_h"]) < 0))
# print("Cold fluid heats?", np.all(np.diff(result["T_c"]) > 0))


A_f = 0.05 # m2, guess inicial
N_seg = 20
P_losses = 0.01
D = 0.005 # m
R_f = 0.0
k_wall = 0.0
delta_T_min_target = 10.0

Q_low = 1e2
Q_high = Q_max

def residual_Q(Q_tot):

    result = UA_segments(
        A_f=A_f,
        Q_tot=Q_tot,
        mdot_storage=mdot_storage,
        mdot_in_wf=mdot_in_wf,
        fluid_storage=fluid_storage,
        fluid_wf=fluid_wf,
        T_h_in=T_h_in,
        T_c_in=T_c_in,
        P_in_wf=P_in_wf,
        N_seg=N_seg,
        P_losses=P_losses,
        D=D,
        R_f=R_f,
        k_wall=k_wall
    )

    return result["delta_T_min"] - delta_T_min_target


Q_solution = brentq(
    residual_Q,
    Q_low,
    Q_high
)

print("Q solution =", Q_solution / 1e6, "MW")

# Evaluate HX at the final solution
result = UA_segments(
    A_f=A_f,
    Q_tot=Q_solution,
    mdot_storage=mdot_storage,
    mdot_in_wf=mdot_in_wf,
    fluid_storage=fluid_storage,
    fluid_wf=fluid_wf,
    T_h_in=T_h_in,
    T_c_in=T_c_in,
    P_in_wf=P_in_wf,
    N_seg=N_seg,
    P_losses=P_losses,
    D=D,
    R_f=R_f,
    k_wall=k_wall
)

print("Delta T min =", result["delta_T_min"], "K")
print("P loss =", result["P_losses_calc"] * 100, "%")
print("Delta P =", result["delta_p_total"] / 1e3, "kPa")
print("L =", result["L_total"], "m")
print("A =", result["A_total"], "m2")
print("UA =", result["UA_total"] / 1e6, "MW/K")
print("Iterations =", result["iterations"])

print("T hot out =", result["T_h"][-1], "K")
print("T cold out =", result["T_c"][-1], "K")
print("TTD high T side =", result["T_h"][0] - result["T_c"][-1], "K")
print("TTD low T side =", result["T_h"][-1] - result["T_c"][0], "K")

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
T_storage_spatial = result["T_h"]
T_wf_spatial = result["T_c"][::-1]

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
C_st = mdot_storage * result["cp_st"]
C_wf = mdot_in_wf * result["cp_wf"]

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