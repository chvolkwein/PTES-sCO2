
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

esol_properties = Incompressible()


T_in_storage = 650.0 # K, temperature of storage fluid entering the heat exchanger
T_in_wf = 300.0 # K, temperature of working fluid entering the heat exchanger
N = 20


fluid_storage_minT = 288.1 # K, min temperature of Dowtherm A
T_h_in = T_in_storage
T_c_in = T_in_wf
mdot_storage = 96.
mdot_in_wf = 120.
fluid_storage = "Dowtherm A"
fluid_wf = 'CO2'
P_in_wf = 8e6 # Pa, pressure of working fluid entering the heat exchanger
eff = 0.98


Q_storage_max = mdot_storage*(esol_properties.enthalpy(fluid = fluid_storage, T=T_h_in, T_ref=fluid_storage_minT) - esol_properties.enthalpy(fluid = fluid_storage, T=T_c_in, T_ref=fluid_storage_minT))
Q_wf_max = mdot_in_wf*(fluid_properties.enthalpy(fluid = fluid_wf, T=T_h_in, P=P_in_wf) - fluid_properties.enthalpy(fluid = fluid_wf, T=T_c_in, P=P_in_wf))
Q_max = min(Q_storage_max, Q_wf_max)

Q_design = eff * Q_max
q_i = Q_design / N

h_conv_storage = 1000.
h_conv_wf = 1000.

h_h = np.zeros(N + 1)
h_c = np.zeros(N + 1)
T_h = np.zeros(N + 1)
T_c = np.zeros(N + 1)

h_h_in = esol_properties.enthalpy(fluid_storage, T=T_h_in, T_ref=fluid_storage_minT)
h_c_in = fluid_properties.enthalpy(fluid_wf, T=T_c_in, P=P_in_wf)

h_h[0] = h_h_in
h_c[0] = h_c_in
T_h[0] = T_h_in
T_c[0] = T_c_in

def T_from_h_storage(h):
    x = h / 1e6

    T = (
        -27.06574572*x**6
        +151.21088901*x**5
        -340.13167310*x**4
        +411.02036213*x**3
        -388.55934518*x**2
        +645.42026919*x
        +288.26451925
    )

    return T


for i in range(N):
    h_h[i+1] = h_h[i] - (q_i / (mdot_storage))
    h_c[i+1] = h_c[i] + (q_i / (mdot_in_wf))

    T_h[i+1] = T_from_h_storage(h_h[i+1])
    T_c[i+1] = fluid_properties.temperature(fluid_wf, h = h_c[i+1],P=P_in_wf )


# node:          0        1        2      ...       N

# hot:          ---> ---> ---> ---> ---> --->

# cold:         <--- <--- <--- <--- <--- <---

# arrays:
# T_h           node 0 ----------------> node N
# T_c_spatial   node 0 ----------------> node N

T_c_spatial = T_c[::-1]
h_c_spatial = h_c[::-1]
delta_T = T_h - T_c_spatial
print("T_h:", T_h)
print("T_c:", T_c_spatial)
print("delta_T:", delta_T)

C_h_seg = np.zeros(N)
C_c_seg = np.zeros(N)
C_min_seg = np.zeros(N)
C_max_seg = np.zeros(N)
C_r_seg = np.zeros(N)

for i in range(N):

    C_h_seg[i] = mdot_storage*(h_h[i]-h_h[i+1]) / (T_h[i] - T_h[i+1])

    C_c_seg[i] = mdot_in_wf*(h_c_spatial[i]-h_c_spatial[i+1]) / (
        T_c_spatial[i] - T_c_spatial[i+1]
    )

    C_min_seg[i] = min(C_h_seg[i], C_c_seg[i])
    C_max_seg[i] = max(C_h_seg[i], C_c_seg[i])

    C_r_seg[i] = C_min_seg[i] / C_max_seg[i]

epsilon_seg = np.zeros(N)
NTU_seg = np.zeros(N)
UA_seg = np.zeros(N)

for i in range(N):

    Q_max_i = C_min_seg[i] * (
        T_h[i] - T_c_spatial[i+1]
    )

    epsilon_seg[i] = q_i / Q_max_i

    if np.isclose(C_r_seg[i], 1.0):
        NTU_seg[i] = epsilon_seg[i] / (1.0 - epsilon_seg[i])

    else:
        NTU_seg[i] = (
            1.0 / (1.0 - C_r_seg[i])
        ) * np.log(
            (1.0 - epsilon_seg[i] * C_r_seg[i]) /
            (1.0 - epsilon_seg[i])
        )

    UA_seg[i] = NTU_seg[i] * C_min_seg[i]

UA_total = np.sum(UA_seg)

print("UA_total:", UA_total)
print("UA_seg:", UA_seg)
print("NTU_seg:", NTU_seg)
print("epsilon_seg:", epsilon_seg)
print("C_r_seg:", C_r_seg)
print("C_min_seg:", C_min_seg)

# print("C_h:", C_h)
# print("C_c:", C_c)


# print("h_h:", h_h)
# print("h_c:", h_c)






#######################
####Curve fit
#######################

# P_wf = 8e6 # Pa, pressure of working fluid
# T_ref_storage = 288.1 # K, reference temperature for storage fluid enthalpy calculations
# T_storage = np.linspace(288.1, 900, 100)   # K
# T_wf = np.linspace(288.1, 900, 100)        # K

# h_storage = np.array([
#     esol_properties.enthalpy(
#         fluid=fluid_storage,
#         T=T,
#         T_ref=T_ref_storage
#     )
#     for T in T_storage
# ])

# h_wf = np.array([
#     fluid_properties.enthalpy(
#         fluid=fluid_wf,
#         T=T,
#         P=P_wf
#     )
#     for T in T_wf
# ])

# print("Storage:")
# print(np.column_stack((T_storage, h_storage)))

# print("\nWorking fluid:")
# print(np.column_stack((T_wf, h_wf)))

# np.savetxt(
#     "storage_T_h.csv",
#     np.column_stack((T_storage, h_storage)),
#     delimiter=",",
#     header="T,h",
#     comments=""
# )

# np.savetxt(
#     "wf_T_h.csv",
#     np.column_stack((T_wf, h_wf)),
#     delimiter=",",
#     header="T,h",
#     comments=""
# )

# ============================================================
# PLOTS
# ============================================================

# Spatial coordinate:
# x = 0 -> hot-fluid inlet
# x = 1 -> hot-fluid outlet
x = np.linspace(0, 1, N + 1)

# Cold-fluid arrays were generated following the cold-fluid flow direction.
# Reverse them to put both fluids in the same physical coordinate.
T_c_spatial = T_c[::-1]
h_c_spatial = h_c[::-1]

# ------------------------------------------------------------
# 1. Temperature profiles
# ------------------------------------------------------------

plt.figure()

plt.plot(x, T_h, marker='o', label='Storage fluid')
plt.plot(x, T_c_spatial, marker='o', label='Working fluid')

plt.xlabel('Normalized HX position')
plt.ylabel('Temperature [K]')
plt.title('Heat Exchanger Temperature Profiles')

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 2. Local temperature difference
# ------------------------------------------------------------

delta_T = T_h - T_c_spatial

plt.figure()

plt.plot(x, delta_T, marker='o')

plt.xlabel('Normalized HX position')
plt.ylabel(r'$\Delta T$ [K]')
plt.title('Local Temperature Difference')

plt.grid(True)
plt.tight_layout()
plt.show()

print("Minimum delta T =", np.min(delta_T), "K")
print("Pinch node =", np.argmin(delta_T))


# ------------------------------------------------------------
# 3. Enthalpy profiles
# ------------------------------------------------------------

plt.figure()

plt.plot(x, h_h / 1000, marker='o', label='Storage fluid')
plt.plot(x, h_c_spatial / 1000, marker='o', label='Working fluid')

plt.xlabel('Normalized HX position')
plt.ylabel('Specific enthalpy [kJ/kg]')
plt.title('Heat Exchanger Enthalpy Profiles')

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 4. UA profiles
# ------------------------------------------------------------

plt.figure()
x_seg = 0.5 * (x[:-1] + x[1:])
plt.plot(x_seg, UA_seg, marker='o')

plt.xlabel('Normalized HX position')
plt.ylabel('UA [W/K]')
plt.title('UA Distribution')

plt.grid(True)
plt.tight_layout()
plt.show()

if np.any(epsilon_seg <= 0) or np.any(epsilon_seg >= 1):
    print("WARNING: epsilon outside physical range")

if np.any(C_min_seg <= 0):
    print("WARNING: non-positive C_min")
