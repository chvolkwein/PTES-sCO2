import PTES_sCO2_HX_solver_off_comp_PRchg_PRdis as p

PR_chg = 2.146136
PR_dis = 2.067154

chg = p.charge(PR_chg)
dis = p.discharge(PR_dis, 30.0, chg)

error_CH = dis["T_TES_CH_new"] - p.T_TES_CH_ini
error_HL = dis["T_TES_HL_new"] - p.T_TES_HL_ini

Wchg = chg["W_comp"] - chg["W_turb"]
Wdis = dis["W_turb"] - dis["W_comp"]

print("\n===== RESULTS =====")
print(f"PR_chg = {PR_chg:.6f}")
print(f"PR_dis = {PR_dis:.6f}")
print(f"CH error = {error_CH:.6f} K")
print(f"HL error = {error_HL:.6f} K")
print(f"RTE = {Wdis/Wchg:.6f}")
print(f"W_net_chg = {Wchg/1000:.3f} kJ/kg")
print(f"W_net_dis = {Wdis/1000:.3f} kJ/kg")
print(f"T_TES_HH = {chg['T_TES_HH']:.3f} K")
print(f"T_TES_CL = {chg['T_TES_CL']:.3f} K")

if max(abs(error_CH), abs(error_HL)) > 0.1:
    print("WARNING: TES closure exceeds 0.1 K")