
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

fluid = 'INCOMP::NaK'
fluid_ees = "Salt (60 NaNO3, 40 KNO3)"
P_ref = 101325
T_low = 305 + 273
T_high = 550 + 273
T_range = np.linspace(T_low, T_high, 10)
print(T_range)

cp_coolprops = fluid_properties.specheat(fluid, T = T_range, P = P_ref)
print("cp_coolprops", cp_coolprops)

cp_ees = esol_properties.specheat(fluid_ees, T = T_range)
print("cp_ees", cp_ees)

