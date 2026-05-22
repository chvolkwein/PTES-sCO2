from CoolProp.CoolProp import PropsSI
from state import State

FLUID = "CO2"

def state_from_TP(T: float, P: float) -> State:
    h = PropsSI("H", "T", T, "P", P, FLUID)
    s = PropsSI("S", "T", T, "P", P, FLUID)
    rho = PropsSI("D", "T", T, "P", P, FLUID)
    v = 1/rho

    return State(T=T, P=P, h=h, s=s, v=v)


def state_from_PS(P: float, S: float) -> State:
    h = PropsSI("H", "S", S, "P", P, FLUID)
    t = PropsSI("T", "S", S, "P", P, FLUID)
    rho = PropsSI("D", "S", S, "P", P, FLUID)
    v = 1/rho

    return State(T=t, P=P, h=h, s=S, v=v)

def state_from_PH(P: float, H: float) -> State:
    s = PropsSI("S", "H", H, "P", P, FLUID)
    t = PropsSI("T", "H", H, "P", P, FLUID)
    rho = PropsSI("D", "H", H, "P", P, FLUID)
    v = 1/rho

    return State(T=t, P=P, h=H, s=s, v=v)



class CoolPropStorageFluid:
    def __init__(self, fluid_name, P_ref=101325.0):
        self.fluid_name = fluid_name
        self.P_ref = P_ref

    def cp(self, T):
        return PropsSI("C", "T", T, "P", self.P_ref, self.fluid_name)

    def rho(self, T):
        return PropsSI("D", "T", T, "P", self.P_ref, self.fluid_name)

    def h(self, T):
        return PropsSI("H", "T", T, "P", self.P_ref, self.fluid_name)

class EESStorageFluid:
    def __init__(self, fluid_name):
        self.fluid_name = fluid_name

    def rho(self, T):
        return 4.79*T-2616.15

    def h(self, T):
        return 2855.58*T-1455264
