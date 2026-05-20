from dataclasses import dataclass
from state import State
from properties import state_from_PS
from properties import state_from_PH


class Turbine:
    def __init__(self, eta, P_out):
        self.eta = eta
        self.P_out = P_out

    def solve(self, state_in):
        P2 = self.P_out
        s2s = state_in.s
        h2s = state_from_PS(P2, s2s).h
        W_dot_turb_s_per_m_dot = state_in.h - h2s
        W_dot_turb_per_m_dot = W_dot_turb_s_per_m_dot * self.eta

        h2 = state_in.h - W_dot_turb_per_m_dot
        state_out = state_from_PH(P=P2, h=h2)

        return state_out