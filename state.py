from dataclasses import dataclass

@dataclass
class State:
    T: float      # K
    P: float      # Pa
    h: float      # J/kg
    s: float      # J/kg-K
    v: float      # m3/kg