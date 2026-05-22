from properties import CoolPropStorageFluid

class Tank:
    def __init__(self, T, mass_current = 0.0, mass_max = None):
        self.T = T
        self.mass_current = mass_current
        self.mass_max = mass_max

    def can_remove(self, mass):
        return self.mass_current >= mass

    def can_add(self, mass):
        if self.mass_max is None:
            return True
        return self.mass_current + mass <= self.mass_max

    def remove_mass(self, mass):
        if not self.can_remove(mass):
            raise ValueError("Not enough mass in tank.")
        self.mass_current -= mass

    def add_mass(self, mass):
        if not self.can_add(mass):
            raise ValueError("Tank capacity exceeded.")
        self.mass_current += mass

class TwoTankTES:
    def __init__(self, high_tank, low_tank, fluid):
        self.high_tank = high_tank
        self.low_tank = low_tank
        self.fluid = fluid

    def mass_required_for_energy(self, Q): #Receives the Q/m_dot*m_dot*t
        """
        Q [J]. Returns required storage-fluid mass [kg].
        """

        h_low = self.fluid.h(self.low_tank.T)
        h_high = self.fluid.h(self.high_tank.T)

        specific_energy = abs(h_high - h_low)  # J/kg

        return abs(Q) / specific_energy

    def move_low_to_high(self, mass):
        """
        Moves storage fluid from low-temperature tank to high-temperature tank.
        This means the TES gains thermal energy.
        """
        self.low_tank.remove_mass(mass)
        self.high_tank.add_mass(mass)

    def move_high_to_low(self, mass):
        """
        Moves storage fluid from high-temperature tank to low-temperature tank.
        This means the TES releases thermal energy.
        """
        self.high_tank.remove_mass(mass)
        self.low_tank.add_mass(mass)

    def exchange_heat(self, Q_to_tes):
        """
        Q_to_tes > 0: TES receives heat, mass moves low -> high.
        Q_to_tes < 0: TES releases heat, mass moves high -> low.
        """
        mass = self.mass_required_for_energy(Q_to_tes)

        if Q_to_tes > 0:  
            self.move_low_to_high(mass)
        elif Q_to_tes < 0: 
            self.move_high_to_low(mass)

        return mass
    

