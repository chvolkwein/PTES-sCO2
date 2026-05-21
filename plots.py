import numpy as np
import matplotlib.pyplot as plt
from properties import state_from_TP


def get_isobaric_path(state_a, state_b, n=50):
    """
    Creates a constant-pressure path between state_a and state_b.
    Uses pressure from state_a.
    """
    T_values = np.linspace(state_a.T, state_b.T, n)
    P = state_a.P

    states = [state_from_TP(T=T, P=P) for T in T_values]

    s_values = [st.s for st in states]
    T_values = [st.T for st in states]

    return s_values, T_values


def plot_ts_diagram(charge_results, discharge_results):
    plt.figure(figsize=(9, 6))

    for results, label, linestyle in [
        (charge_results, "Charge", "-"),
        (discharge_results, "Discharge", "--"),
    ]:
        states = results["states"]

        s1, T1 = states["1"].s, states["1"].T
        s2, T2 = states["2"].s, states["2"].T
        s3, T3 = states["3"].s, states["3"].T
        s4, T4 = states["4"].s, states["4"].T

        # Compressor: 1 -> 2
        plt.plot([s1, s2], [T1, T2], linestyle=linestyle)

        # High-side HX: 2 -> 3, constant high pressure
        s_hx_high, T_hx_high = get_isobaric_path(states["2"], states["3"])
        plt.plot(s_hx_high, T_hx_high, linestyle=linestyle)

        # Turbine: 3 -> 4
        plt.plot([s3, s4], [T3, T4], linestyle=linestyle)

        # Low-side HX: 4 -> 1, constant low pressure
        s_hx_low, T_hx_low = get_isobaric_path(states["4"], states["1"])
        plt.plot(s_hx_low, T_hx_low, linestyle=linestyle, label=label)

        # State points
        for k in ["1", "2", "3", "4"]:
            plt.scatter(states[k].s, states[k].T)
            plt.annotate(
                f"{k}{label[0].lower()}",
                (states[k].s, states[k].T),
                textcoords="offset points",
                xytext=(6, 6),
            )

    plt.xlabel("Entropy, s [J/kg-K]")
    plt.ylabel("Temperature [K]")
    plt.title("T-s Diagram")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()