import matplotlib.pyplot as plt


def plot_ts_diagram(charge_results, discharge_results, relative_entropy=True, save_path=None):
    charge_states = charge_results["states"]
    discharge_states = discharge_results["states"]

    order = ["1", "2", "3", "4", "1"]

    s_charge = [charge_states[k].s for k in order]
    T_charge = [charge_states[k].T for k in order]

    s_discharge = [discharge_states[k].s for k in order]
    T_discharge = [discharge_states[k].T for k in order]

    if relative_entropy:
        s_charge_ref = charge_states["1"].s
        s_discharge_ref = discharge_states["1"].s

        s_charge = [s - s_charge_ref for s in s_charge]
        s_discharge = [s - s_discharge_ref for s in s_discharge]

        xlabel = r"Entropy change, $s - s_1$ [J/kg-K]"
    else:
        xlabel = r"Entropy, $s$ [J/kg-K]"

    plt.figure(figsize=(8, 6))

    plt.plot(s_charge, T_charge, marker="o", linestyle="-", label="Carga")
    plt.plot(s_discharge, T_discharge, marker="o", linestyle="--", label="Descarga")

    # Etiquetas de estados
    for k in ["1", "2", "3", "4"]:
        if relative_entropy:
            x_c = charge_states[k].s - charge_states["1"].s
            x_d = discharge_states[k].s - discharge_states["1"].s
        else:
            x_c = charge_states[k].s
            x_d = discharge_states[k].s

        y_c = charge_states[k].T
        y_d = discharge_states[k].T

        plt.annotate(f"{k}c", (x_c, y_c), textcoords="offset points", xytext=(5, 5))
        plt.annotate(f"{k}d", (x_d, y_d), textcoords="offset points", xytext=(5, -10))

    plt.xlabel(xlabel)
    plt.ylabel("Temperature [K]")
    plt.title("T-s Diagram")
    plt.grid(True)
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()