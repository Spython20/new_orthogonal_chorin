import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config import *
from egorov import MC_sample
from vel_verlet import flow_vel_verlet
from chorin import chorin_step_one, chorin_step_two

output_directory = Path.cwd() / "phase_space_plots"
output_directory.mkdir(exist_ok=True)


def position_1_observable(q1, q2, p1, p2):
    return q1


def position_2_observable(q1, q2, p1, p2):
    return q2


def momentum_1_observable(q1, q2, p1, p2):
    return p1


def momentum_2_observable(q1, q2, p1, p2):
    return p2


def kinetic_energy_observable(q1, q2, p1, p2):
    return 0.5 * (p1**2 + p2**2)


def potential_energy_observable(q1, q2, p1, p2):
    return 0.5 * (q1**2 + q2**2) + lamb * (q1**2 * q2 - q2**3 / 3.0)


def total_energy_observable(q1, q2, p1, p2):
    return (kinetic_energy_observable(q1, q2, p1, p2)
            + potential_energy_observable(q1, q2, p1, p2))


observables = {
    "q1": position_1_observable,
    "q2": position_2_observable,
    "p1": momentum_1_observable,
    "p2": momentum_2_observable,
    "kinetic": kinetic_energy_observable,
    "potential": potential_energy_observable,
    "total": total_energy_observable,
}

# Chorin: estimate coefficients on an independent first batch.
rng_first = np.random.default_rng(42)
q1_first, q2_first, p1_first, p2_first = MC_sample(N, rng_first)
flow_first = flow_vel_verlet(q1_first, q2_first, p1_first, p2_first, dt, n_steps)

# Compute Egorov and corrected Chorin estimates on the SAME second batch.
rng_second = np.random.default_rng(43)
q1_second, q2_second, p1_second, p2_second = MC_sample(N, rng_second)
flow_second = flow_vel_verlet(q1_second, q2_second, p1_second, p2_second, dt, n_steps)

results = {}
for name, observable in observables.items():
    first_values = observable(*flow_first)
    coefficients = chorin_step_one(first_values,
                                   q1_first, q2_first, p1_first, p2_first)
    second_values = observable(*flow_second)
    egorov_values = np.mean(second_values, axis=1)
    chorin_values = chorin_step_two(second_values, coefficients,
                                   q1_second, q2_second, p1_second, p2_second)
    results[name] = (egorov_values, chorin_values)

times = np.arange(n_steps + 1) * dt


def plot_observable(name, ylabel, title, filename):
    egorov_values, chorin_values = results[name]
    plt.figure()
    plt.plot(times, egorov_values, label="Egorov")
    plt.plot(times, chorin_values, "--", label="Chorin")
    plt.xlabel("time")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_directory / filename, dpi=200)
    plt.close()


plot_observable("q1", r"$\langle q_1 \rangle$", "Position expectation", "position_q1.png")
plot_observable("q2", r"$\langle q_2 \rangle$", "Position expectation", "position_q2.png")
plot_observable("p1", r"$\langle p_1 \rangle$", "Momentum expectation", "momentum_p1.png")
plot_observable("p2", r"$\langle p_2 \rangle$", "Momentum expectation", "momentum_p2.png")

plt.figure(figsize=(9, 6))
for name, label in (("kinetic", "kinetic"), ("potential", "potential"), ("total", "total")):
    egorov_values, chorin_values = results[name]
    plt.plot(times, egorov_values, label=f"Egorov {label}")
    plt.plot(times, chorin_values, "--", label=f"Chorin {label}")
plt.xlabel("time")
plt.ylabel("energy expectation")
plt.title("Energy expectations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(output_directory / "energy.png", dpi=200)
plt.close()

print(f"Saved dynamics plots to {output_directory}")

