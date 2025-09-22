import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pathlib import Path

def run_simulation(Np, p_min=-0.01, p_max=0.01, J=0.90, gamma=0.08, I0=0.02/12, cfl=0.4, max_steps=100_000):
    I_theory = I0 / (1 - J)
    p_grid = np.linspace(p_min, p_max, Np)
    dp = (p_max - p_min) / (Np - 1)
    P = np.ones(Np) / (p_max - p_min)
    I_values = []
    t = 0.0

    for step in range(max_steps):
        p_mean = np.sum(p_grid * P) * dp
        P_min = P[0]
        denom = 1 - J * (p_max - p_min) * P_min
        if denom <= 0:
            break
        I = (I0 + J * gamma * (p_max - p_mean)) / denom
        I_values.append(I)
        dt = cfl * dp / max(abs(I), 1e-12)
        t += dt

        dPdp = (P[1:] - P[:-1]) / dp
        P_new = np.empty_like(P)
        P_new[:-1] = P[:-1] + dt * (I * dPdp - gamma * P[:-1])
        P_new[-1] = P_new[0] + gamma / I
        P_new /= np.sum(P_new) * dp
        P = P_new

        if step > 2000 and abs(I_values[-1] - I_values[-2]) < 1e-12:
            break

    I_final = I_values[-1]
    rel_error_I = abs(I_final - I_theory) / I_theory

    # Slope analysis
    mask = (p_grid > p_min + 0.002) & (p_grid < p_max - 0.002)
    slope_empirical, *_ = linregress(p_grid[mask], np.log(P[mask]))
    slope_theory = gamma / I_final
    rel_error_slope = abs(slope_empirical - slope_theory) / abs(slope_theory)

    return rel_error_I, rel_error_slope   # ✅ Correct return

# Define grid sizes
Np_values = np.arange(500, 5001, 500)

errors_I = []
errors_slope = []

for Np in Np_values:
    err_I, err_slope = run_simulation(Np)
    errors_I.append(err_I)
    errors_slope.append(err_slope)
    print(f"Np={Np}, Rel_Error_I={err_I:.4e}, Rel_Error_Slope={err_slope:.4e}")

# Plot
plt.figure()
plt.plot(Np_values, errors_I, 'o-', label="Relative Error in I_final")
plt.plot(Np_values, errors_slope, 's-', label="Relative Error in Slope")
plt.xlabel("Np (Grid Points)")
plt.ylabel("Relative Error")
plt.title("Error vs Grid Resolution")
plt.legend()
plt.grid(True)
plt.tight_layout()
Path("figs").mkdir(exist_ok=True)                         # ensure folder exists
plt.savefig("J=0.9,gamma=0.08, band=0.02.png", dpi=300, bbox_inches="tight")  # save PNG
plt.show()

