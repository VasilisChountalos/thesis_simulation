import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pathlib import Path
# Parameters
p_min, p_max = -0.01, 0.01
J      = 0.90
gamma  = 0.08               # per month
I0     = 0.02/12            # per month (2%/yr)
Np     = 3000              # number of p‑grid points
p_grid = np.linspace(p_min, p_max, Np)
dp     = (p_max - p_min)/(Np - 1)

# Uniform initial PDF
P = np.ones(Np)/(p_max - p_min)

# Storage for inflation rate and time
I_values   = []
t_values   = []
t          = 0.0
cfl        = 0.4            # Courant number for stability
max_steps  = 500_000

# Time‑stepping loop
for step in range(max_steps):
    # Current mean price and P(p_-)
    p_mean = np.sum(p_grid * P) * dp
    P_min  = P[0]

    # Mean‑field inflation rate I(t) from (14)
    denom  = 1 - J*(p_max - p_min)*P_min
    if denom <= 0:
        raise RuntimeError("Instability: 1 - J*(p_max-p_min)*P_min <= 0")
    I      = (I0 + J*gamma*(p_max - p_mean))/denom

    I_values.append(I)
    t_values.append(t)

    # Choose time step to satisfy CFL: dt ≤ cfl*dp/|I|
    dt = cfl * dp/max(abs(I),1e-12)
    t  += dt

    # Allocate next PDF and update interior points using forward difference
    P_new = np.empty_like(P)
    dPdp  = (P[1:] - P[:-1])/dp           # forward derivative
    P_new[:-1] = P[:-1] + dt*(I*dPdp - gamma*P[:-1])

    # Reinjection at p_max using the updated P_new[0]
    P_new[-1] = P_new[0] + gamma/I

    # Renormalize so that ∫P(p)dp = 1
    P_new /= np.sum(P_new)*dp
    P = P_new

    # Stop if I(t) has converged
    if step > 2000 and abs(I_values[-1] - I_values[-2]) < 1e-12:
        break

# Analyse stationary distribution
I_final       = I_values[-1]
p_mean_final  = np.sum(p_grid * P)*dp
P_theory      = np.exp((gamma/I_final)*p_grid)
P_theory     /= np.sum(P_theory)*dp

# Narrow-band mean-field stationary inflation and slope
I_star        = I0/(1 - J)               
slope_star    = gamma / I_star           
P_theory_star = np.exp((gamma/I_star)*p_grid)  
P_theory_star/= np.sum(P_theory_star)*dp

# Empirical slope and theoretical slope
mask = (p_grid > p_min+0.002) & (p_grid < p_max-0.002)
slope_empirical, *_ = linregress(p_grid[mask], np.log(P[mask]))
slope_theory = gamma/I_final

print(f"Theoretical I_st = {I0/(1-J):.6f}")
print(f"Simulated  I_st = {I_final:.6f}")
print(f"I_st Error = {abs((I_final-(I0/(1-J))))/(I0/(1-J)):.6f}")
print(f"I_st Error Percenatge= {abs((I_final-(I0/(1-J))))/(I0/(1-J))*100:.6f}%")
print(f"Mean price     = {p_mean_final:.6f}")
print(f"Empirical slope = {slope_empirical:.3f}, Expected slope = {slope_theory:.3f}, Theoritical slope={slope_star:.3f}")
print(f"P(p_-) = {P[0]:.5e},  P(p_+) = {P[-1]:.5e}")


plt.semilogy(p_grid, P, label=f'Simulated={slope_empirical:.3f}')
plt.semilogy(p_grid, P_theory, '--', label=f'Analytical (exp)={slope_theory:.3f}')
plt.semilogy(p_grid, P_theory_star, ':', label=f"Analytical via I* = I0/(1-J) (γ/I* = {slope_star:.3f})")
plt.xlabel("Price p")
plt.ylabel("P(p) [log-scale]")
plt.title("Stationary Price Distribution (Semilog)")
plt.legend()
plt.grid(True)

plt.tight_layout()
Path("figs").mkdir(exist_ok=True)                         # ensure folder exists
plt.savefig("J=0.9,gamma=0.08, band=0.02(b).png", dpi=300, bbox_inches="tight")  # save PNG
plt.show()
# Convert time to days
days = np.array(t_values) * 30

# Plot I(t)
plt.figure()
plt.plot(days, I_values, label="Inflation rate I(t)")
plt.axvline(days[-1], color='red', linestyle='--', label=f"Stopped at day {days[-1]:.1f}")
plt.xlabel("Time [days]")
plt.ylabel("Inflation rate I(t)")
plt.title("Inflation Convergence Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
Path("figs").mkdir(exist_ok=True)                         # ensure folder exists
plt.savefig("J=0.9,gamma=0.08, band=0.02(c).png", dpi=300, bbox_inches="tight")  # save PNG
plt.show()

# Print final step
print(f"\nSimulation stopped at:")
print(f"  Step         = {step}")
print(f"  Time         = {t:.6f} months")
print(f"  Time         = {t * 30:.2f} days")

