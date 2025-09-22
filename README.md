# Thesis Simulation

Agent-based and mean-field simulations for my thesis on price dynamics and inflation.

## Contents

- `src/abm_sawtooth.py` – **Agent-based saw-tooth model**  
  N firms’ log-prices drift down inside a band `[p_-, p_+]`, then reset (Poisson with rate γ or forced at the floor).  
  A mean-field feedback updates the inflation rate  
  \[
    I = I_0 + J\big(\gamma (p_+ - \langle p\rangle) + \text{flux}_{\text{forced}}\cdot (p_+ - p_-)\big).
  \]  
  After burn-in it estimates stationary \(I^*\), builds a time-averaged histogram, fits the **semi-log PDF** interior, and saves:
  - `figs/J=..., ...(traj_zoom).png` (zoomed saw-tooth trajectories)  
  - `figs/J=..., ...(semilog).png` (semi-log stationary PDF with fitted & theoretical slopes)

- `src/mf_grid_convergence.py` – **Mean-field PDE: grid-resolution study**  
  Solves the master equation on a uniform grid and plots **relative errors vs. Np** for
  - stationary inflation \(I^*\) (vs \(I_0/(1-J)\)),
  - semi-log slope \(\gamma/I^*\).

- `src/mf_stationary_semilog.py` – **Mean-field PDE: stationary PDF & I(t)**  
  Single high-resolution run; compares the simulated stationary PDF against analytics
  (using simulated \(I^*\) and narrow-band \(I_0/(1-J)\)) and plots \(I(t)\) convergence.

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
# Thesis Simulation

Agent-based and mean-field simulations for my thesis on price dynamics and inflation.

## Contents

- `src/abm_sawtooth.py` – **Agent-based saw-tooth model**  
  N firms’ log-prices drift down inside a band `[p_-, p_+]`, then reset (Poisson with rate γ or forced at the floor).  
  A mean-field feedback updates the inflation rate  
  \[
    I = I_0 + J\big(\gamma (p_+ - \langle p\rangle) + \text{flux}_{\text{forced}}\cdot (p_+ - p_-)\big).
  \]  
  After burn-in it estimates stationary \(I^*\), builds a time-averaged histogram, fits the **semi-log PDF** interior, and saves:
  - `figs/J=..., ...(traj_zoom).png` (zoomed saw-tooth trajectories)  
  - `figs/J=..., ...(semilog).png` (semi-log stationary PDF with fitted & theoretical slopes)

- `src/mf_grid_convergence.py` – **Mean-field PDE: grid-resolution study**  
  Solves the master equation on a uniform grid and plots **relative errors vs. Np** for
  - stationary inflation \(I^*\) (vs \(I_0/(1-J)\)),
  - semi-log slope \(\gamma/I^*\).

- `src/mf_stationary_semilog.py` – **Mean-field PDE: stationary PDF & I(t)**  
  Single high-resolution run; compares the simulated stationary PDF against analytics
  (using simulated \(I^*\) and narrow-band \(I_0/(1-J)\)) and plots \(I(t)\) convergence.

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Agent-based model
python src/abm_sawtooth.py

# Mean-field: grid convergence
python src/mf_grid_optimization.py

# Mean-field: stationary PDF & I(t)
python src/mf_stationary_semilog.py
