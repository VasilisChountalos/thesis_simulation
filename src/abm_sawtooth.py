import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pathlib import Path
import datetime

# ------------------------------------------------------------
# 1) PARAMETERS
# ------------------------------------------------------------
N          = 100000
band       = 0.02
p_plus     =  band / 2
p_minus    = -band / 2
gamma      = 0.08                 # per month
I0_annual  = 0.02                 # 2 % per year
I0         = I0_annual / 12
J          = 0.90

dt_days    = 0.1                  # simulation step (days)
dt         = dt_days / 30.0       # months
total_days = 3000
burnin_days= 1000
rng_seed   = 0

# histogram / fitting options
NBINS      = 300                  # time-averaged histogram bins
TRIM_FRAC  = 0.08                 # trim 8% from each boundary for fitting
DESYNC_RESETS = True              # desynchronize within-step resets

steps        = int(total_days / dt_days)
burnin_steps = int(burnin_days / dt_days)
rng          = np.random.default_rng(rng_seed)

# -------- output folder & filename prefix --------
outdir = Path("figs")
outdir.mkdir(exist_ok=True)  # ensure folder exists
fname_prefix = f"J={J},gamma={gamma},band={band}"

# ------------------------------------------------------------
# 2) INITIAL CONDITIONS
# ------------------------------------------------------------
p         = rng.uniform(p_minus, p_plus, size=N)
I         = I0
infl_hist = []

traj_to_plot = 15
trajectories = np.zeros((steps, traj_to_plot))
trajectories[0] = p[:traj_to_plot]

# pre-define bins and accumulator for a time-averaged stationary PDF
bins    = np.linspace(p_minus, p_plus, NBINS + 1)
bin_w   = np.diff(bins)           # constant here
hist_acc = np.zeros(NBINS, dtype=np.float64)

# ------------------------------------------------------------
# 3) MAIN LOOP
# ------------------------------------------------------------
for k in range(1, steps):
    # deterministic drift
    p -= I * dt

    # spontaneous repricings (Poisson in discrete time)
    reset_poiss = rng.random(N) < gamma * dt

    # firms that crossed the lower boundary
    reset_floor = p < p_minus

    # reset: either Poisson or forced
    reset = reset_poiss | reset_floor

    if DESYNC_RESETS:
        # Spread reset times uniformly within the step so they don't all land at p_plus
        n = reset.sum()
        if n:
            ages = rng.uniform(0.0, dt, size=n)  # when in this step the reset happened
            # After reset at time t*, drift acts for (dt - t*) in this step:
            p[reset] = p_plus - I * ages
    else:
        # classic: place exactly at p_plus
        p[reset] = p_plus

    # store some trajectories
    if k < steps:
        trajectories[k] = p[:traj_to_plot]

    # -------- mean-field feedback --------
    gap_sp      = p_plus - p.mean()
    gap_for     = p_plus - p_minus
    frac_forced = reset_floor.mean()
    flux_forced = frac_forced / dt            # per month
    I = I0 + J * (gamma * gap_sp + flux_forced * gap_for)

    # -------- collect stats after burn-in --------
    if k >= burnin_steps:
        infl_hist.append(I)
        # time-average histogram (counts, not density)
        hist_acc += np.histogram(p, bins=bins, density=False)[0]

# ------------------------------------------------------------
# 4) RESULTS
# ------------------------------------------------------------
I_star        = float(np.mean(infl_hist))           # empirical I* (time-avg)
I_theory      = I0 / (1 - J)                        # theoretical I*
slope_emp     = gamma / I_star
slope_theory  = gamma / I_theory

# Build time-averaged density and centers
centers = 0.5 * (bins[:-1] + bins[1:])
dens    = hist_acc / (hist_acc.sum() * bin_w)       # proper PDF estimate

# Trim boundaries (exclude bins near p_- and p_+ and empty bins)
lo = p_minus + TRIM_FRAC * band
hi = p_plus  - TRIM_FRAC * band
core = (centers > lo) & (centers < hi) & (dens > 0)

# Linear fit on semi-log of the interior
slope_fit, intercept_fit, r_value, _, _ = linregress(centers[core], np.log(dens[core]))
r2 = r_value**2
I_from_fit = gamma / slope_fit                       # I* inferred from fitted slope

# --- Errors (absolute and percent) ---
err_timeavg_abs = abs(I_star - I_theory)
err_timeavg_pct = 100 * err_timeavg_abs / I_theory
err_fit_abs     = abs(I_from_fit - I_theory)
err_fit_pct     = 100 * err_fit_abs / I_theory

print('----------------------------')
print(f'Stationary I* (time-avg): {I_star:.6f} per month')
print(f'I* from log-PDF fit    : {I_from_fit:.6f} per month')
print(f'Theory I* = I0/(1-J)   : {I_theory:.6f} per month')
print('--- Errors vs theory ---')
print(f'|I*_timeavg - I*_theory| = {err_timeavg_abs:.6e}  ({err_timeavg_pct:.4f} %)')
print(f'|I*_fit     - I*_theory| = {err_fit_abs:.6e}  ({err_fit_pct:.4f} %)')
print('--- Slopes ---')
print(f'Empirical slope γ/I*    : {slope_emp:.6f}')
print(f'Fit slope (log-PDF)     : {slope_fit:.6f}  (R^2={r2:.4f})')
print(f'Theoretical slope γ/I_th: {slope_theory:.6f}')
print('----------------------------')

# ------------------------------------------------------------
# 5) PLOTS (saved to figs/)
# ------------------------------------------------------------

# ---- (A) Saw-tooth trajectories: zoom into a short window (~2 periods) ----
period_days   = (band / I_star) * 30.0
window_days   = 2.0 * period_days
window_steps  = max(1, int(window_days / dt_days))

k1 = steps                                  # end of simulation
k0 = max(0, k1 - window_steps)              # start index for the window
t_window = np.arange(k0, k1) * dt_days      # time axis in days

plt.figure(figsize=(10, 4.2))
for i in range(traj_to_plot):
    plt.plot(t_window, trajectories[k0:k1, i], lw=1)
plt.axhline(p_plus,  ls='--', label='p_plus')
plt.axhline(p_minus, ls='--', label='p_minus')
plt.title('Bouchaud saw-tooth log-price (zoomed window ~2 periods)')
plt.xlabel('time (days)')
plt.ylabel('log price $p$')
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.savefig(outdir / f"{fname_prefix}(traj_zoom).png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ---- (B) Semi-log check using time-averaged histogram ----
plt.figure(figsize=(7.0, 4.8))
plt.scatter(centers[core], np.log(dens[core]), s=14,
            label='time-averaged log-hist (interior)')

xline = centers[core]
# fitted line
plt.plot(xline, intercept_fit + slope_fit * xline, lw=2,
         label=f'Fit slope = {slope_fit:.3f} (R²={r2:.3f})')

# overlay lines with empirical/theoretical slopes, anchored at the same intercept
b0 = intercept_fit
plt.plot(xline, b0 + slope_emp * xline, '--',
         label=f'γ/I* = {slope_emp:.3f}')
plt.plot(xline, b0 + slope_theory * xline, ':',
         label=f'γ/(I0/(1−J)) = {slope_theory:.3f}')

# visualize trimmed region
plt.axvspan(p_minus, lo, color='k', alpha=0.06)
plt.axvspan(hi, p_plus, color='k', alpha=0.06)

plt.xlabel('log price $p$')
plt.ylabel('$\\log P(p)$')
plt.title(f'Semi-log stationary PDF (J = {J}) — time-averaged, trimmed interior')
plt.legend()
plt.tight_layout()
plt.savefig(outdir / f"{fname_prefix}(semilog).png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

