# Output files and columns

After a complete run (or `python FP.py finish` / `python IBM.py finish`), each
method writes to `FP/data/` and `IBM/data/` respectively. Radii are **standardized**
(`sigma_m`, `sigma_w` = $\tilde\sigma_\mathrm{m}$, $\tilde\sigma_\mathrm{w}$) unless
a column name ends in `_dim`. One row is one grid cell; the grid is 60 × 60
log-spaced radii in $[0.1, 10]$ for each dimension `D`.

---

## FP (`FP/data/`)

### `fp_rmax_{1,2,3}d.csv` — one row per cell

| Column | Meaning |
|:--|:--|
| `method` | `"FP"` |
| `D` | habitat dimension (1, 2, 3) |
| `sigma_m`, `sigma_w` | standardized radii $\tilde\sigma_\mathrm{m}$, $\tilde\sigma_\mathrm{w}$ |
| `sigma_m_dim`, `sigma_w_dim` | dimensional radii ($\tilde\sigma \cdot \lambda^{-1/n}$) |
| `lam` | mean-field density $\lambda$ (400) |
| `length_unit` | $\lambda^{-1/n}$ |
| `Neq_std` | **standardized carrying capacity $\tilde N = N/\lambda$ (Figure 2)** |
| `Neq_raw` | equilibrium density $N$ (dimensional) |
| `g0` | **zero-distance pair correlation $g(0)$ (Figure 3)**; equals `g0_point` here |
| `converged` | 1 if the solver reached tolerance; 0 if it stopped at `maxit` |
| `res_q` | steady-state residual $\max\lvert dq/dt\rvert$ at the solution |
| `status` | `survival`, `extinct`, or `nonconv` |
| `g0_ball` | $g(0)$ averaged over the ball $\tilde r \le 0.5$ (the IBM's own estimator) |
| `g0_point` | $g(0)$ at exactly $\tilde r = 0$ |
| `wall_s` | solve time (s) |
| `dispersal`, `comp_trunc`, `g0_mode` | the kernel/estimator options used (`gaussian`, `0`, `point`) |
| `model_tag` | a one-string fingerprint of all solver settings |
| `iters` | fixed-point iterations to convergence |

### `fp_structure_{1,2,3}d.csv` — one row per cell

| Column | Meaning |
|:--|:--|
| `D`, `sigma_m`, `sigma_w`, `status` | as above |
| `q0` | $g(0) - 1$ |
| `r_half` | **half-saturation distance $\tilde r_{1/2}$ (Figure 4)**; `NaN` where the half level is not reached within the box |
| `S` | **spatial-structure degree $S_n$ over all distances (Figure 5)** |
| `S_signed` | signed version, $\int \log_{10} g\, dV$ (no absolute value) |
| `S_trunc` | $S_n$ integrated only to the IBM's largest distance ($\tilde r = 15/9/3.3$), for like-for-like comparison with the IBM |

### `fp_curves_points.csv` — full pair-correlation curves at the 7 marked points

| Column | Meaning |
|:--|:--|
| `D`, `k` | dimension and marked-point index (1–7, Table 2 / Figure 3) |
| `sigma_m`, `sigma_w` | radii of that point |
| `r` | standardized distance $\tilde r$ (shell centres, 0.02 wide, up to 3) |
| `g` | $g(\tilde r) = C(\tilde r)/N^2$ |

---

## IBM (`IBM/data/`)

### `ibm_std_data_D{1,2,3}.csv` — one row per cell

| Column | Meaning |
|:--|:--|
| `method` | `"IBM"` |
| `D`, `sigma_m`, `sigma_w`, `sigma_m_dim`, `sigma_w_dim`, `lam`, `length_unit` | as for FP |
| `Neq_std` | **standardized carrying capacity $\tilde N$ (Figure 2)**, floored at $10^{-3}$ |
| `Neq_std_unfloored` | $\tilde N$ without the floor (exact 0 for a fully extinct cell) |
| `Neq_raw` | mean equilibrium count $N$ |
| `Neq_sd_between_reps` | standard deviation of $\tilde N$ across realizations |
| `g0` | **zero-distance pair correlation $g(0)$ (Figure 3)** |
| `g0_sd` | standard error of $g(0)$ (noise-inflated; see below) |
| `g0_ball` | $g(0)$ averaged over $\tilde r \le 0.5$ |
| `extinct` | 1 if the cell went extinct |
| `survival` | fraction of realizations with survivors at $T_\text{END}$ |
| `n_reps` | realizations pooled for this cell |
| `neq_half1`, `neq_half2` | mean $N$ over the first and second half of $[T_\text{EQUIL},T_\text{END}]$ |
| `drift` | $(\text{half2}-\text{half1})/\text{mean}$ — the equilibration diagnostic |
| `equilibrated` | 1 if $\lvert$`drift`$\rvert < 0.10$; **0 flags cells still drifting** (chiefly clustered 1D) |
| `t_extinct_mean` | mean extinction time (where applicable) |
| `n_final_mean` | mean final count |
| `clustered_late_window` | 1 if only the late snapshots ($t \ge$ `T_SNAP_LATE`) were used for the pair statistics |
| `t_snap` | first snapshot time actually used |
| `n_snaps` | number of snapshots pooled |
| `gr_pairs` | total pairs counted |
| `habitat_std` | habitat side in standardized lengths |
| `cpu_hours` | CPU time for the cell |

### `ibm_structure_D{1,2,3}.csv` — one row per cell

| Column | Meaning |
|:--|:--|
| `D`, `sigma_m`, `sigma_w`, `status`, `extinct` | as above; `N` repeats $\tilde N$, `drift` repeats the diagnostic |
| `q0`, `sd_q0` | $g(0)-1$ and its standard error |
| `r_half` | **half-saturation distance $\tilde r_{1/2}$ (Figure 4)**, from a fit to the whole curve |
| `r_half_lo`, `r_half_hi` | 16–84 % bootstrap range of `r_half` |
| `r_half_reliable` | 1 if resolved and well inside the sampled range (the "reliable" flag used in the figure mask) |
| `r_half_capped` | 1 if the curve had not decayed within the sampled range (a lower bound) |
| `r_half_snr`, `r_half_state` | signal-to-noise of the crossing; 0/1/2 = unresolved/uncertain/reliable |
| `r_half_bin`, `r_half_bin_capped`, `r_half_bin_reliable` | the simpler bin-by-bin estimate (diagnostic) |
| `q0_fit`, `sd_q0_fit` | $g(0)-1$ from the fitted curve and its bootstrap sd |
| `S` | **spatial-structure degree $S_n$ (Figure 5)**, noise-corrected |
| `S_noise` | the subtracted noise contribution (diagnostic) |

### `ibm_pair_counts_D{1,2,3}.csv` — the raw pair counts (every cell)

Everything above can be recomputed from these.

| Column | Meaning |
|:--|:--|
| `sigma_m`, `sigma_w` | the cell |
| `r` | standardized distance bin centre (width 0.05) |
| `obs_pairs` | observed ordered-pair count, pooled over snapshots and realizations |
| `exp_pairs` | count expected for the same individuals placed at random |
| `C_over_N2` | `obs_pairs / exp_pairs` = $g(\tilde r)$ |

### `ibm_curves_points.csv` — pair-correlation curves at the 7 marked points

| Column | Meaning |
|:--|:--|
| `D`, `k`, `sigma_m`, `sigma_w` | dimension, marked-point index, radii |
| `r`, `g` | standardized distance and $g(\tilde r)$ (up to $\tilde r = 3$) |
| `pairs` | observed pairs in that bin |

---

## `run_settings.csv` (both methods)

A two-column (`setting`, `value`) record of the exact configuration of the run —
closure weights, kernel options, $b$, $d$, $d'$, $\lambda$, the grid, the
simulation windows, the habitat sizes and the largest distances sampled. **Archive
this file with the data**: it is the provenance of every number.

---

## Reconstructing the comparison

The paper's cell-by-cell comparison (its Table 3 and Appendix B) is obtained by
joining the FP and IBM `*_D{D}.csv` tables on (`D`, `sigma_m`, `sigma_w`) and
taking, over the off-diagonal cells with `equilibrated == 1`:

* $\lvert\tilde N_\text{FP} - \tilde N_\text{IBM}\rvert$ (percentage points),
* $\lvert\log_{10} g(0)_\text{FP} - \log_{10} g(0)_\text{IBM}\rvert$,
* $\lvert\log_{10} \tilde r_{1/2,\text{FP}} - \log_{10} \tilde r_{1/2,\text{IBM}}\rvert$ over cells where the IBM value is reliable,
* $\lvert\log_{10} S_\text{trunc,FP} - \log_{10} S_\text{IBM}\rvert$ (note: `S_trunc`, integrated over the same distances as the IBM).
