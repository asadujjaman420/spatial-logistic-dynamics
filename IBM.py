"""IBM.py  --  individual-based model (IBM): data and Figures 2-5 of the draft, IBM panels.

ONE script does everything.  Run it with run_IBM.slurm (sbatch run_IBM.slurm); all settings are at the
top of that Slurm file.  Results go to the folder IBM/ (next to this script):

  IBM/data/ibm_std_data_D{1,2,3}.csv     one row per (sigma~_m, sigma~_w): N~ (Figure 2), C(0)/N^2 at zero
                                         distance (Figure 3), r~ <= 0.5 average, drift / equilibrated, ...
  IBM/data/ibm_structure_D{1,2,3}.csv    half-saturation distance r_1/2 (Figure 4, with its 16-84% range and
                                         a reliability flag) and spatial-structure degree S (Figure 5)
  IBM/data/ibm_pair_counts_D{1,2,3}.csv  the pooled pair counts behind C(r~)/N^2 for every cell (observed,
                                         expected, ratio) -- everything above can be recomputed from these
  IBM/data/ibm_curves_points.csv         C(r~)/N^2 at the 7 marked points (Figure 3 g-i)
  IBM/data/run_settings.csv              the settings of the run
  IBM/figures/Figure2..Figure5 *_IBM.png and .pdf
  IBM/work/                              partial results of the array tasks (the run resumes from these)
  IBM/logs/                              Slurm logs

Model (the draft): b = 0.4, d = 0, d' = 0.001 (lambda = 400), n-dimensional Gaussian dispersal and
competition kernels (competition cut at KCUT = 4 sigma, renormalized), exact Gillespie dynamics in a
periodic unit habitat (lambda individuals: 400 / 20 / 7.4 standardized lengths wide in 1 / 2 / 3 D).
Grid: 60 x 60 standardized radii sigma~_m, sigma~_w (log scale 0.1 .. 10) in 1, 2 and 3 dimensions.

For every cell: REPS realizations (default 150, three times the earlier 50), each from t = 0 to T_END:
  N~      N averaged over T_EQUIL <= t <= T_END and over the realizations, divided by lambda; the drift
          between the two halves of that window flags cells that are not yet equilibrated (> 10%).
  C(r~)/N^2  ordered-pair counts in distance bins of 0.05 (standardized), pooled over snapshots taken
          every SNAP_DT from T_SNAP, divided by the counts expected for the same number of individuals
          placed at random.  The counts of the late window t >= T_SNAP_LATE are kept separately: where the
          IBM is strongly clustered (late C(0)/N^2 >= CLUSTER_G0 and sigma_m < sigma_w) clustering keeps
          developing after N has settled, so only the late window is used there.
  C(0)/N^2   zero-distance value: Gaussian-weighted pair counts with width max(0.05, 0.25 min sigma~)
          (over which C(r~)/N^2 is flat), the same quantity as the moment-closure map of Figure 3.
  r_1/2   the moment-closure rule (the smallest r~ beyond which |C(r~)/N^2 - 1| stays below half of
          |C(0)/N^2 - 1|), applied to a smooth curve fitted to the whole of C(r~)/N^2 - 1 (a weighted sum
          of Gaussian shapes, the narrowest 0.6 times the smaller radius), so that all pairs are used;
          16-84% range from a parametric bootstrap.  No value (light grey in Figure 4) where C(0)/N^2 is
          within 3 standard errors of 1, the half level lies beyond the largest distance sampled, or the
          range is wider than +-25%.
  S       integral over R^n of |log10 C(r~)/N^2| dV~ up to the largest distance sampled (r~ = 15 / 9 / 3.3),
          minus the expected contribution of counting noise.
Counting errors are inflated by NOISE_INFL (snapshots one time unit apart are correlated), calibrated on
the diagonal sigma_m = sigma_w, where C(r~)/N^2 = 1 exactly.

Command line (run_IBM.slurm calls "run"; the rest is for checking by hand):
  python IBM.py run        this array task's share of the simulations; the task that completes the last
                           share then writes IBM/data and IBM/figures automatically
  python IBM.py finish     write IBM/data and IBM/figures from whatever has been computed
  python IBM.py status     progress
  python IBM.py estimate   time a few cells (one realization each) -> CPU hours and a suggested array size
  python IBM.py test       quick local check: 4 x 4 cells, 4 realizations, t <= 40 (output in IBM_test/)
"""
import os, sys, csv, glob, time, math
# ---- the draft model (can be overridden from the Slurm script) ----
os.environ.setdefault("DISPERSAL", "gaussian")      # n-dimensional Gaussian dispersal
os.environ.setdefault("D_RATE", "0.0")              # d = 0 -> lambda = (b - d) / d' = 400
os.environ.setdefault("KCUT", "4.0")                # competition kernel cut at 4 sigma
os.environ.setdefault("KERNEL_MODE", "auto")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize, LogNorm, ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter, binary_dilation, binary_closing, binary_opening
from scipy.special import erf
from scipy.spatial import cKDTree
from itertools import product
from multiprocessing import Pool, cpu_count


# =====================================================================================================
# PART 1: the individual-based model -- functions of your ibm_sweep_snap.py (ibm_sweep_all.py + snapshots), unchanged
# =====================================================================================================
# Competition kernel under periodic boundaries.
#   "auto" (DEFAULT, matches points.c):
#        R_d <= L/2  -> minimum image, truncated at 3 sigma  (exact, cell list)
#        R_d >  L/2  -> sum over periodic images, truncated at 3 sigma.  This is
#                       what points.c does: survival_rate() scans
#                       delta_maxcell = 1+floor(int_mn_rad/cell_size) cells each
#                       way and get_cell() returns a compensating offset per
#                       wrap, so the same neighbour is counted once per image.
#        If the image count exceeds KERNEL_IMG_MAX the separable wrapped
#        Gaussian is used instead; that only happens in the far field, where it
#        agrees with the truncated image sum to < 0.5% and the kernel is flat to
#        CV < 0.2% (i.e. the cell is mean-field regardless).
#   "never"  = old behaviour, minimum image everywhere.  WRONG for R_d > L/2:
#              on the 60-pt [0.1,10] axis that is 0% of D=1, 32% of D=2 and 50%
#              of D=3 rows.  Kept only for reproducing the earlier runs.
#   "always" = separable wrapped Gaussian everywhere.
KERNEL_MODE = os.environ.get("KERNEL_MODE", "auto")

# Above this image count the exact truncated image sum is replaced by the
# separable wrapped Gaussian.  Measured at D=3, sigma~_w=5.179 (R_w/L=2.66,
# 179 pruned images): exact gives Neq_raw=198.5, wrapped gives 198.4 -- a 0.05%
# difference, far below the stochastic noise -- but 34x faster (10.1s vs 0.3s
# per realization).  The `images` branch has no cell list (when R_w > L/2 every
# particle is a neighbour, so a cell list buys nothing), which is why it is the
# bottleneck.  200 keeps the exact sum wherever it is affordable and switches to
# the fast form only in the far field, where the kernel is flat anyway.
KERNEL_IMG_MAX = int(os.environ.get("KERNEL_IMG_MAX", 200))

THETA_TABLE_N = 4096

# ---- thresholds: every cell gets a finite, positive value -------------------
# A replicate that dies contributes 0 to the abundance mean (see run_ibm), so a
# fully extinct cell has Neq_std = 0 exactly.  Floor it, and report g(0) at the
# cap there: an extinct cell has no pairs to measure and N sits at the floor, so
# c(0)/N^2 is unbounded rather than undefined.  Nothing is ever NaN or inf, so
# no cell is dropped from the figure.
NEQ_FLOOR = float(os.environ.get("NEQ_FLOOR", 1e-3))

G0_CAP    = float(os.environ.get("G0_CAP", 2000.0))

# |drift| above this means the cell never settled inside [T_EQUIL, T_END].
EQ_DRIFT_TOL = float(os.environ.get("EQ_DRIFT_TOL", 0.10))

# ============================================================
# PAPER PARAMETERS (Fig 14.4) + NONDIMENSIONALIZATION
# ============================================================
b  = 0.4          # intrinsic birth rate

d  = float(os.environ.get("D_RATE", 0.0))   # intrinsic death rate (draft: 0)

dp = 0.001        # d' = density-dependent death coefficient

N0 = None         # initial population: set below to lambda * L^n (env N0)

L  = 1.0          # habitat side length

LAM = (b - d) / dp                     # mean-field equilibrium density, lambda

N0 = int(os.environ.get("N0", round(LAM * L)))   # L^n = 1 for L = 1

DISPERSAL = os.environ.get("DISPERSAL", "gaussian").lower()   # gaussian | halfnormal

KCUT = float(os.environ.get("KCUT", 4.0))         # competition kernel cut-off, in sigma

def length_unit(D):
    """L_unit = lambda^(-1/n)."""
    return LAM ** (-1.0 / D)

def kernel_norm(sigma, D):
    """Normalization of the Gaussian kernel TRUNCATED at KCUT sigma."""
    A = KCUT
    if D == 1:
        return sigma * math.sqrt(2 * math.pi) * erf(A / math.sqrt(2))
    if D == 2:
        return 2 * math.pi * sigma**2 * (1 - math.exp(-A * A / 2))
    if D == 3:
        return 4 * math.pi * sigma**3 * (
            math.sqrt(math.pi / 2) * erf(A / math.sqrt(2))
            - A * math.exp(-A * A / 2)
        )
    raise ValueError("D must be 1, 2, or 3")

def theta_table(sigma, L, G=THETA_TABLE_N):
    """1-D periodic (wrapped) Gaussian, tabulated on x in [0, L/2].

    The product over D axes is the D-dimensional periodic Gaussian divided by
    the FULL normalization (2 pi sigma^2)^(D/2) -- NOT by kernel_norm, which is
    the 3-sigma TRUNCATED normalization.  Both integrate to 1 over the torus, so
    both give the same mean-field equilibrium; they differ only in shape.  Mixing
    them would shift competition by 0.27% / 1.11% / 2.93% in D = 1 / 2 / 3 -- a
    DIMENSION-DEPENDENT bias, which is exactly what this sweep must not have.
    Callers therefore use this branch only in the far field (see KERNEL_MODE).
    """
    n_img = int(math.ceil(6.0 * sigma / L)) + 1
    x = np.linspace(0.0, L / 2.0, G)
    m = np.arange(-n_img, n_img + 1)
    xx = x[:, None] + m[None, :] * L
    return (np.exp(-xx * xx / (2.0 * sigma * sigma)).sum(axis=1)
            / (sigma * math.sqrt(2.0 * math.pi)))

def sample_dispersal(sigma_b, R_b_max, D, rng):
    if DISPERSAL == "gaussian":
        # draft model: n-dimensional isotropic Gaussian, untruncated
        return rng.standard_normal(D) * sigma_b
    while True:
        r = abs(rng.standard_normal() * sigma_b)
        if r <= R_b_max:
            break
    if D == 1:
        sign = 1.0 if rng.random() < 0.5 else -1.0
        return np.array([sign * r])
    if D == 2:
        th = 2 * math.pi * rng.random()
        return np.array([r * math.cos(th), r * math.sin(th)])
    if D == 3:
        cph = 2.0 * rng.random() - 1.0
        sph = math.sqrt(max(0.0, 1.0 - cph * cph))
        th = 2 * math.pi * rng.random()
        return np.array([r * sph * math.cos(th),
                         r * sph * math.sin(th),
                         r * cph])
    raise ValueError

def run_ibm(sigma_b, sigma_d, D=2, t_end=100.0, seed=1,
            record_dt=0.5, L=1.0, N0=N0,
            b=b, d=d, dp=dp, verbose=False,
            kernel_mode=KERNEL_MODE, snap_from=None, snap_dt=1.0):
    """sigma_b, sigma_d are DIMENSIONAL radii.

    Exact Gillespie birth-death with the points.c rates:
        birth  = b                              (rpr_mn_icf == 0: no birth interaction)
        death  = d + d' * sum_j w(r_ij)         (death_model == 2)
        w(r)   = exp(-r^2 / 2 sigma_d^2) / Z ,  r <= 3 sigma_d ,  Z = kernel_norm
    The positions/rates are held in numpy arrays and the per-event weight vector
    is computed with one vectorized call, which is what makes the full sweep
    affordable; the event sequence is identical to the original loop.

    points.c refreshes rates only every update_events = 10 events and reads
    survival_rate on individuals born in between, before it is ever assigned.
    That is a bug in the C, not a modelling choice, so rates here are exact.

    HISTORY: samples are recorded on the regular record_dt grid all the way to
    t_end, PADDED WITH ZEROS after extinction.  The original stopped recording at
    the extinction time, so `N_h[t_h >= T_EQUIL].mean()` either averaged only the
    pre-extinction decay or -- if the run died before T_EQUIL -- fell back to
    `N_h.mean()`, reporting ~50-100 individuals for a population that is gone.
    With the padding that same expression is now correct.
    """
    rng = np.random.default_rng(seed)
    run_ibm.snaps = []

    R_b = 3 * sigma_b
    R_d = KCUT * sigma_d
    Z_d = kernel_norm(sigma_d, D)
    inv_2sig2 = 1.0 / (2 * sigma_d * sigma_d)
    half_L = L / 2.0
    R_d2 = R_d * R_d

    # ---------------- choose kernel treatment -------------------------------
    n_img_axis = int(math.floor((R_d + half_L) / L))
    n_img_tot = (2 * n_img_axis + 1) ** D
    if kernel_mode == "never":
        mode = "minimg"
    elif kernel_mode == "always":
        mode = "wrapped"
    else:                                          # "auto" == points.c
        if R_d <= half_L:
            mode = "minimg"                        # minimum image IS the image sum
        elif n_img_tot <= KERNEL_IMG_MAX:
            mode = "images"                        # exact truncated image sum
        else:
            mode = "wrapped"                       # far field; kernel is flat here
    periodic = (mode != "minimg")
    run_ibm.last_periodic = periodic
    run_ibm.last_mode = mode
    run_ibm.last_n_img = n_img_tot if mode == "images" else 1
    if verbose:
        print(f"    D={D} R_d={R_d:.4f} R_d/(L/2)={R_d/half_L:.2f} mode={mode} "
              f"images={n_img_tot}")

    if mode == "images":
        # Prune images that cannot contribute.  After minimum image every
        # component satisfies |delta_k| <= L/2, so the closest that image m can
        # ever come is  sum_k max(0, |m_k| L - L/2)^2 .  If that already exceeds
        # R_d^2 the image is outside the 3-sigma cutoff for EVERY pair.
        _all = np.array(list(product(range(-n_img_axis, n_img_axis + 1),
                                     repeat=D)), dtype=float)
        _lo = np.maximum(np.abs(_all) * L - half_L, 0.0)
        SHIFTS = (_all[(_lo ** 2).sum(axis=1) <= R_d2] * L)
        run_ibm.last_n_img = int(len(SHIFTS))
    elif mode == "wrapped":
        TAB = theta_table(sigma_d, L)
        GT = len(TAB)
        inv_step = (GT - 1) / half_L

    # a cell list only helps, and is only valid, when the interaction fits in it
    use_cell_list = (mode == "minimg")
    if use_cell_list:
        cs0 = max(R_d, 0.02)
        ncell = max(int(math.floor(L / cs0)), 1)
        cs = L / ncell
        assert cs >= R_d - 1e-12
        use_cell_list = ncell >= 4          # below that it degenerates to all-pairs
    if not use_cell_list:
        ncell, cs = 1, L
    OFFSETS = list(product((-1, 0, 1), repeat=D))

    cap = max(4 * N0, 64)
    pos = np.empty((cap, D))
    pos[:N0] = rng.random((N0, D)) * L
    n = N0
    drate = np.empty(cap)

    # ---------------- cell list (indices bucketed by cell) ------------------
    if use_cell_list:
        cellid = np.empty(cap, dtype=np.int64)
        buckets = {}

        def _cid(p):
            k = 0
            for a in range(D):
                k = k * ncell + (int(p[a] / cs) % ncell)
            return k

        def _nbr_keys(p):
            base = [int(p[a] / cs) % ncell for a in range(D)]
            out = set()
            for off in OFFSETS:
                k = 0
                for a in range(D):
                    k = k * ncell + ((base[a] + off[a]) % ncell)
                out.add(k)
            return out

        def _add(i):
            k = _cid(pos[i]); cellid[i] = k
            buckets.setdefault(k, []).append(i)

        def _drop(i):
            buckets[int(cellid[i])].remove(i)

        for i in range(n):
            _add(i)

    def _cand(p0):
        """Indices that can possibly interact with p0 (None == all of them)."""
        if not use_cell_list:
            return None
        out = []
        for k in _nbr_keys(p0):
            bk = buckets.get(k)
            if bk:
                out.extend(bk)
        return np.fromiter(out, dtype=np.int64, count=len(out))

    def _wvec(p0, upto, skip=-1):
        """Competition weights between p0 and the candidate set.

        Returns (w, idx) with idx None meaning 'indices 0..upto-1'.
        """
        idx = _cand(p0)
        P = pos[:upto] if idx is None else pos[idx]
        if len(P) == 0:
            return np.zeros(0), idx
        dd = P - p0
        if mode == "wrapped":
            # separable periodic Gaussian: product of 1-D wrapped Gaussians
            a = np.abs(dd)
            a = np.minimum(a, L - a)
            f = a * inv_step
            i0 = f.astype(np.int64)
            np.clip(i0, 0, GT - 2, out=i0)
            fr = f - i0
            w = np.prod(TAB[i0] * (1.0 - fr) + TAB[i0 + 1] * fr, axis=1)
        else:
            dd = dd - L * np.round(dd / L)                    # minimum image
            if mode == "images":
                w = np.zeros(len(P))
                for s in SHIFTS:
                    e = dd + s
                    r2 = np.einsum("ij,ij->i", e, e)
                    w += np.where(r2 <= R_d2, np.exp(-r2 * inv_2sig2), 0.0)
            else:
                r2 = np.einsum("ij,ij->i", dd, dd)
                w = np.where(r2 <= R_d2, np.exp(-r2 * inv_2sig2), 0.0)
            w = w / Z_d
        if skip >= 0:
            if idx is None:
                if skip < upto:
                    w[skip] = 0.0
            else:
                w[idx == skip] = 0.0
        return w, idx

    for i in range(n):
        w, _ = _wvec(pos[i], n, skip=i)
        drate[i] = d + dp * float(w.sum())
    dsum = float(drate[:n].sum())

    t = 0.0
    t_hist = [0.0]
    N_hist = [N0]
    next_sample = record_dt
    RESUM_EVERY = 20000
    ev = 0

    while t < t_end and n > 0:
        bsum = b * n
        R_tot = bsum + max(dsum, 0.0)
        if R_tot <= 0:
            break

        dt = -math.log(max(rng.random(), 1e-15)) / R_tot
        t += dt
        if t >= t_end:
            break
        u = rng.random() * R_tot

        if u < bsum:
            # ---------------- birth -------------------------------------
            p = int(rng.random() * n)
            disp = sample_dispersal(sigma_b, R_b, D, rng)
            newpos = (pos[p] + disp) % L
            if n >= cap:
                newcap = 2 * cap
                p2 = np.empty((newcap, D)); p2[:n] = pos[:n]; pos = p2
                d2 = np.empty(newcap); d2[:n] = drate[:n]; drate = d2
                if use_cell_list:
                    c2 = np.empty(newcap, dtype=np.int64); c2[:n] = cellid[:n]
                    cellid = c2
                cap = newcap
            w, idx = _wvec(newpos, n)
            sw = float(w.sum())
            pos[n] = newpos
            if idx is None:
                drate[:n] += dp * w
            else:
                drate[idx] += dp * w
            drate[n] = d + dp * sw
            dsum += dp * 2.0 * sw + d
            if use_cell_list:
                _add(n)
            n += 1
        else:
            # ---------------- death -------------------------------------
            u2 = u - bsum
            c = np.cumsum(drate[:n])
            victim = int(np.searchsorted(c, u2))
            if victim >= n:
                victim = n - 1
            w, idx = _wvec(pos[victim], n, skip=victim)
            sw = float(w.sum())
            if idx is None:
                drate[:n] -= dp * w
            else:
                drate[idx] -= dp * w
            dsum -= dp * 2.0 * sw + d
            if use_cell_list:
                _drop(victim)
            n -= 1
            if victim != n:
                pos[victim] = pos[n]
                drate[victim] = drate[n]
                if use_cell_list:
                    kk = int(cellid[n])
                    buckets[kk].remove(n)
                    buckets[kk].append(victim)
                    cellid[victim] = kk

        ev += 1
        if ev % RESUM_EVERY == 0 and n > 0:
            dsum = float(drate[:n].sum())          # guard against fp drift

        while t >= next_sample:
            t_hist.append(next_sample)
            N_hist.append(n)
            if snap_from is not None and next_sample >= snap_from - 1e-9 and \
                    abs((next_sample - snap_from) / snap_dt - round((next_sample - snap_from) / snap_dt)) < 1e-6:
                run_ibm.snaps.append(np.array(pos[:n]))
            next_sample += record_dt

    # FIX: pad the record grid to t_end with zeros after extinction, so the
    # equilibrium average sees the zeros instead of stopping at the decay.
    if n == 0:
        while next_sample <= t_end:
            t_hist.append(next_sample)
            N_hist.append(0)
            next_sample += record_dt
    t_hist.append(min(t, t_end))
    N_hist.append(n)

    final_pos = np.array(pos[:n]) if n else np.zeros((0, D))
    if verbose:
        print(f"    D={D}  t_final={t:.2f}  N_final={n}")
    return np.array(t_hist), np.array(N_hist), final_pos

def ball_volume(r, D):
    """Volume of the D-ball of radius r.  D=1: 2r, D=2: pi r^2, D=3: 4/3 pi r^3."""
    return math.pi ** (D / 2.0) * np.asarray(r, float) ** D / math.gamma(D / 2.0 + 1.0)

def shell_volume(bins, D):
    """EXACT volume of each spherical shell, as a difference of ball volumes.

    The midpoint form  S(D) * r_centre^(D-1) * bin_width  is exact for D = 1 and
    D = 2 but NOT for D = 3, where the innermost shell comes out
        pi*dr^3   instead of   (4/3)*pi*dr^3        -- 25% too small.
    Over the two bins that make g(0) that is 6.25% too small, so g(0) was biased
    HIGH by 6.7% in D = 3 ONLY.  A dimension-dependent bias in precisely the
    quantity the three panels are compared on, so it has to be the exact form.
    """
    return np.diff(ball_volume(bins, D))

def _cost(D, sm, sw):
    """Rough relative cost of a cell, for balancing the shards."""
    swd = sw * length_unit(D)
    cut = min(KCUT * swd, 0.49 * L)
    neigh = min(float(ball_volume(cut, D)) / (L ** D), 1.0)
    return LAM * LAM * neigh * (1.0 + (0.5 / sm) ** D)


# =====================================================================================================
# PART 2: C(r)/N^2 -> C(0)/N^2, half-saturation distance (fit to the whole curve), structure degree
# =====================================================================================================
BW = 0.05

# Snapshots one time unit apart are correlated, so Poisson counting errors are too small.  Calibrated on
# the diagonal cells (where C(r)/N^2 = 1 exactly): the scatter of the smoothed curve is 2.6 / 2.2 / 2.7 times
# the nominal error in 1 / 2 / 3 D.  All counting errors below are multiplied by this factor.
NOISE_INFL = {1: 2.6, 2: 2.2, 3: 2.7}

RH_SNR_MIN = 4.0          # half level >= 4 (calibrated) standard errors at the crossing

RH_RANGE_FRAC = 0.6       # crossing within 60% of the largest sampled distance

def ball(r, D):
    return {1: 2 * r, 2: np.pi * r ** 2, 3: 4 / 3 * np.pi * r ** 3}[D]

def smooth_counts(x, r):
    """Gaussian smoothing with r-dependent width (in bins)."""
    sig = (0.05 + 0.1 * r) / BW
    out = np.empty_like(x)
    idx = np.arange(len(x))
    for i in range(len(x)):
        w = np.exp(-0.5 * ((idx - i) / sig[i]) ** 2)
        w[idx < 0] = 0
        out[i] = (w * x).sum() / w.sum()
    return out, sig

def g0_origin(r, obs, exp, r0):
    """C(0)/N^2 from pair counts with Gaussian weights exp(-r^2 / 2 r0^2)."""
    w = np.exp(-0.5 * (r / r0) ** 2)
    o, e = (w * obs).sum(), (w * exp).sum()
    if o <= 0 or e <= 0:
        return np.nan, np.nan
    g = o / e
    neff = o ** 2 / (w ** 2 * obs).sum()          # effective number of pairs
    return float(g), float(g / np.sqrt(neff))

def analyse(D, r, obs, exp, sm=None, sw=None):
    nan = dict(q0=np.nan, sd_q0=np.nan, r_half=np.nan, r_half_capped=0, r_half_snr=np.nan, r_half_reliable=0,
               r_half_state=0, r_half_lo=np.nan, r_half_hi=np.nan, q0_fit=np.nan, sd_q0_fit=np.nan,
               r_half_bin=np.nan, r_half_bin_capped=0, r_half_bin_reliable=0, S=np.nan, S_noise=np.nan)
    if exp.sum() == 0 or obs.sum() < 50:
        return nan
    r0 = max(BW, 0.25 * min(sm, sw)) if sm is not None else BW
    g0, sdg = g0_origin(r, obs, exp, r0)
    sdg = sdg * NOISE_INFL[D]
    q0 = g0 - 1.0
    os_, sig = smooth_counts(obs, r); es_, _ = smooth_counts(exp, r)
    gs = os_ / np.where(es_ > 0, es_, np.nan)
    neff = os_ * np.sqrt(2 * np.pi) * sig    # pairs inside the smoothing window
    sd_log = NOISE_INFL[D] / (np.log(10) * np.sqrt(np.maximum(neff, 1.0)))
    sd_lin = gs * np.log(10) * sd_log
    # r_half on the smoothed curve, significant deviations only
    a = np.abs(gs - 1.0); tgt = 0.5 * abs(q0)
    above = np.nonzero((a > tgt) & (a > 2 * sd_lin))[0]
    capped = 0
    if abs(q0) < 3 * sdg:
        rh = np.nan                          # C(0)/N^2 indistinguishable from 1: undefined
    elif len(above) < 3:
        rh = np.nan                          # fewer than 3 significant bins: not resolved
    elif above[-1] >= len(a) - 1:
        rh, capped = float(r[-1]), 1         # not decayed within the sampled range: lower bound
    else:
        i = above[-1]
        rh = float(np.interp(tgt, [a[i + 1], a[i]], [r[i + 1], r[i]]))
    # reliability: the half level must stand clearly above the noise where the curve crosses it, and the
    # crossing must lie well inside the sampled range (the smoothing window and the habitat edge)
    snr, rel = np.nan, 0
    if np.isfinite(rh) and not capped:
        j = int(np.clip(np.searchsorted(r, rh), 0, len(r) - 1))
        snr = float(tgt / max(sd_lin[j], 1e-12))
        rel = int(snr >= RH_SNR_MIN and rh <= RH_RANGE_FRAC * r[-1])
    # r_half from a fit to the whole curve (rhalf_fit.py): uses all pairs, same rule as the FP.
    # The bin-by-bin value above is kept as r_half_bin (diagnostic).
    rh_bin, cap_bin, snr_bin, rel_bin = rh, capped, snr, rel
    e = estimate(D, r, obs, exp, nb=60, seed=7, smin=max(0.04, 0.6 * min(sm, sw)) if sm is not None else 0.04)
    state, spread = classify(D, e)
    rh = e["rh_fit"] if state > 0 else np.nan
    capped = int(e["rh_fit_capped"] and abs(e["q0_fit"]) >= Q0_SIG * FIT_INFL[D] * e["sd_q0_fit"])
    rel = int(state == 2); snr = spread
    lg = np.abs(np.log10(np.clip(gs, 1e-6, None)))
    vol = np.diff(ball(np.r_[0.0, r + BW / 2], D))
    nz = sd_log * np.sqrt(2 / np.pi)
    S = max(float(((lg - nz) * vol).sum()), 0.0)
    return dict(q0=q0, sd_q0=sdg, r_half=rh, r_half_capped=capped, r_half_snr=snr, r_half_reliable=rel,
                r_half_state=state, r_half_lo=e["rh_lo"], r_half_hi=e["rh_hi"], q0_fit=e["q0_fit"], sd_q0_fit=e["sd_q0_fit"],
                r_half_bin=rh_bin, r_half_bin_capped=cap_bin, r_half_bin_reliable=rel_bin,
                S=S, S_noise=float((nz * vol).sum()))


def basis(r, s):
    return np.exp(-0.5 * (r[:, None] / s[None, :]) ** 2)

def widths(rmax, n=22, smin=0.04):
    return np.geomspace(smin, 1.2 * rmax, n)

def rhalf_curve(rf, q, q0):
    """FP rule on a curve q(r) = C/N^2 - 1 sampled on rf.  Returns (r_half, capped)."""
    a = np.abs(q); tgt = 0.5 * abs(q0)
    above = np.nonzero(a > tgt)[0]
    if len(above) == 0:
        return float(rf[0]), 0
    i = above[-1]
    if i >= len(a) - 1:
        return float(rf[-1]), 1
    return float(np.interp(tgt, [a[i + 1], a[i]], [rf[i + 1], rf[i]])), 0

class Fit:
    def __init__(self, D, r, obs, exp, nbw=22, smin=0.04):
        ok = exp > 0
        self.D, self.r, self.obs, self.exp = D, r[ok], obs[ok].astype(float), exp[ok].astype(float)
        self.rmax = float(r[-1]) + 0.025
        self.s = widths(self.rmax, nbw, smin)
        self.Phi = basis(self.r, self.s)
        self.rf = np.linspace(0, self.rmax, 1201)
        self.Pf = basis(self.rf, self.s)
        self.y = self.obs / self.exp - 1.0
        # error of g in each bin: first pass with g = 1, second with the fitted g
        self.sd = NOISE_INFL[D] * np.sqrt(np.maximum(self.exp, 1.0)) / self.exp
        self.alpha = None
        a = self._solve(self.y)
        g = np.clip(1 + self.Phi @ a, 0.02, None)
        self.sd = NOISE_INFL[D] * np.sqrt(np.maximum(self.exp * g, 1.0)) / self.exp
        self.alpha = None
        self.a = self._solve(self.y)

    def _solve(self, y):
        X = self.Phi / self.sd[:, None]; z = y / self.sd
        # column scaling so the ridge treats all shapes alike
        cs = np.sqrt((X ** 2).sum(0)) + 1e-300; Xs = X / cs
        U, sv, Vt = np.linalg.svd(Xs, full_matrices=False)
        uz = U.T @ z
        if self.alpha is None:
            n = len(z); best = None
            for al in np.geomspace(1e-6, 1e2, 41) * sv[0] ** 2:
                f = sv ** 2 / (sv ** 2 + al)
                rss = float(((z - U @ (f * uz)) ** 2).sum()); df = f.sum()
                gcv = n * rss / (n - df) ** 2
                if best is None or gcv < best[0]:
                    best = (gcv, al)
            self.alpha = best[1]
            self._svd = (U, sv, Vt, cs)
        U, sv, Vt, cs = self._svd
        uz = U.T @ z
        return (Vt.T @ (sv / (sv ** 2 + self.alpha) * uz)) / cs

    def curve(self, a=None):
        return self.Pf @ (self.a if a is None else a)

    def rhalf(self, a=None):
        q = self.curve(a)
        return rhalf_curve(self.rf, q, q[0]) + (float(q[0]),)

    def bootstrap(self, nb=60, seed=0):
        rng = np.random.default_rng(seed)
        mu = self.Phi @ self.a
        out = []
        for _ in range(nb):
            ys = mu + rng.normal(0.0, self.sd)
            out.append(self.rhalf(self._solve(ys)))
        return np.array(out)          # columns r_half, capped, q0

def estimate(D, r, obs, exp, nb=60, seed=0, smin=0.04):
    """r_half (median of fit), 16/84% bootstrap range, q0 and its bootstrap sd, capped flag."""
    if exp.sum() == 0 or obs.sum() < 50:
        return dict(rh_fit=np.nan, rh_lo=np.nan, rh_hi=np.nan, q0_fit=np.nan, sd_q0_fit=np.nan, rh_fit_capped=0, rh_fit_ok=0)
    F = Fit(D, r, obs, exp, smin=smin)
    rh, cap, q0 = F.rhalf()
    B = F.bootstrap(nb, seed)
    lo, hi = np.percentile(B[:, 0], [16, 84]); sdq = float(np.std(B[:, 2]))
    capfrac = float(B[:, 1].mean())
    return dict(rh_fit=rh, rh_lo=float(lo), rh_hi=float(hi), q0_fit=q0, sd_q0_fit=sdq,
                rh_fit_capped=int(cap or capfrac > 0.16), rh_fit_ok=0)

# The bootstrap treats distance bins as independent; on the diagonal cells (no structure) the fitted
# C(0)/N^2 - 1 scatters 1.4 / 1.0 / 1.0 times the bootstrap sd in 1 / 2 / 3 D -> errors scaled by this.
FIT_INFL = {1: 1.4, 2: 1.0, 3: 1.0}

SPREAD_OK = np.log(1.25)       # 16-84% range within +-25%: reliable

SPREAD_UNC = np.log(1.6)       # within +-60%: shown, marked uncertain

Q0_SIG = 3.0                   # |C(0)/N^2 - 1| >= 3 sd, otherwise undefined (no structure to halve)

def classify(D, e):
    """0 = no value (no significant structure / beyond the sampled range / too uncertain),
    1 = uncertain, 2 = reliable.  Also returns the scaled half-width of the 16-84% range (log)."""
    f = FIT_INFL[D]
    if not np.isfinite(e["rh_fit"]) or not np.isfinite(e["q0_fit"]):
        return 0, np.nan
    spread = f * np.log(e["rh_hi"] / e["rh_lo"]) / 2 if e["rh_lo"] > 0 else np.inf
    if abs(e["q0_fit"]) < Q0_SIG * f * e["sd_q0_fit"] or e["rh_fit_capped"]:
        return 0, spread
    return (2 if spread <= SPREAD_OK else 1 if spread <= SPREAD_UNC else 0), spread


# =====================================================================================================
# PART 3: drawing Figures 2-5 (the functions used for the draft figures)
# =====================================================================================================
# ======================================================================= settings
DIMS = (1, 2, 3)

DIMNAME = {1: "One-dimensional habitats", 2: "Two-dimensional habitats", 3: "Three-dimensional habitats"}

NF = 720                                  # fine display grid per axis (smooth maps)

EXT_COLOR = "0.80"                        # extinct / undefined cells

DPI = 300

# marker points of the draft's Fig. 3 (sigma~_m, sigma~_w)
DRAFT_PTS = [(0.125, 8), (0.18, 2.3), (0.27, 0.7), (0.155, 0.16), (0.68, 0.28), (2.3, 0.19), (8, 0.127)]

CURVE_RX = 3.0                            # x-range of Fig. 3(g-i)

SM = r"$\widetilde{\sigma}_{\mathrm{m}}$"

SW = r"$\widetilde{\sigma}_{\mathrm{w}}$"

# ======================================================================= helpers
OVER_COLOR = "#6b0000"

SHOW_UNCERTAIN = os.environ.get("SHOW_UNCERTAIN", "0") == "1"   # IBM r_half: show noisy-but-measured cells, hatched

SMOOTH_N = None                              # smoothing of the IBM N~ map (None: same as --ibm-smooth)

def cmap_N():
    """Colour scale of the carrying-capacity maps: nipy_spectral up to 90 % of the range (as in the
    draft); the top 10 % (whose light grey would look like missing data) fades from red to dark red,
    and values above the range (IBM, N~ > 600 %) are drawn in the same dark red."""
    base = plt.get_cmap("nipy_spectral")
    lo = base(np.linspace(0, 0.9, 461))
    c0, c1 = np.array(base(0.9)), np.array(matplotlib.colors.to_rgba(OVER_COLOR))
    hi = c0 + (c1 - c0) * np.linspace(0, 1, 52)[1:, None]
    cm = ListedColormap(np.vstack([lo, hi]), name="nps_N")
    cm.set_over(OVER_COLOR)
    return cm

def cmap(top=1.0):
    return ListedColormap(plt.get_cmap("nipy_spectral")(np.linspace(0, top, 512)))

def pivot(df, col, sm, sw):
    return (df.pivot_table(index="sigma_w", columns="sigma_m", values=col, aggfunc="first", dropna=False)
              .reindex(index=sw, columns=sm).to_numpy())

def axes_of(df):
    return np.sort(df.sigma_m.unique()), np.sort(df.sigma_w.unique())

def fill_nan(F):
    """Fill NaN cells by repeated linear extrapolation from their neighbours."""
    S = F.astype(float).copy()
    ny, nx = S.shape
    for _ in range(4 * max(ny, nx)):
        nan = np.isnan(S)
        if not nan.any():
            break
        new = S.copy()
        for i, j in zip(*np.where(nan)):
            est = []
            for di, dj in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                i1, j1, i2, j2 = i + di, j + dj, i + 2 * di, j + 2 * dj
                if 0 <= i2 < ny and 0 <= j2 < nx and np.isfinite(S[i1, j1]) and np.isfinite(S[i2, j2]):
                    est.append(2 * S[i1, j1] - S[i2, j2])
                elif 0 <= i1 < ny and 0 <= j1 < nx and np.isfinite(S[i1, j1]):
                    est.append(S[i1, j1])
            if est:
                new[i, j] = float(np.mean(est))
        if np.array_equal(np.isnan(new), nan):
            new[np.isnan(new)] = np.nanmean(S)
        S = new
    return S

class Grid:
    """Fine log-spaced display grid spanning the sweep axis."""
    def __init__(self, lo, hi):
        self.lf = np.linspace(np.log10(lo), np.log10(hi), NF)
        self.x = 10 ** self.lf

    def refine(self, V, sm, sw):
        """Bicubic refinement in log-sigma with a local (4-cell) limiter: smooth, no overshoot."""
        lx, ly = np.log10(sm), np.log10(sw)
        R = RectBivariateSpline(ly, lx, V, kx=3, ky=3, s=0)(self.lf, self.lf)
        ix = np.clip(np.searchsorted(lx, self.lf) - 1, 0, len(lx) - 2)
        iy = np.clip(np.searchsorted(ly, self.lf) - 1, 0, len(ly) - 2)
        c4 = np.stack([V[np.ix_(iy, ix)], V[np.ix_(iy + 1, ix)], V[np.ix_(iy, ix + 1)], V[np.ix_(iy + 1, ix + 1)]])
        return np.clip(R, c4.min(0), c4.max(0))

def ibm_fields(D, G, ibm_dir, smooth):
    df = pd.read_csv(os.path.join(ibm_dir, f"ibm_std_data_D{D}.csv"))
    sm, sw = axes_of(df)
    ncol = "Neq_std_unfloored" if "Neq_std_unfloored" in df else "Neq_std"
    N = pivot(df, ncol, sm, sw).astype(float) * 100
    g = pivot(df, "g0", sm, sw).astype(float)            # C(0)/N^2 as in Figure 3 (zero distance)
    # r~ <= 0.5 average, for the like-for-like comparison with the FP ball average (Figs B2, B3);
    # older IBM files only have the ball average, stored in "g0"
    gball = pivot(df, "g0_ball", sm, sw).astype(float) if "g0_ball" in df else g.copy()
    ext = pivot(df, "extinct", sm, sw).astype(float) > 0 if "extinct" in df else N <= 0.1
    nrep = pivot(df, "n_gr_reps", sm, sw).astype(float) if "n_gr_reps" in df else np.full(N.shape, 99.0)
    gbad = ext | ~np.isfinite(g) | (g >= 1999) | (g <= 1.001e-3) | (nrep < 5)
    eq = pivot(df, "equilibrated", sm, sw).astype(float) > 0 if "equilibrated" in df else np.ones(N.shape, bool)
    g_raw = np.where(gbad, np.nan, g)
    sN = SMOOTH_N if SMOOTH_N is not None else smooth
    Ns = gaussian_filter(N, sN, mode="nearest") if sN > 0 else N
    lg = np.log10(np.where(gbad, 50.0, np.clip(g, 1e-3, 50.0)))       # no pairs: top of scale
    lg = gaussian_filter(lg, smooth, mode="nearest") if smooth > 0 else lg
    gbb = ext | ~np.isfinite(gball) | (gball >= 1999) | (gball <= 1.001e-3) | (nrep < 5)
    lgb = np.log10(np.where(gbb, 50.0, np.clip(gball, 1e-3, 50.0)))
    lgb = gaussian_filter(lgb, smooth, mode="nearest") if smooth > 0 else lgb
    return dict(df=df, sm=sm, sw=sw, N=G.refine(Ns, sm, sw), g=10 ** G.refine(lg, sm, sw),
                g_ball=10 ** G.refine(lgb, sm, sw), raw_g_ball=np.where(gbb, np.nan, gball),
                dead=G.refine(ext.astype(float), sm, sw) > 0.5, ext=ext, eq=eq,
                raw_N=N, raw_g=g_raw, gbad=gbad)

def ibm_struct_fields(D, G, st):
    """IBM structure table (from ibm_structure.py) -> what figure4/figure5_l1 need."""
    sm, sw = axes_of(st)
    ext = (pivot(st, "extinct", sm, sw).astype(float) > 0) if "extinct" in st else np.zeros((len(sw), len(sm)), bool)
    g0 = 1.0 + pivot(st, "q0", sm, sw).astype(float)
    g0[ext | ~(g0 > 0)] = np.nan
    # the C(0)/N^2 = 1 line is drawn from a lightly smoothed field (counting noise)
    g0 = 10 ** gaussian_filter(fill_nan(np.log10(g0)), 1.0, mode="nearest")
    return dict(sm=sm, sw=sw, raw_g=g0, ext=ext, dead=G.refine(ext.astype(float), sm, sw) > 0.5)

# ======================================================================= curves at the draft points
def snap(sm_axis, sw_axis):
    return [(float(sm_axis[np.argmin(abs(np.log(sm_axis / a)))]), float(sw_axis[np.argmin(abs(np.log(sw_axis / c)))]))
            for a, c in DRAFT_PTS]

# ======================================================================= drawing primitives
def heat(ax, G, Z, norm, cm, dead=None, diag=True, lx=True, ly=True):
    Zm = np.ma.masked_where(~np.isfinite(Z) | (dead if dead is not None else False), Z)
    ax.set_facecolor(EXT_COLOR)
    ax.pcolormesh(G.x, G.x, Zm, cmap=cm, norm=norm, shading="gouraud", rasterized=True)
    if dead is not None and dead.any():
        ax.contour(G.x, G.x, dead.astype(float), levels=[0.5], colors="k", linewidths=0.6)
    if diag:
        ax.plot([G.x[0], G.x[-1]], [G.x[0], G.x[-1]], "k-", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(G.x[0], G.x[-1]); ax.set_ylim(G.x[0], G.x[-1]); ax.set_aspect("equal")
    t = [v for v in (0.1, 1, 10) if G.x[0] * 0.999 <= v <= G.x[-1] * 1.001]
    ax.set_xticks(t); ax.set_xticklabels([f"{v:g}" for v in t])
    ax.set_yticks(t); ax.set_yticklabels([f"{v:g}" for v in t]); ax.minorticks_off()
    if lx: ax.set_xlabel(SM)
    if ly: ax.set_ylabel(SW)

def surface(ax, G, Z, norm, cm, zf, zt, ztl, dead=None, zlabel=None, diag=True, edge=12):
    s = slice(None, None, 3)
    X, Y = np.meshgrid(G.lf[s], G.lf[s])
    Zs = np.clip(np.where(np.isfinite(Z), Z, norm.vmin), norm.vmin * (1.0001 if norm.vmin > 0 else 1), None)
    # light smoothing of the 3-D surface only (removes aliasing of narrow ridges)
    if isinstance(norm, LogNorm):
        Zs = 10 ** gaussian_filter(np.log10(Zs), 4, mode="nearest")
    else:
        Zs = gaussian_filter(Zs, 4, mode="nearest")
    Zc = Zs                                   # colours: unclipped (over-range -> cmap "over" colour)
    Zs = np.minimum(Zs, norm.vmax)            # heights: clipped to the axis
    hide = binary_dilation(dead, iterations=edge) if dead is not None else np.zeros(Z.shape, bool)
    fc = cm(norm(Zc[s, s]))
    m = hide[s, s].copy(); m[:-1, :-1] |= m[1:, 1:] | m[1:, :-1] | m[:-1, 1:]
    fc[m] = (1, 1, 1, 0)
    ax.plot_surface(X, Y, zf(Zs[s, s]), facecolors=fc, rstride=1, cstride=1, linewidth=0,
                    antialiased=False, shade=False)
    if diag:
        k = np.arange(0, NF, 3)
        dz = np.where(hide[k, k], np.nan, zf(Zs[k, k]))
        ax.plot(G.lf[k], G.lf[k], dz, "k-", lw=1.0, zorder=10)
    t = [v for v in (-1, 0, 1) if G.lf[0] - 1e-9 <= v <= G.lf[-1] + 1e-9]
    ax.set_xticks(t); ax.set_xticklabels([f"{10**v:g}" for v in t])
    ax.set_yticks(t); ax.set_yticklabels([f"{10**v:g}" for v in t])
    ax.set_zticks(zt); ax.set_zticklabels(ztl); ax.set_zlim(zt[0], zt[-1])
    ax.set_xlabel(SM, labelpad=2); ax.set_ylabel(SW, labelpad=2)
    if zlabel: ax.set_zlabel(zlabel, labelpad=8)
    ax.view_init(elev=22, azim=-125); ax.tick_params(pad=0, labelsize=10)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((0.97, 0.97, 0.97, 1))

def two_row_figure():
    fig = plt.figure(figsize=(14, 9.6))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.06], height_ratios=[1, 1.05], left=0.06,
                          right=0.91, top=0.91, bottom=0.04, wspace=0.18, hspace=0.18)
    return fig, gs

def colorbar(fig, ax, norm, cm, ticks, label, ref=None, fmt="{:g}", tickpos=None, extend="neither"):
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm), cax=ax, extend=extend,
                      ticks=ticks if tickpos is None else tickpos)
    cb.ax.set_yticklabels([fmt.format(t) for t in ticks]); cb.ax.minorticks_off()
    if ref is not None:
        cb.ax.axhline(ref, color="k", lw=1.2)
    cb.set_label(label, rotation=270, labelpad=20)
    return cb

def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".png"), dpi=DPI, facecolor="white")
    fig.savefig(os.path.join(out, name + ".pdf"), dpi=DPI, facecolor="white")
    plt.close(fig)
    print("  wrote", name, flush=True)

# ======================================================================= Figure 2
def figure2(fields, method, out):
    G = fields["G"]; cm = cmap_N(); norm = Normalize(0, 600)
    cm.set_over(OVER_COLOR)                   # N~ > 600 % (draft scale): dark red, not the grey top
    over = any(np.nanmax(fields[method][D]["N"]) > 600 for D in DIMS)
    fig, gs = two_row_figure()
    for j, D in enumerate(DIMS):
        f = fields[method][D]
        Z = np.clip(f["N"], 0, None)
        ax = fig.add_subplot(gs[0, j])
        heat(ax, G, Z, norm, cm, ly=(j == 0))
        ax.set_title(f"({'abc'[j]})\n{DIMNAME[D]}")
        a3 = fig.add_subplot(gs[1, j], projection="3d")
        surface(a3, G, Z, norm, cm, lambda z: z, [0, 100, 200, 300, 400, 500, 600],
                ["0", "100", "200", "300", "400", "500", "600"],
                zlabel=(r"Standardized carrying capacity, $\widetilde{N}$ (%)" if j == 0 else None))
        a3.set_title(f"({'def'[j]})", y=0.98)
    colorbar(fig, fig.add_subplot(gs[:, 3]), norm, cm, list(range(0, 601, 100)),
             r"Standardized carrying capacity, $\widetilde{N}$ (%)", ref=100,
             extend=("max" if over else "neither"))
    fig.suptitle(TITLE[method], fontsize=16, fontweight="bold", y=0.985)
    save(fig, out, f"Figure2_carrying_capacity_{method}")

# ======================================================================= Figure 3
def figure3(fields, method, cfp, cibm, out):
    G = fields["G"]
    GMIN, GMAX = 0.03, 50
    norm = LogNorm(GMIN, GMAX); cm = cmap(0.90)
    TICKS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50]
    fig = plt.figure(figsize=(14, 13.8))
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 0.06], height_ratios=[1.05, 1, 0.95], left=0.07,
                          right=0.91, top=0.93, bottom=0.05, wspace=0.2, hspace=0.25)
    for j, D in enumerate(DIMS):
        f = fields[method][D]
        Z = f["g"].copy()
        if method == "FP":
            Z = np.where(f["dead"], GMAX, Z)         # extinct: C(0)/N^2 unbounded -> top
        Z = np.clip(Z, GMIN * 1.001, GMAX * 0.999)
        a3 = fig.add_subplot(gs[0, j], projection="3d")
        surface(a3, G, Z, norm, cm, np.log10, list(np.log10([0.05, 0.1, 1, 10, 50])), ["0.05", "0.1", "1", "10", "50"],
                zlabel=("Standardized zero-distance\npair density, $C(0)/N^2$" if j == 0 else None))
        a3.set_title(f"({'abc'[j]})\n{DIMNAME[D]}", y=1.0)
        ax = fig.add_subplot(gs[1, j])
        heat(ax, G, Z, norm, cm, lx=False, ly=False)
        if j == 1: ax.set_xlabel("Standardized dispersal radius, " + SM)
        if j == 0: ax.set_ylabel("Standardized competition radius, " + SW)
        ax.set_title(f"({'def'[j]})")
        pts = snap(f["sm"], f["sw"]); cols = []
        for k, (a, c) in enumerate(pts):
            i, jj = int(np.argmin(abs(f["sm"] - a))), int(np.argmin(abs(f["sw"] - c)))
            v = f["raw_g"][jj, i]
            col = cm(norm(np.clip(v if np.isfinite(v) else GMAX, GMIN, GMAX)))
            cols.append(col)
            ax.plot(a, c, "o", ms=17, mfc=col, mec="white", mew=1.6)
            ax.text(a, c, str(k + 1), color="white", ha="center", va="center", fontsize=10, fontweight="bold")
        ax = fig.add_subplot(gs[2, j])
        ax.set_facecolor("lightgrey"); ax.axhline(1, color="k", lw=0.8)
        labs, missing, few = [], [], []
        for k in range(1, 8):
            if method == "FP":
                p = cfp[(cfp.D == D) & (cfp.k == k)]
                r, g = p.r.to_numpy(), p.g.to_numpy()
            else:
                p = cibm[(cibm.D == D) & (cibm.k == k)] if cibm is not None else pd.DataFrame()
                r, g = (p.r.to_numpy(), p.g.to_numpy()) if len(p) else (np.array([]), np.array([]))
                if len(g):
                    w = p.pairs.to_numpy().astype(float); ok = w >= 30
                    if ok.sum() < 0.6 * len(w):
                        few.append(k); g = np.array([])
                    else:
                        lg = np.log10(np.clip(g, 1e-3, None)); ker = np.array([0.25, 0.6, 1, 0.6, 0.25])
                        sm_ = np.full(len(lg), np.nan)
                        for i in range(len(lg)):
                            lo, hi = max(0, i - 2), min(len(lg), i + 3)
                            kk = ker[2 - (i - lo):2 + (hi - i)] * w[lo:hi] * ok[lo:hi]
                            if ok[i] and kk.sum() > 0:
                                sm_[i] = kk @ lg[lo:hi] / kk.sum()
                        r, g = r[ok], 10 ** sm_[ok]
            if len(g) == 0:
                if k not in few: missing.append(k)
                continue
            ax.plot(r, g, "-", color=cols[k - 1], lw=1.9, solid_capstyle="round")
            labs.append([k, float(np.clip(g[0], 0.03, 600)), cols[k - 1]])
        labs.sort(key=lambda t: t[1])
        ly = [np.log10(t[1]) for t in labs]
        for i in range(1, len(ly)):
            ly[i] = max(ly[i], ly[i - 1] + 0.16)
        for i in range(len(ly) - 1, -1, -1):
            ly[i] = min(ly[i], np.log10(700) - 0.16 * (len(ly) - 1 - i))
        for (kk, _, col), yy in zip(labs, ly):
            ax.text(0.04, 10 ** yy, str(kk), color=col, fontsize=10, ha="left", va="center", fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        note = []
        if missing: note.append(", ".join(map(str, missing)) + ": population extinct")
        if few: note.append(", ".join(map(str, few)) + ": too few surviving pairs")
        if note:
            ax.text(0.97, 0.04, "\n".join(note), transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="0.25")
        ax.set_yscale("log"); ax.set_ylim(0.02, 1000); ax.set_xlim(0, CURVE_RX)
        yt = [0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500]
        ax.set_yticks(yt); ax.set_yticklabels([f"{t:g}" for t in yt]); ax.minorticks_off()
        ax.set_title(f"({'ghi'[j]})")
        if j == 0: ax.set_ylabel(r"Standardized pair density, $C(\widetilde{r})/N^2$")
        if j == 1: ax.set_xlabel(r"Distance, $\widetilde{r}$")
    colorbar(fig, fig.add_subplot(gs[:, 3]), norm, cm, TICKS,
             r"Standardized zero-distance pair density, $C(0)/N^2$", ref=1)
    fig.suptitle(TITLE[method], fontsize=16, fontweight="bold", y=0.985)
    save(fig, out, f"Figure3_zero_distance_pair_density_{method}")

# ======================================================================= Figures 4 and 5
def fill_harmonic(F, iters=2000):
    """Fill NaN cells with the average of their neighbours (discrete Laplace equation):
    bounded by the surrounding values, no extrapolation."""
    S = F.astype(float).copy(); nan = np.isnan(S)
    if not nan.any():
        return S
    S[nan] = np.nanmean(F)
    for _ in range(iters):
        P = np.pad(S, 1, mode="edge")
        S[nan] = 0.25 * (P[:-2, 1:-1] + P[2:, 1:-1] + P[1:-1, :-2] + P[1:-1, 2:])[nan]
    return S

def struct_field(G, st, col, transform, inverse, smooth=0.0, qband=None, fill="linear"):
    sm, sw = axes_of(st)
    V = transform(pivot(st, col, sm, sw).astype(float))
    if qband is not None:
        V[np.abs(pivot(st, "q0", sm, sw).astype(float)) < qband] = np.nan
    V = np.where(np.isfinite(V), V, np.nan)
    V = fill_harmonic(V) if fill == "harmonic" else fill_nan(V)
    if smooth > 0:
        V = gaussian_filter(V, smooth, mode="nearest")
    return inverse(G.refine(V, sm, sw))

def undefined_line(G, f):
    """C(0)/N^2 = 1 contour (where r_1/2 is undefined), from the full FP data."""
    Q = G.refine(fill_nan(np.log10(f["raw_g"])), f["sm"], f["sw"])
    Q = np.where(f["dead"], np.nan, Q)
    c = plt.figure(); cs = c.add_subplot().contour(G.lf, G.lf, Q, levels=[0.0]); plt.close(c)
    return [s for s in cs.allsegs[0] if len(s) > 5]

def ibm_unresolved(G, st):
    """fine-grid mask of IBM cells without a usable half-saturation distance (off the diagonal)"""
    sm_, sw_ = axes_of(st)
    Uc = ~np.isfinite(pivot(st, "r_half", sm_, sw_).astype(float))
    Uc &= ~np.eye(len(sm_), dtype=bool)                   # the diagonal is "undefined", not unresolved
    Uc = binary_opening(Uc, iterations=1) | (Uc & binary_closing(Uc, iterations=1))   # no isolated single cells
    Uc = binary_closing(Uc, iterations=1) & ~np.eye(len(sm_), dtype=bool)
    Uc = binary_opening(Uc, iterations=1)
    return G.refine(gaussian_filter(Uc.astype(float), 0.6), sm_, sw_) > 0.5

def figure4(fields, structs, out, lo=0.1, hi=10.0, suffix="", method="FP", qband=0.01):
    # colour range 0.1-10: the recomputed half-saturation distances span 0.12-6 (FP).  The draft's
    # original range 0.008-3.5 belonged to values ~10^(-1/n) smaller (distances standardized with a
    # different lambda), see Appendix B.
    G = fields["G"]; cm = cmap()
    src = fields["FP"] if method == "FP" else fields["IBMS"]
    norm = LogNorm(lo, hi)
    ticks = [t for t in (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 50, 100) if lo <= t <= hi and (suffix or t not in (0.15, 0.3, 3))]
    label = r"Half-saturation distance, $\widetilde{r}_{1/2}$"
    fig, gs = two_row_figure()
    for j, D in enumerate(DIMS):
        f = src[D]
        st4 = structs[D]
        if method != "FP":                   # on the diagonal C(0)/N^2 = 1 exactly (Appendix C): undefined
            st4 = st4.copy(); st4.loc[np.isclose(st4.sigma_m, st4.sigma_w), "r_half"] = np.nan
            if "r_half_capped" in st4:       # not decayed within the sampled range: only a lower bound -> unresolved
                st4.loc[st4.r_half_capped == 1, "r_half"] = np.nan
            H = None
            if "r_half_reliable" in st4:     # half level not clearly above the noise / crossing near the range end
                if SHOW_UNCERTAIN:           # keep the measured value, mark it as uncertain (hatched)
                    unc = (st4.r_half_reliable == 0) & np.isfinite(st4.r_half)
                    sm_u, sw_u = axes_of(st4)
                    Hc = pivot(st4.assign(u=unc.astype(float)), "u", sm_u, sw_u).astype(float)
                    H = G.refine(gaussian_filter(np.nan_to_num(Hc), 0.6), sm_u, sw_u) > 0.5
                else:
                    st4.loc[st4.r_half_reliable == 0, "r_half"] = np.nan
        Z = struct_field(G, st4, "r_half", np.log10, lambda x: 10 ** x,
                         smooth=(0.4 if method == "FP" else 0.5), qband=qband,   # verified: map within data (IBM: within counting noise)
                         fill=("linear" if method == "FP" else "harmonic"))
        Z = np.clip(Z, lo * 1.001, hi * 0.999)
        U = None
        if method != "FP":                   # cells the simulations cannot resolve: shown as such, not interpolated
            U = ibm_unresolved(G, st4)
        ax = fig.add_subplot(gs[0, j])
        heat(ax, G, Z, norm, cm, dead=(f["dead"] | U) if U is not None else f["dead"], diag=False, ly=(j == 0))
        if method != "FP" and SHOW_UNCERTAIN and H is not None and H.any():
            ax.contourf(G.x, G.x, H.astype(float), levels=[0.5, 1.5], colors="none", hatches=["////"])
            ax.contour(G.x, G.x, H.astype(float), levels=[0.5], colors="white", linewidths=0.8)
        ax.set_title(f"({'abc'[j]})\n{DIMNAME[D]}")
        a3 = fig.add_subplot(gs[1, j], projection="3d")
        zt = [t for t in (0.2, 0.5, 1, 2, 5) if lo <= t <= hi] if suffix else [t for t in (0.01, 0.1, 1, 10) if lo <= t <= hi]
        surface(a3, G, Z, norm, cm, np.log10, list(np.log10(zt)), [f"{t:g}" for t in zt],
                dead=(f["dead"] | U) if U is not None else f["dead"], zlabel=(label if j == 0 else None), diag=False,
                edge=(2 if U is not None else 12))
        for seg in undefined_line(G, f):
            if U is not None:                # not drawn inside the unresolved region
                xi = np.clip(np.searchsorted(G.lf, seg[:, 0]), 0, NF - 1)
                yi = np.clip(np.searchsorted(G.lf, seg[:, 1]), 0, NF - 1)
                seg = np.where(U[yi, xi][:, None], np.nan, seg)
            ax.plot(10 ** seg[:, 0], 10 ** seg[:, 1], color="grey", lw=4.5, solid_capstyle="round")
            xi = np.clip(np.searchsorted(G.lf, seg[:, 0]), 0, NF - 1)
            yi = np.clip(np.searchsorted(G.lf, seg[:, 1]), 0, NF - 1)
            if U is not None:
                seg = np.where(U[yi, xi][:, None], np.nan, seg)
            a3.plot(seg[:, 0], seg[:, 1], np.log10(Z[yi, xi]), color="grey", lw=2.5, zorder=10)
        a3.set_title(f"({'def'[j]})", y=0.98)
    colorbar(fig, fig.add_subplot(gs[:, 3]), norm, cm, ticks, label)
    fig.suptitle(TITLE[method], fontsize=16, fontweight="bold", y=0.985)
    if method != "FP":
        fig.text(0.45, 0.012, ("Hatched: uncertain (bootstrap range of the fitted curve wider than ±25%).  Light grey: no measurable value."
                 if SHOW_UNCERTAIN else
                 "Light grey: not resolvable from the simulations (C(0)/N² not measurably different from 1, "
                 "or the half level reached beyond the largest distance the habitat allows)"), ha="center", fontsize=10.5, color="0.25")
    save(fig, out, f"Figure4_half_saturation_distance_{method}" + suffix)

def figure5_l1(fields, structs, out, lo=0.01, hi=50.0, method="FP"):
    """Draft definition with the L1 norm:  integral over R^n |log10 C(r)/N^2| dV~  (log colour scale)."""
    G = fields["G"]; cm = cmap()
    src = fields["FP"] if method == "FP" else fields["IBMS"]
    norm = LogNorm(lo, hi)
    ticks = [t for t in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000) if lo <= t <= hi]
    zt = [t for t in (0.01, 0.1, 1, 10, 50, 100, 1000) if lo <= t <= hi]
    label = r"Spatial-structure degree, $\int|\log_{10}\,C(\widetilde{r})/N^2|\,d\widetilde{V}$"
    fig, gs = two_row_figure()
    for j, D in enumerate(DIMS):
        f = src[D]
        # S vanishes on the diagonal (V-shaped valley); interpolate S/|log10(sw/sm)|, which is finite and
        # smooth there, and multiply back: exact zero line, no smearing of the valley.  FP: no smoothing
        # (deterministic); IBM: half a grid cell against counting noise.
        st = structs[D].copy()
        dl = np.abs(np.log10(st.sigma_w / st.sigma_m))
        st["h"] = np.where(dl > 1e-9, st.S / np.where(dl > 1e-9, dl, 1), np.nan)
        H = struct_field(G, st, "h", lambda v: np.log10(np.clip(v, 1e-2, None)), lambda t: 10 ** t,
                         smooth=(0.0 if method == "FP" else 0.5))
        Z = H * np.abs(G.lf[:, None] - G.lf[None, :])
        Z = np.clip(Z, lo * 1.001, hi * 0.999)
        ax = fig.add_subplot(gs[0, j])
        heat(ax, G, Z, norm, cm, dead=f["dead"], ly=(j == 0))
        ax.set_title(f"({'abc'[j]})\n{DIMNAME[D]}")
        a3 = fig.add_subplot(gs[1, j], projection="3d")
        surface(a3, G, Z, norm, cm, np.log10, list(np.log10(zt)), [f"{t:g}" for t in zt], dead=f["dead"],
                zlabel=("Spatial-structure degree" if j == 0 else None), edge=30)
        a3.set_title(f"({'def'[j]})", y=0.98)
    colorbar(fig, fig.add_subplot(gs[:, 3]), norm, cm, ticks, label)
    fig.suptitle(TITLE[method], fontsize=16, fontweight="bold", y=0.985)
    save(fig, out, f"Figure5_spatial_structure_degree_{method}_L1")

TITLE = {"FP": "Moment closure (FP)", "IBM": "Individual-based model (IBM)"}
# =====================================================================================================
# PART 4: the run -- settings, simulations of one array task, combining the results, CSV files, figures
# =====================================================================================================
OUT_DIR = os.environ.get("OUT_DIR", "IBM")                   # results folder
REPS = int(os.environ.get("REPS", 150))                       # realizations per cell (3 x the earlier 50)
REP_BLOCK = int(os.environ.get("REP_BLOCK", 50))              # realizations per work unit
N_AXIS = int(os.environ.get("N_AXIS", 60))                    # grid points per axis (0.1 .. 10)
STRIDE = int(os.environ.get("STRIDE", 1))                     # every STRIDE-th grid point (2 -> 31 x 31)
RUN_DIMS = [int(x) for x in os.environ.get("DIMS", "1,2,3").split(",")]
T_EQUIL = float(os.environ.get("T_EQUIL", 30))                # N averaged over T_EQUIL .. T_END
T_END = float(os.environ.get("T_END", 100))
T_SNAP = float(os.environ.get("T_SNAP", 30))                  # pair-count snapshots from here ...
SNAP_DT = float(os.environ.get("SNAP_DT", 1.0))               # ... every SNAP_DT
T_SNAP_LATE = float(os.environ.get("T_SNAP_LATE", 70))        # late window (strongly clustered cells)
CLUSTER_G0 = float(os.environ.get("CLUSTER_G0", 2.5))
SEED_BASE = int(os.environ.get("SEED_BASE", 20261015))
BOX_STD = {k: float(os.environ.get(f"BOX_STD_{k}D", 0)) for k in (1, 2, 3)}   # 0 = unit habitat
IBM_SMOOTH = float(os.environ.get("IBM_SMOOTH", 0.7))         # display smoothing of the C(0)/N^2 map (cells)
NPROC = int(os.environ.get("NPROC", os.environ.get("SLURM_CPUS_PER_TASK", cpu_count())))
TASK_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
N_TASKS = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 0)) or \
    (int(os.environ["SLURM_ARRAY_TASK_MAX"]) + 1 if "SLURM_ARRAY_TASK_MAX" in os.environ else 1)
SMOOTH_N = 0.0                                                # N~ map: no smoothing (precise enough)

AX = np.logspace(np.log10(0.1), np.log10(10.0), N_AXIS)
IDX = sorted(set(list(range(0, N_AXIS, STRIDE)) + [N_AXIS - 1]))


def box_L(D):
    """habitat side (dimensional); never smaller than the unit habitat"""
    return max(1.0, BOX_STD[D] * length_unit(D))


def rmax_std(D):
    """largest pair distance counted (standardized): below half the habitat side, at most 15"""
    return min(15.0, 0.45 * box_L(D) / length_unit(D))


def snap_cells():
    """grid cells of the 7 marked points of Figure 3"""
    ax = AX[IDX]
    return [(IDX[int(np.argmin(np.abs(np.log(ax / a))))], IDX[int(np.argmin(np.abs(np.log(ax / c))))]) for a, c in DRAFT_PTS]


def seed_of(D, im, iw, rep):
    """an independent random stream for every (cell, realization)"""
    return int(np.random.SeedSequence([SEED_BASE, D, im, iw, rep]).generate_state(1)[0])


def pair_counts(P, e_dim, L):
    """ordered pairs (self pairs excluded) per distance bin, periodic habitat of side L"""
    if len(P) < 2:
        return np.zeros(len(e_dim) - 1)
    T = cKDTree(np.mod(P, L), boxsize=L)
    return np.diff(T.count_neighbors(T, e_dim).astype(float) - len(P))


# ---------------------------------------------------------------- one work unit: REP_BLOCK realizations of one cell
def block(task):
    D, im, iw, r0, r1 = task
    sm_std, sw_std = float(AX[im]), float(AX[iw]); u = length_unit(D); L = box_L(D)
    n0 = int(round(LAM * L ** D)); t_mid = 0.5 * (T_EQUIL + T_END)
    ed = np.arange(0, rmax_std(D) + 1e-9, BW) * u; sv = shell_volume(ed, D); nb = len(ed) - 1
    cnt = {w: dict(obs=np.zeros(nb), exp=np.zeros(nb), n=0) for w in ("early", "late")}
    per = dict(N_eq=[], h1=[], h2=[], t_ext=[], n_final=[])
    t0 = time.time()
    for rep in range(r0, r1):
        t_h, N_h, pos = run_ibm(sigma_b=sm_std * u, sigma_d=sw_std * u, D=D, t_end=T_END, seed=seed_of(D, im, iw, rep),
                                record_dt=0.5, L=L, N0=n0, b=b, d=d, dp=dp, snap_from=T_SNAP, snap_dt=SNAP_DT)
        mask = t_h >= T_EQUIL
        per["N_eq"].append(float(N_h[mask].mean()) if np.any(mask) else 0.0)
        m1 = mask & (t_h < t_mid); m2 = mask & (t_h >= t_mid)
        per["h1"].append(float(N_h[m1].mean()) if np.any(m1) else 0.0)
        per["h2"].append(float(N_h[m2].mean()) if np.any(m2) else 0.0)
        nz = np.nonzero(N_h)[0]
        per["t_ext"].append(float(t_h[nz[-1]]) if (len(nz) and N_h[-1] == 0) else float("nan"))
        per["n_final"].append(float(N_h[-1]))
        for j, P in enumerate(run_ibm.snaps):                 # snapshot j is taken at T_SNAP + j * SNAP_DT
            w = "late" if T_SNAP + j * SNAP_DT >= T_SNAP_LATE - 1e-9 else "early"
            n = len(P)
            cnt[w]["obs"] += pair_counts(P, ed, L); cnt[w]["exp"] += n * (n - 1) * sv / L ** D; cnt[w]["n"] += 1
    return dict(D=D, im=im, iw=iw, r0=r0, r1=r1, L=L, wall_s=time.time() - t0, per=per, cnt=cnt)


def all_blocks():
    """every work unit, most expensive first (dealt round-robin to the array tasks)"""
    blocks = [(r, min(r + REP_BLOCK, REPS)) for r in range(0, REPS, REP_BLOCK)]
    t = [(D, im, iw, a, c) for D in RUN_DIMS for iw in IDX for im in IDX for a, c in blocks]
    t.sort(key=lambda x: -_cost(x[0], AX[x[1]], AX[x[2]]) * (x[4] - x[3]))
    return t


# ---------------------------------------------------------------- work files (CSV, one pair per array task)
FIELDS = ["D", "im", "iw", "r0", "r1", "sigma_m", "sigma_w", "habitat_L", "wall_s", "n_snap_early", "n_snap_late",
          "N_eq", "h1", "h2", "t_ext", "n_final", "obs_early", "exp_early", "obs_late", "exp_late"]
ARR = lambda v: " ".join(f"{x:.10g}" for x in v)


def work_dir():
    return os.path.join(OUT_DIR, "work")


def done_keys():
    """(D, im, iw, r0, r1) of every finished work unit, from all tasks' index files"""
    keys = set()
    for f in glob.glob(os.path.join(work_dir(), "done_task*.csv")):
        try:
            t = pd.read_csv(f, on_bad_lines="skip")
            keys |= set(map(tuple, t[["D", "im", "iw", "r0", "r1"]].astype(int).to_numpy()))
        except Exception:
            pass
    return keys


def write_block(fw, fd, bl):
    c, p = bl["cnt"], bl["per"]
    fw.writerow([bl["D"], bl["im"], bl["iw"], bl["r0"], bl["r1"], AX[bl["im"]], AX[bl["iw"]], bl["L"], round(bl["wall_s"], 2),
                 c["early"]["n"], c["late"]["n"], ARR(p["N_eq"]), ARR(p["h1"]), ARR(p["h2"]), ARR(p["t_ext"]), ARR(p["n_final"]),
                 ARR(c["early"]["obs"]), ARR(c["early"]["exp"]), ARR(c["late"]["obs"]), ARR(c["late"]["exp"])])
    fd.writerow([bl["D"], bl["im"], bl["iw"], bl["r0"], bl["r1"]])


def run():
    os.makedirs(work_dir(), exist_ok=True)
    mine = all_blocks()[TASK_ID::N_TASKS]
    have = done_keys()
    todo = [t for t in mine if t not in have]
    print(f"task {TASK_ID} of {N_TASKS}: {len(mine)} work units, {len(mine) - len(todo)} already done, {len(todo)} to run | "
          f"{REPS} realizations per cell (units of {REP_BLOCK}), t = 0..{T_END:g}, N averaged from {T_EQUIL:g}, snapshots every "
          f"{SNAP_DT:g} from {T_SNAP:g} (late window from {T_SNAP_LATE:g}) | {NPROC} processes | d = {d:g}, lambda = {LAM:g}, "
          f"dispersal = {DISPERSAL}, competition cut = {KCUT:g} sigma", flush=True)
    if todo:
        wf = os.path.join(work_dir(), f"blocks_task{TASK_ID:04d}.csv"); df_ = os.path.join(work_dir(), f"done_task{TASK_ID:04d}.csv")
        new_w, new_d = not os.path.exists(wf), not os.path.exists(df_)
        t0 = time.time()
        with open(wf, "a", newline="") as hw, open(df_, "a", newline="") as hd, Pool(NPROC) as pool:
            fw, fd = csv.writer(hw), csv.writer(hd)
            if new_w: fw.writerow(FIELDS)
            if new_d: fd.writerow(["D", "im", "iw", "r0", "r1"])
            for k, bl in enumerate(pool.imap_unordered(block, todo), 1):
                write_block(fw, fd, bl); hw.flush(); hd.flush()           # the block row first, then its index entry
                if k % 20 == 0 or k == len(todo):
                    print(f"  {k}/{len(todo)} work units  {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"task {TASK_ID} finished its share", flush=True)
    maybe_finish()


def maybe_finish():
    """the task that completes the last work unit writes the data files and figures (once)"""
    n_done, n_all = len(done_keys() & set(all_blocks())), len(all_blocks())
    print(f"overall: {n_done} of {n_all} work units done", flush=True)
    if n_done < n_all:
        print("other tasks are still running (or must be resubmitted); the last one writes IBM/data and IBM/figures", flush=True)
        return
    try:
        os.mkdir(os.path.join(work_dir(), "finish.lock"))
    except FileExistsError:
        print("another task is already writing the results (if it failed: python IBM.py finish)", flush=True)
        return
    finish()


# ---------------------------------------------------------------- combine -> CSV data files
def read_work():
    """all finished work units, grouped by cell"""
    by = {}
    for f in sorted(glob.glob(os.path.join(work_dir(), "blocks_task*.csv"))):
        for _, row in pd.read_csv(f, on_bad_lines="skip", dtype={c: str for c in FIELDS[11:]}).iterrows():
            key = (int(row.D), int(row.im), int(row.iw))
            bl = by.setdefault(key, {})
            bl[(int(row.r0), int(row.r1))] = row                  # duplicates (re-run units) are counted once
    return by


def _arr(s):
    return np.array(str(s).split(), dtype=float) if isinstance(s, str) and s.strip() else np.zeros(0)


def analyse_cell(item):
    """one cell: combine its work units, choose the snapshot window, C(0)/N^2, r_1/2, S"""
    (D, im, iw), rows = item
    rows = [rows[k] for k in sorted(rows)]
    per = {k: np.concatenate([_arr(r[k]) for r in rows]) for k in ("N_eq", "h1", "h2", "t_ext", "n_final")}
    cnt = {w: dict(obs=sum(_arr(r[f"obs_{w}"]) for r in rows), exp=sum(_arr(r[f"exp_{w}"]) for r in rows),
                   n=int(sum(int(r[f"n_snap_{w}"]) for r in rows))) for w in ("early", "late")}
    nrep = int(sum(int(r.r1) - int(r.r0) for r in rows)); L = float(rows[0].habitat_L)
    sm_std, sw_std = float(AX[im]), float(AX[iw]); u = length_unit(D)
    N_eq = float(per["N_eq"].mean()); h1, h2 = float(per["h1"].mean()), float(per["h2"].mean())
    drift = (h2 - h1) / max(0.5 * (h1 + h2), 1e-12)
    neq = N_eq / (LAM * L ** D); extinct = int(neq < NEQ_FLOOR)
    rc = np.arange(len(cnt["late"]["obs"])) * BW + BW / 2
    # snapshot window: late only where the IBM itself is strongly clustered, otherwise both windows pooled
    g_late, _ = g0_origin(rc, cnt["late"]["obs"], cnt["late"]["exp"], max(BW, 0.25 * min(sm_std, sw_std)))
    clustered = (not extinct) and np.isfinite(g_late) and g_late >= CLUSTER_G0 and sm_std < sw_std
    if clustered:
        obs, exp, nsn, t_snap = cnt["late"]["obs"], cnt["late"]["exp"], cnt["late"]["n"], T_SNAP_LATE
    else:
        obs, exp = cnt["early"]["obs"] + cnt["late"]["obs"], cnt["early"]["exp"] + cnt["late"]["exp"]
        nsn, t_snap = cnt["early"]["n"] + cnt["late"]["n"], T_SNAP
    st = analyse(D, rc, obs, exp, sm_std, sw_std)
    inner = rc <= 0.5 + 1e-9
    te = per["t_ext"][np.isfinite(per["t_ext"])]
    row = dict(method="IBM", D=D, sigma_m=sm_std, sigma_w=sw_std, sigma_m_dim=sm_std * u, sigma_w_dim=sw_std * u, lam=LAM,
               length_unit=u, Neq_std=max(neq, NEQ_FLOOR), Neq_std_unfloored=neq, Neq_raw=N_eq,
               Neq_sd_between_reps=float(per["N_eq"].std() / (LAM * L ** D)),
               g0=(1.0 + st["q0"]) if (not extinct and np.isfinite(st["q0"])) else G0_CAP, g0_sd=st["sd_q0"],
               g0_ball=float(obs[inner].sum() / exp[inner].sum()) if exp[inner].sum() > 0 else np.nan,
               extinct=extinct, survival=float(np.mean(per["n_final"] > 0)), n_reps=nrep, neq_half1=h1, neq_half2=h2,
               drift=drift, equilibrated=int(abs(drift) < EQ_DRIFT_TOL), t_extinct_mean=float(te.mean()) if len(te) else np.nan,
               n_final_mean=float(per["n_final"].mean()), clustered_late_window=int(clustered), t_snap=t_snap,
               n_snaps=int(nsn), n_gr_reps=int(nsn), gr_pairs=float(obs.sum()), habitat_std=L / u,
               cpu_hours=float(sum(float(r.wall_s) for r in rows) / 3600))
    srow = dict(D=D, sigma_m=sm_std, sigma_w=sw_std, N=neq, drift=drift, extinct=extinct,
                status="extinct" if extinct else "ok", **st)
    return D, im, iw, row, srow, rc, obs, exp


def finish():
    t0 = time.time()
    data, figs = os.path.join(OUT_DIR, "data"), os.path.join(OUT_DIR, "figures")
    os.makedirs(data, exist_ok=True); os.makedirs(figs, exist_ok=True)
    by = read_work()
    print(f"[finish] {len(by)} cells with results; analysing on {NPROC} processes ...", flush=True)
    with Pool(NPROC) as pool:
        res = pool.map(analyse_cell, sorted(by.items()), chunksize=4)
    pts = snap_cells(); cpts = []
    for D in RUN_DIMS:
        r = [x for x in res if x[0] == D]
        short = sum(1 for x in r if x[3]["n_reps"] < REPS)
        print(f"  D={D}: {len(r)} of {len(IDX) ** 2} cells" + (f"  ({short} with fewer than {REPS} realizations so far)" if short else ""), flush=True)
        if not r:
            continue
        pd.DataFrame([x[3] for x in r]).sort_values(["sigma_w", "sigma_m"]).to_csv(f"{data}/ibm_std_data_D{D}.csv", index=False)
        pd.DataFrame([x[4] for x in r]).sort_values(["sigma_w", "sigma_m"]).to_csv(f"{data}/ibm_structure_D{D}.csv", index=False)
        pc = []
        for _, im, iw, row, _, rc, obs, exp in sorted(r, key=lambda x: (x[2], x[1])):
            g = np.where(exp > 0, obs / np.where(exp > 0, exp, 1), np.nan)
            pc.append(pd.DataFrame(dict(sigma_m=row["sigma_m"], sigma_w=row["sigma_w"], r=np.round(rc, 4), obs_pairs=obs,
                                        exp_pairs=np.round(exp, 4), C_over_N2=g)))
            for k in [i + 1 for i, p in enumerate(pts) if p == (im, iw)]:       # marked point(s) in this cell
                for j in np.nonzero(rc <= CURVE_RX)[0]:
                    cpts.append(dict(D=D, k=k, sigma_m=row["sigma_m"], sigma_w=row["sigma_w"], r=rc[j], g=g[j], pairs=obs[j]))
        pd.concat(pc).to_csv(f"{data}/ibm_pair_counts_D{D}.csv", index=False)
    pd.DataFrame(cpts, columns=["D", "k", "sigma_m", "sigma_w", "r", "g", "pairs"]).to_csv(f"{data}/ibm_curves_points.csv", index=False)
    settings = dict(REPS=REPS, REP_BLOCK=REP_BLOCK, N_AXIS=N_AXIS, STRIDE=STRIDE, T_EQUIL=T_EQUIL, T_END=T_END, T_SNAP=T_SNAP,
                    SNAP_DT=SNAP_DT, T_SNAP_LATE=T_SNAP_LATE, CLUSTER_G0=CLUSTER_G0, SEED_BASE=SEED_BASE, b=b, d=d, dprime=dp,
                    lam=LAM, dispersal=DISPERSAL, kcut=KCUT, pair_bin_width=BW,
                    **{f"habitat_std_{D}D": box_L(D) / length_unit(D) for D in RUN_DIMS},
                    **{f"rmax_std_{D}D": rmax_std(D) for D in RUN_DIMS})
    pd.DataFrame(list(settings.items()), columns=["setting", "value"]).to_csv(f"{data}/run_settings.csv", index=False)
    print(f"[finish] data written to {data}/  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    if any(sum(1 for x in res if x[0] == D) < len(IDX) ** 2 for D in RUN_DIMS):
        print("[finish] figures skipped: not all cells have results yet (run 'python IBM.py status')", flush=True)
        return
    figures(data, figs)
    print(f"[finish] done: {data}/ and {figs}/", flush=True)


# ---------------------------------------------------------------- figures 2-5 (IBM)
def figures(data, figs):
    d0 = pd.read_csv(os.path.join(data, f"ibm_std_data_D{RUN_DIMS[0]}.csv"))
    if len(RUN_DIMS) < 3 or d0.sigma_m.nunique() < 4:
        print("[figures] skipped: the figures need all three dimensions and at least 4 x 4 cells", flush=True)
        return
    G = Grid(d0.sigma_m.min(), d0.sigma_m.max())
    fields = {"G": G, "FP": {}, "IBM": {D: ibm_fields(D, G, data, IBM_SMOOTH) for D in DIMS}}
    print("[figures] IBM", flush=True)
    figure2(fields, "IBM", figs)
    pi = os.path.join(data, "ibm_curves_points.csv")
    figure3(fields, "IBM", None, pd.read_csv(pi) if os.path.exists(pi) else None, figs)
    istr = {D: pd.read_csv(os.path.join(data, f"ibm_structure_D{D}.csv")) for D in DIMS}
    fields["IBMS"] = {D: ibm_struct_fields(D, G, istr[D]) for D in DIMS}
    figure4(fields, istr, figs, method="IBM")
    figure5_l1(fields, istr, figs, method="IBM")


# ---------------------------------------------------------------- progress, cost estimate, self-test
def status():
    n_done, n_all = len(done_keys() & set(all_blocks())), len(all_blocks())
    print(f"{n_done} of {n_all} work units done ({100 * n_done / max(n_all, 1):.1f}%) -- {REPS} realizations per cell in "
          f"units of {REP_BLOCK}; results in {OUT_DIR}/work/")


def estimate_cost(n_per_dim=8, cpus=32, hours=24.0):
    sample = []
    for D in RUN_DIMS:
        cells = sorted(((im, iw) for iw in IDX for im in IDX), key=lambda c: _cost(D, AX[c[0]], AX[c[1]]))
        sample += [(D, *cells[i], 0, 1) for i in np.unique(np.linspace(0, len(cells) - 1, n_per_dim).astype(int))]
    print(f"timing {len(sample)} cells (one realization each, t = 0..{T_END:g}) on {NPROC} processes ...", flush=True)
    with Pool(NPROC) as pool:
        res = list(pool.imap_unordered(block, sample))
    tot = 0.0
    for D in RUN_DIMS:
        r = [x for x in res if x["D"] == D]
        ratio = np.median([x["wall_s"] / _cost(D, AX[x["im"]], AX[x["iw"]]) for x in r])
        sec = ratio * sum(_cost(D, AX[im], AX[iw]) for iw in IDX for im in IDX) * REPS; tot += sec
        print(f"  {D}D: {min(x['wall_s'] for x in r):.1f}-{max(x['wall_s'] for x in r):.1f} s per realization -> about "
              f"{sec / 3600:,.0f} CPU hours ({len(IDX) ** 2} cells x {REPS} realizations)", flush=True)
    n = int(np.ceil(tot / 3600 / (cpus * hours * 0.8)))
    print(f"total about {tot / 3600:,.0f} CPU hours (rough, +-50%): at least {n} array tasks of {cpus} CPUs x {hours:g} h; "
          f"more tasks finish sooner. The run is resumable, so too few only means resubmitting.")


def self_test():
    global IDX, REPS, REP_BLOCK, T_END, T_EQUIL, T_SNAP, T_SNAP_LATE, OUT_DIR, N_TASKS, TASK_ID
    IDX, REPS, REP_BLOCK, T_END, T_EQUIL, T_SNAP, T_SNAP_LATE = [0, 20, 40, 59], 4, 2, 40.0, 15.0, 15.0, 28.0
    OUT_DIR, N_TASKS, TASK_ID = "IBM_test", 1, 0
    print("self-test: 4 x 4 cells per dimension, 4 realizations, t <= 40 -> IBM_test/", flush=True)
    run()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        run()
    elif cmd == "finish":
        finish()
    elif cmd == "status":
        status()
    elif cmd == "estimate":
        estimate_cost()
    elif cmd == "test":
        self_test()
    else:
        sys.exit(__doc__)
