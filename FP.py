"""FP.py  --  moment closure (fixed point, FP): data and Figures 2-5 of the draft, FP panels.

ONE script does everything.  Run it with run_FP.slurm (sbatch run_FP.slurm); all settings are at the top
of that Slurm file.  Results go to the folder FP/ (next to this script):

  FP/data/fp_rmax_{1,2,3}d.csv        one row per (sigma~_m, sigma~_w): N~ (Figure 2), C(0)/N^2 at zero
                                      distance (Figure 3), r~ <= 0.5 average, solver status, ...
  FP/data/fp_structure_{1,2,3}d.csv   half-saturation distance r_1/2 (Figure 4) and spatial-structure
                                      degree S (Figure 5); S_trunc = S integrated only up to the IBM's largest
                                      distance (r~ = 15 / 9 / 3.3), for comparisons with the IBM
  FP/data/fp_curves_points.csv        C(r~)/N^2 at the 7 marked points (Figure 3 g-i)
  FP/data/run_settings.csv            the settings of the run
  FP/figures/Figure2..Figure5 *_FP.png and .pdf
  FP/work/                            partial results of the array tasks (the run resumes from these)
  FP/logs/                            Slurm logs

Model (the draft): b = 0.4, d = 0, d' = 0.001 (lambda = 400), n-dimensional Gaussian dispersal and
untruncated Gaussian competition, moment closure with weights 4:1:1 (your fp_rmax.py; the solver functions
below are taken from it unchanged).  Grid: 60 x 60 standardized radii sigma~_m, sigma~_w (log scale
0.1 .. 10) in 1, 2 and 3 dimensions.  The FP is deterministic: one solve per cell (no realizations).

  N~        the fixed point N divided by lambda
  C(0)/N^2  the solved pair density at zero distance (point value)
  r_1/2     the smallest r~ beyond which |C(r~)/N^2 - 1| stays below half of |C(0)/N^2 - 1|
  S         integral over R^n of |log10 C(r~)/N^2| dV~

Command line (run_FP.slurm calls "run"; the rest is for checking by hand):
  python FP.py run        this array task's share of the cells; the task that completes the last share then
                          writes FP/data and FP/figures automatically
  python FP.py finish     write FP/data and FP/figures from whatever has been computed
  python FP.py status     progress
  python FP.py test       quick local check: 4 x 4 cells per dimension (output in FP_test/)
"""
import os, sys, csv, glob, time, math
# ---- the draft model (can be overridden from the Slurm script) ----
os.environ.setdefault("DISPERSAL", "gaussian")      # n-dimensional Gaussian dispersal
os.environ.setdefault("COMP_TRUNC", "0")            # untruncated Gaussian competition
os.environ.setdefault("G0_MODE", "point")           # C(0)/N^2 at zero distance
os.environ.setdefault("D_RATE", "0.0")              # d = 0 -> lambda = (b - d) / d' = 400
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
from scipy.special import jv, j0, jn_zeros, gamma, erf, gammainc
from multiprocessing import Pool, cpu_count


# =====================================================================================================
# PART 1: the moment-closure solver -- functions of your fp_rmax.py, unchanged (only d = 0 for the draft model)
# =====================================================================================================
# ==========================================================================
# ===============  MODEL PARAMETERS  =======================================
# ==========================================================================
b, d, s = 0.4, float(os.environ.get("D_RATE", 0.0)), 0.001   # draft model: d = 0 (lambda = 400)

# Asymmetric power-2 closure weights (alpha, beta, gamma).
#   4,1,1   Law, Murrell & Dieckmann (2003), Eq. 6 -- the weights of the 2x2
#           IBM-vs-deterministic notebook; DEFAULT.  Against the IBM this
#           fixes the D1 over-estimate (e.g. sigma~=1.04/1.12: IBM 86,
#           old 160, now 76) and the clustered D2 cells (IBM 116 at
#           sigma=0.05/0.05: old 159, now 109).
#   1,1,0.5 the weights of the previous fp/td code:  CLOSURE=1,1,0.5
al, be, ga = (float(v) for v in os.environ.get("CLOSURE", "4,1,1").split(","))

ag = al + ga

N_MF = (b - d) / s

# ==========================================================================
# ===============  NONDIMENSIONALIZATION + COMMON SWEEP AXIS  ==============
# ==========================================================================
DPRIME = s

LAM = (b - d) / DPRIME                         # mean-field equilibrium density = 200

def length_unit(D):     return LAM ** (-1.0 / D)

# ==========================================================================
# ===============  KERNELS: ALIGNED WITH THE IBM (ibm_sweep_all.py)  =======
# ==========================================================================
# DISPERSAL  "ibm"      (default) exactly the IBM / points.c birth event,
#                       moverad_model 2: offspring DISTANCE r = |N(0,sigma_m)|
#                       truncated at 3 sigma_m, direction uniform.  In n dims the
#                       displacement density is  p(r) / (S_n r^(n-1)),  which is
#                       a truncated Gaussian in n=1 but NOT a Gaussian in n=2,3.
#            "gaussian" the previous deterministic kernel, an n-dim isotropic
#                       Gaussian.  At equal sigma~_m it disperses 1.44x (n=2) and
#                       1.75x (n=3) further (rms) than the IBM does, so the two
#                       models were not being compared at the same parameters.
DISPERSAL = os.environ.get("DISPERSAL", "ibm").lower()

# COMP_TRUNC 1 (default) competition kernel truncated at 3 sigma_w and
#              renormalised, as in the IBM (kernel_norm) and points.c (int_norm).
#            0 untruncated Gaussian (previous deterministic code): 0.27% / 1.1% /
#              3.0% weaker inside 3 sigma in n = 1 / 2 / 3 -- a bias that grows
#              with dimension.
COMP_TRUNC = int(os.environ.get("COMP_TRUNC", 1))

# G0_MODE    "ball" (default) the IBM's own estimator: pair density averaged over
#                   r~ <= G0_BALL_STD (= GR_N0_BINS * GR_BINW_STD = 2 * 0.25 in
#                   ibm_sweep_all.py), i.e. ratio of pooled pair counts.
#            "point" q(0)+1 at zero distance (previous deterministic code).  With
#                   the IBM dispersal law the pair density is (integrably)
#                   SINGULAR at r = 0 in n = 2, 3, so the pointwise value does not
#                   exist there -- only the ball average can be compared with the
#                   IBM.  Both are always written to the CSV.
G0_MODE = os.environ.get("G0_MODE", "ball").lower()

G0_BALL_STD = float(os.environ.get("G0_BALL_STD", 0.5))

# ==========================================================================
# ===============  RADIAL GRID  (replaces the capped Cartesian grid)  ======
# ==========================================================================
# Both kernels are isotropic, so the equilibrium q(x) is radial and every
# convolution in the closure is done with a Hankel transform of order n/2-1 on a
# 1-D radial grid (same algorithm for n = 1, 2, 3; only the Bessel order
# changes).  The old n-dim Cartesian grid was capped at 1601 / 301 / 81-121
# points per axis, so the narrower kernel fell below ONE grid step in 8% / 52% /
# 81% of the D1 / D2 / D3 cells (RMAX_STD=50), and the cap switches drew the
# straight edges.  Here the step is dr = sigma_min / PPS in EVERY cell -- the
# discretisation error is then a smooth function of (sigma_m, sigma_w) and
# cannot draw edges -- and the box is R >= R_MULT * sigma_max.
PPS = {1: float(os.environ.get("PPS_1D", 4)),        # points per sigma_min
       2: float(os.environ.get("PPS_2D", 8)),
       3: float(os.environ.get("PPS_3D", 8))}

R_MULT = float(os.environ.get("R_MULT", 8.0))        # box / sigma_max

R_MIN_MULT = 6.0                                     # never below this

# Optional STANDARDIZED floor on the box, r~_max.  Default 0 = off: measured,
# N and g0 are identical to 5 significant figures for R_MULT = 6, 12, 24, 48,
# so a bigger box buys nothing -- resolution was the problem, not range.
RMAX_STD = float(os.environ.get("RMAX_STD", 0.0))

NMAX = int(os.environ.get("NMAX", 6144))

N_LADDER = (128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192)

MODEL_TAG = (f"radial1|clos={al:g},{be:g},{ga:g}|disp={DISPERSAL}|trunc={COMP_TRUNC}|g0={G0_MODE}"
             f"|pps={PPS[1]:g},{PPS[2]:g},{PPS[3]:g}|rmult={R_MULT:g}|rmax={RMAX_STD:g}")

# positivity / extinction thresholds (on the BOUNDED abundance N and on g0)
N_EXT     = float(os.environ.get("N_EXT", 1e-2))    # N < N_EXT      -> extinct (N->0)

G_EXPLODE = float(os.environ.get("G_EXPLODE", 1e8)) # g0 > G_EXPLODE -> extinct (q->inf)

N_CAP     = float(os.environ.get("N_CAP", 1e6))     # N > N_CAP      -> runaway (nonconv guard)

G0_CAP    = float(os.environ.get("G0_CAP", 2000.0))    # extinct g(0)=+inf recorded here

# ==========================================================================
# ===============  RADIAL SPECTRAL ENGINE  =================================
# ==========================================================================
# n-dimensional convolution of isotropic functions by the quasi-discrete Hankel
# transform (Guizar-Sicairos & Gutierrez-Vega, JOSA A 21, 53, 2004), order
# nu = n/2 - 1:  n=1 -> nu=-1/2 (cosine transform), n=2 -> nu=0, n=3 -> nu=+1/2
# (sine transform).
#     f^(k) = (2 pi)^(n/2) k^-nu  H_nu[r^nu f](k),   (f*g)^ = f^ g^
# Nodes are the zeros j_i of J_nu:  r_i = j_i R/S,  k_i = j_i/R,  S = j_(N+1).
# The transform matrix J_nu(j_m j_i / S) depends on (n, N) only, not on R, so it
# is built once per size and cached.  Checked: Gaussian transforms exact to
# 1e-15, a 50x scale-ratio convolution exact to 1e-14 in every n, and the full
# closure reproduces the old Cartesian solver to 8-16 digits wherever that
# solver was resolved (see --check).
_A_CACHE = {}

_Z_CACHE = {}

def _bessel_zeros(D, n):
    key = (D, n)
    if key not in _Z_CACHE:
        i = np.arange(1, n + 1, dtype=float)
        if D == 1:
            z = (i - 0.5) * math.pi              # J_{-1/2}(z) ~ cos z / sqrt z
        elif D == 3:
            z = i * math.pi                      # J_{+1/2}(z) ~ sin z / sqrt z
        else:
            z = jn_zeros(0, n)
        _Z_CACHE[key] = z
    return _Z_CACHE[key]

def _hankel_matrix(D, N):
    key = (D, N)
    if key not in _A_CACHE:
        z = _bessel_zeros(D, N + 1)
        S = z[N]; jz = z[:N]
        X = np.outer(jz, jz) / S
        if D == 1:
            A = np.sqrt(2.0 / (math.pi * X)) * np.cos(X)
        elif D == 3:
            A = np.sqrt(2.0 / (math.pi * X)) * np.sin(X)
        else:
            A = j0(X)
        _A_CACHE[key] = np.ascontiguousarray(A)
    return _A_CACHE[key]

class Radial:
    """Isotropic functions on R^D sampled on the Bessel-zero radial grid."""

    def __init__(self, D, R, N):
        self.D, self.R, self.N = D, float(R), int(N)
        nu = D / 2.0 - 1.0
        self.nu = nu
        z = _bessel_zeros(D, N + 1)
        S = z[N]; jz = z[:N]
        K = S / R
        self.r = jz / K
        self.k = jz / R
        Jp = np.abs(jv(nu + 1.0, jz))
        alpha = 2.0 / (K * K * Jp * Jp)
        beta = 2.0 / (R * R * Jp * Jp)
        self.A = _hankel_matrix(D, N)
        c = (2.0 * math.pi) ** (D / 2.0)
        self.af = alpha * self.r ** nu
        self.cf = c * self.k ** (-nu)
        self.ai = beta * self.k ** nu
        self.ci = self.r ** (-nu) / c
        self.W = c / (2.0 ** nu * gamma(nu + 1.0)) * alpha * self.r ** (2 * nu)   # int f dV
        self.w0 = beta * self.k ** (2 * nu) / (c * 2.0 ** nu * gamma(nu + 1.0))   # f(0)

    def fwd(self, f):
        if f.ndim == 1:
            return self.cf * (self.A @ (self.af * f))
        return self.cf[:, None] * (self.A @ (self.af[:, None] * f))

    def inv(self, F):
        if F.ndim == 1:
            return self.ci * (self.A @ (self.ai * F))
        return self.ci[:, None] * (self.A @ (self.ai[:, None] * F))

    def integral(self, f):
        return float(np.dot(self.W, f))

    def value_at_zero(self, F):
        return float(np.dot(self.w0, F))

    def eval_from_hat(self, F, rr):
        rr = np.asarray(rr, float)
        c = (2.0 * math.pi) ** (self.D / 2.0)
        return rr ** (-self.nu) / c * (jv(self.nu, np.outer(rr, self.k)) @ (self.ai * F))

    def ball_mean(self, F, Rb, nq=96):
        """(1/|B_Rb|) * integral of f over the ball of radius Rb, from f^."""
        D = self.D
        x, wq = np.polynomial.legendre.leggauss(nq)
        rr = 0.5 * Rb * (x + 1.0); ww = 0.5 * Rb * wq
        fv = self.eval_from_hat(F, rr)
        vol = math.pi ** (D / 2.0) * Rb ** D / math.gamma(D / 2.0 + 1.0)
        return float(np.sum(ww * _S_D(D) * rr ** (D - 1) * fv) / vol)

def _S_D(D):
    return {1: 2.0, 2: 2.0 * math.pi, 3: 4.0 * math.pi}[D]

# ==========================================================================
# ===============  KERNELS ON THE RADIAL GRID  =============================
# ==========================================================================
class Kern:
    """Grid + both kernels (values and transforms) for one cell."""
    pass

def grid_size(D, sm, sw):
    """(N, R, resolved) with dr = sigma_min/PPS exactly and R >= R_MULT sigma_max."""
    smin, smax = min(sm, sw), max(sm, sw)
    dr = smin / PPS[D]
    R_want = max(R_MULT * smax, RMAX_STD * length_unit(D))
    need = int(math.ceil(R_want / dr))
    N = next((n for n in N_LADDER if n >= need), N_LADDER[-1])
    N = min(N, NMAX)
    S = _bessel_zeros(D, N + 1)[N]
    R = dr * S / math.pi                        # node spacing pi R / S == dr
    resolved = 1
    if R < R_MIN_MULT * smax:                   # size cap hit: keep the box, coarsen dr
        R = R_MIN_MULT * smax; resolved = 0
    return N, R, resolved

def build_kernels(D, sm, sw):
    """Dimensional radii sm (dispersal), sw (competition) -> Kern.

    Point samples wherever a kernel is smooth (spectral accuracy); exact cell
    mass / cell volume in the shells that contain a non-smooth point -- the
    3-sigma truncation radius, and r = 0 where the IBM dispersal density is
    singular (n >= 2).  Both kernels are then normalised to integrate to 1 on
    the grid, so the mean-field limit N -> (b-d)/s is exact.
    """
    N, R, resolved = grid_size(D, sm, sw)
    g = Radial(D, R, N)
    r = g.r
    e = np.empty(N + 1); e[0] = 0.0
    e[1:-1] = 0.5 * (r[1:] + r[:-1]); e[-1] = min(R, r[-1] + 0.5 * (r[-1] - r[-2]))

    def shell_of(x):
        """Shell whose INTERIOR contains x, or None.  With dr = sigma_min/PPS the
        3-sigma cut of the narrower kernel falls exactly ON a shell edge in n=1
        (3 sigma = 3 PPS dr); point samples are then already exact on both sides,
        and letting round-off pick a shell would make the result jump between
        cells."""
        i = int(np.searchsorted(e, x) - 1)
        if not 0 <= i < N:
            return None
        tol = 1e-9 * (e[i + 1] - e[i])
        if x - e[i] <= tol or e[i + 1] - x <= tol:
            return None
        return i

    if DISPERSAL == "ibm":
        Z = erf(3.0 / math.sqrt(2.0))
        cdf = lambda x: erf(np.minimum(x, 3 * sm) / (sm * math.sqrt(2.0))) / Z
        p = math.sqrt(2.0 / math.pi) / sm * np.exp(-r * r / (2 * sm * sm)) / Z
        m = np.where(r <= 3 * sm, p / (_S_D(D) * r ** (D - 1)), 0.0)
        special = {shell_of(3 * sm)} | ({0} if D >= 2 else set())
    elif DISPERSAL == "gaussian":
        cdf = lambda x: gammainc(D / 2.0, x * x / (2 * sm * sm))
        m = np.exp(-r * r / (2 * sm * sm)) / (2 * math.pi * sm * sm) ** (D / 2.0)
        special = set()
    else:
        raise ValueError(f"DISPERSAL must be 'ibm' or 'gaussian', not {DISPERSAL!r}")
    # special shells: exact mass / QUADRATURE weight, so that the grid integral
    # of that shell returns exactly its mass (dividing by the geometric shell
    # volume loses 5% (n=2) / 11% (n=3) of the innermost shell's mass).
    for i in special:
        if i is not None:
            m[i] = (cdf(e[i + 1]) - cdf(e[i])) / g.W[i]

    w = np.exp(-r * r / (2 * sw * sw)) / (2 * math.pi * sw * sw) ** (D / 2.0)
    if COMP_TRUNC:
        w = np.where(r <= 3 * sw, w, 0.0)
        i = shell_of(3 * sw)
        if i is not None:
            # mass of the (unit-mass) Gaussian inside the shell, up to 3 sigma
            cw = lambda x: gammainc(D / 2.0, np.minimum(x, 3 * sw) ** 2 / (2 * sw * sw))
            w[i] = (cw(e[i + 1]) - cw(e[i])) / g.W[i]

    m = m / g.integral(m)
    w = w / g.integral(w)
    K = Kern()
    K.D, K.g, K.m, K.w = D, g, m, w
    K.mh, K.wh = g.fwd(m), g.fwd(w)
    K.info = dict(n_grid=N, r_max_std=R / length_unit(D),
                  dr_over_sigma=(math.pi * R / _bessel_zeros(D, N + 1)[N]) / min(sm, sw),
                  resolved=resolved)
    return K

# ==========================================================================
# ===============  CLOSURE OPERATORS  (model math unchanged)  ==============
# ==========================================================================
def _convolutions(q, K):
    """m*q, w*q, (q w)*q  -- two stacked transforms per call."""
    g = K.g
    H = g.fwd(np.column_stack([q, q * K.w]))
    qh, qwh = H[:, 0], H[:, 1]
    C = g.inv(np.column_stack([K.mh * qh, K.wh * qh, qwh * qh]))
    return C[:, 0], C[:, 1], C[:, 2]

def picard_map(q, K):
    """The fixed-point map of the original _picard_step, on the radial grid."""
    m, w = K.m, K.w
    Y = K.g.integral(w * q)
    N = (b - d) / (s * (1 + Y)) if 1 + Y > 1e-300 else np.inf
    mq, wq, qwq = _convolutions(q, K)
    D_ = ag * (d + s * w) + al * (b - d) + be * s * N
    qn = (ag / D_) * (b * m / N + b * mq - s * w) \
        - (s * N / D_) * ((be * (q + 1) + ga) * wq + ga * qwq)
    return qn, N

def td_rhs(q, N, K):
    """Right-hand side of the original solve_td_positive (dq/dt, dN/dt)."""
    m, w = K.m, K.w
    N2 = N ** 2
    Y = K.g.integral(w * q)
    mq, wq, qwq = _convolutions(q, K)
    C = N2 * (q + 1); Gamma = N2 * (1 + Y)
    mC = N2 * (mq + 1); wC = N2 * (wq + 1)
    B = N2 ** 2 * (qwq + wq + Y + 1)
    dN = (b - d) * N - s * Gamma
    tri = (1.0 / ag) * (al * C * Gamma / N + be * C * wC / N
                        + ga * B / N - be * N ** 3)
    dC = 2.0 * (b * m * N + b * mC - (d + s * w) * C - s * tri)
    dq = (dC - 2 * N * dN * (q + 1)) / N2
    return dq, dN, Y, wq

def td_residual(q, N, K):
    """Steady-state residual max|dq/dt| (only meaningful for a finite positive state)."""
    dq, dN, _, _ = td_rhs(q, N, K)
    return float(np.max(np.abs(dq))), float(abs(dN) / max(abs(N), 1e-15))

def g0_values(q, K):
    """(g0_ball, g0_point): IBM estimator (pair density over r~ <= G0_BALL_STD)
    and the zero-distance value 1+q(0)."""
    qh = K.g.fwd(q)
    ball = 1.0 + K.g.ball_mean(qh, G0_BALL_STD * length_unit(K.D))
    point = 1.0 + K.g.value_at_zero(qh)
    return ball, point

# ==========================================================================
# ===============  FIXED-POINT SOLVER  =====================================
# ==========================================================================
def solve_fp_positive(K, tol=1e-10, maxit=40000,
                      omegas=(0.3, 0.15, 0.08, 0.04),
                      aa_switch=1e-3, aa_depth=5, aa_beta=1.0):
    """Damped Picard with the hard positivity clamp (q >= -1), extinction
    detection and omega fallback of the ORIGINAL solver, plus an Anderson(5)
    finisher.

    Near the extinction edge the map's contraction factor tends to 1 (critical
    slowing down): the original Picard left a band of `nonconv` cells there,
    which is part of the staircase along the red boundary.  Anderson is switched
    on only once the damped Picard step is already below `aa_switch`; a misstep
    reverts to the last good iterate, and after 3 missteps (or 500 iterations
    without converging) Anderson is retired for the cell and plain damped
    Picard finishes -- so Anderson can only speed things up, and the global
    behaviour, including which cells go extinct, is the original algorithm's.

    Returns (q, N, iters, status) with status in {"survival","extinct","nonconv"}.
    """
    n = K.g.N
    q = np.zeros(n); N = float(N_MF); it = 0
    for om in omegas:
        q = np.zeros(n)
        Xh, Fh = [], []
        use_aa = False; best = np.inf; q_safe = q
        aa_fail = 0; aa_start = 0; aa_ok = True
        blew = False
        for it in range(maxit):
            qn, N = picard_map(q, K)
            bad = not (np.all(np.isfinite(qn)) and np.isfinite(N)) or N > N_CAP
            if bad and use_aa:                              # an Anderson overshoot
                use_aa = False; Xh.clear(); Fh.clear()      # (e.g. q pushed onto the
                aa_fail += 1; aa_ok = aa_fail < 3           #  clamp, 1+Y -> 0): back off
                q = q_safe.copy()
                continue
            if bad:
                blew = True; break
            qn = np.maximum(qn, -1.0)                       # POSITIVITY: q >= -1
            f = qn - q
            err = float(np.max(np.abs(f))) / max(1.0, float(np.max(np.abs(q))))
            if not use_aa and (N < N_EXT or 1.0 + float(qn[0]) > G_EXPLODE):
                return q, 0.0, it + 1, "extinct"
            if err < tol and N >= N_EXT:
                return qn, picard_map(qn, K)[1], it + 1, "survival"
            if use_aa and (err > 10.0 * best or N < N_EXT
                           or it - aa_start > 500):
                use_aa = False; Xh.clear(); Fh.clear()      # misstep: back off
                aa_fail += 1; aa_ok = aa_fail < 3
                q = q_safe.copy()
                continue
            if not use_aa:
                if aa_ok and err < aa_switch:
                    use_aa = True; best = err; q_safe = q.copy(); aa_start = it
                else:
                    q = np.maximum(q + om * f, -1.0)        # the original damped step
                    continue
            if err < best:
                best = err; q_safe = q.copy()
            Xh.append(q.copy()); Fh.append(f.copy())
            if len(Xh) > aa_depth + 1:
                Xh.pop(0); Fh.pop(0)
            if len(Xh) >= 2:
                dX = np.diff(np.array(Xh), axis=0).T
                dF = np.diff(np.array(Fh), axis=0).T
                gam = np.linalg.lstsq(dF, f, rcond=None)[0]
                q = np.maximum(q + aa_beta * f - (dX + aa_beta * dF) @ gam, -1.0)
            else:
                q = np.maximum(q + om * f, -1.0)
        if not blew:
            return q, N, maxit, "nonconv"
    return q, N, maxit, "nonconv"

FP_SETTINGS = {
    1: dict(tol=1e-10, maxit=40000),
    2: dict(tol=1e-10, maxit=40000),
    3: dict(tol=1e-10, maxit=40000),
}

def solve_cell(K):
    """Method hook used by one_cell and --check.  Returns (q, N, status, info)."""
    cfg = FP_SETTINGS[K.D]
    q, N, it, status = solve_fp_positive(K, tol=cfg["tol"], maxit=cfg["maxit"])
    return q, N, status, dict(iters=it)


# =====================================================================================================
# PART 2: drawing Figures 2-5 (the functions used for the draft figures)
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

# ======================================================================= data
def fp_fields(D, G, fp_dir):
    df = pd.read_csv(os.path.join(fp_dir, f"fp_rmax_{D}d.csv"))
    sm, sw = axes_of(df)
    N = pivot(df, "Neq_std", sm, sw).astype(float) * 100
    gcol = "g0"                         # = g0_ball or g0_point, per the run's G0_MODE
    g = pivot(df, gcol, sm, sw).astype(float)
    ext = pivot(df, "status", sm, sw) == "extinct" if "status" in df else (N <= 0)
    g[ext | ~np.isfinite(g) | (g <= 0)] = np.nan
    # extinction edge at sub-cell resolution: N^2 continued linearly into the extinct
    # cells (negative there); its zero level is the edge, sqrt gives the onset
    S = N ** 2; S[ext] = np.nan
    S = fill_nan(S); S[ext] = np.minimum(S[ext], -1e-6)
    Sf = G.refine(S, sm, sw)
    dead = Sf <= 0
    Nf = np.sqrt(np.clip(Sf, 0, None))
    gf = 10 ** G.refine(fill_nan(np.log10(g)), sm, sw)
    # ball-averaged C(0)/N^2 (r~ <= 0.5) = the IBM's estimator, for IBM-vs-FP comparisons
    gb = pivot(df, "g0_ball", sm, sw).astype(float) if "g0_ball" in df else g.copy()
    gb[ext | ~np.isfinite(gb) | (gb <= 0)] = np.nan
    gbf = 10 ** G.refine(fill_nan(np.log10(gb)), sm, sw)
    return dict(df=df, sm=sm, sw=sw, N=Nf, g=gf, g_ball=gbf, dead=dead, ext=ext, raw_N=N, raw_g=g,
                raw_g_ball=gb)

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
# PART 3: the run -- settings, solves of one array task, combining the results, CSV files, figures
# =====================================================================================================
OUT_DIR = os.environ.get("OUT_DIR", "FP")                     # results folder
N_AXIS = int(os.environ.get("N_AXIS", 60))                    # grid points per axis (0.1 .. 10)
STRIDE = int(os.environ.get("STRIDE", 1))                     # every STRIDE-th grid point (2 -> 31 x 31)
RUN_DIMS = [int(x) for x in os.environ.get("DIMS", "1,2,3").split(",")]
RMAX_IBM = {k: float(x) for k, x in zip((1, 2, 3), os.environ.get("RMAX_IBM", "15,9,3.3").split(","))}
NPROC = int(os.environ.get("NPROC", os.environ.get("SLURM_CPUS_PER_TASK", cpu_count())))
TASK_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
N_TASKS = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 0)) or \
    (int(os.environ["SLURM_ARRAY_TASK_MAX"]) + 1 if "SLURM_ARRAY_TASK_MAX" in os.environ else 1)

AX = np.logspace(np.log10(0.1), np.log10(10.0), N_AXIS)
IDX = sorted(set(list(range(0, N_AXIS, STRIDE)) + [N_AXIS - 1]))


def snap_cells():
    """grid cells of the 7 marked points of Figure 3"""
    ax = AX[IDX]
    return [(IDX[int(np.argmin(np.abs(np.log(ax / a))))], IDX[int(np.argmin(np.abs(np.log(ax / c))))]) for a, c in DRAFT_PTS]


# ---------------------------------------------------------------- one cell
def cell(task):
    D, im, iw = task
    sm_s, sw_s = float(AX[im]), float(AX[iw]); u = length_unit(D); t0 = time.time()
    base = dict(method="FP", D=D, sigma_m=sm_s, sigma_w=sw_s, sigma_m_dim=sm_s * u, sigma_w_dim=sw_s * u, lam=LAM, length_unit=u)
    K = build_kernels(D, sm_s * u, sw_s * u)
    q, N, status, info = solve_cell(K)
    diag = dict(dispersal=DISPERSAL, comp_trunc=COMP_TRUNC, g0_mode=G0_MODE, model_tag=MODEL_TAG, iters=info.get("iters", np.nan))
    if status == "extinct":
        row = dict(base, Neq_std=0.0, Neq_raw=0.0, g0=G0_CAP, converged=1, res_q=np.nan, status="extinct",
                   g0_ball=G0_CAP, g0_point=G0_CAP, wall_s=time.time() - t0, **diag)
        st = dict(D=D, sigma_m=sm_s, sigma_w=sw_s, status=status, q0=np.nan, r_half=np.nan, S=np.nan, S_signed=np.nan, S_trunc=np.nan)
        return dict(D=D, im=im, iw=iw, row=row, st=st, curve="")
    N = float(N); gb, gp = g0_values(q, K); res, _ = td_residual(q, N, K)
    g0 = gp if G0_MODE == "point" else gb
    row = dict(base, Neq_std=N * DPRIME / (b - d), Neq_raw=N, g0=g0, converged=int(status == "survival"),
               res_q=res, status=status, g0_ball=gb, g0_point=gp, wall_s=time.time() - t0, **diag)
    # half-saturation distance and structure degree from the solved C(r)/N^2 = 1 + q(r)
    q0 = g0 - 1.0; r = K.g.r / u; a = np.abs(q); tgt = 0.5 * abs(q0)
    above = np.nonzero(a > tgt)[0]
    if len(above) == 0:
        rh = float(r[0])
    elif above[-1] == len(a) - 1:
        rh = np.nan
    else:
        i = above[-1]; rh = float(np.interp(tgt, [a[i + 1], a[i]], [r[i + 1], r[i]]))
    lg = np.log10(np.maximum(1.0 + q, 1e-6)); W = K.g.W / u ** D; inside = r <= RMAX_IBM[D]
    st = dict(D=D, sigma_m=sm_s, sigma_w=sw_s, status=status, q0=q0, r_half=rh, S=float(W @ np.abs(lg)),
              S_signed=float(W @ lg), S_trunc=float(W[inside] @ np.abs(lg[inside])))
    curve = ""
    if (im, iw) in snap_cells():             # Figure 3(g-i): shell means of C(r)/N^2 over 0.02-wide shells
        qh = K.g.fwd(q); x, wq = np.polynomial.legendre.leggauss(24); pts = []
        for a0 in np.arange(0, CURVE_RX, 0.02):
            rr = 0.01 * (x + 1) + a0; ww = 0.01 * wq * rr ** (D - 1)
            v = K.g.eval_from_hat(qh, rr * u)
            pts.append(f"{a0 + 0.01:.4f}:{1 + float(ww @ v / ww.sum()):.10g}")
        curve = " ".join(pts)
    return dict(D=D, im=im, iw=iw, row=row, st=st, curve=curve)


def all_cells():
    """every cell, most expensive first (dealt round-robin to the array tasks)"""
    t = [(D, im, iw) for D in RUN_DIMS for iw in IDX for im in IDX]
    t.sort(key=lambda x: -grid_size(x[0], AX[x[1]] * length_unit(x[0]), AX[x[2]] * length_unit(x[0]))[0] * x[0])
    return t


# ---------------------------------------------------------------- work files (CSV, one per array task)
ROW_KEYS = ["method", "D", "sigma_m", "sigma_w", "sigma_m_dim", "sigma_w_dim", "lam", "length_unit", "Neq_std", "Neq_raw", "g0",
            "converged", "res_q", "status", "g0_ball", "g0_point", "wall_s", "dispersal", "comp_trunc", "g0_mode", "model_tag", "iters"]
ST_KEYS = ["q0", "r_half", "S", "S_signed", "S_trunc"]
FIELDS = ["im", "iw"] + ROW_KEYS + ST_KEYS + ["curve"]


def work_dir():
    return os.path.join(OUT_DIR, "work")


def read_work():
    frames = [pd.read_csv(f, on_bad_lines="skip", dtype={"curve": str})
              for f in sorted(glob.glob(os.path.join(work_dir(), "cells_task*.csv")))]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=FIELDS)
    return pd.concat(frames).drop_duplicates(["D", "im", "iw"], keep="last")


def done_keys():
    w = read_work()
    return set(zip(w.D.astype(int), w.im.astype(int), w.iw.astype(int)))


def run():
    os.makedirs(work_dir(), exist_ok=True)
    mine = all_cells()[TASK_ID::N_TASKS]
    have = done_keys()
    todo = [t for t in mine if t not in have]
    print(f"task {TASK_ID} of {N_TASKS}: {len(mine)} cells, {len(mine) - len(todo)} already done, {len(todo)} to run | "
          f"{NPROC} processes | {MODEL_TAG} | b = {b:g}, d = {d:g}, d' = {s:g}, lambda = {LAM:g}", flush=True)
    if todo:
        wf = os.path.join(work_dir(), f"cells_task{TASK_ID:04d}.csv"); new = not os.path.exists(wf)
        t0 = time.time()
        with open(wf, "a", newline="") as h, Pool(NPROC) as pool:
            fw = csv.writer(h)
            if new: fw.writerow(FIELDS)
            for k, c in enumerate(pool.imap_unordered(cell, todo), 1):
                fw.writerow([c["im"], c["iw"]] + [c["row"].get(x, "") for x in ROW_KEYS] + [c["st"][x] for x in ST_KEYS] + [c["curve"]])
                h.flush()
                if k % 50 == 0 or k == len(todo):
                    print(f"  {k}/{len(todo)} cells  {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"task {TASK_ID} finished its share", flush=True)
    maybe_finish()


def maybe_finish():
    """the task that completes the last cell writes the data files and figures (once)"""
    n_done, n_all = len(done_keys() & set(all_cells())), len(all_cells())
    print(f"overall: {n_done} of {n_all} cells done", flush=True)
    if n_done < n_all:
        print("other tasks are still running (or must be resubmitted); the last one writes FP/data and FP/figures", flush=True)
        return
    try:
        os.mkdir(os.path.join(work_dir(), "finish.lock"))
    except FileExistsError:
        print("another task is already writing the results (if it failed: python FP.py finish)", flush=True)
        return
    finish()


# ---------------------------------------------------------------- combine -> CSV data files, figures
def finish():
    data, figs = os.path.join(OUT_DIR, "data"), os.path.join(OUT_DIR, "figures")
    os.makedirs(data, exist_ok=True); os.makedirs(figs, exist_ok=True)
    w = read_work(); pts = snap_cells(); cpts = []; complete = True
    for D in RUN_DIMS:
        t = w[w.D == D].copy()
        # radii from the grid indices: tasks on different nodes can differ in the last digit of sigma
        t["sigma_m"] = AX[t.im.astype(int).to_numpy()]; t["sigma_w"] = AX[t.iw.astype(int).to_numpy()]
        t["sigma_m_dim"] = t.sigma_m * length_unit(D); t["sigma_w_dim"] = t.sigma_w * length_unit(D)
        t = t.sort_values(["sigma_w", "sigma_m"])
        print(f"  D={D}: {len(t)} of {len(IDX) ** 2} cells", flush=True)
        complete &= len(t) == len(IDX) ** 2
        if not len(t):
            continue
        t[ROW_KEYS].to_csv(f"{data}/fp_rmax_{D}d.csv", index=False)
        t[["D", "sigma_m", "sigma_w", "status"] + ST_KEYS].to_csv(f"{data}/fp_structure_{D}d.csv", index=False)
        for _, c in t.iterrows():
            if not isinstance(c.curve, str):
                continue
            for k in [i + 1 for i, p in enumerate(pts) if p == (int(c.im), int(c.iw))]:   # marked point(s) in this cell
                for p in c.curve.split():
                    r, g = p.split(":")
                    cpts.append(dict(D=D, k=k, sigma_m=c.sigma_m, sigma_w=c.sigma_w, r=float(r), g=float(g)))
    pd.DataFrame(cpts, columns=["D", "k", "sigma_m", "sigma_w", "r", "g"]).to_csv(f"{data}/fp_curves_points.csv", index=False)
    settings = dict(N_AXIS=N_AXIS, STRIDE=STRIDE, b=b, d=d, dprime=s, lam=LAM, closure_weights=f"{al:g}:{be:g}:{ga:g}",
                    dispersal=DISPERSAL, comp_trunc=COMP_TRUNC, g0_mode=G0_MODE, model_tag=MODEL_TAG,
                    **{f"rmax_ibm_{k}D": v for k, v in RMAX_IBM.items()})
    pd.DataFrame(list(settings.items()), columns=["setting", "value"]).to_csv(f"{data}/run_settings.csv", index=False)
    print(f"[finish] data written to {data}/", flush=True)
    if not complete:
        print("[finish] figures skipped: not all cells have results yet (run 'python FP.py status')", flush=True)
        return
    figures(data, figs)
    print(f"[finish] done: {data}/ and {figs}/", flush=True)


def figures(data, figs):
    if len(RUN_DIMS) < 3 or len(IDX) < 4:
        print("[figures] skipped: the figures need all three dimensions and at least 4 x 4 cells", flush=True)
        return
    d0 = pd.read_csv(os.path.join(data, "fp_rmax_1d.csv"))
    G = Grid(d0.sigma_m.min(), d0.sigma_m.max())
    fields = {"G": G, "FP": {D: fp_fields(D, G, data) for D in DIMS}, "IBM": {}}
    print("[figures] FP", flush=True)
    figure2(fields, "FP", figs)
    figure3(fields, "FP", pd.read_csv(os.path.join(data, "fp_curves_points.csv")), None, figs)
    structs = {}
    for D in DIMS:                          # snap to the axis of the data (guards against float round-off)
        st = pd.read_csv(os.path.join(data, f"fp_structure_{D}d.csv"))
        sm_ax, sw_ax = fields["FP"][D]["sm"], fields["FP"][D]["sw"]
        st["sigma_m"] = sm_ax[np.abs(np.log(st.sigma_m.to_numpy()[:, None] / sm_ax[None, :])).argmin(1)]
        st["sigma_w"] = sw_ax[np.abs(np.log(st.sigma_w.to_numpy()[:, None] / sw_ax[None, :])).argmin(1)]
        structs[D] = st.drop_duplicates(["sigma_m", "sigma_w"], keep="last")
    figure4(fields, structs, figs)
    figure5_l1(fields, structs, figs)


def status():
    n_done, n_all = len(done_keys() & set(all_cells())), len(all_cells())
    print(f"{n_done} of {n_all} cells done ({100 * n_done / max(n_all, 1):.1f}%); results in {OUT_DIR}/work/")


def self_test():
    global IDX, OUT_DIR, N_TASKS, TASK_ID
    IDX, OUT_DIR, N_TASKS, TASK_ID = [0, 20, 40, 59], "FP_test", 1, 0
    print("self-test: 4 x 4 cells per dimension -> FP_test/", flush=True)
    run()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        run()
    elif cmd == "finish":
        finish()
    elif cmd == "status":
        status()
    elif cmd == "test":
        self_test()
    else:
        sys.exit(__doc__)
