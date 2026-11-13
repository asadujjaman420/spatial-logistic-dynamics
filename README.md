# Spatial logistic dynamics in one, two, and three dimensions

Equilibria of the spatial logistic model of Bolker & Pacala (1997) and Law &
Dieckmann (2000), computed in habitats of dimension **n = 1, 2, 3** by two
independent methods and compared cell by cell:

* **IBM** — an exact, event-driven (Gillespie) simulation of the
  individual-based model;
* **FP** — the **fixed point** of the second-order spatial-moment equations
  closed with the asymmetric power-2 closure (weights **4 : 1 : 1**).

Each method sweeps a **60 × 60** grid of the two standardized interaction radii
— the dispersal radius $\tilde{\sigma}_\mathrm{m}$ and the competition radius
$\tilde{\sigma}_\mathrm{w}$, logarithmically spaced over $[0.1, 10]$ — in all
three dimensions, and produces four comparison figures (carrying capacity,
zero-distance pair density, half-saturation distance, and spatial-structure
degree) together with the CSV data behind them.

This repository accompanies the paper *Effects of Competition and Dispersal
Distances on Carrying Capacities of Spatial Logistic Dynamics in One-, Two-, and
Three-Dimensional Habitats* (Nikolaev, Galkin, Nikitin & Dieckmann). The two
scripts are the exact, validated versions that produced the figures in the
paper; they are included here **unmodified**.

> **Please review before publishing:** confirm the author list and affiliations
> in [`CITATION.cff`](CITATION.cff), choose a license (an [MIT](LICENSE)
> placeholder is included), and add the repository URL and a release
> DOI (e.g. via Zenodo). Lines marked `TODO` in those files flag each decision.

---

## The model and the equations

A population of **immobile** individuals in $\mathbb{R}^n$ reproduces and dies in
continuous time (the spatial logistic model of Bolker & Pacala 1997 and Law &
Dieckmann 2000). An individual is fully described by its location; the population
is the set $X$ of occupied locations.

**Individual-based model (IBM).** An individual at $x$ gives birth at rate $b$,
placing offspring at $x'$ with probability density $m(x'-x)$, so the rate density
of births at a location $y$ is

$$B(y \mid X) = b \sum_{x \in X} m(y - x). \tag{1}$$

An individual at $x$ dies at a rate made of a background term and competition from
every **other** individual:

$$D(x \mid X) = d + d' \sum_{x' \in X,\; x' \ne x} w(x - x'). \tag{2}$$

Both kernels are isotropic $n$-dimensional Gaussians of standard deviation
$\sigma_\mathrm{m}$ (dispersal) and $\sigma_\mathrm{w}$ (competition):

$$m(x) = \frac{\exp\!\big(-\lVert x\rVert^2 / 2\sigma_\mathrm{m}^2\big)}{(2\pi\sigma_\mathrm{m}^2)^{n/2}},
\qquad
w(x) = \frac{\exp\!\big(-\lVert x\rVert^2 / 2\sigma_\mathrm{w}^2\big)}{(2\pi\sigma_\mathrm{w}^2)^{n/2}}. \tag{3}$$

Equations (1)–(2) are rates, not probabilities: together they define a
continuous-time Markov birth–death process, simulated exactly by `IBM.py` with the
Gillespie (1976) algorithm. In the simulation the competition kernel is truncated
at $4\sigma$ (< 0.12 % of its mass); the moment solver uses untruncated Gaussians.

**Spatial moments.** Writing $p(\xi)=\sum_{x\in X}\delta(\xi-x)$ for the density
function of a configuration, the first and second moments are the expectations of

$$N(p) = \frac{1}{|A|}\int_A p(\xi)\,d\xi,
\qquad
C(x,p) = \frac{1}{|A|}\int_A p(\xi)\big[p(\xi+x)-\delta(x)\big]\,d\xi, \tag{A2, A3}$$

the mean density of individuals and of ordered pairs at displacement $x$; the
$\delta(x)$ removes self-pairing. The **pair correlation** is $g(r)=C(r)/N^2$
(= 1 with no structure, > 1 clustered, < 1 overdispersed). The third moment
$T(x,y)$ is the density of triplets.

**Moment dynamics (what `FP.py` solves at equilibrium).** The density equation
couples to the pair density, and the pair equation couples to the triplet
(Law & Dieckmann 2000):

$$\frac{dN}{dt} = (b-d)N - d'\!\int w(y)\,C(y)\,dy, \tag{A4}$$

$$\frac{\partial C(x)}{\partial t} = 2\Big[\,b\,m(x)N + b\!\int m(x-y)C(y)\,dy - \big(d + d'w(x)\big)C(x) - d'\!\int w(y)\,T(x,y)\,dy\,\Big]. \tag{A5}$$

**Closure.** The hierarchy is closed with the asymmetric power-2 closure
(Law, Murrell & Dieckmann 2003; Murrell et al. 2004), which approximates the
triplet by pairs:

$$T(x,y) = \frac{1}{\alpha+\gamma}\Big[\alpha\frac{C(x)C(y)}{N} + \beta\frac{C(x)C(x-y)}{N} + \gamma\frac{C(y)C(x-y)}{N} - \beta N^3\Big]. \tag{A6}$$

This code uses **weights 4 : 1 : 1** ($\alpha=\tfrac23,\ \beta=\gamma=\tfrac16$;
`CLOSURE=4,1,1`); the subtractive term keeps the mean-field limit exact
($T=N^3$ when $C=N^2$), and $1:0:0$ recovers the simpler closure $T=C(x)C(y)/N$.
Substituting (A6) into the equilibrium form of (A5) gives the nonlinear integral
equation that `FP.py` solves by fixed-point iteration on a radial grid (the full
equilibrium equation is Eq. A7 in [`model.md`](model.md)).

**Equilibrium identity.** At a positive equilibrium the density equation closes
**exactly**, with no approximation:

$$\frac{N}{\lambda} = \left[\int_{\mathbb{R}^n} w(x)\,g(x)\,dx\right]^{-1}, \tag{5}$$

where $\lambda=(b-d)/d'$ is the mean-field density. Overdispersion ($g<1$ where
$w$ is large) raises the carrying capacity above the mean field; clustering lowers
it. The full derivation, the nondimensionalization, the exact equilibrium
equation (A7) and operator (A11), and the definitions of $\tilde r_{1/2}$ and
$S_n$ are in **[`model.md`](model.md)**.

---

## What the code computes

For every combination of standardized radii $(\tilde{\sigma}_\mathrm{m},
\tilde{\sigma}_\mathrm{w})$ and each dimension, both methods report four
characteristics of the stationary state:

| Symbol | Name | Figure |
|:--|:--|:--|
| $\tilde{N} = N/\lambda$ | standardized carrying capacity (equilibrium density relative to the mean field) | 2 |
| $g(0) = C(0)/N^2$ | standardized zero-distance pair density (clustering if $>1$, overdispersion if $<1$) | 3 |
| $\tilde{r}_{1/2}$ | half-saturation distance (extent of the spatial structure) | 4 |
| $S_n$ | spatial-structure degree ($L_1$ norm of $\log_{10} g$) | 5 |

The mathematics — the model, the Gillespie algorithm, the moment hierarchy and
triplets, the closure, the nondimensionalization, and the definitions of the
four characteristics — is documented in **[`model.md`](model.md)**. The
exact columns of every output file are documented in
**[`outputs.md`](outputs.md)**.

### Main findings of the paper (reproduced by this code)

1. Overdispersion raises the carrying capacity above its mean-field value when
   competition is shorter-ranged than dispersal, and clustering lowers it in the
   opposite case; for equal radii and zero intrinsic death rate the spatial
   structure vanishes and the model reaches the mean field in every dimension.
2. These effects have the same qualitative behaviour in all dimensions, but grow
   stronger with dimensionality for small radii and weaker for large radii.
3. The half-saturation distance is set by the smaller of the two radii and
   barely depends on dimensionality.
4. Swapping the two radii does not mirror the pair density on a logarithmic
   scale: clustering produces far more spatial structure than overdispersion.
5. The moment equations reproduce all of these patterns, but underestimate the
   changes in carrying capacity when individuals interact with few neighbours —
   most strongly for clustered populations in one dimension.

---

## Repository layout

```
spatial-logistic-dynamics/
├── README.md              # this file
├── model.md               # model, algorithm, moments, closure, definitions (the theory)
├── outputs.md             # every CSV file and column
├── FP.py                  # moment-closure fixed-point solver + figures (unmodified)
├── IBM.py                 # individual-based Gillespie simulation + figures (unmodified)
├── run_FP.slurm           # Slurm array job for FP.py (all settings at the top)
├── run_IBM.slurm          # Slurm array job for IBM.py (all settings at the top)
├── requirements.txt       # pip dependencies
├── environment.yml        # conda environment (equivalent)
├── CITATION.cff           # how to cite — review before release
└── LICENSE                # MIT placeholder — confirm or replace
```

Each script writes its results **next to itself**, into a folder named after the
method:

```
FP/data/   FP/figures/   FP/work/   FP/logs/
IBM/data/  IBM/figures/  IBM/work/  IBM/logs/
```

`work/` holds the partial results of the array tasks and makes every run
**resumable**; `data/` holds the CSV files; `figures/` holds the PNG and PDF
panels. These output folders are git-ignored (see `.gitignore`).

---

## Requirements

Python ≥ 3.9 with NumPy, SciPy, pandas, and Matplotlib. Install with either:

```bash
pip install -r requirements.txt
# or
conda env create -f environment.yml && conda activate spatial-logistic
```

The full sweep is heavy (≈ 16,000 CPU-hours for the IBM; see below) and is meant
for a cluster with Slurm. The FP sweep and all post-processing run comfortably on
a single workstation. Neither script needs a GPU or network access.

---

## Quick start (a few minutes, on a laptop)

Each script has a built-in self-test on a coarse 4 × 4 grid:

```bash
python FP.py test      # -> FP_test/
python IBM.py test     # -> IBM_test/
```

These check that the code runs end to end and produce small CSV files you can
inspect. They do **not** reproduce the paper (the grid is too coarse for
figures).

Run one method by hand without Slurm (uses all local cores; `finish` then writes
the data and, if the grid is complete, the figures):

```bash
python FP.py run && python FP.py finish       # FP sweep: minutes to ~1 h
python IBM.py status                          # progress of an interrupted run
```

---

## Reproducing the paper (on a cluster)

All run settings live at the top of the two Slurm files — edit them there, not in
the Python. Submit each job from the folder that contains the matching script:

```bash
sbatch run_FP.slurm      # deterministic: 10,800 cells (60×60×3), ~1 node-day
sbatch run_IBM.slurm     # stochastic: 150 realizations/cell, ~16,000 CPU-hours
```

The array tasks share the work. **The task that completes the last piece writes
`data/` and `figures/` automatically.** If some tasks stop at the time limit,
**submit the same file again** — finished work is kept and only the rest is run.
To force the write-up after a partial run:

```bash
python FP.py finish
python IBM.py finish
```

To estimate the IBM cost and a sensible array size on your hardware first:

```bash
python IBM.py estimate
```

### Key settings (Slurm file → meaning)

| Setting | FP | IBM | Meaning |
|:--|:--:|:--:|:--|
| `N_AXIS` | 60 | 60 | grid points per axis, $\tilde{\sigma}\in[0.1,10]$ (log) |
| `D_RATE` | 0.0 | 0.0 | intrinsic death rate $d$ (⇒ $\lambda = (b-d)/d' = 400$) |
| `CLOSURE` | 4,1,1 | — | asymmetric power-2 closure weights $\alpha,\beta,\gamma$ |
| `COMP_TRUNC` | 0 | — | FP competition kernel untruncated |
| `DISPERSAL` | gaussian | gaussian | $n$-dimensional Gaussian dispersal |
| `G0_MODE` | point | — | $g(0)$ taken at zero distance |
| `KCUT` | — | 4 | IBM competition kernel cut at $4\sigma$ (< 0.12 % of its mass) |
| `REPS` | — | 150 | realizations per cell |
| `T_END` | — | 100 | simulated time per realization |
| `T_EQUIL` | — | 30 | average $N$ over $[T_\text{EQUIL}, T_\text{END}]$ |
| `T_SNAP`, `SNAP_DT` | — | 30, 1 | pair-count snapshots: from $t=30$, every 1 time unit |
| `T_SNAP_LATE` | — | 70 | strongly clustered cells: use only snapshots from $t=70$ |

`b = 0.4` and `d' = 0.001` are fixed inside the scripts; with `D_RATE=0.0` this
gives the mean-field density $\lambda = 400$. Both scripts also accept these as
environment variables, so the Slurm files are the single place to change a run.

> **Note on the clustered one-dimensional cells.** At `T_END=100` the most
> strongly clustered populations in one dimension have not fully equilibrated
> (their density was still drifting down by a few percent between the two halves
> of the averaging window). The IBM flags these cells (`equilibrated = 0` in
> `ibm_std_data_D1.csv`), the paper treats their carrying capacities as upper
> bounds, and the comparison medians exclude them. To turn them into converged
> values, raise `T_END` (e.g. to 300) and resubmit `run_IBM.slurm`; the run
> resumes and only the missing work is redone.

---

## Outputs

After a complete run each method writes, per dimension `D ∈ {1,2,3}`:

* `*/data/…_D{1,2,3}.csv` — one row per grid cell, with $\tilde{N}$, $g(0)$,
  $\tilde{r}_{1/2}$, $S_n$ and diagnostics;
* `*/data/…_curves_points.csv` — the full pair-correlation curves $g(\tilde r)$
  at the seven marked points of Figure 3;
* `IBM/data/ibm_pair_counts_D*.csv` — the raw pooled pair counts, from which
  every IBM characteristic can be recomputed;
* `*/data/run_settings.csv` — the exact settings of the run (archive this);
* `*/figures/Figure2…Figure5_*.{png,pdf}` — the four comparison figures.

See **[`outputs.md`](outputs.md)** for the full column dictionary.

---

## How the two methods differ (and why they agree)

* **IBM** samples the stochastic process exactly (continuous-time Gillespie),
  averages over realizations, and is limited by the size of the habitat (a unit
  volume holding $\lambda = 400$ individuals, i.e. $400 / 20 / 7.4$ standardized
  lengths wide in 1D / 2D / 3D) and by finite run time.
* **FP** solves the closed moment equations deterministically on a radial grid
  using a quasi-discrete Hankel transform (Guizar-Sicairos & Gutiérrez-Vega
  2004), so it is cheap and covers all distances, but its accuracy depends on the
  closure.

The closure is exact in the mean-field limit and wherever individuals interact
with many neighbours, and least accurate where they interact with few — which is
exactly the regime (clustered 1D, smallest competition radii) that the
comparison in the paper maps out. See `model.md` for the details.

---

## Citing

If you use this code, please cite the paper and the software — see
[`CITATION.cff`](CITATION.cff). Please also cite the sources of the method:
Bolker & Pacala (1997); Law & Dieckmann (2000); Law, Murrell & Dieckmann (2003)
for the closure; and Guizar-Sicairos & Gutiérrez-Vega (2004) for the radial
transform.

## License

Released under the MIT License (see [`LICENSE`](LICENSE)) — **confirm or replace
this before publishing**, as your institution may require a specific license.
