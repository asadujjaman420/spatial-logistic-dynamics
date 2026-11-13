# Spatial Logistic Dynamics in One, Two, and Three Dimensions

Equilibria of the spatial logistic model of Bolker & Pacala (1997) and Law & Dieckmann (2000), computed in habitats of dimension **n = 1, 2, and 3** using two independent methods and compared cell by cell:

- **IBM** — an exact, event-driven (Gillespie) simulation of the individual-based model.
- **FP** — the **fixed point** of the second-order spatial-moment equations, closed using the asymmetric power-2 closure with weights **4 : 1 : 1**.

Both methods sweep a **60 × 60 parameter grid** of the two standardized interaction radii:

- Dispersal radius: \(\tilde{\sigma}_{\mathrm{m}}\)
- Competition radius: \(\tilde{\sigma}_{\mathrm{w}}\)

The radii are logarithmically spaced over \([0.1,10]\), and the sweep is performed independently in all three spatial dimensions.

For each dimension, the two methods produce four comparison figures:

1. Carrying capacity.
2. Zero-distance pair density.
3. Half-saturation distance.
4. Spatial-structure degree.

The corresponding CSV data underlying all figures are also provided.

This repository accompanies the paper:

*Effects of Competition and Dispersal Distances on Carrying Capacities of Spatial Logistic Dynamics in One-, Two-, and Three-Dimensional Habitats*

**Authors:** Nikolaev, Galkin, Nikitin & Dieckmann.

The two scripts included in this repository are the exact, validated versions used to produce the figures in the paper. They are provided **unmodified**.

> **Before publishing:** Please review the author list and affiliations in [`CITATION.cff`](CITATION.cff), select an appropriate software license (an MIT placeholder is currently included in [`LICENSE`](LICENSE)), and add the repository URL and a release DOI (e.g., through Zenodo). All decisions requiring attention are marked with `TODO` in the corresponding files.

---

## 1. The Model and the Equations

We consider a population of **immobile individuals** distributed in \(\mathbb{R}^n\), where individuals reproduce and die continuously in time according to the spatial logistic model of Bolker & Pacala (1997) and Law & Dieckmann (2000).

Each individual is fully described by its spatial location, and the population is represented by the set \(X\) of occupied locations.

### 1.1 Individual-Based Model (IBM)

An individual at location \(x\) gives birth at rate \(b\), placing its offspring at location \(x'\) according to the dispersal probability density \(m(x'-x)\).

Consequently, the rate density of births at location \(y\) is

\[
B(y\mid X)
=
b\sum_{x\in X}m(y-x).
\tag{1}
\]

An individual at location \(x\) dies at a rate consisting of a background mortality term and competition from every **other** individual:

\[
D(x\mid X)
=
d+d'\sum_{\substack{x'\in X\\x'\neq x}}
w(x-x').
\tag{2}
\]

Both interaction kernels are isotropic \(n\)-dimensional Gaussian functions, with standard deviations \(\sigma_{\mathrm{m}}\) for dispersal and \(\sigma_{\mathrm{w}}\) for competition:

\[
m(x)
=
\frac{
\exp\left(-\frac{\lVert x\rVert^2}{2\sigma_{\mathrm{m}}^2}\right)
}{
(2\pi\sigma_{\mathrm{m}}^2)^{n/2}
},
\]

\[
w(x)
=
\frac{
\exp\left(-\frac{\lVert x\rVert^2}{2\sigma_{\mathrm{w}}^2}\right)
}{
(2\pi\sigma_{\mathrm{w}}^2)^{n/2}
}.
\tag{3}
\]

Equations (1)–(2) define **rates, not probabilities**. Together, they specify a continuous-time Markov birth–death process, which is simulated exactly by `IBM.py` using the Gillespie (1976) algorithm.

In the IBM implementation, the competition kernel is truncated at \(4\sigma\), corresponding to less than **0.12% of its total mass**. The moment solver, in contrast, uses untruncated Gaussian kernels.

### 1.2 Spatial Moments

Let

\[
p(\xi)=\sum_{x\in X}\delta(\xi-x)
\]

denote the density function associated with a population configuration.

The first- and second-order spatial moments are defined as

\[
N(p)
=
\frac{1}{|A|}
\int_A p(\xi)\,d\xi,
\]

\[
C(x,p)
=
\frac{1}{|A|}
\int_A
p(\xi)
\left[p(\xi+x)-\delta(x)\right]
\,d\xi.
\tag{A2, A3}
\]

Here:

- \(N\) represents the mean density of individuals.
- \(C(x)\) represents the density of ordered pairs separated by displacement \(x\).
- The Dirac delta term \(\delta(x)\) removes self-pairing.

The **pair-correlation function** is defined by

\[
g(r)=\frac{C(r)}{N^2}.
\]

Its interpretation is:

| Pair correlation | Spatial interpretation |
|---|---|
| \(g(r)=1\) | No spatial structure (random spatial distribution) |
| \(g(r)>1\) | Clustering |
| \(g(r)<1\) | Overdispersion |

The third spatial moment \(T(x,y)\) describes the density of triplets.

### 1.3 Equilibrium Moment Equations

The fixed-point solver, `FP.py`, computes the **stationary state** of the spatial moment equations.

At equilibrium, the spatial moments no longer change with time. Therefore,

\[
\frac{dN}{dt}=0,
\qquad
\frac{\partial C(x)}{\partial t}=0.
\]

Setting the left-hand sides of the moment dynamics (Law & Dieckmann, 2000) to zero gives the equilibrium conditions.

**Density balance**

The equilibrium density equation relates the mean density to the pair density integrated against the competition kernel:

\[
(b-d)N
=
d'\int w(y)C(y)\,dy.
\tag{A4}
\]

**Pair-density balance**

The equilibrium pair-density equation couples the second moment to the third moment:

\[
\begin{aligned}
b\,m(x)N
&+
b\int m(x-y)C(y)\,dy\\
&-
\big(d+d'w(x)\big)C(x)\\
&-
d'\int w(y)T(x,y)\,dy
=0.
\end{aligned}
\tag{A5}
\]

### 1.4 Asymmetric Power-2 Closure

The moment hierarchy is not closed because the pair-density equation depends on the third-order moment \(T(x,y)\).

We close the hierarchy using the **asymmetric power-2 closure** (Law, Murrell & Dieckmann, 2003; Murrell et al., 2004), which approximates the triplet density in terms of the first- and second-order moments:

\[
\begin{aligned}
T(x,y)
=
\frac{1}{\alpha+\gamma}
\Bigg[
&\alpha\frac{C(x)C(y)}{N}
+
\beta\frac{C(x)C(x-y)}{N}\\
&+
\gamma\frac{C(y)C(x-y)}{N}
-
\beta N^3
\Bigg].
\end{aligned}
\tag{A6}
\]

The implementation uses the weights

\[
\boxed{\alpha:\beta:\gamma=4:1:1}
\]

corresponding to

```python
CLOSURE = (4, 1, 1)
```

Only the relative ratios matter. Thus, these weights are equivalent to the normalized values used in the paper:

\[
\alpha=\frac{2}{3},
\qquad
\beta=\gamma=\frac{1}{6}.
\]

The subtractive term ensures that the mean-field limit is exact:

\[
T=N^3
\qquad\text{when}\qquad C=N^2.
\]

The special case

\[
1:0:0
\]

recovers the simpler closure

\[
T(x,y)=\frac{C(x)C(y)}{N}.
\]

Substituting the closure (A6) into the pair-density balance (A5), and eliminating \(N\) using the density balance (A4), yields a single nonlinear integral equation for \(C\).

This equation is solved by `FP.py` using fixed-point iteration on a radial grid.

The complete equilibrium equation is given in **Equation A7** of [`model.md`](model.md).

### 1.5 Exact Equilibrium Identity

At a positive equilibrium, the density equation yields the following exact identity, independently of the moment closure:

\[
\boxed{
\frac{N}{\lambda}
=
\left[
\int_{\mathbb{R}^n}w(x)g(x)\,dx
\right]^{-1}
}
\tag{5}
\]

where

\[
\lambda=\frac{b-d}{d'}
\]

is the mean-field density.

This identity provides a direct connection between spatial structure and carrying capacity.

- **Overdispersion** (\(g<1\) where \(w\) is large) reduces the effective competition and raises the carrying capacity above the mean-field value.
- **Clustering** increases the effective competition and lowers the carrying capacity.

The complete derivation, nondimensionalization, exact equilibrium equation (A7), operator (A11), and definitions of \(\tilde r_{1/2}\) and \(S_n\) are documented in **[`model.md`](model.md)**.

---

## 2. What the Code Computes

For every combination of standardized interaction radii,

\[
\left(
\tilde{\sigma}_{\mathrm{m}},
\tilde{\sigma}_{\mathrm{w}}
\right),
\]

and each spatial dimension \(n\in\{1,2,3\}\), both methods report four characteristics of the stationary state.

| Symbol | Quantity | Description | Figure |
|:---:|---|---|:---:|
| \(\tilde N=N/\lambda\) | Standardized carrying capacity | Equilibrium density relative to the mean-field density | 2 |
| \(g(0)=C(0)/N^2\) | Zero-distance pair density | Indicates clustering (\(>1\)) or overdispersion (\(<1\)) | 3 |
| \(\tilde r_{1/2}\) | Half-saturation distance | Characteristic spatial extent of the structure | 4 |
| \(S_n\) | Spatial-structure degree | \(L_1\) norm of \(\log_{10}g\) | 5 |

The mathematical formulation—including the model, Gillespie algorithm, moment hierarchy, triplet approximation, closure, nondimensionalization, and definitions of the four characteristics—is documented in **[`model.md`](model.md)**.

The exact columns and definitions of every output file are provided in **[`outputs.md`](outputs.md)**.

### 2.1 Main Findings of the Paper

The following findings are reproduced by the code:

**1. Spatial structure determines carrying capacity.**

Overdispersion raises the carrying capacity above its mean-field value when competition is shorter-ranged than dispersal. Conversely, clustering lowers the carrying capacity when competition is longer-ranged than dispersal.

For equal interaction radii and zero intrinsic death rate, spatial structure vanishes, and the model reaches the mean-field equilibrium in every dimension.

**2. Spatial effects become stronger with dimensionality.**

The qualitative behaviour of the spatial effects is consistent across all dimensions. However, these effects become stronger with increasing dimensionality for small interaction radii and weaker for large interaction radii.

**3. The half-saturation distance is controlled by the shorter interaction radius.**

The half-saturation distance is primarily determined by the smaller of the dispersal and competition radii and exhibits very little dependence on spatial dimensionality.

**4. Clustering and overdispersion are not symmetric.**

Swapping the dispersal and competition radii does not produce a mirrored pair-density structure on a logarithmic scale. Clustering generates substantially more spatial structure than overdispersion.

**5. Moment equations reproduce the overall patterns but have limitations.**

The moment equations reproduce all the major qualitative patterns. However, they underestimate changes in carrying capacity when individuals interact with relatively few neighbours. This discrepancy is most pronounced for clustered populations in one dimension.

---

## 3. Repository Structure

The repository is organized as follows:

```text
spatial-logistic-dynamics/
│
├── README.md              # Project overview and instructions
├── model.md               # Model, algorithm, moments, closure, definitions
├── outputs.md             # Complete CSV file and column documentation
│
├── FP.py                  # Moment-closure fixed-point solver + figures
├── IBM.py                 # Individual-based Gillespie simulation + figures
│
├── run_FP.slurm           # Slurm array job for FP.py
├── run_IBM.slurm          # Slurm array job for IBM.py
│
├── requirements.txt       # Python dependencies
├── environment.yml        # Equivalent Conda environment
│
├── CITATION.cff           # Software citation metadata
└── LICENSE                # MIT license placeholder
```

The two main Python scripts, `FP.py` and `IBM.py`, are provided unmodified.

### 3.1 Output Directories

Each script writes its results next to itself, into a directory named after the corresponding method.

```text
FP/
├── data/
├── figures/
├── work/
└── logs/

IBM/
├── data/
├── figures/
├── work/
└── logs/
```

The directories serve the following purposes:

| Directory | Contents |
|---|---|
| `work/` | Partial results from individual array tasks; enables resumable execution |
| `data/` | Final CSV files and numerical results |
| `figures/` | Generated comparison figures in PNG and PDF formats |
| `logs/` | Execution logs |

All generated output directories are excluded from version control through `.gitignore`.

---

## 4. Requirements and Installation

The code requires:

- Python ≥ 3.9
- NumPy
- SciPy
- pandas
- Matplotlib

Install the dependencies using either pip or Conda.

**Option 1: pip**

```bash
pip install -r requirements.txt
```

**Option 2: Conda**

```bash
conda env create -f environment.yml
conda activate spatial-logistic
```

### Computational Requirements

The complete parameter sweep is computationally intensive, particularly for the stochastic IBM.

| Method | Computational requirements |
|---|---|
| FP | Runs comfortably on a single workstation |
| IBM | Approximately 16,000 CPU-hours for the complete sweep |

The full IBM computation is intended for a high-performance computing (HPC) cluster equipped with Slurm.

Neither method requires a GPU or network access.

---

## 5. Quick Start: Running on a Laptop

Both scripts include a built-in self-test that runs on a coarse \(4\times4\) parameter grid.

Run the tests using:

```bash
python FP.py test
python IBM.py test
```

The scripts generate:

```text
FP_test/
IBM_test/
```

These tests verify that the code executes end to end and produces small CSV files that can be inspected.

**Important:** The self-tests are intended only for checking the implementation. They do not reproduce the paper because the parameter grid is too coarse for the published figures.

### 5.1 Running the Fixed-Point Solver Locally

To execute the FP parameter sweep without Slurm:

```bash
python FP.py run
python FP.py finish
```

The first command runs the sweep using the available local cores. The second command collects the results and generates the figures once the grid is complete.

The FP sweep typically takes from several minutes to approximately one hour, depending on the hardware.

### 5.2 Checking IBM Progress

To inspect the progress of an interrupted or ongoing IBM computation:

```bash
python IBM.py status
```

---

## 6. Reproducing the Paper on an HPC Cluster

All computational settings are defined at the top of the two Slurm submission scripts.

**Edit the settings in the Slurm files rather than modifying the Python scripts.**

Submit each job from the directory containing its corresponding Python script.

### 6.1 Submit the Fixed-Point Sweep

```bash
sbatch run_FP.slurm
```

- Deterministic computation.
- \(60\times60\times3=10,800\) parameter cells.
- Approximately one node-day.

### 6.2 Submit the Individual-Based Simulations

```bash
sbatch run_IBM.slurm
```

- Stochastic computation.
- 150 realizations per parameter cell.
- Approximately 16,000 CPU-hours.

### 6.3 Resuming Interrupted Jobs

The Slurm array tasks share intermediate results in the `work/` directory.

**The task that completes the final piece automatically writes the final CSV data and figures.**

If some array tasks terminate because they reach their time limits, simply submit the same Slurm file again:

```bash
sbatch run_FP.slurm
```

or

```bash
sbatch run_IBM.slurm
```

Previously completed work is preserved. Only the remaining computations are executed.

To explicitly trigger the final data collection and figure generation after a partial run:

```bash
python FP.py finish
python IBM.py finish
```

### 6.4 Estimating IBM Computational Cost

Before submitting the full IBM sweep, estimate its computational cost and determine a suitable array size for your hardware:

```bash
python IBM.py estimate
```

### 6.5 Key Computational Settings

All settings are defined in the corresponding Slurm file.

| Setting | FP | IBM | Description |
|---|:---:|:---:|---|
| `N_AXIS` | 60 | 60 | Number of grid points per axis; logarithmic spacing over \([0.1,10]\) |
| `D_RATE` | 0.0 | 0.0 | Intrinsic death rate \(d\); gives \(\lambda=(b-d)/d'=400\) |
| `CLOSURE` | 4,1,1 | — | Asymmetric power-2 closure weights \(\alpha,\beta,\gamma\) |
| `COMP_TRUNC` | 0 | — | FP competition kernel remains untruncated |
| `DISPERSAL` | gaussian | gaussian | \(n\)-dimensional Gaussian dispersal kernel |
| `G0_MODE` | point | — | Evaluates \(g(0)\) directly at zero distance |
| `KCUT` | — | 4 | IBM competition kernel truncated at \(4\sigma\), retaining more than 99.88% of its mass |
| `REPS` | — | 150 | Number of realizations per parameter cell |
| `T_END` | — | 100 | Total simulated time per realization |
| `T_EQUIL` | — | 30 | Equilibrium density averaged over \([T_{\mathrm{EQUIL}},T_{\mathrm{END}}]\) |
| `T_SNAP`, `SNAP_DT` | — | 30, 1 | Pair-count snapshots starting at \(t=30\), taken every 1 time unit |
| `T_SNAP_LATE` | — | 70 | For strongly clustered cells, only snapshots from \(t=70\) onward are used |

The birth rate and competition coefficient are fixed inside the scripts:

\[
b=0.4,
\qquad
d'=0.001.
\]

With `D_RATE=0.0`, the resulting mean-field density is

\[
\lambda=\frac{b-d}{d'}=400.
\]

Both scripts also accept these parameters through environment variables, making the Slurm files the central location for configuring a computational run.

### 6.6 Note on Strongly Clustered One-Dimensional Cells

At `T_END=100`, the most strongly clustered populations in one dimension have not fully equilibrated.

Their population densities are still drifting downward by a few percent between the two halves of the averaging window.

The IBM implementation flags these cells using:

```text
equilibrated = 0
```

in `ibm_std_data_D1.csv`.

For these cases:

- The paper treats the computed carrying capacities as **upper bounds**.
- The comparison medians exclude these non-equilibrated cells.
- The remaining density drift is associated with slow convergence in strongly clustered one-dimensional populations.

To obtain more fully converged values, increase the simulation end time, for example:

```text
T_END=300
```

Then resubmit `run_IBM.sl
