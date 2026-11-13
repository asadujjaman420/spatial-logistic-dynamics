# The model, the methods, and the definitions

This document describes the mathematics behind `IBM.py` and `FP.py`: the
individual-based model and its exact simulation, the spatial-moment hierarchy and
its closure, the nondimensionalization, and the four characteristics of spatial
structure that the code measures. Equation numbers of the form (A·) refer to
Appendix A of the accompanying paper; the chapter equations (14.·) are those of
Law & Dieckmann (2000).

Three levels of description must be kept apart, and the code embodies all three:

1. the **stochastic individual-based model** (the IBM, simulated exactly by
   `IBM.py`);
2. the **exact but unclosed hierarchy** of spatial-moment equations;
3. the **closed moment equations**, whose equilibrium is an approximation with an
   accuracy that depends on the closure (solved by `FP.py`).

---

## 1. Notation

| Symbol | Meaning |
|:--|:--|
| $n$ | habitat dimension (1, 2, or 3) |
| $b$ | per capita birth rate (`b = 0.4`) |
| $d$ | per capita death rate from the environment (`D_RATE`, here 0) |
| $d'$ | competition strength (`d' = 0.001`) |
| $m(\cdot)$, $\sigma_\mathrm{m}$ | dispersal kernel and dispersal radius |
| $w(\cdot)$, $\sigma_\mathrm{w}$ | competition kernel and competition radius |
| $\lambda = (b-d)/d'$ | mean-field equilibrium density (here 400) |
| $N$ | first spatial moment (mean density of individuals) |
| $C(r)$ | second spatial moment (density of ordered pairs at separation $r$) |
| $T(r,r')$ | third spatial moment (density of triplets) |
| $g(r) = C(r)/N^2$ | pair correlation function (standardized pair density) |
| $\tilde{\sigma}_\mathrm{m} = \sigma_\mathrm{m}\lambda^{1/n}$ | standardized dispersal radius |
| $\tilde{\sigma}_\mathrm{w} = \sigma_\mathrm{w}\lambda^{1/n}$ | standardized competition radius |
| $\tilde{d} = d/b$ | standardized death rate (here 0) |
| $\tilde{r} = r\lambda^{1/n}$ | standardized distance |
| $\tilde{N} = N/\lambda$ | standardized carrying capacity |
| $g(0)$ | zero-distance pair correlation |
| $\tilde{r}_{1/2}$ | half-saturation distance |
| $S_n$ | spatial-structure degree |

The two radii are the standard deviations of each **coordinate** of the
displacement, not of the distance: the root-mean-square dispersal distance is
$\sqrt{n}\,\sigma_\mathrm{m}$.

---

## 2. The individual-based model

A population of **immobile** individuals reproduces and dies in continuous time.
An individual is fully described by its location; the population is the set $X$
of occupied locations.

**Births.** An individual at $x$ produces offspring at rate $b$, placed at $x'$
with probability density $m(x'-x)$. The rate density of births at a location $y$
is therefore

$$B(y \mid X) = b \sum_{x \in X} m(y - x). \tag{1}$$

**Deaths.** An individual at $x$ dies at a rate that is the sum of a
background term and competition from every other individual:

$$D(x \mid X) = d + d' \sum_{x' \in X,\; x' \ne x} w(x - x'). \tag{2}$$

The sum excludes the focal individual, so nothing competes with itself. This is
Eq. (14.13) of Law & Dieckmann (2000), written as a sum over individuals.

**Kernels.** Both $m$ and $w$ are isotropic $n$-dimensional Gaussians:

$$m(x) = \frac{\exp\!\big(-\lVert x\rVert^2 / 2\sigma_\mathrm{m}^2\big)}{(2\pi\sigma_\mathrm{m}^2)^{n/2}},
\qquad
w(x) = \frac{\exp\!\big(-\lVert x\rVert^2 / 2\sigma_\mathrm{w}^2\big)}{(2\pi\sigma_\mathrm{w}^2)^{n/2}}. \tag{3}$$

In the simulation the competition kernel is truncated at `KCUT` = 4 standard
deviations and renormalized (this removes < 0.12 % of its mass in three
dimensions, less in lower dimensions); dispersal is sampled from the untruncated
Gaussian. In the moment solver both kernels are untruncated Gaussians
(`COMP_TRUNC=0`).

Equations (1) and (2) are **rates**, not probabilities: together they define a
continuous-time Markov birth–death process (Fournier & Méléard 2004). Births
contribute like a Poisson process of rate $b$; competition makes the death rate
depend on the whole configuration.

---

## 3. The Gillespie algorithm (as implemented in `IBM.py`)

`IBM.py` simulates the process **exactly** with the Gillespie (1976)
continuous-time method — no time discretization. One realization:

1. **Initialize.** Place $N_0 = \lambda L^n$ individuals uniformly at random in a
   periodic box of side $L$. Compute each individual's death rate from (2) and
   the totals $B_\text{tot} = bN$ and $D_\text{tot} = \sum_x D(x\mid X)$.
2. **Time to next event.** Draw $\Delta t \sim \mathrm{Exp}(B_\text{tot} +
   D_\text{tot})$ and advance $t \leftarrow t + \Delta t$.
3. **Event type.** With probability $B_\text{tot}/(B_\text{tot}+D_\text{tot})$
   the event is a **birth**, otherwise a **death**.
4. **Birth.** Pick a parent uniformly; draw a displacement from the dispersal
   kernel; place the offspring (wrapped into the periodic box). Compute its death
   rate and add its competitive contribution to the relevant neighbours.
5. **Death.** Pick the victim with probability proportional to its death rate
   (weighted draw over the stored rates); remove it and subtract its competitive
   contribution from the relevant neighbours.
6. Repeat from step 2 until $t = T_\text{END}$.

The implementation keeps positions and per-individual death rates in arrays and
updates only the affected neighbours per event, so the event sequence is exact
while the full sweep stays affordable. Competition is evaluated under periodic
boundaries by whichever of three equivalent treatments is valid for the cell: a
cell list with the minimum-image convention when the interaction range fits
inside half the box ($R_\mathrm{w} \le L/2$), an exact sum over periodic images
when it does not, and a separable wrapped-Gaussian approximation in the far field
(where the kernel is flat and the cell is effectively mean-field). Independent
random streams are seeded per (cell, realization) for reproducibility.

---

## 4. Spatial moments and the pair correlation

Instead of the configuration itself, we track its **spatial moments** — averages
over all realizations at stationarity:

* the **first moment** $N$: the mean density of individuals;
* the **second moment** $C(r)$: the mean density of **ordered pairs** of
  individuals separated by $r$;
* the **third moment** $T(r, r')$: the mean density of **triplets** with
  displacements $r$ and $r'$.

The normalized second moment is the **pair correlation function**

$$g(r) = \frac{C(r)}{N^2}, \tag{4}$$

which equals 1 at all separations when there is no spatial structure (complete
spatial randomness). $g(r) > 1$ means more close pairs than random
(**clustering**); $g(r) < 1$ means fewer (**overdispersion**). Because
individuals far apart become independent, $C(r) \to N^2$ and $g(r) \to 1$ as
$r \to \infty$.

In the simulation, $C(r)$ is estimated from counts of ordered pairs in distance
bins of width 0.05 (standardized), pooled over snapshots and divided by the
counts expected for the same number of individuals placed at random (see §8).

---

## 5. The moment hierarchy and its closure

The dynamics of each moment involve the **next higher** moment (Dieckmann & Law
2000; Law & Dieckmann 2000). The first moment follows a logistic-like equation
that needs the pair density $C$,

$$\frac{dN}{dt} = (b - d)\,N - d' \int_{\mathbb{R}^n} w(y)\,C(y)\,dy, \tag{A4}$$

and the pair density in turn needs the triplet density $T$,

$$\frac{\partial C(x)}{\partial t} = 2\Big[\,b\,m(x)N + b\!\int_{\mathbb{R}^n} m(x-y)\,C(y)\,dy - \big(d + d'w(x)\big)C(x) - d'\!\int_{\mathbb{R}^n} w(y)\,T(x,y)\,dy\,\Big]. \tag{A5}$$

Here $T(x,y)$ is the expected density of triplets whose second and third members
are displaced by $x$ and $y$ from the first. The factor 2 appears because births
and deaths of either member of a pair change the pair density (Law & Dieckmann
2000 write it as $\tfrac12$ on the left); it does not affect equilibria. This is
the **moment hierarchy**: an exact but infinite system.

### The closure

A **moment closure** truncates the hierarchy by approximating $T$ in terms of $N$
and $C$. This code uses the **asymmetric power-2 closure** (Law, Murrell &
Dieckmann 2003; Murrell et al. 2004):

$$T(x, y) = \frac{1}{\alpha + \gamma}\left[\alpha\,\frac{C(x)C(y)}{N} + \beta\,\frac{C(x)C(x - y)}{N} + \gamma\,\frac{C(y)C(x - y)}{N} - \beta N^{3}\right]. \tag{A6}$$

The three terms weight the three pair-of-pairs products that can form a triplet;
the subtractive $\beta N^3$ keeps the mean-field limit exact, so that
$T = N^3$ when $C = N^2$ (no spatial structure). The closure depends only on the
ratios of the weights. This code uses **$\alpha:\beta:\gamma = 4:1:1$**
($\alpha=\tfrac23,\ \beta=\gamma=\tfrac16$; `CLOSURE=4,1,1`). The special case
$1:0:0$ is the simpler closure $T = C(x)C(y)/N$ of Law & Dieckmann (2000,
Eq. 14.10); the family with $\beta=\gamma$ and $\alpha=(2/\delta-1)\gamma$ gives
the symmetric power-2 closures, with $\delta=2/5$ for the weights used here.

### The equilibrium equation

At equilibrium both left-hand sides vanish. Substituting the closure (A6) into the
equilibrium form of (A5), eliminating $N$ through the equilibrium form of (A4)
($N = Y/(b-d)$), and multiplying by $Y$ gives the **equilibrium equation**

$$\left[Y\left(d + \bar{w}\right) + \frac{b - d}{\alpha + \gamma}\left(\alpha Y + \beta\,[\bar{w} * C]\right)\right] C
= \frac{\bar{m}\,Y^{2}}{b - d} + Y\,[\bar{m} * C] - \frac{\gamma\,(b - d)}{\alpha + \gamma}\,[C * (\bar{w}C)] + \frac{\beta\, d'\, Y^{4}}{(\alpha + \gamma)(b - d)^{3}}, \tag{A7}$$

with $\bar w = d'w$, $\bar m = bm$, $Y = \int \bar w(y)C(y)\,dy$, and
$[f * h](x) = \int f(x-y)h(y)\,dy$. Written as a fixed-point equation
$C = \mathcal{A}C$, the operator is (A7) divided through by the bracket on the
left:

$$\mathcal{A}C = \frac{\dfrac{\bar{m}\,Y^{2}}{b - d} + Y\,[\bar{m} * C] - \dfrac{\gamma\,(b - d)}{\alpha + \gamma}\,[C * (\bar{w}C)] + \dfrac{\beta\, d'\, Y^{4}}{(\alpha + \gamma)(b - d)^{3}}}{Y\left(d + \bar{w}\right) + \dfrac{b - d}{\alpha + \gamma}\left(\alpha Y + \beta\,[\bar{w} * C]\right)}. \tag{A11}$$

`FP.py` solves $C = \mathcal{A}C$ by the fixed-point iteration of §7. The three
convolutions $[\bar w * C]$, $[\bar m * C]$, $[C * (\bar w C)]$ are the expensive
part; because the kernels and the solution are radial, they are reduced to one
dimension (§7). The closure and these operators are implemented in `FP.py` in
`picard_map` (the fixed-point map) and `td_rhs` (the time-derivative form, used to
measure the steady-state residual).

The choice of weights matters quantitatively: 4:1:1 corrects the one-dimensional
over-prediction and the clustered two-dimensional cells of the simpler closures,
which is why it is the default here. The closure is **exact in the mean-field
limit** and accurate when interaction neighbourhoods contain many individuals; it
is least accurate when they contain few (see §9).

---

## 6. Nondimensionalization

The model has five parameters ($b$, $d$, $d'$, $\sigma_\mathrm{m}$,
$\sigma_\mathrm{w}$). Choosing a unit of time $1/b$ and a unit of length
$\lambda^{-1/n}$ — where

$$\lambda = \frac{b - d}{d'}$$

is the **mean-field equilibrium density**, the equilibrium of a well-mixed
population with no spatial structure, and $\lambda^{-1/n}$ is the typical distance
between neighbours — reduces the model to **three** standardized parameters:

$$\tilde{\sigma}_\mathrm{m} = \sigma_\mathrm{m}\,\lambda^{1/n}, \qquad
\tilde{\sigma}_\mathrm{w} = \sigma_\mathrm{w}\,\lambda^{1/n}, \qquad
\tilde{d} = d/b.$$

With $\tilde d = 0$ (the case studied here, `D_RATE=0.0`), only the two
standardized radii remain, and the standardized carrying capacity
$\tilde{N} = N/\lambda$ depends on them alone. This is what makes a single
$60 \times 60$ sweep a complete characterization, and what makes the three
dimensions directly comparable. The code fixes $b = 0.4$, $d' = 0.001$, so
$\lambda = 400$; these affect the dimensional values but not the standardized
results.

---

## 7. The equilibrium and the two numerical methods

At a positive equilibrium the density equation (A4) closes **exactly**, without
any approximation, into (Law & Dieckmann 2000, Eq. 14.16):

$$\frac{N}{\lambda} = \left[\int_{\mathbb{R}^n} w(x)\,g(x)\,dx\right]^{-1}. \tag{5}$$

The equilibrium density is the mean-field density divided by the pair correlation
averaged over the competition kernel. Overdispersion ($g < 1$ where $w$ is large)
raises $\tilde N$ above 1; clustering lowers it.

**FP (`FP.py`).** Because both kernels are isotropic, the equilibrium $g$ is
radial, and the solver works entirely on a **one-dimensional radial grid**. Every
convolution in the closed equation is evaluated with a **quasi-discrete Hankel
transform** of order $n/2 - 1$ (Guizar-Sicairos & Gutiérrez-Vega 2004) — a cosine
transform for $n=1$, the order-0 Hankel transform for $n=2$, a sine transform for
$n=3$ — so the same algorithm serves all three dimensions. The grid nodes are the
zeros of the corresponding Bessel function; the node spacing is $\sigma_\min/4$
($n=1$) or $\sigma_\min/8$ ($n=2,3$) and the box extends to at least $8\sigma_\max$.
The fixed point is found by **damped Picard iteration** (successive
substitution) with a hard positivity clamp $g \ge 0$, accelerated by Anderson
mixing once the step is small, to a relative tolerance of $10^{-10}$. Using a
radial grid instead of a capped Cartesian one removes the discretization
artefacts that the earlier solver showed for narrow kernels.

**IBM (`IBM.py`).** The exact Gillespie simulation of §3, averaged over `REPS`
realizations per cell.

The two are compared cell by cell on the common grid; `FP.py` additionally
writes `S_trunc`, the structure degree integrated only out to the largest
distance the IBM habitat can measure, so that the two $S_n$ values are compared
over the same range.

---

## 8. The four characteristics

**Standardized carrying capacity** $\tilde N = N/\lambda$ — Eq. (5); Figure 2.

**Zero-distance pair correlation** $g(0) = \lim_{r\to 0} C(r)/N^2$ — Figure 3.
In the FP solution this is the value at $r=0$ (`G0_MODE=point`). In the
simulation it is a Gaussian-weighted average of the pair counts over the
innermost distances, with weight width $\max(0.05,\,0.25\min(\tilde\sigma_\mathrm{m},
\tilde\sigma_\mathrm{w}))$, over which $g$ is flat.

**Half-saturation distance** $\tilde r_{1/2}$ — Figure 4 — the extent of the
spatial structure: the smallest distance beyond which the deviation of $g$ from 1
stays below half its value at zero distance,

$$\tilde r_{1/2} = \inf\!\Big\{\,\tilde r \ge 0 :\;
\sup_{s \ge \tilde r} \lvert g(s) - 1\rvert \le \tfrac12\,\lvert g(0) - 1\rvert \Big\}. \tag{6}$$

The supremum (rather than a first crossing) handles the non-monotonic $g$ of
clustered populations, which dip below 1 at intermediate distances. In the FP
solution the rule is applied directly to the solved curve; in the simulation it
is applied to a smooth curve fitted to the whole of $g(\tilde r) - 1$ (a
ridge-regularized sum of Gaussian shapes), with a parametric bootstrap for its
uncertainty. It is reported as unresolved where $g(0)$ is statistically
indistinguishable from 1, where the half level is reached only beyond the sampled
range, or where the bootstrap range exceeds $\pm 25\%$.

**Spatial-structure degree** $S_n$ — Figure 5 — the magnitude-and-extent of the
structure:

$$S_n = \int_{\mathbb{R}^n} \big\lvert \log_{10} g(\tilde x)\big\rvert \, d\tilde x
= \omega_n \int_0^\infty \big\lvert \log_{10} g(\tilde r)\big\rvert\, \tilde r^{\,n-1}\, d\tilde r,
\qquad \omega_1 = 2,\ \omega_2 = 2\pi,\ \omega_3 = 4\pi. \tag{7}$$

Clustering and overdispersion both contribute positively (absolute value).
Because the simulation can only integrate out to the largest sampled distance
($\tilde r = 15 / 9 / 3.3$ in 1D / 2D / 3D), the FP solver also reports `S_trunc`
over the same range for a fair comparison; the IBM value is noise-corrected so
that it is zero within noise on the structure-free diagonal.

---

## 9. When the closure is accurate

Both $\tilde N$ and $g$ depart from their mean-field values more strongly when an
individual interacts with fewer neighbours, and the pair-based closure is
accurate exactly when that number is large. In a well-mixed population a kernel of
standardized radius $\tilde\sigma$ covers on average $(\sqrt{4\pi}\,\tilde\sigma)^n$
individuals. This number falls below one — and the closure loses accuracy — for
small radii, the more so the higher the dimension; it is the same in every
dimension at $\tilde\sigma = 1/\sqrt{4\pi} \approx 0.28$, which is where the
effect of dimensionality reverses. The paper maps the resulting accuracy across
the whole plane: the moment equations reproduce every qualitative pattern, and
deviate quantitatively most for clustered populations in one dimension and for
the smallest competition radii.

---

## References

* Bolker B & Pacala SW (1997). Using moment equations to understand
  stochastically driven spatial pattern formation in ecological systems.
  *Theoretical Population Biology* 52:179–197.
* Dieckmann U & Law R (2000). Relaxation projections and the method of moments.
  In: *The Geometry of Ecological Interactions*, pp. 412–455. Cambridge University
  Press.
* Fournier N & Méléard S (2004). A microscopic probabilistic description of a
  locally regulated population and macroscopic approximations. *Annals of Applied
  Probability* 14:1880–1919.
* Gillespie DT (1976). A general method for numerically simulating the stochastic
  time evolution of coupled chemical reactions. *Journal of Computational Physics*
  22:403–434.
* Guizar-Sicairos M & Gutiérrez-Vega JC (2004). Computation of quasi-discrete
  Hankel transforms of integer order for propagating optical wave fields.
  *Journal of the Optical Society of America A* 21:53–58.
* Law R & Dieckmann U (2000). A dynamical system for neighborhoods in plant
  communities. *Ecology* 81:2137–2148. (Chapter 14 of *The Geometry of Ecological
  Interactions*.)
* Law R, Murrell DJ & Dieckmann U (2003). Population growth in space and time:
  spatial logistic equations. *Ecology* 84:252–262.
* Murrell DJ, Dieckmann U & Law R (2004). On moment closures for population
  dynamics in continuous space. *Journal of Theoretical Biology* 229:421–432.
