#!/bin/bash
# =====================================================================================================
# run_FP.slurm -- moment closure (FP): data (CSV) and Figures 2-5 (FP panels)
#
#   Submit from the folder that contains FP.py:       sbatch run_FP.slurm
#   Results:  FP/data/*.csv   FP/figures/*.png, *.pdf   (logs in FP/logs/)
#
# The array tasks share the cells.  The task that completes the last cell writes FP/data and FP/figures
# automatically.  If some tasks stop at the time limit, submit the same file again: finished cells are
# kept and only the rest is solved.  Progress:  python FP.py status
# The FP is deterministic (one solve per cell), so it has no realizations.
# =====================================================================================================
#SBATCH --job-name=FP
#SBATCH --partition=compute
#SBATCH --array=0-9
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=FP/logs/FP_%A_%a.out
#SBATCH --error=FP/logs/FP_%A_%a.err

# ----------------------------------------- settings ---------------------------------------------------
PYTHON="$HOME/miniconda3/envs/ibm-env/bin/python"   # Python with numpy, scipy, pandas, matplotlib

export N_AXIS=60           # 60 x 60 standardized radii, 0.1 .. 10, in 1, 2 and 3 dimensions
export D_RATE=0.0          # the draft model: d = 0 (lambda = 400), b = 0.4 and d' = 0.001 are fixed in FP.py
export DISPERSAL=gaussian  # n-dimensional Gaussian dispersal
export COMP_TRUNC=0        # untruncated Gaussian competition
export G0_MODE=point       # C(0)/N^2 at zero distance
# ------------------------------------------------------------------------------------------------------

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p FP/logs
export NPROC="${SLURM_CPUS_PER_TASK:-4}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg MPLCONFIGDIR="$PWD/FP/.mplcache"
"$PYTHON" -c 'import numpy, scipy, pandas, matplotlib' || { echo "FATAL: $PYTHON lacks numpy/scipy/pandas/matplotlib" >&2; exit 1; }
"$PYTHON" FP.py run
