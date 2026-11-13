#!/bin/bash
# =====================================================================================================
# run_IBM.slurm -- individual-based model: data (CSV) and Figures 2-5 (IBM panels)
#
#   Submit from the folder that contains IBM.py:      sbatch run_IBM.slurm
#   Results:  IBM/data/*.csv   IBM/figures/*.png, *.pdf   (logs in IBM/logs/)
#
# The array tasks share the work.  The task that completes the last piece writes IBM/data and
# IBM/figures automatically.  If some tasks stop at the time limit, submit the same file again: finished
# work is kept and only the rest is run.  Progress:  python IBM.py status
# =====================================================================================================
#SBATCH --job-name=IBM
#SBATCH --partition=compute
#SBATCH --array=0-59%30            # 60 tasks (~9 h each), at most 30 running at a time (~16,000 CPU h in total)
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=IBM/logs/IBM_%A_%a.out
#SBATCH --error=IBM/logs/IBM_%A_%a.err

# ----------------------------------------- settings ---------------------------------------------------
PYTHON="$HOME/miniconda3/envs/ibm-env/bin/python"   # Python with numpy, scipy, pandas, matplotlib

export REPS=150            # realizations per cell (3 x the earlier 50)
export REP_BLOCK=50        # realizations per piece of work (each cell = REPS / REP_BLOCK pieces)
export N_AXIS=60           # 60 x 60 standardized radii, 0.1 .. 10, in 1, 2 and 3 dimensions
export T_END=100           # simulated time
export T_EQUIL=30          # N~ = mean N over T_EQUIL .. T_END
export T_SNAP=30           # pair-count snapshots from T_SNAP ...
export SNAP_DT=1           # ... every SNAP_DT
export T_SNAP_LATE=70      # strongly clustered cells: only snapshots from T_SNAP_LATE (still developing earlier)
export D_RATE=0.0          # the draft model: d = 0 (lambda = 400), b = 0.4 and d' = 0.001 are fixed in IBM.py
# ------------------------------------------------------------------------------------------------------

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p IBM/logs
export NPROC="${SLURM_CPUS_PER_TASK:-4}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg MPLCONFIGDIR="$PWD/IBM/.mplcache"
"$PYTHON" -c 'import numpy, scipy, pandas, matplotlib' || { echo "FATAL: $PYTHON lacks numpy/scipy/pandas/matplotlib" >&2; exit 1; }
"$PYTHON" IBM.py run
