# magellano_common.sh — pezzo comune agli sbatch (source-ato, non eseguito).
# Replica le convenzioni iSEEDs su Magellano (INFN Pisa): module python 3.13,
# venv dedicato, threading controllato. Progetto: fattorizzazione PCE/DoTS.
BASE="/gpfs/ddn/cta/cappu/factorization"
CODE_DIR="$BASE/code"                 # qui vive l'albero del repo (FAO_DoTS/...)
PCE_DIR="$CODE_DIR/FAO_DoTS/PCE"      # runner run_local.py e src/
VENV="/gpfs/ddn/cta/cappu/factorization_venv"

echo "═══════════════════════════════════════════════════════════════"
echo "  job $SLURM_JOB_ID${SLURM_ARRAY_TASK_ID:+ task $SLURM_ARRAY_TASK_ID}  host=$(hostname)  cpu=$SLURM_CPUS_PER_TASK  mem=${SLURM_MEM_PER_NODE:-?}MB"
echo "  $(date)"
echo "═══════════════════════════════════════════════════════════════"

# ── ambiente ──
module purge
module load python/3.13.5-gcc-11.5.0
source "$VENV/bin/activate" || { echo "✗ venv assente: eseguire setup_magellano.sh"; exit 1; }

# ── threading: la simulazione è statevector Aer (CPU), BLAS/OMP su pochi core ──
# La modalità Simulation è single-circuit e leggera; il costo è nell'ottimizzatore
# classico (DE/PSO). Teniamo i thread BLAS = cpus-per-task per non oversubscrivere.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS=1
export QISKIT_NUM_PROCS=1

cd "$PCE_DIR" || { echo "✗ manca $PCE_DIR"; exit 1; }
