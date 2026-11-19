#!/usr/bin/env bash
# magellano_pce.sbatch — fattorizzazione PCE/DoTS (modalità Simulation, Aer
# statevector) su Magellano (INFN Pisa). Array job: un task per semiprimo di
# Numbers_test.txt. Ogni task lancia run_local.py.
#
# Parametri (override da riga di comando, es.  sbatch --export=ALL,K=4 ...):
#   K        ordine di compressione PCE (default 3)
#   MAXITER  iterazioni massime dell'ottimizzatore (default 300)
#   OPT      DIFFERENTIALEVOLUTION | PSO | QDPSO (default DIFFERENTIALEVOLUTION)
#
# Avvio:
#   cd /gpfs/ddn/cta/cappu/factorization/slurm
#   sbatch magellano_pce.sbatch
#   # sottoinsieme (solo i primi 10):  sbatch --array=0-9 magellano_pce.sbatch
#SBATCH --job-name=pce_fact
#SBATCH --account=default
#SBATCH --partition=levante
#SBATCH --array=0-27%8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/ddn/cta/cappu/factorization/slurm/logs/pce_%A_%a.out
#SBATCH --error=/gpfs/ddn/cta/cappu/factorization/slurm/logs/pce_%A_%a.err

source /gpfs/ddn/cta/cappu/factorization/slurm/magellano_common.sh

: "${K:=3}"
: "${MAXITER:=300}"
: "${OPT:=DIFFERENTIALEVOLUTION}"

NUMBERS="$CODE_DIR/Numbers_test.txt"
[ -f "$NUMBERS" ] || { echo "✗ manca $NUMBERS"; exit 1; }

# N per questo task: il numero prima di '=' nella riga di indice SLURM_ARRAY_TASK_ID
N=$(python - "$NUMBERS" "$SLURM_ARRAY_TASK_ID" << 'PY'
import sys
path, idx = sys.argv[1], int(sys.argv[2])
rows = []
for line in open(path, encoding="utf-8"):
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    lhs = line.split("=", 1)[0]
    rows.append(int(lhs.split()[-1]))   # token prima di '='
if idx >= len(rows):
    sys.exit(f"indice {idx} fuori range (semiprimi: {len(rows)})")
print(rows[idx])
PY
) || exit 1

echo "  task $SLURM_ARRAY_TASK_ID → N=$N  (k=$K, opt=$OPT, maxiter=$MAXITER)"
t0=$(date +%s)
python run_local.py --num "$N" --k "$K" --optimizer "$OPT" --maxiter "$MAXITER"
RC=$?
echo "  N=$N terminato rc=$RC in $(( $(date +%s) - t0 ))s"

# I risultati finiscono in $PCE_DIR/Resultados (su GPFS, sotto $BASE/code).
# Li consolidiamo sotto $BASE/output per il rientro.
mkdir -p "$BASE/output"
rsync -a "$PCE_DIR/Resultados/" "$BASE/output/Resultados/" 2>/dev/null || true
exit $RC
