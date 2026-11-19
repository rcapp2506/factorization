#!/usr/bin/env bash
# magellano_pce.sbatch — fattorizzazione PCE/DoTS (modalità Simulation, Aer
# statevector) su Magellano (INFN Pisa). Array job su (semiprimo × seed): R run
# indipendenti e riproducibili per ogni N di Numbers_test.txt, per stimare la
# probabilità di successo ps e il TTS.
#
# Indice del task:  id = n_idx * R + seed   (n_idx = riga di Numbers_test.txt)
# L'header è dimensionato per 28 semiprimi × R=10 seed = 280 task (0-279).
# Per un R diverso:  sbatch --array=0-$((28*R-1))%64 --export=ALL,R=<R> magellano_pce.sbatch
#
# Parametri (override con --export=ALL,...):
#   R        numero di seed per semiprimo (default 10)
#   K        ordine di compressione PCE (default 3)
#   MAXITER  iterazioni massime dell'ottimizzatore (default 300)
#   OPT      DIFFERENTIALEVOLUTION | PSO | QDPSO (default DIFFERENTIALEVOLUTION)
#
# Output: $BASE/output/k<K>_<OPT>_it<MAXITER>/N<N>/seed_<s>/Resultados/...
#
# Avvio:
#   cd /gpfs/ddn/cta/cappu/factorization/slurm
#   sbatch magellano_pce.sbatch
#   sbatch --array=0-139%64 magellano_pce.sbatch   # solo i primi 14 semiprimi (R=10)
#SBATCH --job-name=pce_fact
#SBATCH --account=default
#SBATCH --partition=levante
#SBATCH --array=0-279%64
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/ddn/cta/cappu/factorization/slurm/logs/pce_%A_%a.out
#SBATCH --error=/gpfs/ddn/cta/cappu/factorization/slurm/logs/pce_%A_%a.err

source /gpfs/ddn/cta/cappu/factorization/slurm/magellano_common.sh

: "${R:=10}"
: "${K:=3}"
: "${MAXITER:=300}"
: "${OPT:=DIFFERENTIALEVOLUTION}"

NUMBERS="$CODE_DIR/Numbers_test.txt"
[ -f "$NUMBERS" ] || { echo "✗ manca $NUMBERS"; exit 1; }

N_IDX=$(( SLURM_ARRAY_TASK_ID / R ))
SEED=$(( SLURM_ARRAY_TASK_ID % R ))

# N = numero prima di '=' nella riga N_IDX (stesso parser di benchmark/mapping.py)
N=$(python - "$NUMBERS" "$N_IDX" << 'PY'
import re, sys
path, idx = sys.argv[1], int(sys.argv[2])
row = re.compile(r"(\d+)\s*=\s*(\d+)\s*[×xX*]\s*(\d+)")
rows = []
for line in open(path, encoding="utf-8"):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    m = row.search(line)
    if m:
        rows.append(int(m.group(1)))
if idx >= len(rows):
    sys.exit(f"indice semiprimo {idx} fuori range (semiprimi: {len(rows)}); controllare --array e R")
print(rows[idx])
PY
) || exit 1

OUT="$BASE/output/k${K}_${OPT}_it${MAXITER}/N${N}/seed_${SEED}"
echo "  task $SLURM_ARRAY_TASK_ID → N=$N (riga $N_IDX)  seed=$SEED  k=$K  opt=$OPT  maxiter=$MAXITER"
echo "  output: $OUT"
t0=$(date +%s)
python run_local.py --num "$N" --k "$K" --optimizer "$OPT" --maxiter "$MAXITER" \
       --seed "$SEED" --outdir "$OUT"
RC=$?
echo "  N=$N seed=$SEED terminato rc=$RC in $(( $(date +%s) - t0 ))s"
exit $RC
