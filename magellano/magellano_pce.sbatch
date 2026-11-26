#!/usr/bin/env bash
# magellano_pce.sbatch — fattorizzazione PCE/DoTS (simulazione statevector)
# su Magellano (INFN Pisa). Array job su (semiprimo × seed): R run indipendenti
# e riproducibili per ogni N del file NUMBERS, per stimare la probabilità di
# successo ps e il TTS.
#
# Indice del task:  id = n_idx * R + seed   (n_idx = riga del file NUMBERS)
# L'header è dimensionato per 28 semiprimi × R=10 seed = 280 task (0-279).
# Per un R diverso:  sbatch --array=0-$((28*R-1))%64 --export=ALL,R=<R> magellano_pce.sbatch
#
# Parametri (override con --export=ALL,...):
#   R        numero di seed per semiprimo (default 10)
#   K        ordine di compressione PCE (default 3)
#   MAXITER  iterazioni massime dell'ottimizzatore (default 300)
#   OPT      DIFFERENTIALEVOLUTION | PSO | QDPSO (default DIFFERENTIALEVOLUTION)
#   BACKEND  numpy (default, src/fast_sim.py, ~5-9x più veloce) | aer
#   NUMBERS  file dei semiprimi, assoluto o relativo a $CODE_DIR
#            (default Numbers_test.txt; es. magellano/numbers_cesga_k4.txt)
#   TAG      nome della cartella di output (default k<K>_<OPT>_it<MAXITER>)
#   VARIANT  serial (default, FAO_DoTS/PCE) | parallel (FAO_DoTS/PCE_parallel,
#            la versione con cui sono stati prodotti quasi tutti i risultati
#            k=4 del CESGA). Con parallel la popolazione di DE è valutata su
#            SLURM_CPUS_PER_TASK processi (1 thread BLAS ciascuno): chiedere i
#            core con --cpus-per-task e la memoria con --mem (~0.3 GB a core).
#
# Output: $BASE/output/<TAG>/N<N>/seed_<s>/Resultados/...
#
# Campagna CESGA k=4 (14 semiprimi × 20 seed, budget 10000, run fino a ~1 giorno):
#   sbatch --array=0-279%280 --time=48:00:00 \
#          --export=ALL,R=20,K=4,MAXITER=10000,NUMBERS=magellano/numbers_cesga_k4.txt,TAG=cesga_k4_it10000 \
#          magellano_pce.sbatch
#
# Campagna con la variante parallela (stessi semiprimi, 16 core per run):
#   sbatch --array=<indici> --cpus-per-task=16 --mem=8G --time=12:00:00 \
#          --export=ALL,R=50,K=4,MAXITER=10000,VARIANT=parallel,NUMBERS=magellano/numbers_cesga_k4.txt,TAG=cesga_k4_it10000_parallel \
#          magellano_pce.sbatch
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
: "${BACKEND:=numpy}"
: "${NUMBERS:=Numbers_test.txt}"
: "${TAG:=k${K}_${OPT}_it${MAXITER}}"
: "${VARIANT:=serial}"

WORKERS=1
if [ "$VARIANT" = parallel ]; then
    WORKERS="${SLURM_CPUS_PER_TASK:-1}"
    # un processo per core, ciascuno con un solo thread BLAS/OpenMP
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
fi

[[ "$NUMBERS" = /* ]] || NUMBERS="$CODE_DIR/$NUMBERS"
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

OUT="$BASE/output/${TAG}/N${N}/seed_${SEED}"
echo "  task $SLURM_ARRAY_TASK_ID → N=$N (riga $N_IDX di $(basename "$NUMBERS"))  seed=$SEED  k=$K  opt=$OPT  maxiter=$MAXITER  backend=$BACKEND  variant=$VARIANT  workers=$WORKERS"
echo "  output: $OUT"
t0=$(date +%s)
python run_local.py --num "$N" --k "$K" --optimizer "$OPT" --maxiter "$MAXITER" \
       --seed "$SEED" --outdir "$OUT" --backend "$BACKEND" \
       --variant "$VARIANT" --workers "$WORKERS"
RC=$?
echo "  N=$N seed=$SEED terminato rc=$RC in $(( $(date +%s) - t0 ))s"
exit $RC
