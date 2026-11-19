#!/usr/bin/env bash
# ============================================================================
# setup_magellano.sh — configurazione UNA TANTUM dell'ambiente di
# fattorizzazione PCE/DoTS su Magellano (INFN Pisa).
#
# Da eseguire sul login node (magellano.pi.infn.it). Di norma il repo è già
# stato copiato in $BASE/code da push_to_magellano.sh:
#     cd /gpfs/ddn/cta/cappu/factorization/code && bash magellano/setup_magellano.sh
# Se lo lanci da un'altra copia del repo, il codice viene sincronizzato in $BASE/code.
#
# Cosa fa:
#   1. crea l'albero di lavoro in /gpfs/ddn/cta/cappu/factorization
#   2. crea il venv dedicato factorization_venv (python 3.13 da module) con le
#      dipendenze di requirements.txt
#   3. sincronizza il codice in $BASE/code (solo se lanciato da un'altra copia)
#      e copia sbatch e common in $BASE/slurm
#   4. verifica gli import (niente calcolo sul login node: lo smoke è uno sbatch)
# Rieseguibile: non ricrea ciò che esiste.
# ============================================================================
set -eo pipefail   # niente -u: module/Lmod e activate possono usare variabili non definite

BASE="/gpfs/ddn/cta/cappu/factorization"
VENV="/gpfs/ddn/cta/cappu/factorization_venv"
PYMOD="python/3.13.5-gcc-11.5.0"
SRC="$(cd "$(dirname "$0")/.." && pwd)"   # radice del repo (contiene FAO_DoTS/)

echo "══ 1. directory in $BASE"
mkdir -p "$BASE"/{code,slurm/logs,output}

echo "══ 2. venv $VENV"
module purge
module load "$PYMOD"
if [ ! -x "$VENV/bin/python" ]; then
    python -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install --upgrade pip -q
pip install -q -r "$SRC/requirements.txt"

echo "══ 3. codice e sbatch"
if [ "$(realpath "$SRC")" != "$(realpath "$BASE/code")" ]; then
    rsync -a --delete \
        --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
        --exclude 'Resultados' --exclude 'magellano_results' \
        "$SRC"/ "$BASE/code/"
    echo "   codice sincronizzato da $SRC"
else
    echo "   codice già in $BASE/code"
fi
cp -v "$SRC"/magellano/magellano_*.sbatch "$SRC"/magellano/magellano_common.sh "$BASE/slurm/"

echo "══ 4. verifica import"
cd "$BASE/code/FAO_DoTS/PCE"
python - << 'PY'
import qiskit, qiskit_aer, numpy, scipy
from src.exe_experiments import ejecutar_experimentos  # noqa: F401
print(f"   qiskit {qiskit.__version__}  aer {qiskit_aer.__version__}  "
      f"numpy {numpy.__version__}  scipy {scipy.__version__}  — import progetto OK")
PY

cat << MSG

Setup completato. Prossimi passi:
  cd $BASE/slurm
  sbatch magellano_smoke.sbatch        # verifica su nodo di calcolo (~minuti)
  sbatch magellano_pce.sbatch          # array 28 semiprimi × 10 seed
  squeue -u \$USER
MSG
