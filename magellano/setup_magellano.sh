#!/usr/bin/env bash
# ============================================================================
# setup_magellano.sh — configurazione UNA TANTUM dell'ambiente di
# fattorizzazione PCE/DoTS su Magellano (INFN Pisa).
#
# Da eseguire sul login node (magellano.pi.infn.it), dalla cartella in cui hai
# copiato il repo (vedi push_to_magellano.sh):
#     bash setup_magellano.sh
#
# Cosa fa:
#   1. crea l'albero di lavoro in /gpfs/ddn/cta/cappu/factorization
#   2. crea il venv dedicato factorization_venv (python 3.13 da module) con
#      qiskit, qiskit-aer, scipy, numpy, pandas, networkx (requirements.txt)
#   3. copia il codice del repo al suo posto ($BASE/code)
#   4. verifica l'ambiente (import qiskit/aer + uno smoke factor piccolo)
# Rieseguibile: non ricrea ciò che esiste.
# ============================================================================
set -euo pipefail

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
# Dipendenze dal requirements.txt del repo (qiskit>=1.3, aer>=0.15, ...).
pip install -q -r "$SRC/requirements.txt"
python - << 'PY'
import qiskit, qiskit_aer, numpy, scipy
print(f"   qiskit {qiskit.__version__}  aer {qiskit_aer.__version__}  "
      f"numpy {numpy.__version__}  scipy {scipy.__version__}")
PY

echo "══ 3. codice del repo in $BASE/code"
# rsync del repo escludendo git, cache e risultati locali
rsync -a --delete \
    --exclude '.git' --exclude '__pycache__' --exclude 'Resultados' \
    --exclude '*.pyc' \
    "$SRC"/ "$BASE/code/"
# gli sbatch e il common vivono in slurm/
cp -v "$SRC"/magellano/magellano_*.sbatch "$SRC"/magellano/magellano_common.sh "$BASE/slurm/"

echo "══ 4. verifica: smoke factor 899 = 29 × 31 (Simulation)"
cd "$BASE/code/FAO_DoTS/PCE"
python run_local.py --num 899 --k 3 --optimizer DIFFERENTIALEVOLUTION --maxiter 100

cat << MSG

Setup completato. Prossimi passi:
  cd $BASE/slurm
  sbatch magellano_pce.sbatch          # array sui semiprimi di Numbers_test.txt
  squeue -u \$USER
  tail -f logs/pce_*_*.out
MSG
