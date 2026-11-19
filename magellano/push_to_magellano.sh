#!/usr/bin/env bash
# ============================================================================
# push_to_magellano.sh — dal PORTATILE (WSL): copia il repo di fattorizzazione
# su Magellano (INFN Pisa).
#     USER_MAG=<tuo_username> bash magellano/push_to_magellano.sh
# Richiede la VPN INFN Pisa (o la rete interna della Sezione).
# Solo codice (~pochi MB): risultati locali e rientrati sono esclusi.
# ============================================================================
set -euo pipefail
USER_MAG="${USER_MAG:?USER_MAG=<username> bash push_to_magellano.sh}"
HOST="magellano.pi.infn.it"
BASE="/gpfs/ddn/cta/cappu/factorization"
SRC="$(cd "$(dirname "$0")/.." && pwd)"   # radice del repo

ssh "$USER_MAG@$HOST" "mkdir -p $BASE/{code,slurm/logs,output}"
rsync -avh --delete \
      --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
      --exclude 'Resultados' --exclude 'magellano_results' \
      "$SRC"/ "$USER_MAG@$HOST:$BASE/code/"
rsync -avh "$SRC"/magellano/magellano_*.sbatch "$SRC"/magellano/magellano_common.sh \
      "$USER_MAG@$HOST:$BASE/slurm/"
echo "fatto. Ora su magellano:  cd $BASE/code && bash magellano/setup_magellano.sh"
