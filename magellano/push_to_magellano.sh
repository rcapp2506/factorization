#!/usr/bin/env bash
# ============================================================================
# push_to_magellano.sh — dal PORTATILE (WSL): copia il repo di fattorizzazione
# su Magellano (INFN Pisa).
#     USER_MAG=<tuo_username> bash push_to_magellano.sh
# Richiede la VPN INFN Pisa (o la rete interna della Sezione).
# Niente dati pesanti: il codice è ~pochi MB. I risultati si rientrano a parte
# (vedi pull_results.sh).
# ============================================================================
set -euo pipefail
USER_MAG="${USER_MAG:?USER_MAG=<username> bash push_to_magellano.sh}"
HOST="magellano.pi.infn.it"
BASE="/gpfs/ddn/cta/cappu/factorization"
SRC="$(cd "$(dirname "$0")/.." && pwd)"   # radice del repo

ssh "$USER_MAG@$HOST" "mkdir -p $BASE/{code,slurm/logs,output}"
# repo intero (escludendo git/cache/risultati locali) in code/
rsync -avh --delete \
      --exclude '.git' --exclude '__pycache__' --exclude 'Resultados' --exclude '*.pyc' \
      "$SRC"/ "$USER_MAG@$HOST:$BASE/code/"
# sbatch e common anche in slurm/ (setup_magellano.sh li ricopia, ma così sono subito pronti)
rsync -avh "$SRC"/magellano/magellano_*.sbatch "$SRC"/magellano/magellano_common.sh \
      "$USER_MAG@$HOST:$BASE/slurm/"
echo "fatto. Ora su magellano:  cd $BASE/code && bash setup_magellano.sh"
