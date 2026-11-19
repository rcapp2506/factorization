#!/usr/bin/env bash
# pull_results.sh — dal PORTATILE (WSL): rientro dei risultati da Magellano.
#     USER_MAG=<tuo_username> bash pull_results.sh
set -euo pipefail
USER_MAG="${USER_MAG:?USER_MAG=<username> bash pull_results.sh}"
HOST="magellano.pi.infn.it"
BASE="/gpfs/ddn/cta/cappu/factorization"
DEST="${DEST:-$(cd "$(dirname "$0")/.." && pwd)/magellano_results}"
mkdir -p "$DEST"
rsync -avh --progress "$USER_MAG@$HOST:$BASE/output/Resultados/" "$DEST/"
echo "risultati in $DEST"
