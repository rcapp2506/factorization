#!/usr/bin/env bash
# magellano_smoke.sbatch — smoke test (~pochi minuti) su un nodo di calcolo:
# due semiprimi piccoli con seed fisso, per verificare venv, moduli, tempi e
# riproducibilità prima dell'array completo.
#SBATCH --job-name=pce_smoke
#SBATCH --account=default
#SBATCH --partition=levante
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/ddn/cta/cappu/factorization/slurm/logs/smoke_%j.out
#SBATCH --error=/gpfs/ddn/cta/cappu/factorization/slurm/logs/smoke_%j.err

source /gpfs/ddn/cta/cappu/factorization/slurm/magellano_common.sh

OUT="$BASE/output/smoke_$SLURM_JOB_ID"
echo "  smoke: 899 = 29 × 31  e  4009 = 19 × 211  (seed 0)  → $OUT"
python run_local.py --num 899  --k 3 --optimizer DIFFERENTIALEVOLUTION --maxiter 300 --seed 0 --outdir "$OUT" || exit 1
python run_local.py --num 4009 --k 4 --optimizer DIFFERENTIALEVOLUTION --maxiter 300 --seed 0 --outdir "$OUT" || exit 1
echo "  smoke OK"
