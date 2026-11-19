#!/usr/bin/env bash
# magellano_smoke.sbatch — smoke test (~pochi minuti): un solo semiprimo piccolo
# in modalità Simulation, per verificare venv, moduli e tempi prima dell'array.
#SBATCH --job-name=pce_smoke
#SBATCH --account=default
#SBATCH --partition=levante
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/ddn/cta/cappu/factorization/slurm/logs/smoke_%j.out
#SBATCH --error=/gpfs/ddn/cta/cappu/factorization/slurm/logs/smoke_%j.err

source /gpfs/ddn/cta/cappu/factorization/slurm/magellano_common.sh

echo "  smoke: 899 = 29 × 31  e  4009 = 19 × 211"
python run_local.py --num 899  --k 3 --optimizer DIFFERENTIALEVOLUTION --maxiter 300
python run_local.py --num 4009 --k 4 --optimizer DIFFERENTIALEVOLUTION --maxiter 300
echo "  smoke OK"
