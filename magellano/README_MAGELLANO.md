# Fattorizzazione PCE/DoTS su Magellano (INFN Pisa) — partizione `levante`

Esegue la modalità **Simulation** (Aer statevector, nessun CUNQA) del ramo
`FAO_DoTS/PCE` come job SLURM, un semiprimo per task d'array. Stesse convenzioni
dei job iSEEDs: nodi Lenovo SR665 V3 (2×AMD EPYC 9654, 192 core, 1.5 TiB RAM),
area di gruppo `/gpfs/ddn/cta/cappu/`, modulo `python/3.13.5-gcc-11.5.0`, venv
dedicato. Accesso: `ssh <user>@magellano.pi.infn.it` dalla rete di Sezione o via
VPN INFN.

```
magellano/
├── setup_magellano.sh     una tantum sul login node: directory, venv (qiskit+aer da requirements.txt), copia repo, smoke
├── magellano_common.sh    pezzo comune agli sbatch: module + venv + threading
├── magellano_smoke.sbatch 899 e 4009 (~minuti): verifica e stima tempi
├── magellano_pce.sbatch   array sui semiprimi di Numbers_test.txt (default 0-27%8)
├── push_to_magellano.sh   dal portatile (WSL): rsync del repo (~pochi MB)
└── pull_results.sh        dal portatile: rientro dei Resultados
```

## Procedura

```bash
# 1. dal portatile (WSL), dalla radice del repo
USER_MAG=<username> bash magellano/push_to_magellano.sh

# 2. su magellano, una volta
ssh <username>@magellano.pi.infn.it
cd /gpfs/ddn/cta/cappu/factorization/code && bash magellano/setup_magellano.sh

# 3. smoke test (pochi minuti), poi leggi il log
cd /gpfs/ddn/cta/cappu/factorization/slurm
sbatch magellano_smoke.sbatch
tail -f logs/smoke_<jobid>.out

# 4. array completo (28 semiprimi, max 8 in parallelo)
sbatch magellano_pce.sbatch
# varianti:
sbatch --array=0-9 magellano_pce.sbatch                 # solo i primi 10
sbatch --export=ALL,K=4,MAXITER=500 magellano_pce.sbatch # altri parametri
squeue -u $USER

# 5. rientro risultati (dal portatile)
USER_MAG=<username> bash magellano/pull_results.sh
```

## Dimensionamento

La modalità Simulation è leggera: `num_qubits(2·bits(N), k)` resta sotto ~10
qubit anche per N a 38 bit (compressione PCE), quindi lo statevector è piccolo.
Il costo sta nell'ottimizzatore classico (DE/PSO) × valutazioni di loss. Default
prudenti: `--cpus-per-task=8`, `--mem=16G`, `--time=12:00:00`. Se i run saturano
il walltime, alzare `--time` (partizione `levante`: max 30 giorni) o ridurre
`MAXITER`. I numeri più grandi di `Numbers_test.txt` (ultimi indici) sono i più
lenti: conviene un primo giro con `--array=0-13` e poi la coda alta a parte.

## Output

`run_local.py` scrive in `FAO_DoTS/PCE/Resultados/Factorization/k_<k>/Simulation/num_<N>/<OPT>/`
(CSV per iterazione + JSON riepilogo). Ogni task fa rsync di `Resultados/` sotto
`$BASE/output/Resultados/`; `pull_results.sh` li riporta sul portatile.

> Nota: questo gira la **formulazione DoTS** (cost function diretta del paper),
> non il QUBO di `benchmark/mapping.py`. Il QUBO comune serve agli altri due
> solver del benchmark (OpenJij, SQOA); vedi `benchmark/`.
