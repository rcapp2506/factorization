# Fattorizzazione PCE/DoTS su Magellano (INFN Pisa) — partizione `levante`

Esegue la modalità **Simulation** (Aer statevector, nessun CUNQA) del ramo
`FAO_DoTS/PCE` come job SLURM. L'array copre **semiprimo × seed**: R run
indipendenti e riproducibili per ogni N di `Numbers_test.txt`, da cui stimare la
probabilità di successo `ps` e il TTS. Stesse convenzioni dei job iSEEDs: nodi
Lenovo SR665 V3 (2×AMD EPYC 9654, 192 core, 1.5 TiB RAM), area di gruppo
`/gpfs/ddn/cta/cappu/`, modulo `python/3.13.5-gcc-11.5.0`, venv dedicato.
Accesso: `ssh <user>@magellano.pi.infn.it` dalla rete di Sezione o via VPN INFN.

```
magellano/
├── setup_magellano.sh     una tantum sul login node: directory, venv (requirements.txt), sbatch, verifica import
├── magellano_common.sh    pezzo comune agli sbatch: module + venv + threading
├── magellano_smoke.sbatch 899 e 4009 con seed 0 (~minuti) su nodo di calcolo
├── magellano_pce.sbatch   array 28 semiprimi × R=10 seed (0-279, max 64 in parallelo)
├── push_to_magellano.sh   dal portatile (WSL): rsync del codice (~pochi MB)
└── pull_results.sh        dal portatile: rientro di $BASE/output in magellano_results/
```

## Procedura

```bash
# 1. dal portatile (WSL), dalla radice del repo
USER_MAG=<username> bash magellano/push_to_magellano.sh

# 2. su magellano, una volta
ssh <username>@magellano.pi.infn.it
cd /gpfs/ddn/cta/cappu/factorization/code && bash magellano/setup_magellano.sh

# 3. smoke test su nodo di calcolo (pochi minuti), poi leggi il log
cd /gpfs/ddn/cta/cappu/factorization/slurm
sbatch magellano_smoke.sbatch
tail -f logs/smoke_<jobid>.out

# 4. array completo: 28 semiprimi × 10 seed
sbatch magellano_pce.sbatch
# varianti:
sbatch --array=0-139%64 magellano_pce.sbatch                       # primi 14 semiprimi
sbatch --array=0-139%64 --export=ALL,R=5 magellano_pce.sbatch      # 28 × 5 seed
sbatch --export=ALL,K=4,MAXITER=500 magellano_pce.sbatch           # altri parametri
squeue -u $USER

# 5. rientro risultati (dal portatile)
USER_MAG=<username> bash magellano/pull_results.sh
```

L'indice del task è `id = n_idx * R + seed`: con R diverso da 10 l'intervallo
`--array` va adattato a `0-(28*R-1)`; se l'indice eccede le righe del file il
task esce con errore esplicito.

## Riproducibilità

Ogni task chiama `run_local.py --seed <s> --outdir <dir>`. Il seed è passato a
`scipy.optimize.differential_evolution` (e al generatore globale di numpy per
PSO/QDPSO): rilanciare lo stesso (N, seed, K, OPT, MAXITER) riproduce il run.
Ogni run scrive in una cartella propria, quindi i CSV di iterazione non si
sovrascrivono tra seed.

## Dimensionamento

La modalità Simulation è leggera: `num_qubits(2·bits(N), k)` resta sotto ~10
qubit anche per N a 38 bit, quindi lo statevector è piccolo e il costo sta
nell'ottimizzatore classico, che è seriale (un candidato alla volta): un run usa
un solo core, e la velocità del singolo core conta più del numero di core. Default
per task: 1 core, 4G, 6h; fino a 64 task in parallelo. Riferimento dallo smoke
(seed 0): 899 in 9 s, 4009 (k=4) in 82 s. I numeri più grandi (ultime righe)
sono i più lenti e non ancora misurati: conviene un primo giro sulle prime 14
righe (`--array=0-139%64`), misurare i tempi, poi la coda alta con un walltime
adeguato. Partizione `levante`: walltime massimo 30 giorni.

## Output

```
$BASE/output/k<K>_<OPT>_it<MAXITER>/N<N>/seed_<s>/Resultados/Factorization/k_<K>/Simulation/num_<N>/<OPT>/
    Fact_num_<N>_<OPT>_<K>.json     riepilogo (Sol_Reached, p, q, nfev, tempo, ...)
    Fact_num_<N>_<OPT>_<K>.csv      storia delle iterazioni
```

`pull_results.sh` porta `$BASE/output/` in `<repo>/magellano_results/`, che è
ignorata da git ed esclusa da `push_to_magellano.sh`.

> Nota: questo gira la **formulazione DoTS** (cost function diretta del paper),
> non il QUBO di `benchmark/mapping.py`. Il QUBO comune serve agli altri due
> solver del benchmark (OpenJij, SQOA); vedi `benchmark/`.
