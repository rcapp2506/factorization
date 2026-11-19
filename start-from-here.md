# start-from-here.md — handoff per Claude Code

Questo file orienta una sessione di Claude Code che riprende il progetto da
zero. Leggilo per intero prima di agire, poi procedi.

## Cos'è questo repo

Fork di
[`csampron/CESGA-Quantum-Spain-PCE-Factorisation`](https://github.com/csampron/CESGA-Quantum-Spain-PCE-Factorisation)
— codice dell'articolo *"Can PCE solve the factorisation problem via
optimisation?"* (Alonso, Samprón, Veiga, Gómez — CESGA, arXiv:2607.23727).

Obiettivo del proprietario (Roberto Cappuccio, `rcapp2506`): usarlo come base
per un **benchmark a tre vie** sulla fattorizzazione di semiprimi riformulata
come ottimizzazione, confrontando a parità di istanze e di metriche:

1. **PCE / DoTS** (questo codice) — correlazioni di Pauli, pochi qubit.
2. **OpenJij** — SQA (path-integral MC) e A-PSA (multistart SA). È il filone di
   lavoro del proprietario (benchmark APS Open Science, Curato et al., Intesa
   Sanpaolo, 2026).
3. **SQOA** — Sampling-based Quantum Optimization Algorithm di JIJ, su IBM
   Qiskit Functions (`jij/sqoa-solver`): quantum relaxation + sample-based
   diagonalization, la QPU solo campiona.

Tesi di fondo da verificare empiricamente: tutti e tre attaccano lo **stesso**
QUBO; la leva vera non è il numero di qubit ma la **funzione costo** e il
landscape. A risorse normalizzate, ci si attende che SQOA si allinei a
SQA/A-PSA; la differenza dovrebbe emergere dalla formulazione (DoTS, carry) più
che dal solver.

## Stato attuale (cosa è già fatto)

- Il fork gira **in locale senza CUNQA** (l'HPC del CESGA). Gli import di
  `cunqa` in `loss_functions.py` sono resi lazy: servono solo nelle modalità
  `Shots`/`Circuits`. La modalità `Simulation` usa `qiskit-aer` (statevector).
- Aggiunto `FAO_DoTS/PCE/run_local.py` come runner CLI (il `main_simul.py`
  originale contiene una riga placeholder `Your_route/` non eseguibile — non
  usarlo).
- Verificato su **qiskit 2.6.2 / aer 0.18.4** (il codice era scritto per
  qiskit 1.2.4):
  - `899 = 29 × 31` — k=3, DE — ~5 s
  - `4009 = 19 × 211` — k=4, DE — ~46 s

## Setup e avvio (verificato)

```bash
pip install -r requirements.txt
cd FAO_DoTS/PCE
python run_local.py --num 899 --k 3 --optimizer DIFFERENTIALEVOLUTION --maxiter 300
```

Il DE si ferma da solo quando l'indicatore (base loss) raggiunge 0. I semiprimi
di test (bilanciati, 10–38 bit) sono in `Numbers_test.txt`. Output per run in
`FAO_DoTS/PCE/Resultados/...` (ignorato da git).

## Mappa del codice (DoTS, il ramo buono)

Tutto in `FAO_DoTS/PCE/src/`:

- `tensor_exp_value.py` — **cuore PCE**. Costruisce il sign-tensor dalla base
  computazionale × combinazioni di qubit di grado k e calcola le correlazioni
  di Pauli ⟨P⟩ per contrazione tensoriale con la distribuzione. In `Simulation`
  calcola lo statevector una volta in base Z e ricava X/Y con H e S†H.
- `loss_functions.py` — **funzione costo DoTS**. `loss = loss_pq + loss_ab +
  100·reg`:
  - `loss_pq = Hamming(N, p·q)² + N·(penalità triviali)` (via `r = N XOR p·q`,
    `r.bit_count()`);
  - `loss_ab = (D mod N)² + N·(penalità)`, con `D = (a−b)(a+b)`;
  - `reg = mean(⟨P⟩²)²` con peso `nu=100` — **NON è nel paper**, è tuning
    anti-saturazione ereditato dal MaxCut. Prima leva da studiare/variare.
  - `p,q` recuperati da `gcd(a−b, N)` (DoTS).
- `circuit_builder.py` — ansatz brickwork hardware-efficient (RX/RY/RZ +
  RXX/RYY/RZZ). Profondità: `layers = ceil(1.5·qubits^floor(k/2))`.
- `auxiliar.py` — `num_qubits(m,k)` (compressione PCE), conversioni bit/int,
  `extract_factors` (gcd), `alt_extract_primes`.
- `utilities.py` — `run_vqe_optimization`: DE / PSO / QDPSO + default (DE:
  best1exp, popsize 10, maxiter, init halton). Callback DE si ferma a base
  loss = 0.
- `exe_fact.py` / `exe_experiments.py` — orchestrazione (carica istanza,
  dimensiona circuito, ottimizza, salva JSON/CSV).

`FAO_ORG/` è l'approccio Basic (più debole nel paper). `*_parallel/` usa worker
MPI/CUNQA e SaCeSS: **non verificato in locale**, e il ramo `Shots` ha un bug a
monte (`QPUs` non definito nella versione seriale, sistemato nel ramo lazy ma
non testato).

## Convenzioni

- **Python/Qiskit**: usare l'ultima Qiskit (progetto scritto per 1.2.4, gira su
  2.6.x — attenzione a eventuali deprecazioni Aer/`save_statevector`).
- **Commit**: autore `Roberto Cappuccio <rcapp2506@users.noreply.github.com>`.
  Chiudere i messaggi con le righe di attribuzione Claude Code se presenti nel
  system prompt della sessione.
- Non committare `Resultados/` né `__pycache__/` (già in `.gitignore`).
- Provenienza upstream: il repo parte da un commit radice pulito (il checkout
  era shallow); hash upstream `1826113` citato nel commit e in `FORK_NOTES.md`.

## Prossimi passi (TODO benchmark)

1. **`benchmark/mapping.py`** — estrarre in un modulo unico la costruzione
   dell'istanza di fattorizzazione → QUBO/Ising (riuso di Burges/QUBOVERT come
   nel lavoro OpenJij), così la stessa istanza alimenta i tre solver.
2. **`benchmark/solver_pce.py`** — wrapper attorno a `ejecutar_fact` (DoTS, già
   pronto).
3. **`benchmark/solver_openjij.py`** — SQA + A-PSA con OpenJij sulle stesse
   istanze.
4. **`benchmark/solver_sqoa.py`** — `jij/sqoa-solver` via Qiskit Functions
   (richiede piano IBM Premium/Flex/On-Prem; attenzione: QUBO di fattorizzazione
   è denso e con alto range dinamico dei coupling).
5. **`benchmark/metrics.py`** — probabilità di successo `ps` e TTS normalizzato
   per risorse; tuning bayesiano del budget come nel lavoro OpenJij.
6. Idee sulla cost function (leva principale): formulazione con variabili di
   **carry**, **warm-start** tra semiprimi di taglia simile.

## Come sincronizzarsi e proseguire

```bash
git clone https://github.com/rcapp2506/factorization
cd factorization
git checkout -b benchmark-setup        # lavora su un branch
# ...modifiche...
git push -u origin benchmark-setup     # poi PR verso main
```
