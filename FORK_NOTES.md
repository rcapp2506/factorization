# Note del fork (de-HPC + benchmark)

Questo repository è un fork di
[`csampron/CESGA-Quantum-Spain-PCE-Factorisation`](https://github.com/csampron/CESGA-Quantum-Spain-PCE-Factorisation),
codice a corredo dell'articolo *"Can PCE solve the factorisation problem via
optimisation?"* (Alonso, Samprón, Veiga, Gómez — CESGA, arXiv:2607.23727).

Tutto il merito del metodo e del codice originale è degli autori CESGA. Questo
fork ne estende l'uso in due direzioni:

1. **Esecuzione locale (de-HPC).** Il codice originale è cucito sull'ambiente
   HPC QMIO del CESGA (CUNQA + moduli `module load`). Qui la modalità
   `Simulation` (qiskit-aer, statevector esatto) gira su qualsiasi macchina.
2. **Base per un benchmark a tre vie** su fattorizzazione di semiprimi via
   ottimizzazione: PCE/DoTS vs OpenJij (SQA / A-PSA) vs SQOA.

## Modifiche rispetto all'upstream

- `FAO_DoTS/PCE/src/loss_functions.py` — gli import di `cunqa` sono resi
  **lazy**: vengono eseguiti solo nelle modalità `Shots`/`Circuits`. La
  modalità `Simulation` non richiede più CUNQA. Aggiunta inizializzazione
  esplicita delle QPU nel ramo distribuito.
- `FAO_DoTS/PCE/run_local.py` — runner da riga di comando che sostituisce
  `main_simul.py` (quest'ultimo contiene un placeholder non eseguibile).
- `requirements.txt` — dipendenze per l'esecuzione locale.
- `FAO_DoTS/PCE/src/` — pulizie dopo la review del 24/11/2026, **senza effetto
  sui risultati** (stesso seed → stesso JSON, verificato su run che convergono
  e su run che esauriscono il budget):
  - `alpha` e `beta` tolti da tutta la catena di chiamate
    (`ejecutar_experimentos`, `ejecutar_fact`, `run_vqe_optimization`,
    `loss_func_estimator`) e dal JSON di output: non entravano in nessun calcolo.
    I JSON originali del CESGA hanno ancora i due campi (sempre 0.0).
  - `loss_pq` scritta per quello che vale: `p = gcd(a−b, N)` e `q = N // p`
    danno sempre `p·q = N`, quindi il termine `popcount(N XOR p·q)²` era
    identicamente 0 e `loss_pq` vale `2N` (fattorizzazione banale) oppure 0.
  - Il criterio di arresto di DE legge la loss base del migliore da
    `experiment_result` invece di rivalutare il circuito: prima ogni generazione
    costava una valutazione in più non contata in `function_evaluations`.
  - QDPSO: `np.NaN` → `np.nan` (NumPy 2) e storia per iterazione scritta in
    `_iter.csv`. `--seed` ora inizializza anche il modulo `random`, da cui
    PSO/QDPSO estraggono davvero.
  - Backend numpy: errore esplicito oltre 12 qubit (usare `--backend aer`).

### Limite noto dell'upstream, lasciato invariato

Le m = 2·bit(N) variabili sono ripartite in ⌊m/3⌋ stringhe X, ⌊m/3⌋ Y e le
restanti Z, ma il numero di qubit garantisce solo 3·C(q,k) ≥ m. Quando
m = 3·C(q,k) − 1 la base Z dovrebbe fornire C(q,k)+1 valori e ne ha uno in meno:
il bit più significativo di `b` non è codificato e `b` ha un bit in meno, senza
errori. Con k=4 (fino a 7 qubit) succede per N a 7, 22 e 52 bit; nella campagna CESGA k=4 solo
per N = 3819047 (22 bit, `b` su 21 bit). Il comportamento non è stato corretto
per restare confrontabili con i risultati CESGA: `ejecutar_fact` stampa un
avviso e scrive nel JSON di output il campo `b_bits_missing` (numero di bit di
`b` non codificati, 0 nel caso normale), da usare per marcare o escludere quei
run nell'analisi. I JSON prodotti prima di questa modifica non hanno il campo.

## Esecuzione locale

```bash
pip install -r requirements.txt
cd FAO_DoTS/PCE
python run_local.py --num 899 --k 3 --optimizer DIFFERENTIALEVOLUTION --maxiter 300
```

Istanze di test in `Numbers_test.txt` (semiprimi bilanciati da 10 a 38 bit).

### Verificato

| N | fattori | bit | k | optimizer | esito | tempo |
|---|---|---|---|---|---|---|
| 899 | 29 × 31 | 10 | 3 | DE | ✅ | ~5 s |
| 4009 | 19 × 211 | 12 | 4 | DE | ✅ | ~46 s |

(qiskit 2.6.2, qiskit-aer 0.18.4, CPU singola.)

## TODO benchmark

- [ ] Estrarre il mapping dell'istanza di fattorizzazione in un modulo comune
      condiviso dai tre solver.
- [ ] Wrapper OpenJij (SQA / A-PSA) sulle stesse istanze.
- [ ] Wrapper SQOA (`jij/sqoa-solver`) sulle stesse istanze.
- [ ] Metriche comuni: probabilità di successo e TTS normalizzato per risorse.
