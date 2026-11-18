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
