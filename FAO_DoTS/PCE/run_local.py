#!/usr/bin/env python3
"""
Runner locale (fork de-HPC) per l'approccio DoTS con simulazione statevector
esatta (backend numpy o qiskit-aer). Sostituisce main_simul.py per l'esecuzione
fuori dall'ambiente HPC CUNQA.

Uso:
    python run_local.py --num 899 --k 3 --optimizer DIFFERENTIALEVOLUTION --maxiter 300
    python run_local.py --num 899 --seed 7 --outdir runs/seed_7   # run riproducibile e isolato

--seed   rende il run riproducibile: è passato a scipy differential_evolution
         (DE) e inizializza i generatori globali di `random` (usato da PSO/QDPSO)
         e di numpy.
--outdir cartella in cui scrivere Resultados/ (default: cartella corrente). Run
         con seed diversi sullo stesso N vanno in outdir diversi, altrimenti i CSV
         di iterazione si sovrascrivono.
--backend numpy (default) | aer. "numpy" usa src/fast_sim.py: stesse probabilità
         esatte di Aer, ~5-9x più veloce (tests/check_numpy_backend.py). I
         risultati finiscono in Resultados/.../Numpy/ invece di .../Simulation/.
--variant serial (default) | parallel. Riproduce le due versioni del CESGA:
         serial = FAO_DoTS/PCE (costo con regolarizzazione, DE "immediate"),
         parallel = FAO_DoTS/PCE_parallel (senza regolarizzazione, penalità su
         (a-b) = 0 mod N, DE "deferred", decodifica del migliore di DE).
--variant parallel_l10: come parallel, ma con penalità di peso 10 invece di N e
         senza la penalità su (a-b) = 0 mod N: la configurazione con cui il
         CESGA ha prodotto i run k=4 a 26-32 bit (ricostruita dai loro JSON).
--workers processi per le varianti parallel; con lo stesso seed il run è
         identico per qualsiasi numero di workers. Con il backend numpy la
         popolazione è simulata in blocco (DE "vectorized"), molto più veloce.

Il CSV con una riga per valutazione si scrive solo con PCE_LOG_EVALS=1 (con k=4
è ~1 GB per run); quello per generazione (*_iter.csv) si scrive sempre.
"""
import os, sys, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)

from src.exe_experiments import casuistica_experimento, ejecutar_experimentos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=899, help="Semiprimo da fattorizzare")
    ap.add_argument("--k", type=int, default=3, help="Ordine di compressione PCE")
    ap.add_argument("--optimizer", default="DIFFERENTIALEVOLUTION",
                    choices=["DIFFERENTIALEVOLUTION", "PSO", "QDPSO"])
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--seed", type=int, default=None,
                    help="Seed dell'ottimizzatore (default: nessuno, run non riproducibile)")
    ap.add_argument("--outdir", default=None,
                    help="Cartella in cui scrivere Resultados/ (default: cartella corrente)")
    ap.add_argument("--backend", default="numpy", choices=["numpy", "aer"],
                    help="Simulatore statevector: numpy (default, veloce) o aer (riferimento)")
    ap.add_argument("--variant", default="serial", choices=["serial", "parallel", "parallel_l10"],
                    help="serial = FAO_DoTS/PCE (default); parallel = FAO_DoTS/PCE_parallel, "
                         "con cui sono stati prodotti quasi tutti i risultati k=4 pubblicati; "
                         "parallel_l10 = parallel con penalità di peso 10 e senza penalità su "
                         "(a-b) = 0 mod N, la configurazione dei run CESGA a 26-32 bit")
    ap.add_argument("--workers", type=int, default=1,
                    help="Processi per valutare la popolazione di DE (solo --variant parallel; "
                         "il risultato non dipende da questo numero)")
    args = ap.parse_args()
    if args.workers > 1 and args.variant == "serial":
        ap.error("--workers > 1 richiede una variante parallel")
    mode = {"numpy": "Numpy", "aer": "Simulation"}[args.backend]

    optimizer_params = None
    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)      # PSO/QDPSO estraggono dal modulo random della stdlib
        np.random.seed(args.seed)
        if args.optimizer == "DIFFERENTIALEVOLUTION":
            optimizer_params = {"DIFFERENTIALEVOLUTION": {"seed": args.seed}}

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        os.chdir(args.outdir)   # tutti i percorsi di output upstream sono relativi

    combos = casuistica_experimento(["Factorization"], [args.num], [args.optimizer], [args.k])
    for combo in combos:
        print(f"\n[run_local] esperimento: {combo}  seed={args.seed}  backend={args.backend}  "
              f"variant={args.variant}  workers={args.workers}  outdir={os.getcwd()}")
        ruta_csv, ruta_csv_iter = ejecutar_experimentos(
            exp_list=combo,
            optimizer_params=optimizer_params,
            maxiter=args.maxiter,
            n_shots=1,          # ignorato: statevector esatto (numpy e aer)
            nqpus=None,
            cunqa_str=mode,
            family_name=None,
            variant=args.variant,
            workers=args.workers,
        )
        print(f"[run_local] CSV per generazione: {os.path.join(os.getcwd(), ruta_csv_iter)}")


if __name__ == "__main__":
    main()
