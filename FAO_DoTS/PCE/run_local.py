#!/usr/bin/env python3
"""
Runner locale (fork de-HPC) per l'approccio DoTS in modalità Simulation (qiskit-aer).
Sostituisce main_simul.py per l'esecuzione fuori dall'ambiente HPC CUNQA.

Uso:
    python run_local.py --num 899 --k 3 --optimizer DIFFERENTIALEVOLUTION --maxiter 300
    python run_local.py --num 899 --seed 7 --outdir runs/seed_7   # run riproducibile e isolato

--seed   rende il run riproducibile: è passato a scipy differential_evolution
         (DE) e usato per inizializzare il generatore globale di numpy (PSO/QDPSO).
--outdir cartella in cui scrivere Resultados/ (default: cartella corrente). Run
         con seed diversi sullo stesso N vanno in outdir diversi, altrimenti i CSV
         di iterazione si sovrascrivono.
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
    args = ap.parse_args()

    optimizer_params = None
    if args.seed is not None:
        import numpy as np
        np.random.seed(args.seed)   # PSO/QDPSO usano il generatore globale
        if args.optimizer == "DIFFERENTIALEVOLUTION":
            optimizer_params = {"DIFFERENTIALEVOLUTION": {"seed": args.seed}}

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        os.chdir(args.outdir)   # tutti i percorsi di output upstream sono relativi

    combos = casuistica_experimento(["Factorization"], [args.num], [args.optimizer], [args.k])
    for combo in combos:
        print(f"\n[run_local] esperimento: {combo}  seed={args.seed}  outdir={os.getcwd()}")
        ruta_csv, ruta_csv_iter = ejecutar_experimentos(
            exp_list=combo,
            optimizer_params=optimizer_params,
            alpha=0.0,
            beta=0.0,
            maxiter=args.maxiter,
            n_shots=1,          # ignorato in Simulation (statevector esatto)
            nqpus=None,
            cunqa_str="Simulation",
            family_name=None,
        )
        print(f"[run_local] CSV: {os.path.join(os.getcwd(), ruta_csv)}")


if __name__ == "__main__":
    main()
