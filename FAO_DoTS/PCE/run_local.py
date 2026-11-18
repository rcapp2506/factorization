#!/usr/bin/env python3
"""
Runner locale (fork de-HPC) per l'approccio DoTS in modalità Simulation (qiskit-aer).
Sostituisce main_simul.py per l'esecuzione fuori dall'ambiente HPC CUNQA.

Uso:
    python run_local.py --num 899 --k 3 --optimizer DIFFERENTIALEVOLUTION --maxiter 300
"""
import os, sys, argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.exe_experiments import casuistica_experimento, ejecutar_experimentos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=899, help="Semiprimo da fattorizzare")
    ap.add_argument("--k", type=int, default=3, help="Ordine di compressione PCE")
    ap.add_argument("--optimizer", default="DIFFERENTIALEVOLUTION",
                    choices=["DIFFERENTIALEVOLUTION", "PSO", "QDPSO"])
    ap.add_argument("--maxiter", type=int, default=300)
    args = ap.parse_args()

    combos = casuistica_experimento(["Factorization"], [args.num], [args.optimizer], [args.k])
    for combo in combos:
        print(f"\n[run_local] esperimento: {combo}")
        ruta_csv, ruta_csv_iter = ejecutar_experimentos(
            exp_list=combo,
            optimizer_params=None,
            alpha=0.0,
            beta=0.0,
            maxiter=args.maxiter,
            n_shots=1,          # ignorato in Simulation (statevector esatto)
            nqpus=None,
            cunqa_str="Simulation",
            family_name=None,
        )
        print(f"[run_local] CSV: {ruta_csv}")


if __name__ == "__main__":
    main()
