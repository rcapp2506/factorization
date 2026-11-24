#!/usr/bin/env python3
"""Verifica che il backend "Numpy" (src/fast_sim.py) sia equivalente ad Aer ("Simulation").

Uso (dalla cartella FAO_DoTS/PCE):
    python tests/check_numpy_backend.py          # completo (~qualche minuto)
    python tests/check_numpy_backend.py --quick  # solo valutazioni singole

1. Valutazioni singole: su punti casuali, loss totale e loss base uguali e
   aspettazioni <P> uguali entro 1e-12, per k = 2, 3, 4 e 5-7 qubit.
2. Run DE completi: con lo stesso seed, tutti i campi del JSON finale uguali
   (parametri, loss, nfev, nit, a, b, p, q, Sol_Reached), sia su run che
   convergono sia su run che si fermano a maxiter.
3. Log: senza PCE_LOG_EVALS si scrive solo il CSV per generazione.
Esce con codice 1 al primo controllo fallito.
"""
import os, shutil, sys, tempfile, time
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from qiskit import transpile
from qiskit_aer import AerSimulator
from src.auxiliar import bits_required, num_qubits
from src.circuit_builder import Circuit
from src.tensor_exp_value import build_sign_tensor
from src.loss_functions import loss_func_estimator
from src.fast_sim import PauliRotationSim

FAILED = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg, flush=True)
    if not cond:
        FAILED.append(msg)


def build(N, k):
    nb = bits_required(N); m = 2 * nb; q = num_qubits(m, k)
    L = int(np.ceil(1.5 * (q ** (np.floor(k / 2)))))
    cb = Circuit(size=q, p=L, entanglement="Taylor_efficient", rotation="Taylor_efficient",
                 connectivity="brickwork_single_rotating", mode="custom")
    cb.compile_circuit()
    sim = AerSimulator(method="statevector", seed_simulator=33)
    cz = transpile(cb.get_circuit().copy(), sim, optimization_level=0)
    return nb, m, q, sim, cz, build_sign_tensor(n_circuits=3, n_qubits=q, k_degree=k)


def single_evaluations(n_points=40):
    print("1. valutazioni singole (numpy vs Aer)")
    rng = np.random.default_rng(12345)
    for k, N in [(2, 899), (3, 4009), (3, 16668371), (3, 2956694171), (4, 3819047), (4, 446718216403)]:
        nb, m, q, sim, cz, d_t = build(N, k)
        t = {"Simulation": 0.0, "Numpy": 0.0}
        worst_v, same = 0.0, True
        for _ in range(n_points):
            x = rng.uniform(0, 2 * np.pi, cz.num_parameters)
            out = {}
            for mode in t:
                er = []
                t0 = time.perf_counter()
                tot, base = loss_func_estimator(
                    x=x, ansatz=[cz], sim=sim, N_int=N, num_a_bits=nb,
                    num_b_bits=nb, list_size=m // 3, num_qubits=q, d_t=d_t, n_shots=1,
                    experiment_result=er, CUNQA=mode)
                t[mode] += time.perf_counter() - t0
                out[mode] = (tot, base, np.array(list(er[0]["exp_map"].values())))
            a, b = out["Simulation"], out["Numpy"]
            same &= (a[0] == b[0] or abs(a[0] - b[0]) <= 1e-12 * abs(a[0])) and a[1] == b[1]
            worst_v = max(worst_v, float(np.max(np.abs(a[2] - b[2]))))
        check(same and worst_v < 1e-12,
              f"k={k} N={N} ({nb} bit, {q} qubit, {cz.num_parameters} par): loss uguali, "
              f"max|Δ<P>|={worst_v:.1e}; Aer {1e3*t['Simulation']/n_points:.2f} ms, "
              f"numpy {1e3*t['Numpy']/n_points:.2f} ms")


def full_runs():
    print("2. run DE completi con lo stesso seed (numpy vs Aer)")
    from src.exe_fact import ejecutar_fact
    keys = ["params", "f_loss_value", "function_evaluations", "Number of iterations",
            "a_int", "b_int", "p_int", "q_int", "Sol_Reached"]
    cwd = os.getcwd()
    for N, k, maxiter, seed, what in [(899, 3, 300, 0, "converge"),
                                      (4009, 4, 300, 0, "converge, k=4"),
                                      (881851, 3, 15, 1, "si ferma a maxiter")]:
        res, tt = {}, {}
        for mode in ("Simulation", "Numpy"):
            d = tempfile.mkdtemp(prefix=f"chk_{mode}_")
            os.chdir(d)
            try:
                t0 = time.perf_counter()
                r, _, ruta_csv, ruta_iter = ejecutar_fact(
                    N_int=N, optimizer="DIFFERENTIALEVOLUTION", optimizer_params={"seed": seed},
                    k=k, maxiter=maxiter, n_shots=1, nqpus=None,
                    cunqa_str_arg=mode, family_name=None)
                tt[mode] = time.perf_counter() - t0
                res[mode] = r
                if mode == "Numpy" and "PCE_LOG_EVALS" not in os.environ:
                    check(os.path.exists(ruta_iter) and not os.path.exists(ruta_csv),
                          "3. log: presente solo il CSV per generazione (niente CSV per valutazione)")
            finally:
                os.chdir(cwd)
                shutil.rmtree(d, ignore_errors=True)
        def same(u, v):
            # float (loss finale, che include il termine di regolarizzazione calcolato
            # dalle <P>): uguali entro 1e-12 relativo; interi, stringhe e liste esatti.
            if isinstance(u, float) or isinstance(v, float):
                return abs(u - v) <= 1e-12 * max(1.0, abs(u))
            if isinstance(u, list):
                return len(u) == len(v) and all(same(a, b) for a, b in zip(u, v))
            return u == v
        diff = [kk for kk in keys if not same(res["Simulation"][kk], res["Numpy"][kk])]
        check(not diff, f"N={N} k={k} maxiter={maxiter} seed={seed} ({what}): "
                        f"{len(keys)} campi identici{'' if not diff else ', DIVERSI: ' + str(diff)}; "
                        f"nfev={res['Numpy']['function_evaluations']}, "
                        f"Aer {tt['Simulation']:.1f} s, numpy {tt['Numpy']:.1f} s")


if __name__ == "__main__":
    single_evaluations()
    if "--quick" not in sys.argv:
        full_runs()
    print("\nESITO:", "TUTTO OK" if not FAILED else f"{len(FAILED)} controlli FALLITI")
    sys.exit(1 if FAILED else 0)
