#!/usr/bin/env python3
"""Verifica che `--variant parallel` riproduca FAO_DoTS/PCE_parallel del CESGA.

Uso (dalla cartella FAO_DoTS/PCE):
    python tests/check_parallel_variant.py           # completo (qualche minuto)
    python tests/check_parallel_variant.py --quick   # solo il primo caso

Per ogni caso (N, k, maxiter, seed) esegue, ognuno in un processo separato e in
una cartella temporanea:
  1. il codice originale FAO_DoTS/PCE_parallel (Qiskit Aer); cunqa, pypesto e
     pyscat (servono solo all'HPC del CESGA e a SaCeSS) sono sostituiti da
     moduli vuoti. DE gira con workers=1 invece di -1: con updating="deferred"
     il risultato non dipende dal numero di processi (il pool dell'originale si
     blocca se lanciato con `python -c`);
  2. questo fork con --variant parallel, backend numpy, workers=1;
  3. questo fork con --variant parallel, backend numpy, workers=4.
Controlla che i campi del JSON finale coincidano (parametri, costo, nfev, nit,
a, b, p, q). Con il backend numpy le varianti parallele valutano la
popolazione in blocco (DE "vectorized"), quindi il test copre anche quel percorso.

Variante parallel_l10 (penalità di peso 10, configurazione CESGA a 26-32 bit):
  4. la funzione di costo riproduce i costi registrati nei JSON del CESGA per
     tutti i run falliti a 26-32 bit (k=4), dati a e b del JSON;
  5. numpy in blocco (1 e 4 processi) dà lo stesso run della valutazione
     individuale con Aer.
Esce con codice 1 se un controllo fallisce.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # FAO_DoTS/PCE
ORIG = os.path.join(os.path.dirname(HERE), "PCE_parallel")                     # FAO_DoTS/PCE_parallel

KEYS = ["params", "f_loss_value", "function_evaluations", "Number of iterations",
        "a_int", "b_int", "p_int", "q_int"]

RUN_ORIG = r'''
import sys, types, json, os
sys.path.insert(0, sys.argv[1])
for name in ["cunqa", "cunqa.qpu", "cunqa.qjob", "cunqa.qiskit_deps", "cunqa.qiskit_deps.transpiler",
             "pypesto", "pypesto.objective", "pypesto.problem", "pyscat", "pyscat.sacess"]:
    m = types.ModuleType(name)
    for attr in ["get_QPUs", "run", "gather", "transpiler", "Objective", "Problem", "SacessOptimizer"]:
        setattr(m, attr, None)
    sys.modules[name] = m
from src.exe_fact import ejecutar_fact
N, k, maxiter, seed = map(int, sys.argv[2:6])
r = ejecutar_fact(N_int=N, optimizer="DIFFERENTIALEVOLUTION", optimizer_params={"seed": seed, "workers": 1},
                  k=k, alpha=0.0, beta=0.0, maxiter=maxiter, n_shots=1, nqpus=None,
                  cunqa_str_arg="Simulation", family_name=None)[0]
print("RESULT " + json.dumps(r))
'''

RUN_FORK = r'''
import sys, json
sys.path.insert(0, sys.argv[1])
from src.exe_fact import ejecutar_fact
N, k, maxiter, seed, workers = map(int, sys.argv[2:7])
variant = sys.argv[7] if len(sys.argv) > 7 else "parallel"
mode = sys.argv[8] if len(sys.argv) > 8 else "Numpy"
r = ejecutar_fact(N_int=N, optimizer="DIFFERENTIALEVOLUTION", optimizer_params={"seed": seed},
                  k=k, maxiter=maxiter, n_shots=1, nqpus=None, cunqa_str_arg=mode,
                  family_name=None, variant=variant, workers=workers)[0]
print("RESULT " + json.dumps(r))
'''

# parallel_l10 non ha un codice originale pubblicato: si verifica che la
# funzione di costo riproduca i costi registrati nei JSON del CESGA a 26-32 bit
# (k=4), dove quella configurazione è stata usata, decodificando a e b dai JSON.
CHECK_L10 = r'''
import sys, glob, json, re
import numpy as np
sys.path.insert(0, sys.argv[1])
import src.loss_functions as lf
from src.auxiliar import num_qubits
from src.circuit_builder import Circuit
from src.tensor_exp_value import build_sign_tensor
ok = tot = 0
for f in glob.glob(sys.argv[2] + "/k_4/**/*.json", recursive=True):
    N = int(re.search(r"num_(\d+)", f)[1])
    if N.bit_length() not in (26, 30, 32):
        continue
    nb = N.bit_length(); m = 2 * nb; q = num_qubits(m, 4); L = int(np.ceil(1.5 * q ** 2))
    cb = Circuit(size=q, p=L, entanglement="Taylor_efficient", rotation="Taylor_efficient",
                 connectivity="brickwork_single_rotating", mode="custom"); cb.compile_circuit()
    qc = cb.get_circuit(); d_t = build_sign_tensor(3, q, 4)
    for r in json.load(open(f))["resultados"]:
        fl = float(r["f_loss_value"])
        if fl == 0 or not fl.is_integer():
            continue
        lf.alt_extract_primes = lambda v, na, nbb, a=r["a_int"], b=r["b_int"]: (None, None, a, b)
        tot_loss, _ = lf.loss_func_estimator(
            x=np.zeros(qc.num_parameters), ansatz=[qc], sim=None, N_int=N, num_a_bits=nb,
            num_b_bits=nb, list_size=m // 3, num_qubits=q, d_t=d_t, n_shots=1,
            experiment_result=[], CUNQA="Numpy", variant="parallel_l10")
        tot += 1; ok += (tot_loss == fl)
print("RESULT " + json.dumps({"ok": ok, "tot": tot}))
'''

FAILED = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg, flush=True)
    if not cond:
        FAILED.append(msg)


TIMEOUT = 1800   # secondi per esecuzione: un blocco non deve durare ore


def run(code, args):
    d = tempfile.mkdtemp(prefix="chk_par_")
    try:
        try:
            out = subprocess.run([sys.executable, "-c", code, *map(str, args)], cwd=d,
                                 capture_output=True, text=True, timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            print(f"  timeout dopo {TIMEOUT} s")
            return None
        lines = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
        if out.returncode != 0 or not lines:
            print(out.stdout[-2000:], out.stderr[-2000:])
            return None
        return json.loads(lines[-1][len("RESULT "):])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def same(u, v):
    if isinstance(u, float) or isinstance(v, float):
        return abs(u - v) <= 1e-12 * max(1.0, abs(u))
    if isinstance(u, list):
        return len(u) == len(v) and all(same(a, b) for a, b in zip(u, v))
    return u == v


CASES_L10 = [(4009, 4, 300, 0, "converge, k=4"),
             (881851, 3, 15, 1, "si ferma a maxiter")]

CASES = [(899, 3, 300, 0, "converge"),
         (4009, 4, 300, 0, "converge, k=4"),
         (881851, 3, 15, 1, "si ferma a maxiter")]

if __name__ == "__main__":
    cases = CASES[:1] if "--quick" in sys.argv else CASES
    for N, k, maxiter, seed, what in cases:
        print(f"N={N} k={k} maxiter={maxiter} seed={seed} ({what})", flush=True)
        ref = run(RUN_ORIG, [ORIG, N, k, maxiter, seed])
        check(ref is not None, "  codice originale PCE_parallel eseguito")
        if ref is None:
            continue
        for workers in (1, 4):
            new = run(RUN_FORK, [HERE, N, k, maxiter, seed, workers])
            check(new is not None, f"  fork --variant parallel, workers={workers}, eseguito")
            if new is None:
                continue
            diff = [kk for kk in KEYS if not same(ref[kk], new[kk])]
            check(not diff, f"  workers={workers}: {len(KEYS)} campi identici all'originale"
                            f"{'' if not diff else ', DIVERSI: ' + str(diff)} "
                            f"(nfev={new['function_evaluations']}, nit={new['Number of iterations']}, "
                            f"p={new['p_int']}, q={new['q_int']}, costo={new['f_loss_value']})")
    print("parallel_l10: costi registrati dal CESGA a 26-32 bit (k=4)", flush=True)
    res = run(CHECK_L10, [HERE, os.path.join(os.path.dirname(HERE), "Results", "Resultados", "Factorization")])
    check(res is not None and res["tot"] > 0 and res["ok"] == res["tot"],
          f"  costo ricalcolato = costo registrato in {res and res['ok']}/{res and res['tot']} run falliti")
    for N, k, maxiter, seed, what in (CASES_L10[:1] if "--quick" in sys.argv else CASES_L10):
        print(f"parallel_l10 N={N} k={k} maxiter={maxiter} seed={seed} ({what})", flush=True)
        ref = run(RUN_FORK, [HERE, N, k, maxiter, seed, 1, "parallel_l10", "Simulation"])
        check(ref is not None, "  valutazione individuale con Aer eseguita")
        if ref is None:
            continue
        for workers in (1, 4):
            new = run(RUN_FORK, [HERE, N, k, maxiter, seed, workers, "parallel_l10", "Numpy"])
            diff = None if new is None else [kk for kk in KEYS if not same(ref[kk], new[kk])]
            check(new is not None and not diff,
                  f"  numpy in blocco, workers={workers}: {len(KEYS)} campi identici ad Aer individuale"
                  f"{'' if not diff else ', DIVERSI: ' + str(diff)}"
                  + ("" if new is None else f" (nfev={new['function_evaluations']}, costo={new['f_loss_value']})"))
    print("\nESITO:", "TUTTO OK" if not FAILED else f"{len(FAILED)} controlli FALLITI")
    sys.exit(1 if FAILED else 0)
