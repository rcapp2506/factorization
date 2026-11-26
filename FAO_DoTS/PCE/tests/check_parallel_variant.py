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
a, b, p, q). Esce con codice 1 al primo controllo fallito.
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
r = ejecutar_fact(N_int=N, optimizer="DIFFERENTIALEVOLUTION", optimizer_params={"seed": seed},
                  k=k, maxiter=maxiter, n_shots=1, nqpus=None, cunqa_str_arg="Numpy",
                  family_name=None, variant="parallel", workers=workers)[0]
print("RESULT " + json.dumps(r))
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
    print("\nESITO:", "TUTTO OK" if not FAILED else f"{len(FAILED)} controlli FALLITI")
    sys.exit(1 if FAILED else 0)
