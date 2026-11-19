"""
benchmark/mapping.py — costruzione solver-agnostica dell'istanza di
fattorizzazione di un semiprimo come QUBO / Ising.

Obiettivo: una *stessa* istanza (lo stesso QUBO) alimenta i tre solver del
benchmark a tre vie (PCE/DoTS, OpenJij SQA/A-PSA, SQOA). Questo modulo non
dipende da alcun solver: produce il QUBO come dizionario Python puro e offre
conversioni opzionali (Ising, dimod BQM) solo se le librerie sono presenti.

Formulazione
------------
N = p * q con p, q primi dispari, quindi il bit meno significativo di ciascun
fattore è fissato a 1:

    p = 1 + sum_{i=1..np-1} 2^i p_i ,   q = 1 + sum_{j=1..nq-1} 2^j q_j

Si minimizza l'obiettivo esatto  (N - p*q)^2 . Il prodotto p*q contiene i
termini bilineari p_i q_j; introducendo le variabili prodotto  w_ij = p_i q_j
(AND), l'espressione  p*q  diventa lineare nelle variabili {p_i, q_j, w_ij},
quindi  (N - p*q)^2  è quadratica: un QUBO.

L'uguaglianza w_ij = p_i AND q_j è imposta con la penalità di Rosenberg

    lambda * (p_i q_j - 2 p_i w_ij - 2 q_j w_ij + 3 w_ij)

che vale 0 se e solo se w_ij = p_i q_j, e lambda>0 altrimenti.

NOTA sul benchmark. Il ramo PCE/DoTS di questo repo (FAO_DoTS) NON minimizza
questo QUBO: usa una cost function diretta e non polinomiale (modulo, gcd,
popcount di N XOR p*q). Il QUBO qui è la formulazione *comune* che SQA/A-PSA e
SQOA attaccano nativamente; per un confronto davvero a parità di funzione costo
si può far valutare a PCE l'energia di QUESTO QUBO (vedi `qubo_energy`) invece
della loss DoTS. Tenere separate le due cose è il punto del benchmark.

Avvertenza (rilevante per SQOA): la penalità di default rende il range dinamico
dei coupling ampio (~N^2). È proprio la difficoltà segnalata per i QUBO densi di
fattorizzazione; `penalty` è esposto per poterlo studiare/variare.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from math import ceil, log2
from typing import Dict, List, Optional, Tuple

# Una variabile QUBO è una stringa ("p1", "q2", "w1_3").
Var = str
# QUBO: termini lineari {(v,): coef} e quadratici {(u, v): coef}, più offset.
QUBO = Dict[Tuple[Var, ...], float]


@dataclass
class FactorizationQUBO:
    """Istanza di fattorizzazione come QUBO, con i metadati per decodificare."""
    N: int
    p_bits: int
    q_bits: int
    penalty: float
    Q: QUBO = field(default_factory=dict)
    offset: float = 0.0
    p_vars: List[Var] = field(default_factory=list)   # p_i liberi
    q_vars: List[Var] = field(default_factory=list)   # q_j liberi
    w_vars: List[Var] = field(default_factory=list)   # w_ij = p_i q_j
    base_p: int = 1   # parte costante di p (bit fissati: LSB + eventuale MSB)
    base_q: int = 1
    fix_msb: bool = False

    @property
    def variables(self) -> List[Var]:
        return self.p_vars + self.q_vars + self.w_vars

    @property
    def num_vars(self) -> int:
        return len(self.variables)


# --------------------------------------------------------------------------- #
# Costruzione del QUBO
# --------------------------------------------------------------------------- #
def _add(Q: QUBO, key: Tuple[Var, ...], coef: float) -> None:
    """Accumula un coefficiente su un termine del QUBO (chiave ordinata)."""
    if len(key) == 2:
        key = tuple(sorted(key))  # type: ignore[assignment]
    Q[key] = Q.get(key, 0) + coef


def default_bit_sizes(N: int) -> Tuple[int, int]:
    """Numero di bit (upper bound) per p e q di un semiprimo bilanciato.

    p: bits(N); q: ceil(bits(N)/2). Coerente con worst_case_bits_for_primes
    del ramo PCE. Entrambi >= 2 (serve almeno il bit fissato + 1 libero).
    """
    nbits = N.bit_length()
    p_b = max(2, nbits)
    q_b = max(2, ceil(nbits / 2))
    return p_b, q_b


def build_factorization_qubo(
    N: int,
    p_bits: Optional[int] = None,
    q_bits: Optional[int] = None,
    penalty: Optional[float] = None,
    fix_msb: bool = False,
) -> FactorizationQUBO:
    """Costruisce il QUBO di (N - p*q)^2 con variabili prodotto AND.

    Parametri
    ---------
    N : int
        Semiprimo da fattorizzare (dispari).
    p_bits, q_bits : int, opzionali
        Numero di bit di p e q (incluso il bit LSB fissato a 1). Se None, si
        usano le dimensioni di default per un semiprimo bilanciato.
    penalty : float, opzionale
        Peso lambda della penalità di Rosenberg. Se None, un valore che domina
        con certezza l'obiettivo: (N-1)^2 + 1.
    fix_msb : bool
        Se True fissa anche il bit più significativo di p e q a 1. Questo esclude
        i fattori < 2^(bits-1) e in particolare le fattorizzazioni triviali
        (1, N) e (N, 1), ma richiede p_bits/q_bits pari alla lunghezza esatta in
        bit dei fattori veri (nota solo a posteriori per un semiprimo bilanciato:
        ~ nbits/2 ciascuno). Con dimensioni sbagliate l'istanza diventa
        insoddisfacibile. Default False.

    Ritorna
    -------
    FactorizationQUBO
    """
    if N % 2 == 0:
        raise ValueError("N deve essere dispari (semiprimo di due primi dispari).")
    if p_bits is None or q_bits is None:
        dp, dq = default_bit_sizes(N)
        p_bits = p_bits or dp
        q_bits = q_bits or dq
    if penalty is None:
        penalty = (N - 1) ** 2 + 1   # intero: l'energia resta esatta

    # Bit liberi: LSB (bit 0) sempre fissato a 1; MSB fissato a 1 solo se fix_msb.
    hi = 1 if fix_msb else 0
    p_idx = list(range(1, p_bits - hi))   # indici dei bit liberi di p
    q_idx = list(range(1, q_bits - hi))
    # Parte costante di p e q (bit fissati): LSB + eventuale MSB.
    base_p = 1 + ((1 << (p_bits - 1)) if fix_msb else 0)
    base_q = 1 + ((1 << (q_bits - 1)) if fix_msb else 0)

    p_vars = [f"p{i}" for i in p_idx]
    q_vars = [f"q{j}" for j in q_idx]
    w_vars = [f"w{i}_{j}" for i in p_idx for j in q_idx]

    # Espressione lineare L = p*q nelle variabili {p_i, q_j, w_ij}, con
    # p = base_p + sum 2^i p_i,  q = base_q + sum 2^j q_j:
    #   L = base_p base_q + base_p sum 2^j q_j + base_q sum 2^i p_i
    #       + sum 2^(i+j) w_ij
    # diff = N - L = c0 + sum_k c_k v_k , con c0 = N - base_p base_q.
    lin: Dict[Var, int] = {}
    for i in p_idx:
        lin[f"p{i}"] = lin.get(f"p{i}", 0) - base_q * (1 << i)
    for j in q_idx:
        lin[f"q{j}"] = lin.get(f"q{j}", 0) - base_p * (1 << j)
    for i in p_idx:
        for j in q_idx:
            lin[f"w{i}_{j}"] = lin.get(f"w{i}_{j}", 0) - (1 << (i + j))
    c0 = N - base_p * base_q

    Q: QUBO = {}
    # Obiettivo (c0 + sum c_k v_k)^2, usando v_k^2 = v_k:
    #   lineare v_k : 2 c0 c_k + c_k^2 ; quadratico (k,l): 2 c_k c_l ; offset c0^2
    # Tutto in aritmetica intera: l'energia della soluzione vera è esattamente 0.
    vars_all = list(lin.keys())
    for v in vars_all:
        ck = lin[v]
        _add(Q, (v,), 2 * c0 * ck + ck * ck)
    for a in range(len(vars_all)):
        for b in range(a + 1, len(vars_all)):
            va, vb = vars_all[a], vars_all[b]
            _add(Q, (va, vb), 2 * lin[va] * lin[vb])
    offset = c0 * c0

    # Penalità di Rosenberg per w_ij = p_i AND q_j:
    #   lambda (p_i q_j - 2 p_i w_ij - 2 q_j w_ij + 3 w_ij)
    lam = int(penalty)
    for i in p_idx:
        for j in q_idx:
            pi, qj, wij = f"p{i}", f"q{j}", f"w{i}_{j}"
            _add(Q, (pi, qj), lam)
            _add(Q, (pi, wij), -2 * lam)
            _add(Q, (qj, wij), -2 * lam)
            _add(Q, (wij,), 3 * lam)

    return FactorizationQUBO(
        N=N, p_bits=p_bits, q_bits=q_bits, penalty=lam,
        Q=Q, offset=offset, p_vars=p_vars, q_vars=q_vars, w_vars=w_vars,
        base_p=base_p, base_q=base_q, fix_msb=fix_msb,
    )


# --------------------------------------------------------------------------- #
# Valutazione / decodifica
# --------------------------------------------------------------------------- #
def qubo_energy(fq: FactorizationQUBO, sample: Dict[Var, int]) -> float:
    """Energia QUBO di un assegnamento {var: 0/1} (incluso l'offset)."""
    e = fq.offset
    for key, coef in fq.Q.items():
        if len(key) == 1:
            e += coef * sample[key[0]]
        else:
            e += coef * sample[key[0]] * sample[key[1]]
    return e


def _free_idx(fq: FactorizationQUBO) -> Tuple[List[int], List[int]]:
    """Indici dei bit liberi di p e q (dedotti dai nomi delle variabili)."""
    pi = [int(v[1:]) for v in fq.p_vars]
    qj = [int(v[1:]) for v in fq.q_vars]
    return pi, qj


def decode_factors(fq: FactorizationQUBO, sample: Dict[Var, int]) -> Tuple[int, int]:
    """Da un assegnamento dei bit a (p, q). I w_ij non servono alla decodifica."""
    pi, qj = _free_idx(fq)
    p = fq.base_p + sum((1 << i) * int(sample.get(f"p{i}", 0)) for i in pi)
    q = fq.base_q + sum((1 << j) * int(sample.get(f"q{j}", 0)) for j in qj)
    return p, q


def sample_from_pq(fq: FactorizationQUBO, p: int, q: int) -> Dict[Var, int]:
    """Assegnamento completo (p_i, q_j, w_ij coerenti) a partire da p, q interi.

    Utile per i test: valuta l'energia sulla soluzione vera, che deve essere 0.
    """
    pi, qj = _free_idx(fq)
    sample: Dict[Var, int] = {}
    for i in pi:
        sample[f"p{i}"] = (p >> i) & 1
    for j in qj:
        sample[f"q{j}"] = (q >> j) & 1
    for i in pi:
        for j in qj:
            sample[f"w{i}_{j}"] = ((p >> i) & 1) & ((q >> j) & 1)
    return sample


# --------------------------------------------------------------------------- #
# Conversione in Ising e in dimod (opzionale)
# --------------------------------------------------------------------------- #
def to_ising(fq: FactorizationQUBO):
    """Converte il QUBO in Ising (h, J, offset) con x = (1 - z)/2, z in {-1,+1}."""
    h: Dict[Var, float] = {}
    J: Dict[Tuple[Var, Var], float] = {}
    offset = fq.offset
    for key, coef in fq.Q.items():
        if len(key) == 1:
            v = key[0]
            # coef * x = coef * (1 - z)/2
            h[v] = h.get(v, 0.0) - coef / 2.0
            offset += coef / 2.0
        else:
            u, v = key
            # coef * x_u x_v = coef/4 (1 - z_u)(1 - z_v)
            J[(u, v)] = J.get((u, v), 0.0) + coef / 4.0
            h[u] = h.get(u, 0.0) - coef / 4.0
            h[v] = h.get(v, 0.0) - coef / 4.0
            offset += coef / 4.0
    return h, J, offset


def to_dimod(fq: FactorizationQUBO):
    """Ritorna un dimod.BinaryQuadraticModel se dimod è installato, altrimenti None."""
    try:
        import dimod
    except ImportError:
        return None
    linear = {k[0]: v for k, v in fq.Q.items() if len(k) == 1}
    quadratic = {k: v for k, v in fq.Q.items() if len(k) == 2}
    return dimod.BinaryQuadraticModel(linear, quadratic, fq.offset, dimod.BINARY)


# --------------------------------------------------------------------------- #
# Verifica (brute force) e utilità istanze
# --------------------------------------------------------------------------- #
def brute_force_minimum(fq: FactorizationQUBO) -> Tuple[float, Tuple[int, int], int]:
    """Enumera tutti gli assegnamenti di p_i, q_j (w_ij = p_i q_j) e ritorna
    (energia minima, (p, q) ottimi, numero di minimizzatori a quell'energia).

    Serve per validare la formulazione su istanze piccole: l'energia minima deve
    essere 0 e decodificare nei fattori veri.
    """
    p_free, q_free = _free_idx(fq)
    best_e = float("inf")
    best_pq: Tuple[int, int] = (0, 0)
    count = 0
    for pbits in product((0, 1), repeat=len(p_free)):
        for qbits in product((0, 1), repeat=len(q_free)):
            sample: Dict[Var, int] = {f"p{i}": b for i, b in zip(p_free, pbits)}
            sample.update({f"q{j}": b for j, b in zip(q_free, qbits)})
            for i in p_free:
                for j in q_free:
                    sample[f"w{i}_{j}"] = sample[f"p{i}"] & sample[f"q{j}"]
            e = qubo_energy(fq, sample)
            if e < best_e - 1e-9:
                best_e, best_pq, count = e, decode_factors(fq, sample), 1
            elif abs(e - best_e) <= 1e-9:
                count += 1
    return best_e, best_pq, count


def load_semiprimes(path: str) -> List[Tuple[int, int, int]]:
    """Legge Numbers_test.txt. Ritorna lista di (N, p, q).

    Formato di riga: "<bits> bits <N> = <p> × <q>" (il separatore può essere
    '×' o 'x'/'*'); righe vuote o di commento (#) ignorate.
    """
    out: List[Tuple[int, int, int]] = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            lhs, rhs = line.split("=", 1)
            N = int("".join(ch for ch in lhs if ch.isdigit() or ch == " ").split()[-1])
            for sep in ("×", "x", "X", "*"):
                if sep in rhs:
                    a, b = rhs.split(sep, 1)
                    out.append((N, int(a.strip()), int("".join(
                        ch for ch in b if ch.isdigit()))))
                    break
    return out


if __name__ == "__main__":
    # Smoke test: costruisce e valida il QUBO per 899 = 29 × 31.
    print("— default (solo LSB fissato): ammette i triviali (N,1)/(1,N) —")
    fq = build_factorization_qubo(899)
    print(f"N=899  p_bits={fq.p_bits} q_bits={fq.q_bits}  "
          f"num_vars={fq.num_vars}  penalty={fq.penalty:g}")
    s = sample_from_pq(fq, 29, 31)
    print(f"energia soluzione vera (29,31): {qubo_energy(fq, s):.6g}  (atteso 0)")
    e, pq, n = brute_force_minimum(fq)
    print(f"minimo brute force: energia={e:.6g}  (p,q)={pq}  #minimizzatori={n}")

    # Con fix_msb e dimensioni esatte (29 e 31 sono entrambi a 5 bit) i triviali
    # spariscono: resta solo la fattorizzazione non banale (e il suo scambio).
    print("\n— fix_msb con dimensioni esatte (5,5): niente triviali —")
    fq2 = build_factorization_qubo(899, p_bits=5, q_bits=5, fix_msb=True)
    print(f"num_vars={fq2.num_vars}  base_p={fq2.base_p} base_q={fq2.base_q}")
    e2, pq2, n2 = brute_force_minimum(fq2)
    print(f"minimo brute force: energia={e2:.6g}  (p,q)={pq2}  #minimizzatori={n2}")
