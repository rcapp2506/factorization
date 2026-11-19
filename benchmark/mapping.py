"""
benchmark/mapping.py — costruzione solver-agnostica dell'istanza di
fattorizzazione di un semiprimo come QUBO / Ising.

Obiettivo: una *stessa* istanza (lo stesso QUBO) alimenta i tre solver del
benchmark a tre vie (PCE/DoTS, OpenJij SQA/A-PSA, SQOA). Questo modulo non
dipende da alcun solver: produce il QUBO come dizionario Python puro (coefficienti
interi, energia esatta) e offre conversioni opzionali (Ising, dimod BQM).

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

che vale 0 se e solo se w_ij = p_i q_j, e >= lambda altrimenti. Poiché anche
l'obiettivo è un quadrato (>= 0), QUALSIASI lambda > 0 lascia invariati gli stati
fondamentali: energia 0 esattamente sulle fattorizzazioni rappresentabili con
w_ij coerenti. Il default è quindi lambda = 1; valori diversi cambiano solo il
paesaggio degli stati eccitati (parametro da studiare con i solver euristici).

Convenzione di p e q: p è il fattore maggiore, q il minore (q <= sqrt(N)).
Bit di default: p ha bits(N) - 1 bit (p <= N/3), q ne ha ceil(bits(N)/2). Così
le fattorizzazioni triviali (N, 1) e (1, N) non sono rappresentabili.

LIMITI NUMERICI (importante per il benchmark). L'obiettivo (N - p*q)^2 produce
coefficienti fino a ~2^(2(np+nq)), indipendentemente da lambda: oltre ~20 bit di
N superano 2^53 e non sono rappresentabili esattamente in float64, il formato in
cui lavorano OpenJij, dimod e SQOA; il QUBO è inoltre denso al 100%. Questa
formulazione va quindi usata per istanze piccole e per validazione; per N grandi
serve la formulazione a colonne con variabili di carry (coefficienti limitati,
QUBO sparso). `build_factorization_qubo` emette un warning quando si supera 2^53.

NOTA sul benchmark. Il ramo PCE/DoTS di questo repo (FAO_DoTS) NON minimizza
questo QUBO: usa una cost function diretta e non polinomiale (congruenza di
Fermat a^2 = b^2 mod N, gcd, popcount di N XOR p*q). Per un confronto a parità di
funzione costo si può far valutare a PCE l'energia di QUESTO QUBO
(`qubo_energy`) invece della loss DoTS.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from itertools import product
from math import ceil
from typing import Dict, List, Optional, Tuple

# Una variabile QUBO è una stringa ("p1", "q2", "w1_3").
Var = str
# QUBO: termini lineari {(v,): coef} e quadratici {(u, v): coef}, più offset.
QUBO = Dict[Tuple[Var, ...], int]

FLOAT64_EXACT_LIMIT = 2 ** 53


@dataclass
class FactorizationQUBO:
    """Istanza di fattorizzazione come QUBO, con i metadati per decodificare."""
    N: int
    p_bits: int
    q_bits: int
    penalty: int
    Q: QUBO = field(default_factory=dict)
    offset: int = 0
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

    @property
    def max_abs_coef(self) -> int:
        return max((abs(c) for c in self.Q.values()), default=0)

    @property
    def float64_exact(self) -> bool:
        """True se tutti i coefficienti (e l'offset) sono esatti in float64."""
        return max(self.max_abs_coef, abs(self.offset)) <= FLOAT64_EXACT_LIMIT


# --------------------------------------------------------------------------- #
# Costruzione del QUBO
# --------------------------------------------------------------------------- #
def _add(Q: QUBO, key: Tuple[Var, ...], coef: int) -> None:
    """Accumula un coefficiente su un termine del QUBO (chiave ordinata)."""
    if len(key) == 2:
        key = tuple(sorted(key))  # type: ignore[assignment]
    Q[key] = Q.get(key, 0) + coef


def default_bit_sizes(N: int) -> Tuple[int, int]:
    """Numero di bit (upper bound) per p (maggiore) e q (minore) di N = p*q.

    p <= N/3 < 2^(bits(N)-1)  ->  bits(N) - 1 bit;  q <= sqrt(N) -> ceil(bits(N)/2).
    Con questi limiti né (N, 1) né (1, N) sono rappresentabili. Entrambi >= 2.
    """
    nbits = N.bit_length()
    return max(2, nbits - 1), max(2, ceil(nbits / 2))


def build_factorization_qubo(
    N: int,
    p_bits: Optional[int] = None,
    q_bits: Optional[int] = None,
    penalty: int = 1,
    fix_msb: bool = False,
) -> FactorizationQUBO:
    """Costruisce il QUBO di (N - p*q)^2 con variabili prodotto AND.

    Parametri
    ---------
    N : int
        Semiprimo da fattorizzare (dispari).
    p_bits, q_bits : int, opzionali
        Numero di bit di p (fattore maggiore) e q (minore), incluso il bit LSB
        fissato a 1; ciascuno >= 2. Se None si usano `default_bit_sizes(N)`.
    penalty : int
        Peso lambda > 0 della penalità di Rosenberg. Ogni lambda > 0 preserva gli
        stati fondamentali (vedi docstring del modulo); default 1.
    fix_msb : bool
        Se True fissa anche il bit più significativo di p e q a 1, riducendo le
        variabili. Richiede p_bits/q_bits pari alla lunghezza esatta in bit dei
        fattori veri: con dimensioni sbagliate l'istanza è insoddisfacibile.

    Ritorna
    -------
    FactorizationQUBO
    """
    if N % 2 == 0:
        raise ValueError("N deve essere dispari (semiprimo di due primi dispari).")
    dp, dq = default_bit_sizes(N)
    p_bits = dp if p_bits is None else p_bits
    q_bits = dq if q_bits is None else q_bits
    if p_bits < 2 or q_bits < 2:
        raise ValueError("p_bits e q_bits devono essere >= 2 (LSB fissato + almeno un bit).")
    if penalty <= 0:
        raise ValueError("penalty deve essere > 0.")

    # Bit liberi: LSB (bit 0) sempre fissato a 1; MSB fissato a 1 solo se fix_msb.
    hi = 1 if fix_msb else 0
    p_idx = list(range(1, p_bits - hi))
    q_idx = list(range(1, q_bits - hi))
    base_p = 1 + ((1 << (p_bits - 1)) if fix_msb else 0)
    base_q = 1 + ((1 << (q_bits - 1)) if fix_msb else 0)

    p_vars = [f"p{i}" for i in p_idx]
    q_vars = [f"q{j}" for j in q_idx]
    w_vars = [f"w{i}_{j}" for i in p_idx for j in q_idx]

    # p*q = base_p base_q + base_p sum 2^j q_j + base_q sum 2^i p_i + sum 2^(i+j) w_ij
    # diff = N - p*q = c0 + sum_k c_k v_k , con c0 = N - base_p base_q.
    lin: Dict[Var, int] = {}
    for i in p_idx:
        lin[f"p{i}"] = -base_q * (1 << i)
    for j in q_idx:
        lin[f"q{j}"] = -base_p * (1 << j)
    for i in p_idx:
        for j in q_idx:
            lin[f"w{i}_{j}"] = -(1 << (i + j))
    c0 = N - base_p * base_q

    Q: QUBO = {}
    # (c0 + sum c_k v_k)^2 con v_k^2 = v_k:
    #   lineare 2 c0 c_k + c_k^2 ; quadratico 2 c_k c_l ; offset c0^2  (tutto intero)
    vars_all = list(lin.keys())
    for v in vars_all:
        ck = lin[v]
        _add(Q, (v,), 2 * c0 * ck + ck * ck)
    for a in range(len(vars_all)):
        for b in range(a + 1, len(vars_all)):
            va, vb = vars_all[a], vars_all[b]
            _add(Q, (va, vb), 2 * lin[va] * lin[vb])
    offset = c0 * c0

    # Rosenberg: lambda (p_i q_j - 2 p_i w_ij - 2 q_j w_ij + 3 w_ij)
    lam = int(penalty)
    for i in p_idx:
        for j in q_idx:
            pi, qj, wij = f"p{i}", f"q{j}", f"w{i}_{j}"
            _add(Q, (pi, qj), lam)
            _add(Q, (pi, wij), -2 * lam)
            _add(Q, (qj, wij), -2 * lam)
            _add(Q, (wij,), 3 * lam)

    fq = FactorizationQUBO(
        N=N, p_bits=p_bits, q_bits=q_bits, penalty=lam,
        Q=Q, offset=offset, p_vars=p_vars, q_vars=q_vars, w_vars=w_vars,
        base_p=base_p, base_q=base_q, fix_msb=fix_msb,
    )
    if not fq.float64_exact:
        warnings.warn(
            f"N={N}: coefficienti fino a {fq.max_abs_coef:.3g} > 2^53, non esatti in "
            "float64 (OpenJij/dimod/SQOA). Usare la formulazione a colonne con carry.",
            RuntimeWarning, stacklevel=2,
        )
    return fq


# --------------------------------------------------------------------------- #
# Valutazione / decodifica
# --------------------------------------------------------------------------- #
def qubo_energy(fq: FactorizationQUBO, sample: Dict[Var, int]) -> int:
    """Energia QUBO esatta (intera) di un assegnamento {var: 0/1}, offset incluso."""
    e = fq.offset
    for key, coef in fq.Q.items():
        if len(key) == 1:
            e += coef * int(sample[key[0]])
        else:
            e += coef * int(sample[key[0]]) * int(sample[key[1]])
    return e


def _free_idx(fq: FactorizationQUBO) -> Tuple[List[int], List[int]]:
    """Indici dei bit liberi di p e q (dedotti dai nomi delle variabili)."""
    return [int(v[1:]) for v in fq.p_vars], [int(v[1:]) for v in fq.q_vars]


def decode_factors(fq: FactorizationQUBO, sample: Dict[Var, int]) -> Tuple[int, int]:
    """Da un assegnamento dei bit a (p, q). I w_ij non servono alla decodifica."""
    pi, qj = _free_idx(fq)
    p = fq.base_p + sum((1 << i) * int(sample.get(f"p{i}", 0)) for i in pi)
    q = fq.base_q + sum((1 << j) * int(sample.get(f"q{j}", 0)) for j in qj)
    return p, q


def sample_from_pq(fq: FactorizationQUBO, p: int, q: int) -> Dict[Var, int]:
    """Assegnamento completo (p_i, q_j, w_ij coerenti) a partire da p, q interi.

    Utile per i test: l'energia della soluzione vera deve essere 0.
    Solleva ValueError se p o q non sono rappresentabili con questi bit.
    """
    pi, qj = _free_idx(fq)
    sample: Dict[Var, int] = {}
    for i in pi:
        sample[f"p{i}"] = (p >> i) & 1
    for j in qj:
        sample[f"q{j}"] = (q >> j) & 1
    for i in pi:
        for j in qj:
            sample[f"w{i}_{j}"] = sample[f"p{i}"] & sample[f"q{j}"]
    if decode_factors(fq, sample) != (p, q):
        raise ValueError(
            f"(p, q) = ({p}, {q}) non rappresentabile con p_bits={fq.p_bits}, "
            f"q_bits={fq.q_bits}, fix_msb={fq.fix_msb} (convenzione: p maggiore, q minore)."
        )
    return sample


# --------------------------------------------------------------------------- #
# Conversione in Ising e in dimod (opzionale)
# --------------------------------------------------------------------------- #
def to_ising(fq: FactorizationQUBO, convention: str = "openjij"):
    """Converte il QUBO in Ising (h, J, offset).

    convention:
      "openjij" (default)  x = (1 + s)/2 : convenzione di OpenJij e dimod
                           (s = +1 <-> bit 1). Usare questa per passare h, J ai solver.
      "t4"                 x = (1 - z)/2 : convenzione delle note T4 (z = +1 <-> bit 0).

    L'energia Ising coincide con quella QUBO sullo stato corrispondente.
    """
    if convention == "openjij":
        sg = 1.0
    elif convention == "t4":
        sg = -1.0
    else:
        raise ValueError("convention deve essere 'openjij' o 't4'.")
    h: Dict[Var, float] = {}
    J: Dict[Tuple[Var, Var], float] = {}
    offset = float(fq.offset)
    for key, coef in fq.Q.items():
        if len(key) == 1:
            v = key[0]
            # coef * x = coef/2 (1 + sg s)
            h[v] = h.get(v, 0.0) + sg * coef / 2.0
            offset += coef / 2.0
        else:
            u, v = key
            # coef * x_u x_v = coef/4 (1 + sg s_u)(1 + sg s_v)
            J[(u, v)] = J.get((u, v), 0.0) + coef / 4.0
            h[u] = h.get(u, 0.0) + sg * coef / 4.0
            h[v] = h.get(v, 0.0) + sg * coef / 4.0
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
def brute_force_minimum(fq: FactorizationQUBO) -> Tuple[int, Tuple[int, int], int]:
    """Minimo sul sottospazio con w_ij = p_i q_j (enumera solo p_i, q_j).

    Ritorna (energia minima, (p, q) del primo minimizzatore, numero di
    minimizzatori). Se esiste una fattorizzazione rappresentabile il minimo è 0
    e coincide con il minimo globale del QUBO (ogni violazione costa >= lambda).
    Solo per istanze piccole: costo 2^(#p_i + #q_j).
    """
    p_free, q_free = _free_idx(fq)
    best_e: Optional[int] = None
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
            if best_e is None or e < best_e:
                best_e, best_pq, count = e, decode_factors(fq, sample), 1
            elif e == best_e:
                count += 1
    assert best_e is not None
    return best_e, best_pq, count


_ROW = re.compile(r"(\d+)\s*=\s*(\d+)\s*[×xX*]\s*(\d+)")


def load_semiprimes(path: str) -> List[Tuple[int, int, int]]:
    """Legge Numbers_test.txt. Ritorna lista di (N, p, q) con p >= q.

    Formato di riga: "<bits> bits <N> = <a> × <b>" (separatore '×', 'x' o '*');
    righe vuote o di commento (#) ignorate, testo dopo il secondo fattore ignorato.
    """
    out: List[Tuple[int, int, int]] = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = _ROW.search(line)
            if not m:
                continue
            N, a, b = (int(g) for g in m.groups())
            if a * b != N:
                raise ValueError(f"riga incoerente ({a} × {b} != {N}): {line!r}")
            out.append((N, max(a, b), min(a, b)))
    return out


if __name__ == "__main__":
    # Smoke test: costruisce e valida il QUBO per 899 = 31 × 29.
    fq = build_factorization_qubo(899)
    print(f"N=899  p_bits={fq.p_bits} q_bits={fq.q_bits}  num_vars={fq.num_vars}  "
          f"penalty={fq.penalty}  max|coef|={fq.max_abs_coef:.3g}")
    print(f"energia soluzione vera (31,29): {qubo_energy(fq, sample_from_pq(fq, 31, 29))}  (atteso 0)")
    e, pq, n = brute_force_minimum(fq)
    print(f"minimo brute force: energia={e}  (p,q)={pq}  #minimizzatori={n}  (niente triviali)")
