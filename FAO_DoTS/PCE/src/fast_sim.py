### ========================================================= ###
### Módulo: fast_sim   (fork de-HPC)
### ========================================================= ###
###
### Simulatore statevector numpy per l'ansatz DoTS, alternativo a Aer.
###
### L'ansatz di circuit_builder ("Taylor_efficient") contiene SOLO rotazioni
### di Pauli: RX, RY, RZ, RXX, RYY, RZZ, cioè porte exp(-i θ/2 P) con P stringa
### di Pauli su 1 o 2 qubit. Per ognuna
###     ψ' = cos(θ/2) ψ - i sin(θ/2) Pψ ,      (Pψ)[j] = c[j] ψ[f[j]]
### con f permutazione degli indici (flip dei bit per X/Y) e c fasi,
### precalcolate una volta per circuito.
###
### Con 5-7 qubit lo stato ha 32-128 ampiezze: il costo di Aer è quasi tutto
### overhead per chiamata (3 run per valutazione). Qui una valutazione costa
### ~1-3 ms su un core Magellano contro ~5-19 ms con Aer, con risultati
### identici (aspettazioni uguali a ~1e-15; vedi tests/check_numpy_backend.py).
###
### Basi di misura identiche a loss_functions.py (modalità "Simulation"):
###     X: H⊗n ψ        Y: H⊗n (S†)⊗n ψ        Z: ψ
### Convenzione qiskit little-endian: il qubit q è il bit q dell'indice.
###
### ========================================================= ###

import numpy as np

_PAULI = {"rx": "X", "ry": "Y", "rz": "Z", "rxx": "XX", "ryy": "YY", "rzz": "ZZ"}


class PauliRotationSim:
    """Simulatore di un circuito parametrico composto solo da rotazioni di Pauli.

    I parametri si legano nello stesso ordine usato da loss_functions
    (`qc.parameters`), quindi `probabilities(x)` accetta lo stesso vettore x.
    """

    def __init__(self, qc):
        n = qc.num_qubits
        self.n = n
        dim = 2 ** n
        idx = np.arange(dim)
        pos = {p: i for i, p in enumerate(qc.parameters)}
        self.num_parameters = len(pos)
        self.gates = []          # (indice parametro, permutazione f, fasi c)
        for inst in qc.data:
            name = inst.operation.name
            if name in ("measure", "barrier"):
                continue
            if name not in _PAULI:
                raise ValueError(
                    f"porta '{name}' non supportata dal backend numpy "
                    f"(supportate: {sorted(_PAULI)})")
            qs = [qc.find_bit(q).index for q in inst.qubits]
            f = idx.copy()
            ph = np.ones(dim, dtype=complex)
            for pl, q in zip(_PAULI[name], qs):
                b = (idx >> q) & 1
                if pl in "XY":
                    f = f ^ (1 << q)
                if pl == "Z":
                    ph = ph * np.where(b == 0, 1, -1)
                if pl == "Y":                      # Y|0> = i|1>,  Y|1> = -i|0>
                    ph = ph * np.where(b == 0, 1j, -1j)
            # P|i> = ph[i] |f[i]>  =>  (Pψ)[j] = ph[f[j]] ψ[f[j]]   (f involuzione)
            self.gates.append((pos[inst.operation.params[0]], f, ph[f]))

        H = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
        Hn = np.array([[1.0]])
        for _ in range(n):
            Hn = np.kron(Hn, H)                    # H⊗n è simmetrica: l'ordine non conta
        self.Hn = Hn.astype(complex)
        popcount = np.array([bin(i).count("1") for i in idx])
        self.sdg = (-1j) ** popcount               # (S†)⊗n, diagonale

    def statevector(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (self.num_parameters,):
            raise ValueError(f"attesi {self.num_parameters} parametri, ricevuti {x.shape}")
        h = 0.5 * x
        cs, sn = np.cos(h), np.sin(h)
        psi = np.zeros(2 ** self.n, dtype=complex)
        psi[0] = 1.0
        for k, f, c in self.gates:
            psi = cs[k] * psi - (1j * sn[k]) * (c * psi[f])
        return psi

    def probabilities(self, x):
        """[p_X, p_Y, p_Z]: distribuzioni nelle basi X, Y, Z (stesso formato di
        counts_list in loss_functions, modalità Simulation)."""
        psi = self.statevector(x)
        px = np.abs(self.Hn @ psi) ** 2
        py = np.abs(self.Hn @ (self.sdg * psi)) ** 2
        pz = np.abs(psi) ** 2
        return [px, py, pz]


_CACHE = {}


def get_sim(qc):
    """Simulatore per il circuito qc, costruito una volta e riusato."""
    key = id(qc)
    hit = _CACHE.get(key)
    if hit is None or hit[0] is not qc:
        hit = (qc, PauliRotationSim(qc))   # si tiene qc per evitare il riuso dell'id
        _CACHE[key] = hit
    return hit[1]
