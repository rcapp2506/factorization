### ========================================================= ###
### Función: loss_func_estimator
### ========================================================= ###
###
### Evalúa la función de pérdida (loss) para un conjunto de parámetros
### de un ansatz cuántico, usando tensores precomputados y counts
### obtenidos vía simulador o QPUs (CUNQA).
###
### ========================================================= ###

from src.tensor_exp_value import (
    build_probability_tensor,
    run_with_probabilities,
    select_nodes_from_aux,
    combine_counts_shots,
    combine_counts_circuits
)

from src.auxiliar import kron_delta, alt_extract_primes, extract_factors

import numpy as np
import networkx as nx

# NOTE (fork de-HPC): gli import di CUNQA sono resi "lazy".
# CUNQA è il simulatore distribuito HPC del CESGA e serve solo nelle
# modalità "Shots"/"Circuits". La modalità "Simulation" (qiskit-aer)
# gira su qualsiasi macchina senza CUNQA.

from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit_aer import AerSimulator





def loss_func_estimator(
    x,
    ansatz,
    sim,
    N_int,
    num_a_bits,
    num_b_bits,
    list_size,
    num_qubits,
    d_t,
    n_shots,
    experiment_result,
    CUNQA: str,
    family_name = None,
    variant: str = "serial"
):
    """
    Evalúa la función de pérdida usando un tensor de signos precomputado (d_t)
    y calculando el tensor de probabilidades a partir de counts simulados
    o paralelizados en QPUs.

    Parámetros
    ----------
    x : array-like
        Vector de parámetros del ansatz.
    ansatz : list[QuantumCircuit]
        Lista de circuitos paramétricos para cada base (X, Y, Z).
    sim : Backend
        Simulador cuántico local.
    N_int : np.array
        Número a factorizar.
    n_p_bits : int()
        Número estimado de bits del primo p.
    n_q_bits : int()
        Número estimado de bits del primo q.
    list_size : int
        Número de cadenas de Pauli de cada tipo.
    num_qubits : int
        Número total de qubits/nodos.
    d_t : np.ndarray
        Tensor de signos transpuesto (precomputado).
    n_shots : int
        Número de disparos para la simulación/ejecución en QPUs.
    experiment_result : list
        Lista para almacenar resultados intermedios de cada evaluación.
    CUNQA : str
        Modo de paralelización o None:
        - "Shots" → paralelización por cantidad de shots.
        - "Circuits" → paralelización por circuitos.
        - "Simulation" → simulación local.
    
    family_name : str, optional
        Nombre de la familia de QPUs levantadas con qraise (por defecto None).

    Retorna
    -------
    float
        Valor escalar de la función de pérdida.
    """

    ### ----------------------------------------------------- ###
    ### 1. Bind de parámetros en cada circuito
    ### ----------------------------------------------------- ###

    # NOTE (fork): in modalità "Numpy" il bind non serve (il simulatore numpy
    # prende direttamente x) e da solo costa 0.2-1.7 ms per valutazione.
    if CUNQA != "Numpy":
        bound_circuit_list = [
            qc.assign_parameters({param: val for param, val in zip(qc.parameters, x)})
            for qc in ansatz
        ]
    
    ### ----------------------------------------------------- ###
    ### 2. Obtener counts y construir tensor de probabilidades
    ### ----------------------------------------------------- ###
    if CUNQA in ["Shots", "Circuits"]:

        # Import lazy di CUNQA: richiesto solo per i percorsi distribuiti (HPC CESGA).
        try:
            from cunqa.qpu import get_QPUs, run
            from cunqa.qjob import gather
        except ImportError as e:
            raise RuntimeError(
                "Le modalità CUNQA='Shots'/'Circuits' richiedono il simulatore "
                "distribuito CUNQA (HPC CESGA). Per l'esecuzione locale usa "
                "CUNQA='Simulation' (qiskit-aer)."
            ) from e
        QPUs = get_QPUs(family_name=family_name) if family_name else get_QPUs()

        if CUNQA == "Shots":
            if not QPUs:
                raise RuntimeError("No se encontraron QPUs disponibles")
            shots_per_qpu = [n_shots // len(QPUs)] * len(QPUs)
            shots_per_qpu[0] += n_shots % len(QPUs)
           

            qjobs = [
                run(qc, qpu, shots=shots, method="statevector")
                for qpu, shots in zip(QPUs, shots_per_qpu)
                for qc in bound_circuit_list
            ]
            results = gather(qjobs)
            counts_list = combine_counts_shots(
                results, n_qubits=num_qubits, n_circuits=3, num_qpus=len(QPUs)
            )[1]

        else:  # Circuits
            qjobs = [run(qc, qpu, shots=n_shots, method="statevector") for qc, qpu in zip(bound_circuit_list, QPUs)]
            results = gather(qjobs)

            counts_list = combine_counts_circuits(results, n_qubits=num_qubits)[1]

    elif CUNQA == "Numpy":
        # NOTE (fork): simulatore statevector numpy (src/fast_sim.py), stesse
        # probabilità esatte della modalità "Simulation" (Aer) senza l'overhead
        # di 3 run Aer per valutazione. Verificato in tests/check_numpy_backend.py.
        from src.fast_sim import get_sim
        counts_list = get_sim(ansatz[0]).probabilities(x)
        n_shots = 1

    elif CUNQA == "Simulation":

        def prepare_for_statevector(qc):
            """Elimina medidas y registros clásicos para poder obtener statevector"""
            qc_clean = qc.remove_final_measurements(inplace=False)
            for creg in qc_clean.cregs:
                qc_clean.remove_register(creg)
            # Guardamos el statevector
            qc_clean.save_statevector()
            return qc_clean

        # Preparamos los circuitos     
        # Ejecutar Z completo
        bound_circuit_z = prepare_for_statevector(bound_circuit_list[0])
        state_z = sim.run(bound_circuit_z).result().get_statevector(bound_circuit_z)

        # Probabilidades exactas
        probs_z = np.abs(state_z) ** 2

        # Ejecutar X e Y sobre statevector de Z

        # Ejecutar sobre initial_statevector=state_z
        # X: aplicar H
        qc_x = QuantumCircuit(num_qubits)
        qc_x.set_statevector(state_z)
        for q in range(num_qubits):
            qc_x.h(q)
        qc_x.save_statevector()
        state_x = sim.run(qc_x).result().get_statevector(qc_x)

        # Y: aplicar S† H
        qc_y = QuantumCircuit(num_qubits)
        qc_y.set_statevector(state_z)
        for q in range(num_qubits):
            qc_y.sdg(q)
            qc_y.h(q)
        qc_y.save_statevector()
        state_y = sim.run(qc_y).result().get_statevector(qc_y)


        # Probabilidades exactas
        probs_x = np.abs(state_x) ** 2
        probs_y = np.abs(state_y) ** 2

        # Lista en formato compatible con build_probability_tensor
        counts_list = [probs_x, probs_y, probs_z]

        # Para probabilities exactas, n_shots debe ser 1
        n_shots = 1


    p_t = build_probability_tensor(counts_list, n_shots, num_qubits)

    

    ### ----------------------------------------------------- ###
    ### 3. Calcular valores esperados usando d_t
    ### ----------------------------------------------------- ###
    aux = run_with_probabilities(d_t, p_t)["reshaped_result"]

    ### ----------------------------------------------------- ###
    ### 4. Selección de nodos de interés
    ### ----------------------------------------------------- ###
    m = num_a_bits + num_b_bits


    node_exp_map = select_nodes_from_aux(aux, m, list_size, return_concatenated=True)
    values = np.array(list(node_exp_map.values()))

    #print(values)

    
    #print(f"node_exp_map: {node_exp_map}")
    #print(f"values: {values}")

    a_bits, b_bits, a_int, b_int = alt_extract_primes(values, num_a_bits, num_b_bits)

    D = (a_int - b_int)*(a_int + b_int)

    
    p_int, q_int, p_bits, q_bits = extract_factors(a_int, b_int, N_int)

    ### ----------------------------------------------------- ###
    ### 5. Cálculo de la función de pérdida
    ### ----------------------------------------------------- ###
    prefactor_delta = N_int

    # NOTE (fork): p = gcd(a-b, N) e q = N // p, quindi p*q == N SEMPRE. Il
    # termine originale popcount(N XOR p*q)**2 era identicamente 0 e le quattro
    # delta scattavano a coppie (p == 1 <=> q == N, p == N <=> q == 1): loss_pq
    # vale 2N per una fattorizzazione banale e 0 altrimenti. Stesso valore
    # numerico dell'espressione originale, scritto per quello che è.
    loss_pq = 2 * prefactor_delta * int(p_int == 1 or p_int == N_int)

    #print(f"loss_pq: {loss_pq}")

    loss_ab = (D % N_int)**2 + prefactor_delta*(kron_delta(a_int, b_int) + kron_delta(D, 0))
    
    #print(f"loss_ab: {loss_ab}")

    # NOTE (fork): due varianti del costo, come nelle due versioni del CESGA.
    #  - "serial"   (FAO_DoTS/PCE): loss_pq + loss_ab + regolarizzazione
    #                100 * mean(<P>^2)^2. Il criterio di arresto usa il costo
    #                senza regolarizzazione (base_loss).
    #  - "parallel" (FAO_DoTS/PCE_parallel, con cui sono stati prodotti quasi
    #                tutti i risultati k=4 pubblicati): niente regolarizzazione,
    #                penalità N se (a - b) = 0 (mod N); costo totale = base_loss.
    if variant == "parallel":
        loss_heur_2 = prefactor_delta * kron_delta((a_int - b_int) % N_int, 0)
        base_loss = loss_pq + loss_ab + loss_heur_2
        total_loss = base_loss
    elif variant == "serial":
        base_loss = loss_pq + loss_ab

        # ---------------------------------------------------------
        # 7. Regularización anti-saturación (tipo MaxCut)
        # ---------------------------------------------------------
        reg_term = np.mean([z**2 for z in values]) ** 2

        nu = 100.0   # o 100.0, o 1000.0
        total_loss = base_loss  + nu * reg_term
    else:
        raise ValueError(f"variante sconosciuta: {variant!r} (serial | parallel)")


    ### ----------------------------------------------------- ###
    ### 7. Guardar resultados intermedios
    ### ----------------------------------------------------- ###
    # NOTE (fork): si conserva solo la valutazione migliore. In origine si
    # accodava un dizionario (con tutta la exp_map) a OGNI valutazione: con
    # maxiter=3000 erano >1M dizionari e diversi GB (OUT_OF_MEMORY su Magellano).
    # exe_fact.py usa solo la valutazione di loss minima e, a parità di loss,
    # l'ultima in ordine di tempo: il confronto "<=" riproduce la stessa scelta.
    if not experiment_result or total_loss <= experiment_result[0]["loss"]:
        experiment_result[:] = [{"loss": total_loss, "base": base_loss, "exp_map": node_exp_map}]

    return total_loss, base_loss

 

