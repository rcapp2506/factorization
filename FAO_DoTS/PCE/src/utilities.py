### ========================================================= ###
### Módulo: utilities
### ========================================================= ###
###
### Este módulo proporciona funciones para la optimización variacional
### (VQE) de ansatz cuánticos y manejo de resultados de evaluaciones.
###
### ========================================================= ###

from scipy.optimize import minimize, differential_evolution, brute, OptimizeResult


from src import pso_simple
from src import qpso_simple

import numpy as np
import csv
import os
from functools import partial


# =========================================================
# NOTE (fork): valutazione in parallelo per la variante "parallel"
# =========================================================
# Ogni processo del pool riceve una volta sola (initializer) tutto ciò che
# serve a valutare il costo e costruisce il simulatore una volta sola; poi
# valuta solo vettori x. Con updating="deferred" il risultato di DE non dipende
# dal numero di processi: con lo stesso seed workers=1 e workers=W danno lo
# stesso run.
_WORKER_ARGS = {}


def _worker_init(args):
    _WORKER_ARGS.clear()
    _WORKER_ARGS.update(args)


def _worker_eval(x):
    from src.loss_functions import loss_func_estimator
    total, _ = loss_func_estimator(x=x, experiment_result=[], **_WORKER_ARGS)
    return total


def _worker_batch(X):
    """Costo di un blocco di individui (righe di X), simulati in blocco."""
    from src.loss_functions import loss_func_batch
    return loss_func_batch(X, **_WORKER_ARGS)


PARALLEL_VARIANTS = ("parallel", "parallel_l10")


def run_vqe_optimization(
    sim,
    n_shots,
    compiled_circuit,
    N_int,
    num_a_bits,
    num_b_bits,
    list_size,
    d_t,
    optimizer,
    optimizer_params=None,
    loss_func_estimator=None,
    maxiter=1000,
    log_csv_path=None,
    cunqa_str="None",
    family_name=None,
    variant="serial",
    workers=1
):
    """
    Ejecuta una optimización variacional (VQE) sobre un ansatz cuántico dado.

    variant: "serial" (FAO_DoTS/PCE) o "parallel" (FAO_DoTS/PCE_parallel):
    costo, aggiornamento di DE, criterio di arresto e configurazione finale
    come nelle due versioni del CESGA. workers: processi per valutare la
    popolazione di DE (solo variante "parallel").

    Retorna:
    --------
    result : OptimizeResult
        Resultado del optimizador.
    experiment_result : list
        Lista con un único elemento: la evaluación de pérdida mínima
        ({"loss", "exp_map"}; en caso de empate, la más reciente).
    """

    import os as _os

    class _EvalCounter:
        """Sostituto di cost_history quando il log per valutazione è disattivato:
        conta le valutazioni (len) senza conservarne i valori."""
        def __init__(self):
            self.n = 0
        def append(self, _):
            self.n += 1
        def __len__(self):
            return self.n
        def __iter__(self):
            return iter(())

    # NOTE (fork): il CSV con una riga per OGNI valutazione (k=4: ~33M righe,
    # ~1 GB per run) si scrive solo con PCE_LOG_EVALS=1. Il CSV per
    # iterazione/generazione (_iter.csv) si scrive sempre.
    log_evals = _os.environ.get("PCE_LOG_EVALS", "0") == "1"

    experiment_result = []
    cost_history = [] if log_evals else _EvalCounter()
    iteration_costs = []

    # =========================================================
    # CALLBACKS
    # =========================================================

    def callback(xk):
        value = loss_func(xk)
        iteration_costs.append(value)

    def callback_de(intermediate_result):
        iteration_costs.append(intermediate_result.fun)
        return None
    
    def callback_de_indicator(intermediate_result):
        iteration_costs.append(intermediate_result.fun)

        # NOTE (fork): variante "parallel": il costo non ha regolarizzazione,
        # quindi coincide con l'indicatore; si ferma quando il migliore vale 0
        # (come callback_de_fac in PCE_parallel), senza valutazioni in più.
        if variant in PARALLEL_VARIANTS:
            if intermediate_result.fun == 0.0:
                print("🎯 Se alcanzó exactamente 0.0")
                return True
            return False

        # NOTE (fork): il migliore della popolazione di DE è la valutazione di
        # loss minima vista finora, cioè quella conservata in experiment_result
        # insieme alla sua loss base. La si legge da lì invece di rivalutare il
        # circuito: prima ogni generazione costava una valutazione in più, non
        # contata in nfev. Se le loss non coincidono si ricalcola come prima.
        if experiment_result and experiment_result[0]["loss"] == intermediate_result.fun:
            indicator_value = experiment_result[0]["base"]
        else:
            indicator_value = base_loss_func(intermediate_result.x)

        if indicator_value == 0.0:
            print("🎯 Se alcanzó solución física (indicador = 0.0)")
            return True

        return False

    def callback_spsa(nfev, x, fx, stepsize, accepted):
        cost_history.append(fx)

    # =========================================================
    # LOSS FUNCTION
    # =========================================================

    def make_loss_functions():
        cache = {"last_x": None, "total": None, "base": None}

        def loss_func(x):
            total, base = loss_func_estimator(
                x=x,
                ansatz=compiled_circuit,
                sim=sim,
                N_int=N_int,
                num_a_bits=num_a_bits,
                num_b_bits=num_b_bits,
                list_size=list_size,
                num_qubits=compiled_circuit[0].num_qubits,
                d_t=d_t,
                n_shots=n_shots,
                experiment_result=experiment_result,
                CUNQA=cunqa_str,
                family_name=family_name,
                variant=variant
            )
            cache["last_x"] = tuple(x)
            cache["total"] = total
            cache["base"] = base
            cost_history.append(total)
            return total

        def base_loss_func(x):
            if cache["last_x"] == tuple(x):
                return cache["base"]
            else:
                _, base = loss_func_estimator(
                    x=x,
                    ansatz=compiled_circuit,
                    sim=sim,
                    N_int=N_int,
                    num_a_bits=num_a_bits,
                    num_b_bits=num_b_bits,
                    list_size=list_size,
                    num_qubits=compiled_circuit[0].num_qubits,
                    d_t=d_t,
                    n_shots=n_shots,
                    experiment_result=experiment_result,
                    CUNQA=cunqa_str,
                    family_name=family_name,
                    variant=variant
                )
                return base

        return loss_func, base_loss_func

    loss_func, base_loss_func = make_loss_functions()

    # =========================================================
    # PARAMETROS INICIALES
    # =========================================================

    rng_default = np.random.default_rng(33)
    initial_params = rng_default.random(len(compiled_circuit[0].parameters)) * 2 * np.pi

    # =========================================================
    # OPTIMIZADORES
    # =========================================================

    if isinstance(optimizer, str):

        optimizer_lower = optimizer.lower()

        default_optimizers = {


            "powell": {
                "method": "Powell",
                "options": {
                    "maxiter": maxiter,
                    "maxfev": 50000,
                    "xtol": 1e-9,
                    "ftol": 1e-9,
                    "disp": True,
                    "return_all": True
                },
                "callback": callback
            },

            "cobyla": {
                "method": "COBYLA",
                "options": {
                    "maxiter": maxiter,
                    "rhobeg": 1.0,
                    "tol": 1e-9,
                    "disp": False
                },
                "callback": callback
            },

            "bfgs": {
                "method": "BFGS",
                "options": {
                    "disp": True,
                    "maxiter": maxiter,
                    "gtol": 1e-3
                },
                "callback": callback
            },

            "lbfgsb": {
                "method": "L-BFGS-B",
                "options": {
                    "disp": True,
                    "maxiter": maxiter,
                    "ftol": 1e-6,
                    "gtol": 1e-3,
                    "eps": 1e-3
                },
                "callback": callback
            },

            "tnc": {
                "method": "TNC",
                "options": {
                    "disp": True,
                    "maxiter": maxiter,
                    "ftol": 1e-8,
                    "xtol": 1e-8,
                    "gtol": 1e-4
                },
                "callback": callback
            },

            "slsqp": {
                "method": "SLSQP",
                "options": {
                    "disp": True,
                    "maxiter": maxiter,
                    "ftol": 1e-8,
                    "eps": 1e-6
                },
                "callback": callback
            },

            "differentialevolution": {
                "kwargs": {
                    "strategy": "best1exp",
                    "maxiter": maxiter,
                    "popsize": 10,
                    "tol": 0,
                    "atol": 0,
                    "mutation": (0.5, 1),
                    "recombination": 0.7,
                    "disp": True,
                    "polish": False,
                    "init": "halton"
                }
            },

            "pso": {
                "kwargs": {
                    "num_particles": 100,
                    "maxiter": maxiter,
                    "verbose": True
                }
            },

            "qdpso": {
                "kwargs": {
                    "num_particles": 100,
                    "maxiter": maxiter,
                    "verbose": True,
                    "g": 0.7   # parámetro de dispersión cuántica
                }
            },

            "brute": {}
        }

        if optimizer_lower not in default_optimizers:
            raise ValueError(f"Optimizador desconocido: {optimizer}")

        opt_config = default_optimizers[optimizer_lower]

        # =====================================================
        # Sobrescribir parámetros personalizados
        # =====================================================

        if optimizer_params:
            print(f"⚡ Usando parámetros personalizados para {optimizer.upper()}")

            if optimizer_lower in ["differentialevolution", "pso", "qdpso"]:
                opt_config["kwargs"].update(optimizer_params)
            else:
                opt_config["options"].update(optimizer_params)

        else:
            print(f"⚡ Usando parámetros por defecto para {optimizer.upper()}")

        # =====================================================
        # Ejecutar optimizador
        # =====================================================

        if optimizer_lower == "pso":

            bounds = [(0, 2*np.pi)] * len(initial_params)

            # Ejecutar PSO clásico
            best_value, best_position, iteration_costs = pso_simple.minimize(
                loss_func,                 # función que se optimiza
                base_loss_func,            # función para criterio de parada
                initial_params,
                bounds,
                **opt_config["kwargs"]
            )

            # Asegurar que sea array numpy
            best_position = np.atleast_1d(best_position)

            # Calcular número de evaluaciones
            num_particles = opt_config["kwargs"].get("num_particles", 20)
            maxiter = opt_config["kwargs"].get("maxiter", 1000)
            
            nfev = len(cost_history)
            nit = int(np.ceil(nfev / num_particles))

            # Crear OptimizeResult compatible con SciPy
            result = OptimizeResult(
                x=best_position,
                fun=best_value,
                nfev=nfev,
                nit=nit,
                success=True,
                message="PSO optimization finished."
            )

        elif optimizer_lower == "qdpso":

            bounds = [(0, 2*np.pi)] * len(initial_params)

            # Ejecutar QDPSO
            # NOTE (fork): la storia per iterazione va in iteration_costs (come nel
            # ramo PSO), altrimenti _iter.csv resta con la sola intestazione.
            best_value, best_position, iteration_costs = qpso_simple.minimize_qdpso(
                loss_func=loss_func,          # con regularización
                base_loss_func=base_loss_func,      # sin regularización
                x0=initial_params,
                bounds=bounds,
                **opt_config["kwargs"]
            )

            # Calcular número de evaluaciones
            best_position = np.atleast_1d(best_position)
            num_particles = opt_config["kwargs"].get("num_particles", 20)
            maxiter = opt_config["kwargs"].get("maxiter", 1000)
            
            nfev = len(cost_history)
            nit = int(np.ceil(nfev / num_particles))

            result = OptimizeResult(
                x=best_position,
                fun=best_value,
                nfev=nfev,
                nit=nit,
                success=True,
                message="QDPSO optimization finished."
            )

        elif optimizer_lower == "differentialevolution":

            if variant in PARALLEL_VARIANTS:
                # NOTE (fork): come PCE_parallel: aggiornamento "deferred" della
                # popolazione, valutazioni distribuite su `workers` processi.
                de_kwargs = dict(opt_config["kwargs"], updating="deferred")
                worker_args = dict(
                    ansatz=compiled_circuit, sim=sim, N_int=N_int,
                    num_a_bits=num_a_bits, num_b_bits=num_b_bits,
                    list_size=list_size,
                    num_qubits=compiled_circuit[0].num_qubits, d_t=d_t,
                    n_shots=n_shots, CUNQA=cunqa_str, family_name=family_name,
                    variant=variant)
                if cunqa_str == "Numpy":
                    # NOTE (fork): DE "vectorized": a ogni generazione la
                    # popolazione intera arriva in un'unica chiamata, è divisa in
                    # `workers` blocchi e ogni blocco è simulato in blocco
                    # (loss_func_batch). Stessi valori della valutazione
                    # individuale, quindi stesso run.
                    import contextlib
                    import multiprocessing as mp
                    pool_cm = (mp.get_context("fork").Pool(
                                   workers, initializer=_worker_init,
                                   initargs=(worker_args,))
                               if workers > 1 else contextlib.nullcontext())
                    with pool_cm as pool:
                        if pool is None:
                            _worker_init(worker_args)

                        n_eval = [0]

                        def func_vec(X):
                            XT = np.asarray(X).T
                            n_eval[0] += XT.shape[0]
                            if pool is None:
                                return _worker_batch(XT)
                            parts = np.array_split(XT, workers)
                            return np.concatenate(pool.map(_worker_batch, parts))

                        result = differential_evolution(
                            func_vec,
                            bounds=[(0, 2*np.pi)] * len(initial_params),
                            callback=callback_de_indicator,
                            vectorized=True,
                            **de_kwargs
                        )
                    # in modalità vectorized scipy conta le chiamate, non gli
                    # individui valutati: si riporta il numero di valutazioni
                    result.nfev = n_eval[0]
                elif workers > 1:
                    import multiprocessing as mp
                    with mp.get_context("fork").Pool(
                            workers, initializer=_worker_init,
                            initargs=(worker_args,)) as pool:
                        result = differential_evolution(
                            _worker_eval,
                            bounds=[(0, 2*np.pi)] * len(initial_params),
                            callback=callback_de_indicator,
                            workers=pool.map,
                            **de_kwargs
                        )
                else:
                    result = differential_evolution(
                        loss_func,
                        bounds=[(0, 2*np.pi)] * len(initial_params),
                        callback=callback_de_indicator,
                        workers=1,
                        **de_kwargs
                    )

                # Come PCE_parallel: la configurazione decodificata a fine run
                # è il migliore di DE (result.x), rivalutato qui.
                experiment_result.clear()
                loss_func_estimator(
                    x=result.x, experiment_result=experiment_result,
                    **worker_args)
            else:
                result = differential_evolution(
                    loss_func,
                    bounds=[(0, 2*np.pi)] * len(initial_params),
                    callback=callback_de_indicator,
                    **opt_config["kwargs"]
                )

        elif optimizer_lower == "brute":

            result = brute(
                loss_func,
                ranges=[(0, 2*np.pi)] * len(initial_params)
            )

        else:

            result = minimize(
                loss_func,
                initial_params,
                method=opt_config["method"],
                options=opt_config["options"],
                callback=opt_config["callback"]
            )

    else:
        result = optimizer.minimize(fun=loss_func, x0=initial_params)

    # =========================================================
    # Guardar CSV si se solicita
    # =========================================================

    import fcntl
    import os
    import csv

    if log_csv_path:
        # Asegurarse de que exista la carpeta
        os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)

    # NOTE (fork): senza log per valutazione, un CSV per valutazione rimasto da
    # un run precedente nella stessa cartella apparterrebbe a un altro run
    # (mentre _iter.csv viene sovrascritto): lo si elimina.
    if log_csv_path and not log_evals and os.path.exists(log_csv_path):
        os.remove(log_csv_path)

    if log_csv_path and log_evals:
        # Abrir el CSV con lock
        with open(log_csv_path, "w", newline="", encoding="utf-8") as f:
            # Bloqueo exclusivo
            fcntl.flock(f, fcntl.LOCK_EX)
            writer = csv.writer(f)
            writer.writerow(["iteracion", "valor_coste"])
            for i, val in enumerate(cost_history):
                writer.writerow([i, val])
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)  # desbloquear

    if log_csv_path:
        iter_csv_path = log_csv_path.replace(".csv", "_iter.csv")
        with open(iter_csv_path, "w", newline="", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            writer = csv.writer(f)
            writer.writerow(["iteracion", "valor_coste"])
            for i, val in enumerate(iteration_costs):
                writer.writerow([i, val])
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)

    return result, experiment_result