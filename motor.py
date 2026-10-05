"""
motor.py - Indicadores de concentración de mercado y simulación Monte Carlo.

Funciones públicas: cr_k, ihh, generar_cuotas, simular_mercado.
"""
import numpy as np

# Margen de tolerancia relativo para errores de redondeo decimal
_TOL = 1e-6


def _validar_cuotas(cuotas):
    """
    Valida las cuotas y las normaliza a fracciones.

    Devuelve
    --------
    (fracciones, factor) : (np.ndarray, int)
        `fracciones` suma 1.0. `factor` es 1 si la entrada venía en
        fracciones o 100 si venía en porcentajes.
    """
    arr = np.asarray(cuotas, dtype=float)

    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("`cuotas` debe ser una lista/array 1D no vacío.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("`cuotas` contiene valores NaN o infinitos.")
    if np.any(arr < 0):
        raise ValueError("Las cuotas de mercado no pueden ser negativas.")

    total = arr.sum()
    if abs(total - 1.0) <= _TOL:
        factor = 1
    elif abs(total - 100.0) <= _TOL * 100:
        factor = 100
    else:
        raise ValueError(
            f"Las cuotas deben sumar 1.0 o 100 (suma actual: {total:.8f})."
        )

    return arr / total, factor


def cr_k(cuotas, k):
    """
    Ratio de Concentración CR_k: suma de las k mayores cuotas de mercado.

    Parámetros
    ----------
    cuotas : list | np.ndarray
        Cuotas de las N empresas (suman 1.0 o 100).
    k : int
        Número de mayores empresas a considerar (1 <= k <= N).

    Retorna
    -------
    float
        CR_k en la misma unidad de la entrada (fracción o porcentaje).
    """
    fracciones, factor = _validar_cuotas(cuotas)

    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError("`k` debe ser un entero.")
    if not 1 <= k <= fracciones.size:
        raise ValueError(f"`k` debe estar entre 1 y N={fracciones.size}.")

    # np.partition es O(N) frente a O(N log N) de ordenar todo el arreglo
    top_k = np.partition(fracciones, -k)[-k:]
    return float(top_k.sum() * factor)


def ihh(cuotas, escala="puntos"):
    """
    Índice de Herfindahl-Hirschman: suma de las cuotas al cuadrado.

    Parámetros
    ----------
    cuotas : list | np.ndarray
        Cuotas de las N empresas (suman 1.0 o 100).
    escala : {"puntos", "fraccion"}
        "puntos"   -> rango (0, 10 000]  (cuotas en %, estándar regulatorio)
        "fraccion" -> rango (0, 1]       (cuotas en proporción)

    Retorna
    -------
    float
    """
    fracciones, _ = _validar_cuotas(cuotas)

    if escala == "puntos":
        multiplicador = 10_000.0
    elif escala == "fraccion":
        multiplicador = 1.0
    else:
        raise ValueError('`escala` debe ser "puntos" o "fraccion".')

    return float(np.dot(fracciones, fracciones) * multiplicador)


def generar_cuotas(n_empresas, alpha=1.0, rng=None):
    """
    Genera un vector aleatorio de cuotas de mercado que suma 1.0.

    Parámetros
    ----------
    n_empresas : int
        Número de empresas N (>= 1).
    alpha : float
        Parámetro de concentración de la Dirichlet simétrica.
        alpha = 1  -> uniforme sobre todas las estructuras posibles.
        alpha < 1  -> mercados más concentrados (pocas empresas dominan).
        alpha > 1  -> mercados más equilibrados (cuotas similares).
    rng : np.random.Generator, opcional
        Generador de números aleatorios (para reproducibilidad).

    Retorna
    -------
    np.ndarray de forma (N,), no negativo, con suma 1.0 (error ~1e-16).
    """
    if isinstance(n_empresas, bool) \
            or not isinstance(n_empresas, (int, np.integer)) \
            or n_empresas < 1:
        raise ValueError("`n_empresas` debe ser un entero >= 1.")
    if alpha <= 0:
        raise ValueError("`alpha` debe ser > 0.")

    rng = np.random.default_rng() if rng is None else rng
    cuotas = rng.dirichlet(np.full(n_empresas, alpha))

    # Renormalización final para minimizar el error de redondeo
    return cuotas / cuotas.sum()


def simular_mercado(n_empresas, indice="ihh", k=None,
                    iteraciones=1000, alpha=1.0, semilla=None):
    """
    Simulación Monte Carlo de un indicador de concentración.

    Parámetros
    ----------
    n_empresas : int
        Número de empresas N en el mercado.
    indice : {"ihh", "cr_k"}
        Indicador a calcular en cada iteración.
    k : int, opcional
        Obligatorio si indice == "cr_k" (1 <= k <= N).
    iteraciones : int
        Número de simulaciones (por defecto 1000).
    alpha : float
        Concentración de la Dirichlet (ver `generar_cuotas`).
    semilla : int, opcional
        Semilla para resultados reproducibles.

    Retorna
    -------
    list[float]
        Un valor del indicador por iteración.
        - "ihh"  -> en puntos (0-10 000)
        - "cr_k" -> en fracción (0-1)
    """
    # Validaciones antes de entrar al bucle (fallar rápido)
    if indice not in ("ihh", "cr_k"):
        raise ValueError('`indice` debe ser "ihh" o "cr_k".')
    if indice == "cr_k":
        if k is None:
            raise ValueError('Debes indicar `k` cuando indice="cr_k".')
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) \
                or not 1 <= k <= n_empresas:
            raise ValueError(
                f"`k` debe ser un entero entre 1 y N={n_empresas}.")
    if isinstance(iteraciones, bool) \
            or not isinstance(iteraciones, (int, np.integer)) \
            or iteraciones < 1:
        raise ValueError("`iteraciones` debe ser un entero >= 1.")

    rng = np.random.default_rng(semilla)
    resultados = []

    for _ in range(iteraciones):
        cuotas = generar_cuotas(n_empresas, alpha, rng)
        if indice == "ihh":
            resultados.append(ihh(cuotas, "puntos"))
        else:
            resultados.append(cr_k(cuotas, k))

    return resultados
