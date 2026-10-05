"""
app.py - Simulador Monte Carlo de concentración de mercado (IHH y CRk).

Ejecutar con:  streamlit run app.py
Requiere en la misma carpeta: motor.py
Dependencias: streamlit, numpy, pandas, plotly
"""
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import motor

# ======================================================================
# CONFIGURACIÓN DE PÁGINA Y CONSTANTES
# ======================================================================
st.set_page_config(page_title="Simulador de Concentración de Mercado",
                   layout="wide")
st.title("Simulador Monte Carlo de concentración de mercado")

ITER_MAX = 200_000          # tope de iteraciones permitido
UMBRAL_ITER_ALTO = 50_000   # a partir de aquí se muestra una 2.ª advertencia

NIVELES = ["Baja", "Moderada", "Alta"]

# IHH en puntos: (límite Baja/Moderada, límite Moderada/Alta).
# Confirma los umbrales vigentes de tu autoridad de competencia local.
ESTANDARES_IHH = {
    "Guías de fusiones EE. UU. 1992 / 2023 (1 000 / 1 800)": (1000, 1800),
    "Guías de fusiones EE. UU. 2010 (1 500 / 2 500)": (1500, 2500),
}
# CRk en fracción: clasificación de Shepherd, formulada para CR4.
ESTANDAR_CRK = ("Clasificación de Shepherd para CR4 (40% / 60%)", (0.40, 0.60))


# ======================================================================
# FUNCIONES AUXILIARES
# ======================================================================
def calcular_indicador(cuotas, clave, k):
    """
    Calcula el indicador de un vector de cuotas con las funciones de motor.py
    y lo devuelve en la MISMA escala que la simulación
    (IHH en puntos 0-10 000; CRk en fracción 0-1).
    Lanza ValueError/TypeError si las cuotas no son válidas.
    """
    cuotas = np.asarray(cuotas, dtype=float)
    if clave == "ihh":
        return motor.ihh(cuotas, "puntos")
    valor = motor.cr_k(cuotas, k)      # motor devuelve en la unidad de entrada
    return valor / 100.0 if cuotas.sum() > 1.5 else valor


def parsear_cuotas(texto):
    """Convierte '40, 30, 20, 10' en un array de floats."""
    partes = [p.strip() for p in texto.split(",") if p.strip() != ""]
    try:
        return np.array([float(p) for p in partes], dtype=float)
    except ValueError:
        raise ValueError("Hay valores que no son números. Usa punto como "
                         "separador decimal y comas entre cuotas.")


def cuotas_por_defecto(n):
    """Texto con N cuotas iguales que suman 100."""
    valor = format(round(100.0 / n, 6), "f").rstrip("0").rstrip(".")
    return ", ".join([valor] * n)


def clasificar(valor, limites):
    """Baja si valor < bajo; Moderada si bajo <= valor <= alto; Alta si > alto."""
    bajo, alto = limites
    if valor < bajo:
        return "Baja"
    if valor <= alto:
        return "Moderada"
    return "Alta"


def percentil_caso(datos, valor):
    """% de simulaciones con indicador menor o igual al del caso."""
    return float(np.mean(datos <= valor) * 100)


def formatear_valor(valor, clave):
    return f"{valor:,.2f} puntos" if clave == "ihh" else f"{valor:.2%}"


def texto_umbrales(clave, limites):
    bajo, alto = limites
    if clave == "ihh":
        return (f"- **Baja:** IHH < {bajo:,.0f}\n"
                f"- **Moderada:** {bajo:,.0f} a {alto:,.0f}\n"
                f"- **Alta:** IHH > {alto:,.0f}")
    return (f"- **Baja:** CRk < {bajo:.0%}\n"
            f"- **Moderada:** {bajo:.0%} a {alto:.0%}\n"
            f"- **Alta:** CRk > {alto:.0%}")


def notas_tecnicas(clave, n, k, limites):
    if clave == "ihh":
        minimo = 10_000 / n
        return [
            "El IHH suma los cuadrados de las cuotas (en %), por lo que "
            "pondera más a las empresas grandes.",
            f"Con N = {n} empresas, el IHH mínimo posible es {minimo:,.0f} "
            f"(cuotas iguales), así que el nivel mínimo alcanzable en este "
            f"mercado es **{clasificar(minimo, limites)}**.",
        ]
    notas = ["El CRk suma las k mayores cuotas e ignora cómo se reparte "
             "el resto del mercado."]
    if k != 4:
        notas.append(f"Los umbrales de Shepherd se formularon para CR4; con "
                     f"k = {k} la clasificación es solo orientativa.")
    if k == n:
        notas.append("Con k = N el CRk vale siempre 100%, así que no "
                     "discrimina entre mercados.")
    return notas


# ======================================================================
# 1) SIDEBAR
# ======================================================================
with st.sidebar:
    st.header("Parámetros")

    indicador = st.selectbox(
        "Indicador de concentración",
        options=["IHH", "CRk"],
        help="IHH: Índice de Herfindahl-Hirschman (puntos, 0-10 000). "
             "CRk: suma de las k mayores cuotas.",
    )

    n_empresas = st.number_input(
        "Número de empresas (N)",
        min_value=2, max_value=100, value=5, step=1,
    )

    k = None
    if indicador == "CRk":
        k = st.number_input(
            "k (empresas más grandes)",
            min_value=1, max_value=int(n_empresas),
            value=min(3, int(n_empresas)), step=1,
        )

    iteraciones = st.number_input(
        "Iteraciones de Monte Carlo",
        min_value=100, max_value=ITER_MAX, value=1000, step=1000,
    )

    st.warning(
        "⚠️ **Rendimiento:** el tiempo y el consumo de cómputo crecen de forma "
        "lineal con las iteraciones (y con N). Valores muy altos pueden "
        "volver lenta o bloquear la aplicación, sobre todo en servidores "
        "compartidos."
    )
    if iteraciones > UMBRAL_ITER_ALTO:
        st.warning(
            f"Has superado {UMBRAL_ITER_ALTO:,} iteraciones: la simulación "
            "puede tardar varios segundos o minutos."
        )

    ejecutar = st.button("Ejecutar Simulación", type="primary")

# Parámetros vigentes normalizados
clave = "ihh" if indicador == "IHH" else "cr_k"
n = int(n_empresas)
k_int = int(k) if clave == "cr_k" else None
unidad = "puntos" if clave == "ihh" else "fracción"
sufijo_k = f", k = {k_int}" if k_int else ""
params_actuales = (clave, n, k_int)

# ======================================================================
# EJECUCIÓN DE LA SIMULACIÓN (persistente en session_state)
# ======================================================================
if ejecutar:
    with st.spinner("Simulando..."):
        t0 = time.perf_counter()
        resultados = motor.simular_mercado(
            n, clave, k=k_int, iteraciones=int(iteraciones)
        )
        duracion = time.perf_counter() - t0
    st.session_state["sim"] = {
        "params": params_actuales,
        "datos": np.asarray(resultados, dtype=float),
        "duracion": duracion,
    }

sim = st.session_state.get("sim")
datos_sim = (sim["datos"] if sim is not None
             and sim["params"] == params_actuales else None)

# ======================================================================
# 2) CASO PARTICULAR
# ======================================================================
st.header("1. Caso particular")

modo = st.radio("Origen de las cuotas",
                ["Ingreso manual", "Caso aleatorio"], horizontal=True)

cuotas_caso = None   # vector de cuotas del caso (None si no hay uno válido)
valor_caso = None    # valor del indicador del caso

if modo == "Ingreso manual":
    texto = st.text_input(
        f"Cuotas de las {n} empresas, separadas por comas",
        value=cuotas_por_defecto(n),
        key=f"texto_cuotas_{n}",          # se reinicia si cambia N
        help="Usa punto como separador decimal. Deben sumar 100 "
             "(porcentajes) o 1 (fracciones).",
    )
    try:
        arr = parsear_cuotas(texto)
        if arr.size != n:
            raise ValueError(f"Ingresaste {arr.size} valores y se esperaban "
                             f"{n} (uno por empresa).")
        cuotas_caso = arr
        st.caption(f"Suma actual: {arr.sum():.6g}")
    except ValueError as e:
        st.error(str(e))
else:
    if st.button("🎲 Generar caso aleatorio"):
        # alpha = 1: mismo supuesto que la simulación (uniforme en el simplex)
        st.session_state["caso_aleatorio"] = motor.generar_cuotas(n, 1.0)

    aleatorio = st.session_state.get("caso_aleatorio")
    if aleatorio is not None and len(aleatorio) == n:
        cuotas_caso = aleatorio
        st.dataframe(pd.DataFrame(
            {"Cuota (%)": np.round(aleatorio * 100, 4)},
            index=[f"Empresa {i + 1}" for i in range(n)],
        ))
    else:
        st.info("Pulsa el botón para generar un caso aleatorio.")

if cuotas_caso is not None:
    try:
        valor_caso = calcular_indicador(cuotas_caso, clave, k_int)
        st.metric(f"{indicador} del caso particular ({unidad})",
                  f"{valor_caso:,.4f}")
    except (ValueError, TypeError) as e:
        st.error(f"Caso no válido: {e}")

# ======================================================================
# 3) GRÁFICO: DISTRIBUCIÓN EMPÍRICA + CASO PARTICULAR
# ======================================================================
st.header("2. Distribución empírica (Monte Carlo)")

if sim is None:
    st.info("Configura los parámetros y pulsa **Ejecutar Simulación**.")
elif datos_sim is None:
    st.warning("Cambiaste el indicador, N o k desde la última simulación. "
               "Vuelve a pulsar **Ejecutar Simulación** para actualizar el "
               "gráfico.")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Media", f"{datos_sim.mean():.4f}")
    c2.metric("Desv. estándar", f"{datos_sim.std(ddof=1):.4f}")
    c3.metric("Percentil 5", f"{np.percentile(datos_sim, 5):.4f}")
    c4.metric("Percentil 95", f"{np.percentile(datos_sim, 95):.4f}")

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=datos_sim, nbinsx=40, name="Simulación",
        marker_line_width=1, marker_line_color="white", opacity=0.85,
    ))

    if valor_caso is not None:
        etiqueta = (f"Caso particular: {valor_caso:,.2f}" if clave == "ihh"
                    else f"Caso particular: {valor_caso:.4f}")
        fig.add_vline(
            x=valor_caso, line_width=4, line_dash="dash", line_color="red",
            annotation_text=etiqueta, annotation_position="top right",
            annotation_font_color="red",
        )

    fig.update_layout(
        title=(f"Distribución simulada de {indicador} (N = {n}{sufijo_k}, "
               f"{len(datos_sim):,} iteraciones)"),
        xaxis_title=f"{indicador} ({unidad})",
        yaxis_title="Frecuencia (nº de simulaciones)",
        showlegend=False,
        bargap=0.02,
    )
    st.plotly_chart(fig)

    if valor_caso is not None:
        pct = percentil_caso(datos_sim, valor_caso)
        st.success(f"El caso particular se ubica en el **percentil "
                   f"{pct:.1f}** de la distribución simulada.")
    else:
        st.info("Define un caso particular válido para ver su posición "
                "sobre el histograma.")

    st.caption(f"{len(datos_sim):,} iteraciones en {sim['duracion']:.2f} s.")

# ======================================================================
# 4) EVALUACIÓN PEDAGÓGICA
# ======================================================================
st.header("3. Evaluación")

if valor_caso is None:
    st.info("Define un caso particular válido (sección 1) para poder "
            "evaluarte.")
else:
    st.write("Observa las cuotas del caso particular. "
             "**¿Qué nivel de concentración crees que tiene?**")

    if clave == "ihh":
        nombre_estandar = st.selectbox("Estándar de referencia",
                                       list(ESTANDARES_IHH))
        limites = ESTANDARES_IHH[nombre_estandar]
    else:
        nombre_estandar, limites = ESTANDAR_CRK

    respuesta = st.radio("Tu respuesta", NIVELES, index=None, horizontal=True)

    if st.button("Verificar", disabled=respuesta is None):
        st.session_state["eval"] = {
            "respuesta": respuesta,
            "valor": valor_caso,
            "params": params_actuales,
            "estandar": nombre_estandar,
        }

    # La retroalimentación solo se muestra si sigue vigente
    ev = st.session_state.get("eval")
    vigente = (ev is not None
               and ev["valor"] == valor_caso
               and ev["params"] == params_actuales
               and ev["estandar"] == nombre_estandar)

    if vigente:
        correcta = clasificar(valor_caso, limites)
        if ev["respuesta"] == correcta:
            st.success(f"✅ ¡Correcto! El caso tiene concentración "
                       f"**{correcta}**.")
        else:
            st.error(f"❌ Respondiste **{ev['respuesta']}**, pero el caso "
                     f"tiene concentración **{correcta}**.")

        st.markdown(f"**Valor del caso:** {indicador} = "
                    f"{formatear_valor(valor_caso, clave)}")

        st.subheader("Regla teórica y justificación")
        st.markdown(f"Estándar aplicado: *{nombre_estandar}*\n\n"
                    + texto_umbrales(clave, limites))
        for nota in notas_tecnicas(clave, n, k_int, limites):
            st.markdown(f"- {nota}")

        st.subheader("Posición en la distribución de Monte Carlo")
        if datos_sim is None:
            st.info("Ejecuta la simulación con estos parámetros para ver "
                    "el percentil del caso.")
        else:
            pct = percentil_caso(datos_sim, valor_caso)
            bajo, alto = limites
            mascaras = {
                "Baja": datos_sim < bajo,
                "Moderada": (datos_sim >= bajo) & (datos_sim <= alto),
                "Alta": datos_sim > alto,
            }
            share = float(mascaras[correcta].mean() * 100)
            st.markdown(
                f"- El caso está en el **percentil {pct:.1f}**: el "
                f"{pct:.1f}% de los mercados simulados tiene un {indicador} "
                f"menor o igual.\n"
                f"- El **{share:.1f}%** de los mercados simulados cae en la "
                f"misma categoría (**{correcta}**), lo que indica qué tan "
                f"típico es este nivel bajo el supuesto de cuotas "
                f"uniformes (alpha = 1)."
            )
