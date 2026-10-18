"""Recomendaciones preventivas basadas en el scoring de la cartera completa."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


ORDEN_SEGMENTOS = [
    "Riesgo Bajo",
    "Riesgo Medio",
    "Riesgo Alto",
]

RECOMENDACIONES = {
    ORDEN_SEGMENTOS[0]: {
        "Acción interna": (
            "Mantener el monitoreo habitual del comportamiento crediticio y las comunicaciones generales de seguimiento."
        ),
        "Objetivo": (
            "Conservar una relación positiva con el cliente y promover hábitos financieros saludables."
        ),
        "Comunicación sugerida": (
            "Compartir educación financiera y recordatorios generales sobre la administración del crédito."
        ),
        "Justificación": (
            "Es el segmento con menor probabilidad estimada de incumplimiento dentro de la cartera analizada."
        ),
    },
    ORDEN_SEGMENTOS[1]: {
        "Acción interna": (
            "Dar seguimiento preventivo a posibles señales de deterioro crediticio y poner orientación financiera a disposición del cliente."
        ),
        "Objetivo": (
            "Identificar oportunamente dificultades que puedan afectar el cumplimiento de sus obligaciones."
        ),
        "Comunicación sugerida": (
            "Informar sobre herramientas para administrar pagos, consultar el estado del crédito y acceder a canales de orientación."
        ),
        "Justificación": (
            "Tiene una probabilidad estimada intermedia; el seguimiento preventivo podría resultar útil."
        ),
    },
    ORDEN_SEGMENTOS[2]: {
        "Acción interna": (
            "Priorizar una revisión individual de la cuenta y considerar el comportamiento de pago reciente; cuando corresponda, evaluar alternativas de apoyo disponibles."
        ),
        "Objetivo": (
            "Identificar necesidades de atención temprana y orientar una intervención preventiva personalizada."
        ),
        "Comunicación sugerida": (
            "Establecer un contacto personalizado, respetuoso y empático para conocer si requiere orientación o información sobre alternativas de apoyo."
        ),
        "Justificación": (
            "Es el segmento con mayor probabilidad estimada; se recomienda priorizar su revisión por el equipo de gestión de cartera."
        ),
    },
}


def resumir_segmentos(scoring):
    """Calcula cortes y resumen sobre todas las probabilidades de cartera."""
    requeridas = {
        "customer_id", "Probabilidad estimada", "default_12m", "Partición"
    }
    faltantes = requeridas - set(scoring.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas en el scoring: {sorted(faltantes)}")
    if scoring.empty or scoring["customer_id"].duplicated().any():
        raise ValueError("El scoring debe contener clientes únicos y no estar vacío.")

    probabilidades = scoring["Probabilidad estimada"].to_numpy(dtype=float)
    if not np.isfinite(probabilidades).all() or (
        (probabilidades < 0) | (probabilidades > 1)
    ).any():
        raise ValueError("Todas las probabilidades deben ser valores válidos entre 0 y 1.")

    corte_1, corte_2 = np.quantile(probabilidades, [1 / 3, 2 / 3])
    datos = scoring.copy()
    datos["Segmento"] = np.select(
        [probabilidades <= corte_1, probabilidades <= corte_2],
        [ORDEN_SEGMENTOS[0], ORDEN_SEGMENTOS[1]],
        default=ORDEN_SEGMENTOS[2],
    )
    datos["Segmento"] = pd.Categorical(
        datos["Segmento"], categories=ORDEN_SEGMENTOS, ordered=True
    )

    resumen = (
        datos.groupby("Segmento", observed=True)
        .agg(
            Clientes=("customer_id", "nunique"),
            Participacion=("customer_id", lambda ids: len(ids) / len(datos)),
            Riesgo_estimado=("Probabilidad estimada", "mean"),
            Bad_Rate=("default_12m", "mean"),
        )
        .reindex(ORDEN_SEGMENTOS)
    )
    resumen["Clientes"] = resumen["Clientes"].fillna(0).astype(int)
    resumen["Participacion"] = resumen["Participacion"].fillna(0.0)
    resumen.index.name = "Segmento"
    resumen = resumen.reset_index()

    rangos = {
        ORDEN_SEGMENTOS[0]: f"Hasta {corte_1:.1%}",
        ORDEN_SEGMENTOS[1]: f"Más de {corte_1:.1%} y hasta {corte_2:.1%}",
        ORDEN_SEGMENTOS[2]: f"Más de {corte_2:.1%}",
    }
    resumen["Rango de probabilidad"] = resumen["Segmento"].map(rangos)
    resumen["Probabilidad promedio estimada"] = resumen["Riesgo_estimado"].map(
        lambda valor: f"{valor:.1%}" if pd.notna(valor) else "Sin clientes"
    )
    resumen["Participación"] = resumen["Participacion"].map(
        lambda valor: f"{valor:.1%}"
    )
    resumen["Bad Rate observado"] = resumen["Bad_Rate"].map(
        lambda valor: f"{valor:.1%}" if pd.notna(valor) else "Sin clientes"
    )
    resumen["Segmento"] = pd.Categorical(
        resumen["Segmento"], categories=ORDEN_SEGMENTOS, ordered=True
    )
    resumen = (
        resumen.set_index("Segmento")
        .reindex(ORDEN_SEGMENTOS)
        .reset_index()
    )

    return datos, resumen, (float(corte_1), float(corte_2))


def mostrar(resultados):
    st.header("5. Estadística prescriptiva")
    st.markdown("""
    ### ¿Qué seguimiento preventivo conviene priorizar?

    La vista considera a clientes únicos con préstamos personales activos. Las probabilidades se obtienen con el modelo final de la pestaña **4. Predictiva**; las acciones se enfocan en el seguimiento de créditos existentes y requieren revisión humana.
    """)

    scoring, resumen, cortes = resumir_segmentos(resultados["scoring_cartera"])
    total = len(scoring)

    c1, c2, c3 = st.columns(3)
    c1.metric("Clientes analizados", f"{total:,}")
    c2.metric("Percentil 33", f"{cortes[0]:.1%}")
    c3.metric("Percentil 67", f"{cortes[1]:.1%}")

    st.caption(
        "Los cortes se calculan dinámicamente con los percentiles 33 y 67 de las probabilidades de los 990 clientes. "
        "Describen esta cartera sintética y no son límites regulatorios ni valores universales."
    )

    st.subheader("Resumen por segmento")
    st.dataframe(
        resumen[[
            "Segmento", "Clientes", "Probabilidad promedio estimada",
            "Bad Rate observado",
        ]],
        hide_index=True,
        width="stretch",
        column_config={
            "Bad Rate observado": st.column_config.TextColumn(
                "Bad Rate observado",
                help="Incumplimiento observado entre estos clientes; incluye entrenamiento y prueba, por lo que no es una validación independiente.",
            ),
        },
    )

    st.subheader("Recomendación preventiva")
    segmento_seleccionado = st.selectbox(
        "Selecciona un segmento para consultar la recomendación:",
        ORDEN_SEGMENTOS,
        key="prescriptiva_segmento_recomendacion",
    )
    recomendacion = RECOMENDACIONES[segmento_seleccionado]

    st.markdown(f"### {segmento_seleccionado}")
    st.markdown(f"**Objetivo**  \n{recomendacion['Objetivo']}")
    st.markdown(f"**Justificación**  \n{recomendacion['Justificación']}")

    col_interna, col_cliente = st.columns(2)
    with col_interna:
        with st.container(border=True):
            st.markdown("**Acción interna para el equipo de cartera**")
            st.write(recomendacion["Acción interna"])
    with col_cliente:
        with st.container(border=True):
            st.markdown("**Comunicación sugerida para el cliente**")
            st.write(recomendacion["Comunicación sugerida"])

    st.caption(
        "Las recomendaciones se refieren a préstamos personales existentes. Son orientaciones para revisión humana y no instrucciones automáticas."
    )

    st.subheader("Distribución de la cartera por segmento")
    resumen_grafica = resumen.set_index("Segmento").reindex(ORDEN_SEGMENTOS)
    segmentos = resumen_grafica.index.astype(str).tolist()
    participacion = resumen_grafica["Participacion"].to_numpy() * 100
    bad_rate = resumen_grafica["Bad_Rate"].to_numpy() * 100

    figura = make_subplots(specs=[[{"secondary_y": True}]])
    figura.add_trace(
        go.Bar(
            x=segmentos,
            y=participacion,
            name="Participación de clientes (%)",
            marker_color="#1f77b4",
            hovertemplate="%{x}<br>Participación: %{y:.1f}%<extra></extra>",
        ),
        secondary_y=False,
    )
    figura.add_trace(
        go.Scatter(
            x=segmentos,
            y=bad_rate,
            name="Bad Rate observado (%)",
            mode="lines+markers+text",
            line={"color": "#ff7f0e", "width": 3},
            marker={"color": "#ff7f0e", "size": 10},
            text=[f"{valor:.1f}%" for valor in bad_rate],
            textposition="top center",
            hovertemplate="%{x}<br>Bad Rate observado: %{y:.1f}%<extra></extra>",
        ),
        secondary_y=True,
    )
    figura.update_layout(
        title="Distribución de cartera y Bad Rate por segmento",
        barmode="group",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"t": 95, "b": 55},
        xaxis={"title": "Segmento", "categoryorder": "array", "categoryarray": ORDEN_SEGMENTOS},
    )
    figura.update_yaxes(
        title_text="Participación de clientes (%)",
        ticksuffix="%",
        rangemode="tozero",
        secondary_y=False,
    )
    figura.update_yaxes(
        title_text="Bad Rate observado (%)",
        ticksuffix="%",
        rangemode="tozero",
        secondary_y=True,
    )
    st.plotly_chart(
        figura,
        width="stretch",
        config={"displayModeBar": False},
    )

    with st.expander("Consultar scoring por cliente"):
        vista_clientes = scoring[[
            "customer_id", "Probabilidad estimada", "Segmento", "Partición"
        ]].rename(columns={
            "customer_id": "ID de cliente sintético",
            "Probabilidad estimada": "Probabilidad estimada de incumplimiento",
        })
        st.dataframe(
            vista_clientes,
            hide_index=True,
            width="stretch",
            column_config={
                "Probabilidad estimada de incumplimiento": st.column_config.ProgressColumn(
                    "Probabilidad estimada de incumplimiento",
                    min_value=0.0,
                    max_value=1.0,
                    format="percent",
                )
            },
        )

    st.info(
        "Nota metodológica: el modelo se ajustó con los 693 clientes de entrenamiento; sus probabilidades son *in-sample*. "
        "Las probabilidades de los 297 clientes de prueba son *out-of-sample*. El Bad Rate observado de la cartera completa "
        "es descriptivo y no constituye validación independiente. AUC, Gini y KS de la pestaña predictiva siguen calculándose "
        "exclusivamente con la muestra de prueba. Los datos son sintéticos y los segmentos representan niveles relativos de riesgo "
        "dentro de esta cartera. Estas sugerencias no demuestran que las intervenciones reduzcan el incumplimiento; cualquier acción "
        "real requiere revisión humana, políticas de gestión y validaciones adicionales."
    )
