"""Consulta de recomendaciones para la gestión de cartera activa."""

import streamlit as st

from tabs.prescriptiva import (
    ORDEN_SEGMENTOS,
    RECOMENDACIONES,
    resumir_segmentos,
)


MENSAJES_CLIENTE = {
    ORDEN_SEGMENTOS[0]: (
        "Queremos compartirte algunas recomendaciones para administrar tu crédito y mantener una buena organización de tus pagos. "
        "Recuerda que puedes consultar la información de tu préstamo y nuestros recursos de educación financiera en los canales disponibles."
    ),
    ORDEN_SEGMENTOS[1]: (
        "Queremos ayudarte a mantener una buena administración de tu crédito. Te invitamos a consultar tus próximos pagos y conocer "
        "las herramientas y canales de orientación que tenemos disponibles, en caso de que necesites apoyo."
    ),
    ORDEN_SEGMENTOS[2]: (
        "Nos gustaría conocer si necesitas orientación sobre la administración de tu crédito. Nuestro equipo puede ayudarte a revisar "
        "la información de tu cuenta y explicarte las alternativas de atención disponibles, de acuerdo con tu situación."
    ),
}


def _porcentaje(valor):
    """Formatea proporciones del resumen con el mismo criterio de la pestaña 5."""
    return f"{valor:.1%}"


def explicar_limite_segmento(segmento, cortes):
    """Describe los límites exactos usados por la segmentación de cartera."""
    corte_33, corte_67 = cortes
    if segmento == ORDEN_SEGMENTOS[0]:
        return (
            f"Riesgo Bajo: probabilidad estimada menor o igual a {_porcentaje(corte_33)} "
            "(incluye valores iguales al percentil 33). Corresponde al tercio inferior, "
            "es decir, al grupo de menor riesgo relativo de esta cartera."
        )
    if segmento == ORDEN_SEGMENTOS[1]:
        return (
            f"Riesgo Medio: probabilidad estimada mayor que {_porcentaje(corte_33)} "
            f"y menor o igual a {_porcentaje(corte_67)} "
            "(incluye valores iguales al percentil 67). Corresponde al tercio intermedio "
            "de esta cartera."
        )
    if segmento == ORDEN_SEGMENTOS[2]:
        return (
            f"Riesgo Alto: probabilidad estimada superior a {_porcentaje(corte_67)} "
            "(los valores iguales al percentil 67 pertenecen a Riesgo Medio). "
            "Corresponde al tercio superior, es decir, al grupo de mayor riesgo relativo de esta cartera."
        )
    raise ValueError(f"Segmento desconocido: {segmento}")


def mostrar(resultados):
    """Muestra decisiones sugeridas usando el scoring y reglas existentes."""
    st.header("6. Toma de decisiones")
    st.markdown("""
    ### ¿Qué acción preventiva debemos tomar y cómo comunicarla?

    Consulta recomendaciones para priorizar el seguimiento de préstamos personales activos. Las acciones internas y los mensajes para clientes se muestran por separado.
    """)

    scoring_segmentado, resumen, cortes = resumir_segmentos(
        resultados["scoring_cartera"]
    )
    resumen_por_segmento = resumen.set_index("Segmento")

    st.subheader("Decisiones por segmento de riesgo")
    segmento = st.selectbox(
        "Selecciona un segmento de riesgo:",
        ORDEN_SEGMENTOS,
        key="decisiones_segmento",
    )
    st.info(explicar_limite_segmento(segmento, cortes))
    fila = resumen_por_segmento.loc[segmento]
    recomendacion = RECOMENDACIONES[segmento]

    metrica_1, metrica_2, metrica_3, metrica_4 = st.columns(4)
    metrica_1.metric("Clientes", f"{int(fila['Clientes']):,}")
    metrica_2.metric("Participación", _porcentaje(fila["Participacion"]))
    metrica_3.metric(
        "Probabilidad promedio estimada",
        _porcentaje(fila["Riesgo_estimado"]),
    )
    metrica_4.metric("Bad Rate observado", _porcentaje(fila["Bad_Rate"]))

    st.markdown(f"#### {segmento}")
    st.markdown(f"**Objetivo de seguimiento**  \n{recomendacion['Objetivo']}")

    col_interna, col_mensaje = st.columns(2)
    with col_interna:
        with st.container(border=True):
            st.markdown("**Acción interna recomendada**")
            st.write(recomendacion["Acción interna"])
    with col_mensaje:
        with st.container(border=True):
            st.markdown("**Mensaje sugerido al cliente**")
            st.code(MENSAJES_CLIENTE[segmento], language=None, wrap_lines=True)

    st.caption(
        "Justificación: "
        f"{recomendacion['Justificación']} La probabilidad promedio estimada del segmento es "
        f"{_porcentaje(fila['Riesgo_estimado'])} y su Bad Rate observado es "
        f"{_porcentaje(fila['Bad_Rate'])}."
    )

    st.subheader("Consultar recomendación por cliente")
    clientes = scoring_segmentado["customer_id"].tolist()
    cliente_id = st.selectbox(
        "Selecciona un ID de cliente sintético:",
        clientes,
        key="decisiones_cliente_id",
    )
    cliente = scoring_segmentado.loc[
        scoring_segmentado["customer_id"] == cliente_id
    ].iloc[0]
    segmento_cliente = str(cliente["Segmento"])
    recomendacion_cliente = RECOMENDACIONES[segmento_cliente]
    mensaje_cliente = MENSAJES_CLIENTE[segmento_cliente]
    fila_segmento_cliente = resumen_por_segmento.loc[segmento_cliente]

    st.markdown(f"#### Cliente {cliente_id}")
    cliente_metrica_1, cliente_metrica_2 = st.columns(2)
    cliente_metrica_1.metric(
        "Probabilidad estimada a 12 meses",
        _porcentaje(cliente["Probabilidad estimada"]),
    )
    cliente_metrica_2.metric("Segmento asignado", segmento_cliente)
    st.info(explicar_limite_segmento(segmento_cliente, cortes))
    st.caption(
        f"Nota metodológica: score generado sobre la muestra de "
        f"{str(cliente['Partición']).split(' (')[0].lower()} ({cliente['Partición'].split('(')[-1].rstrip(')')})."
    )

    st.markdown(f"**Acción interna recomendada**  \n{recomendacion_cliente['Acción interna']}")
    st.markdown(f"**Objetivo de seguimiento**  \n{recomendacion_cliente['Objetivo']}")
    st.markdown("**Mensaje sugerido para este cliente**")
    st.code(mensaje_cliente, language=None, wrap_lines=True)
    st.caption(
        "Justificación: "
        f"{recomendacion_cliente['Justificación']} Su probabilidad estimada es "
        f"{_porcentaje(cliente['Probabilidad estimada'])}; el Bad Rate observado de su segmento es "
        f"{_porcentaje(fila_segmento_cliente['Bad_Rate'])}."
    )

    st.info(
        "Los datos y resultados son sintéticos. Behavioral PD se refiere aquí a préstamos existentes y los segmentos son relativos a esta cartera. "
        "La cartera completa combina probabilidades in-sample y out-of-sample. Las recomendaciones son ilustrativas y no demuestran que "
        "las intervenciones reduzcan el incumplimiento. Cualquier acción real requiere revisión humana y validación conforme a las políticas aplicables; "
        "las probabilidades no deben utilizarse para ejecutar decisiones crediticias adversas automáticamente."
    )
