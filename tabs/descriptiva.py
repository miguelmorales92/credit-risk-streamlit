
import streamlit as st
import pandas as pd


# ----------------------------------
# CREAR SEGMENTOS CREDITICIOS
# ----------------------------------

def crear_segmentos(df):
    """
    Genera grupos para el análisis descriptivo.
    Los cortes son ilustrativos para este ejercicio.
    """

    datos = df.copy()

    datos["segmento_dti"] = pd.cut(
        datos["dti_ratio"],
        bins=[-float("inf"), 0.3, 0.5, 0.7, float("inf")],
        labels=["Hasta 30%", "30-50%", "50-70%", "Más de 70%"]
    )

    datos["segmento_utilizacion"] = pd.cut(
        datos["credit_utilization"],
        bins=[-float("inf"), 0.3, 0.6, 0.8, float("inf")],
        labels=["Hasta 30%", "30-60%", "60-80%", "Más de 80%"]
    )

    datos["segmento_consultas"] = pd.cut(
        datos["bureau_inquiries_6m"],
        bins=[-float("inf"), 0, 2, 4, float("inf")],
        labels=["0 consultas", "1-2 consultas",
                "3-4 consultas", "5 o más"]
    )

    datos["segmento_mop"] = datos["worst_mop_12m"].map(
        lambda x: f"MOP {int(x)}"
    )

    return datos


# ----------------------------------
# CALCULAR BAD RATE POR SEGMENTO
# ----------------------------------

def calcular_bad_rate(datos, variable):

    resumen = (
        datos.groupby(variable, observed=True, dropna=False)
        .agg(
            clientes=("customer_id", "count"),
            incumplimientos=("default_12m", "sum")
        )
        .reset_index()
    )

    resumen["bad_rate"] = (
        resumen["incumplimientos"] /
        resumen["clientes"] * 100
    )

    resumen["participacion"] = (
        resumen["clientes"] / len(datos) * 100
    )

    resumen[variable] = resumen[variable].astype(str)

    # Mantener un orden lógico en los segmentos
    ordenes = {
        "segmento_dti": [
            "Hasta 30%", "30-50%", "50-70%", "Más de 70%"
        ],
        "segmento_utilizacion": [
            "Hasta 30%", "30-60%", "60-80%", "Más de 80%"
        ],
        "segmento_consultas": [
            "0 consultas", "1-2 consultas",
            "3-4 consultas", "5 o más"
        ],
        "segmento_mop": [
            "MOP 0", "MOP 1", "MOP 2", "MOP 3", "MOP 4"
        ]
    }

    if variable in ordenes:
        resumen[variable] = pd.Categorical(
            resumen[variable],
            categories=ordenes[variable],
            ordered=True
        )

        resumen = resumen.sort_values(variable)
        resumen[variable] = resumen[variable].astype(str)

    return resumen


# ----------------------------------
# MOSTRAR PESTAÑA DESCRIPTIVA
# ----------------------------------

def mostrar(df):

    st.header("3. Estadística descriptiva")

    st.markdown("""
    ### ¿Cómo se comporta nuestra cartera de préstamos activos?

    Analizamos las características de los clientes y su
    comportamiento de pago para identificar segmentos
    con diferentes niveles de incumplimiento observado.

    **Importante:** En esta etapa describimos relaciones
    históricas; todavía no estamos estimando probabilidades
    individuales ni demostrando causalidad.
    """)

    datos = crear_segmentos(df)

    # ----------------------------------
    # 1. KPIs DE CARTERA
    # ----------------------------------

    st.subheader("1. Indicadores generales de cartera")

    total_clientes = len(datos)
    total_bads = int(datos["default_12m"].sum())
    bad_rate = total_bads / total_clientes * 100

    monto_total = datos["loan_amount"].sum()
    credito_promedio = datos["loan_amount"].mean()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Clientes analizados",
            f"{total_clientes:,}"
        )

    with col2:
        st.metric(
            "Clientes con incumplimiento",
            f"{total_bads:,}"
        )

    with col3:
        st.metric(
            "Bad Rate global",
            f"{bad_rate:.2f}%"
        )

    col4, col5 = st.columns(2)

    with col4:
        st.metric(
            "Monto original total de créditos",
            f"${monto_total:,.0f} MXN"
        )

    with col5:
        st.metric(
            "Monto promedio por crédito",
            f"${credito_promedio:,.0f} MXN"
        )

    st.caption(
        "Bad Rate = clientes con default_12m igual a 1 / "
        "total de clientes. Los montos representan créditos "
        "originales, no exposición vigente."
    )

    # ----------------------------------
    # 2. PERFIL DE LOS CLIENTES
    # ----------------------------------

    st.subheader("2. Perfil financiero y crediticio")

    st.markdown("""
    Comparamos las características promedio de los clientes
    con y sin incumplimiento. Estas diferencias permiten
    formular hipótesis para el análisis predictivo.
    """)

    variables_perfil = [
        "age",
        "monthly_income",
        "loan_amount",
        "dti_ratio",
        "credit_utilization",
        "previous_delinquencies",
        "bureau_inquiries_6m",
        "credit_history_months"
    ]

    perfil = (
        datos.groupby("default_12m")[variables_perfil]
        .mean()
        .T
    )

    perfil.columns = [
        "Sin incumplimiento",
        "Con incumplimiento"
    ]

    st.dataframe(
        perfil.style.format("{:,.2f}"),
        use_container_width=True
    )

    st.caption(
        "Los valores corresponden a promedios descriptivos. "
        "No representan efectos causales ni importancia predictiva."
    )

    # ----------------------------------
    # 3. SEGMENTACION CREDITICIA
    # ----------------------------------

    st.subheader("3. Análisis por segmentos")

    st.markdown("""
    Selecciona una característica para observar cómo cambia
    la tasa de incumplimiento entre diferentes grupos.

    Comparamos el **Bad Rate** y el **número de clientes**
    para interpretar las diferencias considerando el tamaño
    de cada segmento.
    """)

    opciones = {
        "Worst MOP (últimos 12 meses)": "segmento_mop",
        "Consultas al Buró (últimos 6 meses)": "segmento_consultas",
        "Debt-to-Income (DTI)": "segmento_dti",
        "Utilización de crédito": "segmento_utilizacion",
        "Tipo de empleo": "employment_type"
    }

    seleccion = st.selectbox(
        "Selecciona una variable de segmentación:",
        list(opciones.keys()),
        key="segmento_descriptiva"
    )

    variable = opciones[seleccion]
    resumen = calcular_bad_rate(datos, variable)

    resumen_mostrar = resumen.rename(columns={
        variable: "Segmento",
        "clientes": "Clientes",
        "incumplimientos": "Incumplimientos",
        "bad_rate": "Bad Rate (%)",
        "participacion": "Participación (%)"
    })

    st.dataframe(
        resumen_mostrar.style.format({
            "Bad Rate (%)": "{:.2f}%",
            "Participación (%)": "{:.2f}%"
        }),
        hide_index=True,
        use_container_width=True
    )

    # Gráficas complementarias
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Bad Rate por segmento (%)**")

        st.bar_chart(
            resumen.set_index(variable)["bad_rate"],
            y_label="Bad Rate (%)",
            x_label="Segmento",
            sort=False
        )

    with col2:
        st.markdown("**Número de clientes por segmento**")

        st.bar_chart(
            resumen.set_index(variable)["clientes"],
            y_label="Clientes",
            x_label="Segmento",
            sort=False
        )

    # ----------------------------------
    # 4. INTERPRETACION AUTOMATICA
    # ----------------------------------

    st.subheader("4. Interpretación del análisis")

    st.markdown("""
    Para evitar conclusiones basadas en grupos demasiado
    pequeños, identificamos los segmentos que contienen
    al menos 30 clientes.
    """)

    elegibles = resumen[resumen["clientes"] >= 30]

    if not elegibles.empty:

        mayor = elegibles.loc[
            elegibles["bad_rate"].idxmax()
        ]

        menor = elegibles.loc[
            elegibles["bad_rate"].idxmin()
        ]

        st.markdown(
            f"**Mayor Bad Rate observado:** "
            f"{mayor[variable]} "
            f"({mayor['bad_rate']:.2f}%, "
            f"{int(mayor['clientes'])} clientes)."
        )

        st.markdown(
            f"**Menor Bad Rate observado:** "
            f"{menor[variable]} "
            f"({menor['bad_rate']:.2f}%, "
            f"{int(menor['clientes'])} clientes)."
        )

    else:
        st.warning(
            "Ningún segmento alcanza el mínimo exploratorio "
            "de 30 clientes."
        )

    segmentos_pequenos = resumen[
        resumen["clientes"] < 30
    ]

    if not segmentos_pequenos.empty:
        st.warning(
            "Existen segmentos con menos de 30 clientes. "
            "Sus tasas deben interpretarse con cautela "
            "por su reducido tamaño."
        )

        st.dataframe(
            segmentos_pequenos[
                [variable, "clientes", "bad_rate"]
            ].rename(columns={
                variable: "Segmento",
                "clientes": "Clientes",
                "bad_rate": "Bad Rate (%)"
            }).style.format({
                "Bad Rate (%)": "{:.2f}%"
            }),
            hide_index=True,
            use_container_width=True
        )

    st.info(
        "Los segmentos y sus tasas ayudan a identificar "
        "patrones de riesgo, pero no determinan por sí solos "
        "qué clientes incumplirán. La siguiente fase utilizará "
        "un modelo predictivo para combinar la información "
        "de diferentes variables."
    )

    st.caption(
        "Análisis exploratorio con datos sintéticos. "
        "Los cortes de segmentación y el mínimo de 30 clientes "
        "son criterios ilustrativos, no límites regulatorios."
    )
