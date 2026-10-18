
import streamlit as st
import pandas as pd


def mostrar(df):

    # ----------------------------------
    # ENCABEZADO
    # ----------------------------------

    st.header("1. Entender los datos")

    st.markdown("""
    ### ¿Quiénes son nuestros clientes y qué información tenemos?

    Analizamos una cartera sintética de clientes con préstamos
    personales activos para conocer su perfil financiero,
    información contractual e historial de comportamiento crediticio.
    
    La unidad de análisis es el cliente, con un registro por cliente.
    El objetivo es estimar si alcanzará 90 días o más de atraso
    durante los 12 meses posteriores a la fecha de observación.
    """)

    # ----------------------------------
    # DIMENSIONES DEL DATASET
    # ----------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total de registros", len(df))

    with col2:
        st.metric("Número de variables", len(df.columns))

    with col3:
        st.metric(
            "Clientes únicos",
            df["customer_id"].nunique()
        )

    # ----------------------------------
    # VISTA PREVIA
    # ----------------------------------

    st.subheader("Vista previa del dataset")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    # ----------------------------------
    # DICCIONARIO DE VARIABLES
    # ----------------------------------

    st.subheader("Diccionario de datos")

    st.markdown("""
    El dataset contiene información del perfil financiero,
    características del préstamo e historial crediticio.

    Como supuesto académico, las variables explicativas corresponden
    a información disponible en la fecha de observación y
    `default_12m` representa el resultado de los 12 meses posteriores.
    El dataset sintético no tiene fechas ni estructura longitudinal
    para comprobar temporalmente este supuesto.
    """)

    diccionario = pd.DataFrame([
        ["customer_id", "Identificador", "Código anónimo del cliente.", "No aplica"],
        ["age", "Perfil", "Edad del cliente.", "Años"],
        ["monthly_income", "Perfil financiero", "Ingreso mensual del cliente.", "MXN"],
        ["employment_type", "Perfil", "Tipo de actividad laboral.", "Categoría"],
        ["loan_amount", "Préstamo", "Monto original del crédito.", "MXN"],
        ["loan_term", "Préstamo", "Plazo contractual del crédito.", "Meses"],
        ["interest_rate", "Préstamo", "Tasa de interés anual.", "%"],
        ["dti_ratio", "Perfil financiero", "Proporción del ingreso destinada al pago de deudas.", "Proporción 0-1"],
        ["credit_utilization", "Buró de Crédito", "Porcentaje utilizado del crédito revolvente disponible.", "Proporción 0-1"],
        ["previous_delinquencies", "Historial crediticio", "Número de atrasos históricos registrados.", "Conteo"],
        ["months_on_book", "Préstamo", "Antigüedad del préstamo actual.", "Meses"],
        ["bureau_inquiries_3m", "Buró de Crédito", "Consultas al Buró en los últimos 3 meses.", "Conteo"],
        ["bureau_inquiries_6m", "Buró de Crédito", "Consultas al Buró en los últimos 6 meses.", "Conteo"],
        ["revolving_accounts", "Buró de Crédito", "Número de cuentas revolventes.", "Conteo"],
        ["credit_history_months", "Historial crediticio", "Antigüedad desde el primer crédito.", "Meses"],
        ["worst_mop_12m", "Historial crediticio", "Peor nivel de morosidad en los últimos 12 meses, en escala sintética de 0 a 4.", "Ordinal"],
        ["default_12m", "Variable objetivo", "Indica incumplimiento de 90 días o más durante los siguientes 12 meses.", "0 = No / 1 = Sí"]
    ], columns=[
        "Variable",
        "Categoría",
        "Descripción",
        "Unidad / Valores"
    ])

    st.dataframe(
        diccionario,
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "Nota: los datos son sintéticos. La variable worst_mop_12m "
        "utiliza una escala simplificada y no corresponde "
        "a la codificación de Buró de Crédito."
    )

    # ----------------------------------
    # DIAGNÓSTICO
    # ----------------------------------

    st.subheader("Diagnóstico de variables")

    diagnostico = pd.DataFrame({
        "Variable": df.columns,
        "Tipo de dato": df.dtypes.astype(str).values,
        "Valores únicos": df.nunique().values,
        "Valores faltantes": df.isnull().sum().values
    })

    st.dataframe(
        diagnostico,
        use_container_width=True
    )

    # ----------------------------------
    # POSIBLES PROBLEMAS
    # ----------------------------------

    st.subheader("Problemas potenciales de calidad")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Registros duplicados",
            df.duplicated().sum()
        )

    with col2:
        st.metric(
            "Valores faltantes",
            df.isnull().sum().sum()
        )

    st.info(
        "En esta etapa solo identificamos posibles problemas. "
        "Las decisiones de tratamiento se realizarán en "
        "la pestaña 2: Limpiar los datos."
    )

    # ----------------------------------
    # EXPLORADOR DE VARIABLES
    # ----------------------------------

    st.subheader("Explorador de variables")

    variable = st.selectbox(
        "Selecciona una variable para explorar:",
        df.columns,
        key="variable_entender"
    )

    if pd.api.types.is_numeric_dtype(df[variable]):

        st.write("**Tipo:** Variable numérica")

        estadisticas = df[variable].describe()

        st.dataframe(
            estadisticas,
            use_container_width=True
        )

    else:

        st.write("**Tipo:** Variable categórica")

        frecuencias = (
            df[variable]
            .value_counts(dropna=False)
            .reset_index()
        )

        frecuencias.columns = [
            "Valor",
            "Frecuencia"
        ]

        st.dataframe(
            frecuencias,
            use_container_width=True
        )
