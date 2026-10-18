
import streamlit as st
import pandas as pd


# ----------------------------------
# FUNCION PARA PREPARAR LOS DATOS
# ----------------------------------

def preparar(df):
    """
    Crea una copia limpia para el análisis descriptivo.
    Nunca modifica el dataframe original.
    """

    df_limpio = df.copy()

    # Eliminar filas completamente duplicadas
    df_limpio = df_limpio.drop_duplicates().copy()

    # Imputar variables numéricas con la mediana
    for variable in ["monthly_income", "credit_utilization"]:
        mediana = df_limpio[variable].median()
        df_limpio[variable] = df_limpio[variable].fillna(mediana)

    # Conservar los faltantes categóricos como categoría
    df_limpio["employment_type"] = (
        df_limpio["employment_type"].fillna("No informado")
    )

    return df_limpio


# ----------------------------------
# FUNCION PARA MOSTRAR LA PESTAÑA
# ----------------------------------

def mostrar(df, df_limpio):

    st.header("2. Limpiar los datos")

    st.markdown("""
    ### ¿Qué tan confiable es nuestra información crediticia?

    Antes de realizar análisis estadísticos o construir un modelo
    predictivo, necesitamos revisar la calidad de los datos.

    Identificaremos registros duplicados, valores faltantes
    y posibles inconsistencias, documentando las decisiones
    utilizadas para preparar nuestra información.
    """)

    # ----------------------------------
    # DIAGNOSTICO INICIAL
    # ----------------------------------

    st.subheader("Diagnóstico inicial de calidad")

    duplicados = int(df.duplicated().sum())
    faltantes = int(df.isnull().sum().sum())

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Registros originales", len(df))

    with col2:
        st.metric("Duplicados exactos", duplicados)

    with col3:
        st.metric("Valores faltantes", faltantes)

    st.subheader("Valores faltantes por variable")

    tabla_nulos = pd.DataFrame({
        "Variable": df.columns,
        "Valores faltantes": df.isnull().sum().values,
        "Porcentaje (%)": (
            df.isnull().mean().values * 100
        ).round(2)
    })

    tabla_nulos = (
        tabla_nulos[
            tabla_nulos["Valores faltantes"] > 0
        ]
        .sort_values(
            "Valores faltantes",
            ascending=False
        )
    )

    if tabla_nulos.empty:
        st.success("No se identificaron valores faltantes.")
    else:
        st.dataframe(
            tabla_nulos,
            hide_index=True,
            use_container_width=True
        )

        st.bar_chart(
            tabla_nulos.set_index("Variable")["Valores faltantes"]
        )

    # ----------------------------------
    # DECISIONES DE LIMPIEZA
    # ----------------------------------

    st.subheader("Decisiones de tratamiento")

    decisiones = pd.DataFrame([
        [
            "Registros duplicados",
            "Eliminar copias exactas",
            "Evita que un mismo registro tenga mayor peso."
        ],
        [
            "monthly_income",
            "Imputar mediana",
            "Es robusta frente a ingresos extremos."
        ],
        [
            "credit_utilization",
            "Imputar mediana",
            "Conserva los registros sin inventar valores extremos."
        ],
        [
            "employment_type",
            "Crea categoría 'No informado'",
            "Evita asumir un tipo de empleo desconocido."
        ]
    ], columns=[
        "Problema",
        "Tratamiento",
        "Justificación"
    ])

    st.dataframe(
        decisiones,
        hide_index=True,
        use_container_width=True
    )

    st.info(
        "La limpieza se aplica sobre una copia del dataset. "
        "El archivo CSV original permanece sin modificaciones."
    )

    # ----------------------------------
    # VALIDACIONES DE CONSISTENCIA
    # ----------------------------------

    st.subheader("Validaciones de consistencia")

    validaciones = {
        "Edad fuera de 18 a 100 años":
            (~df_limpio["age"].between(18, 100)).sum(),

        "Ingresos menores o iguales a cero":
            (df_limpio["monthly_income"] <= 0).sum(),

        "DTI fuera del rango 0 a 1":
            (~df_limpio["dti_ratio"].between(0, 1)).sum(),

        "Utilización fuera del rango 0 a 1":
            (~df_limpio["credit_utilization"].between(0, 1)).sum(),

        "Consultas 3m mayores que consultas 6m":
            (
                df_limpio["bureau_inquiries_3m"] >
                df_limpio["bureau_inquiries_6m"]
            ).sum(),

        "Antigüedad del préstamo mayor al historial":
            (
                df_limpio["months_on_book"] >
                df_limpio["credit_history_months"]
            ).sum(),

        "Worst MOP fuera del rango 0 a 4":
            (~df_limpio["worst_mop_12m"].between(0, 4)).sum(),

        "Variable objetivo distinta de 0 o 1":
            (~df_limpio["default_12m"].isin([0, 1])).sum()
    }

    tabla_validaciones = pd.DataFrame(
        list(validaciones.items()),
        columns=["Regla de consistencia", "Incumplimientos"]
    )

    st.dataframe(
        tabla_validaciones,
        hide_index=True,
        use_container_width=True
    )

    if tabla_validaciones["Incumplimientos"].sum() == 0:
        st.success(
            "No se detectaron inconsistencias en las reglas evaluadas."
        )
    else:
        st.warning(
            "Se identificaron valores que requieren revisión adicional."
        )

    # ----------------------------------
    # RESULTADOS DE LIMPIEZA
    # ----------------------------------

    st.subheader("Resultados después de la limpieza")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Registros finales",
            len(df_limpio),
            delta=len(df_limpio) - len(df),
            delta_color="off"
        )

    with col2:
        st.metric(
            "Duplicados restantes",
            int(df_limpio.duplicated().sum())
        )

    with col3:
        st.metric(
            "Nulos restantes",
            int(df_limpio.isnull().sum().sum())
        )

    st.subheader("Comparación antes y después")

    comparacion = pd.DataFrame({
        "Indicador": [
            "Registros",
            "Duplicados exactos",
            "Valores faltantes"
        ],
        "Antes": [
            len(df),
            duplicados,
            faltantes
        ],
        "Después": [
            len(df_limpio),
            int(df_limpio.duplicated().sum()),
            int(df_limpio.isnull().sum().sum())
        ]
    })

    st.dataframe(
        comparacion,
        hide_index=True,
        use_container_width=True
    )

    st.markdown("""
    **Interpretación:** La limpieza busca mejorar la calidad
    de los datos sin perder observaciones innecesariamente.
    Los registros duplicados se eliminan y los faltantes
    se tratan de acuerdo con el tipo de variable.
    """)

    st.caption(
        "Nota metodológica: esta imputación sirve para el "
        "análisis descriptivo. En la fase predictiva, las "
        "medianas y transformaciones se aprenderán únicamente "
        "con los datos de entrenamiento mediante un Pipeline."
    )

    # ----------------------------------
    # VISTA PREVIA FINAL
    # ----------------------------------

    with st.expander("Ver muestra del dataset limpio"):
        st.dataframe(
            df_limpio.head(10),
            use_container_width=True
        )
