
import streamlit as st
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss
)


# ----------------------------------
# DIAGNOSTICO DE CORRELACIONES
# ----------------------------------

def diagnosticar_correlaciones(X_train):
    """
    Calcula las correlaciones entre variables numericas
    utilizando solamente la muestra de entrenamiento.
    """

    numericas = X_train.select_dtypes(
        include="number"
    )

    # Correlacion lineal de Pearson
    matriz = numericas.corr(method="pearson")

    # Obtener pares de variables sin repetir
    pares = []

    columnas = matriz.columns.tolist()

    for i in range(len(columnas)):
        for j in range(i + 1, len(columnas)):

            variable_1 = columnas[i]
            variable_2 = columnas[j]

            correlacion = matriz.loc[
                variable_1, variable_2
            ]

            if pd.notna(correlacion):
                pares.append({
                    "Variable 1": variable_1,
                    "Variable 2": variable_2,
                    "Correlacion": correlacion,
                    "Correlacion absoluta": abs(correlacion)
                })

    tabla_pares = pd.DataFrame(pares)

    tabla_pares = tabla_pares.sort_values(
        "Correlacion absoluta",
        ascending=False
    )

    return matriz, tabla_pares


# ----------------------------------
# CONSTRUIR PIPELINE DE MODELACION
# ----------------------------------

def construir_pipeline(X, variables):

    numericas = X[variables].select_dtypes(
        include="number"
    ).columns.tolist()

    categoricas = X[variables].select_dtypes(
        exclude="number"
    ).columns.tolist()

    proceso_numerico = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler())
    ])

    proceso_categorico = Pipeline([
        ("imputar", SimpleImputer(
            strategy="constant",
            fill_value="No informado"
        )),
        ("codificar", OneHotEncoder(
            handle_unknown="ignore"
        ))
    ])

    preprocesamiento = ColumnTransformer([
        ("numericas", proceso_numerico, numericas),
        ("categoricas", proceso_categorico, categoricas)
    ])

    modelo = Pipeline([
        ("preprocesamiento", preprocesamiento),
        ("regresion", LogisticRegression(
            max_iter=2000,
            random_state=42
        ))
    ])

    return modelo


# ----------------------------------
# ENTRENAR Y COMPARAR MODELOS
# ----------------------------------

def entrenar(df):

    # Mantener los datos originales sin imputacion
    datos = df.drop_duplicates().copy()

    X = datos.drop(
        columns=["customer_id", "default_12m"]
    )

    y = datos["default_12m"]

    # Separacion previa a cualquier ajuste
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    # Modelo completo: todas las variables
    variables_completas = X.columns.tolist()

    # Modelo reducido: excluir dos variables correlacionadas
    variables_reducidas = [
        variable for variable in variables_completas
        if variable not in [
            "loan_amount",
            "bureau_inquiries_6m"
        ]
    ]

    configuraciones = {
        "Completo": variables_completas,
        "Reducido": variables_reducidas
    }

    # Validacion cruzada solo sobre entrenamiento
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    comparacion = []

    for nombre, variables in configuraciones.items():

        pipeline = construir_pipeline(X_train, variables)

        auc_folds = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=1
        )

        comparacion.append({
            "Modelo": nombre,
            "Variables": len(variables),
            "AUC promedio": auc_folds.mean(),
            "AUC desviacion": auc_folds.std(),
            "Gini promedio": 2 * auc_folds.mean() - 1
        })

    tabla_comparacion = pd.DataFrame(comparacion)

    # Regla definida antes de evaluar la muestra de prueba
    auc_completo = tabla_comparacion.loc[
        tabla_comparacion["Modelo"] == "Completo",
        "AUC promedio"
    ].iloc[0]

    auc_reducido = tabla_comparacion.loc[
        tabla_comparacion["Modelo"] == "Reducido",
        "AUC promedio"
    ].iloc[0]

    tolerancia = 0.01

    if auc_reducido >= auc_completo - tolerancia:
        modelo_elegido = "Reducido"
    else:
        modelo_elegido = "Completo"

    variables_elegidas = configuraciones[modelo_elegido]

    # Ajustar el modelo elegido con todo el entrenamiento
    modelo = construir_pipeline(
        X_train,
        variables_elegidas
    )

    modelo.fit(X_train, y_train)

    # Evaluacion final en la muestra de prueba
    probabilidades = modelo.predict_proba(X_test)[:, 1]

    predicciones = (
        probabilidades >= 0.50
    ).astype(int)

    auc = roc_auc_score(y_test, probabilidades)
    gini = 2 * auc - 1

    fpr, tpr, thresholds = roc_curve(
        y_test,
        probabilidades
    )

    ks = max(abs(tpr - fpr))

    metricas = {
        "AUC": auc,
        "Gini": gini,
        "KS": ks,
        "Precision": precision_score(
            y_test, predicciones, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predicciones, zero_division=0
        ),
        "F1": f1_score(
            y_test, predicciones, zero_division=0
        ),
        "Brier": brier_score_loss(
            y_test, probabilidades
        )
    }

    # Coeficientes del modelo seleccionado
    nombres = modelo.named_steps[
        "preprocesamiento"
    ].get_feature_names_out()

    coeficientes = pd.DataFrame({
        "Variable": nombres,
        "Coeficiente": modelo.named_steps[
            "regresion"
        ].coef_[0]
    })

    coeficientes["Magnitud"] = (
        coeficientes["Coeficiente"].abs()
    )

    coeficientes = coeficientes.sort_values(
        "Magnitud",
        ascending=False
    )

    return {
        "modelo": modelo,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "probabilidades": probabilidades,
        "predicciones": predicciones,
        "metricas": metricas,
        "coeficientes": coeficientes,
        "comparacion": tabla_comparacion,
        "modelo_elegido": modelo_elegido,
        "variables_elegidas": variables_elegidas,
        "variables_completas": variables_completas,
        "tolerancia": tolerancia
    }


def generar_scoring_cartera(df, resultados):
    """Puntúa los registros únicos con el modelo final ya ajustado."""
    datos = df.drop_duplicates().copy()
    variables = resultados["variables_elegidas"]
    probabilidades = resultados["modelo"].predict_proba(
        datos[variables]
    )[:, 1]

    indices_prueba = set(resultados["X_test"].index)
    particiones = [
        "Prueba (out-of-sample)" if indice in indices_prueba
        else "Entrenamiento (in-sample)"
        for indice in datos.index
    ]

    return pd.DataFrame({
        "customer_id": datos["customer_id"].to_numpy(),
        "Probabilidad estimada": probabilidades,
        "default_12m": datos["default_12m"].to_numpy(),
        "Partición": particiones,
    }, index=datos.index)


# ----------------------------------
# MOSTRAR PESTAÑA PREDICTIVA
# ----------------------------------

def mostrar(df, resultados=None):

    st.header("4. Estadística predictiva")

    st.markdown("""
    ### Behavioral PD: ¿qué clientes con préstamos activos presentan mayor riesgo?

    Utilizamos una regresión logística como Behavioral Probability of Default (PD) Model
    para estimar la probabilidad de que un cliente alcance 90 días o más de atraso
    durante los 12 meses posteriores a la fecha de observación.

    El modelo combina información financiera, contractual y de comportamiento
    crediticio que, como supuesto académico, se considera disponible en esa fecha.
    El dataset sintético no permite verificar este orden temporal longitudinalmente.
    """)

    if resultados is None:
        resultados = entrenar(df)


    # ----------------------------------
    # DIAGNOSTICO DE VARIABLES
    # ----------------------------------

    st.subheader("Diagnóstico de variables predictoras")

    st.markdown("""
    Antes de realizar una selección de variables,
    analizamos posibles relaciones de dependencia
    entre los predictores numéricos.

    Este diagnóstico utiliza únicamente la muestra
    de entrenamiento para preservar la independencia
    de la muestra de prueba.
    """)

    X_train = resultados["X_train"]

    matriz_corr, pares_corr = diagnosticar_correlaciones(
        X_train
    )

    # Resumen de variables originales
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Predictores originales",
            X_train.shape[1]
        )

    with col2:
        st.metric(
            "Variables numéricas",
            len(X_train.select_dtypes(
                include="number"
            ).columns)
        )

    with col3:
        st.metric(
            "Variables categóricas",
            len(X_train.select_dtypes(
                exclude="number"
            ).columns)
        )

    # Matriz de correlacion
    st.markdown("**Matriz de correlación de Pearson**")

    st.dataframe(
        matriz_corr.style.format("{:.2f}").background_gradient(
            cmap="RdBu_r",
            vmin=-1,
            vmax=1
        ),
        use_container_width=True
    )

    st.caption(
        "Los valores cercanos a +1 representan asociación "
        "lineal positiva; los cercanos a -1, asociación "
        "negativa. Los cercanos a cero indican poca "
        "correlación lineal."
    )

    # Pares con mayor correlacion
    st.markdown("**Pares de variables más correlacionados**")

    st.dataframe(
        pares_corr.head(10).style.format({
            "Correlacion": "{:.3f}",
            "Correlacion absoluta": "{:.3f}"
        }),
        hide_index=True,
        use_container_width=True
    )

    # Identificar correlaciones elevadas
    umbral_corr = 0.70

    elevadas = pares_corr[
        pares_corr["Correlacion absoluta"] >= umbral_corr
    ]

    if not elevadas.empty:
        st.warning(
            f"Se identificaron {len(elevadas)} pares "
            f"con correlación absoluta igual o superior "
            f"a {umbral_corr:.2f}. "
            "Estos pares requieren revisión antes "
            "de seleccionar variables."
        )

        st.dataframe(
            elevadas.style.format({
                "Correlacion": "{:.3f}",
                "Correlacion absoluta": "{:.3f}"
            }),
            hide_index=True,
            use_container_width=True
        )
    else:
        st.success(
            "No se identificaron pares de variables "
            "con correlación absoluta de 0.70 o superior."
        )

    st.info(
        "La correlación elevada puede dificultar "
        "la interpretación de coeficientes. Sin embargo, "
        "no implica que debamos eliminar automáticamente "
        "alguna variable. La decisión se evaluará "
        "en el siguiente paso."
    )

    # ----------------------------------
    # VERIFICAR VARIABLES TRANSFORMADAS
    # ----------------------------------

    st.markdown("**Variables utilizadas después del preprocesamiento**")

    preprocesador = resultados["modelo"].named_steps[
        "preprocesamiento"
    ]

    nombres_transformados = (
        preprocesador.get_feature_names_out()
    )

    st.metric(
        "Predictores después de las transformaciones",
        len(nombres_transformados)
    )

    tabla_variables = pd.DataFrame({
        "Variable transformada": nombres_transformados,
        "Coeficiente": resultados["modelo"].named_steps[
            "regresion"
        ].coef_[0]
    })

    st.dataframe(
        tabla_variables.style.format({
            "Coeficiente": "{:.4f}"
        }),
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "Las variables categóricas generan columnas "
        "adicionales mediante One-Hot Encoding. "
        "La regresión inicial utiliza todas las columnas "
        "transformadas, aunque la tabla de coeficientes "
        "principales solo muestre algunas."
    )


    # ----------------------------------
    # COMPARACION Y SELECCION DE MODELOS
    # ----------------------------------

    st.subheader("Comparación y selección de modelos")

    st.markdown("""
    Después de identificar dos pares de variables con
    correlación elevada, comparamos un modelo completo
    contra uno reducido.

    La selección se realiza mediante validación cruzada
    estratificada de cinco particiones, utilizando
    exclusivamente la muestra de entrenamiento.
    """)

    tabla_cv = resultados["comparacion"]

    st.dataframe(
        tabla_cv.style.format({
            "AUC promedio": "{:.4f}",
            "AUC desviacion": "{:.4f}",
            "Gini promedio": "{:.2%}"
        }),
        hide_index=True,
        use_container_width=True
    )

    st.markdown("**AUC promedio por modelo**")

    st.bar_chart(
        tabla_cv.set_index("Modelo")["AUC promedio"],
        y_label="AUC promedio",
        x_label="Modelo",
        sort=False
    )

    st.markdown("**Criterio de selección**")

    st.write(
        "Se prefiere el modelo reducido cuando su AUC "
        "promedio está a no más de 0.01 del modelo "
        "completo. En caso contrario, se conserva "
        "el modelo completo."
    )

    st.success(
        f"Modelo seleccionado: "
        f"{resultados['modelo_elegido']}"
    )

    st.markdown("**Variables utilizadas en el modelo final**")

    st.dataframe(
        pd.DataFrame({
            "Variable": resultados["variables_elegidas"]
        }),
        hide_index=True,
        use_container_width=True
    )

    excluidas = [
        v for v in resultados["variables_completas"]
        if v not in resultados["variables_elegidas"]
    ]

    if excluidas:
        st.info(
            "Variables excluidas en el modelo elegido: "
            + ", ".join(excluidas)
        )
    else:
        st.info(
            "Se conservaron todas las variables originales."
        )

    st.caption(
        "La selección considera desempeño predictivo "
        "y simplicidad. No equivale a una validación "
        "regulatoria ni demuestra significancia estadística."
    )


    # ----------------------------------
    # 1. MUESTRAS DE DESARROLLO
    # ----------------------------------

    st.subheader("1. Preparación del modelo")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Clientes únicos",
            len(df.drop_duplicates())
        )

    with col2:
        st.metric(
            "Entrenamiento (70%)",
            len(resultados["X_train"])
        )

    with col3:
        st.metric(
            "Prueba (30%)",
            len(resultados["X_test"])
        )

    st.markdown("""
    Se realizó una división estratificada para mantener
    una proporción similar de clientes con incumplimiento
    en las muestras de entrenamiento y prueba.
    """)

    distribucion = pd.DataFrame({
        "Muestra": ["Entrenamiento", "Prueba"],
        "Clientes": [
            len(resultados["y_train"]),
            len(resultados["y_test"])
        ],
        "Bad Rate (%)": [
            resultados["y_train"].mean() * 100,
            resultados["y_test"].mean() * 100
        ]
    })

    st.dataframe(
        distribucion.style.format({
            "Bad Rate (%)": "{:.2f}%"
        }),
        hide_index=True,
        use_container_width=True
    )

    # ----------------------------------
    # 2. EVALUACION
    # ----------------------------------

    
    st.subheader("2. Evaluación del modelo")

    m = resultados["metricas"]

    st.markdown("**Capacidad discriminatoria**")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("AUC-ROC", f"{m['AUC']:.3f}")

    with col2:
        st.metric("Gini", f"{m['Gini'] * 100:.2f}%")

    with col3:
        st.metric("KS", f"{m['KS'] * 100:.2f}%")

    st.markdown("**Clasificación y calidad de probabilidades**")

    col4, col5, col6, col7 = st.columns(4)

    with col4:
        st.metric("Precision", f"{m['Precision']:.3f}")

    with col5:
        st.metric("Recall", f"{m['Recall']:.3f}")

    with col6:
        st.metric("F1 Score", f"{m['F1']:.3f}")

    with col7:
        st.metric("Brier Score", f"{m['Brier']:.3f}")


    st.caption(
        "Métricas calculadas en la muestra de prueba. "
        "Para clasificar incumplimiento se utiliza "
        "un umbral ilustrativo de probabilidad del 50%."
    )

    st.markdown("""
    **Cómo interpretar las métricas:**
    - **AUC-ROC:** capacidad de distinguir entre clientes
      con y sin incumplimiento. Un valor de 0.5 equivale
      al desempeño aleatorio.
    - **Gini:** medida de discriminación derivada del AUC.
      Un valor mayor indica mejor separación entre
      clientes con y sin incumplimiento.
    - **KS:** máxima diferencia entre las distribuciones
      acumuladas de buenos y malos, ordenadas por el score.
      Un valor mayor indica mayor separación.
    - **Precision:** proporción de incumplimientos reales
      entre los clientes clasificados como positivos.
    - **Recall:** porcentaje de incumplimientos reales
      que logra identificar el modelo.
    - **F1:** equilibrio entre Precision y Recall.
    - **Brier:** error cuadrático promedio de las
      probabilidades; valores menores son mejores.
    """)

    # ----------------------------------
    # 3. MATRIZ DE CONFUSION
    # ----------------------------------

    st.subheader("3. Matriz de confusión")

    matriz = confusion_matrix(
        resultados["y_test"],
        resultados["predicciones"],
        labels=[0, 1]
    )

    tabla_matriz = pd.DataFrame(
        matriz,
        index=["Real: No default", "Real: Default"],
        columns=[
            "Predicho: No default",
            "Predicho: Default"
        ]
    )

    st.dataframe(
        tabla_matriz,
        use_container_width=True
    )

    st.markdown("""
    Esta matriz permite identificar las predicciones
    correctas y los errores del modelo, especialmente
    los clientes que incumplieron pero no fueron
    identificados como de alto riesgo.
    """)

    # ----------------------------------
    # 4. VARIABLES DEL MODELO
    # ----------------------------------

    st.subheader("4. Variables del modelo")

    st.markdown("""
    La regresión logística utiliza coeficientes para
    relacionar las características de los clientes
    con la probabilidad estimada de incumplimiento.
    """)

    coeficientes = resultados["coeficientes"]

    st.dataframe(
        coeficientes[
            ["Variable", "Coeficiente"]
        ].head(15).style.format({
            "Coeficiente": "{:.4f}"
        }),
        hide_index=True,
        use_container_width=True
    )

    st.markdown("**Coeficientes de mayor magnitud**")

    st.bar_chart(
        coeficientes.head(10).set_index(
            "Variable"
        )["Coeficiente"],
        horizontal=True,
        x_label="Coeficiente logístico"
    )

    st.caption(
        "Los coeficientes indican asociaciones con "
        "los log-odds de incumplimiento. Las variables "
        "numéricas fueron estandarizadas y las categorías "
        "codificadas. Su magnitud no equivale directamente "
        "a causalidad ni a importancia predictiva independiente."
    )

    # ----------------------------------
    # 5. DECISION TECNICA - MOMENTO 4
    # ----------------------------------

    st.subheader("5. Decisión técnica: evitar Data Leakage")

    st.markdown("""
    Durante la limpieza descriptiva calculamos medianas
    sobre la base completa para examinar la cartera.

    Para entrenar el modelo, evitamos utilizar esos valores
    calculados con toda la información.

    **Solución aplicada:**

    1. Eliminar los registros duplicados.
    2. Separar las muestras de entrenamiento y prueba.
    3. Ajustar las imputaciones y el escalamiento
       únicamente con entrenamiento.
    4. Aplicar esas transformaciones a la muestra de
       prueba sin recalcular sus parámetros.
    5. Evaluar el modelo sobre datos no utilizados
       durante el ajuste.
    """)

    st.success(
        "El preprocesamiento y la regresión logística "
        "se encuentran integrados en un Pipeline."
    )

    with st.expander("Verificación del Pipeline"):
        pipeline = resultados["modelo"]

        st.write("**Etapas del modelo:**")
        st.write(list(pipeline.named_steps.keys()))

        st.write("**Predictores utilizados:**")
        st.write(resultados["X_train"].columns.tolist())

        st.write("**Verificaciones:**")

        st.write({
            "Customer ID excluido":
                "customer_id" not in
                resultados["X_train"].columns,

            "Variable objetivo excluida":
                "default_12m" not in
                resultados["X_train"].columns,

            "Muestra de prueba separada":
                resultados["X_train"].index.intersection(
                    resultados["X_test"].index
                ).empty
        })

    # ----------------------------------
    # 6. LIMITACIONES
    # ----------------------------------

    st.subheader("6. Interpretación y limitaciones")

    
    st.info("""
    **Alcance y limitaciones del modelo:**

    - Los datos son sintéticos; los resultados no representan
      el desempeño de un modelo bancario real.
    - La selección de variables se limita a comparar dos
      conjuntos predefinidos mediante validación cruzada.
    - No se realizó un análisis formal de p-values,
      WOE, IV o estabilidad temporal de variables.
    - La evaluación utiliza una división aleatoria,
      no una muestra fuera de tiempo (OOT).
    - No se realizó una validación exhaustiva de
      calibración, sesgos o estabilidad poblacional.
    - Los umbrales de clasificación son ilustrativos
      y necesitarían una justificación económica
      antes de utilizarse en decisiones reales.
    """)
