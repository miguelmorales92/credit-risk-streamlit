
from pathlib import Path

import streamlit as st
import pandas as pd

from tabs import entender, limpiar, descriptiva, predictiva, prescriptiva, decisiones


# ----------------------------------
# CONFIGURACIÓN GENERAL
# ----------------------------------

st.set_page_config(
    page_title="Credit Risk Analytics",
    page_icon="📊",
    layout="wide"
)


# ----------------------------------
# TÍTULO DE LA APLICACIÓN
# ----------------------------------

st.title("📊 Credit Risk Analytics")

st.markdown("""
**Análisis de comportamiento y gestión preventiva de cartera**

Aplicación para analizar una cartera sintética de préstamos personales activos,
estimar el riesgo de incumplimiento a 12 meses a partir del comportamiento
financiero y crediticio, y priorizar acciones preventivas para gestionar
los créditos existentes.

**Pregunta guía:** ¿Qué clientes con préstamos activos podrían alcanzar
90 días o más de atraso durante los próximos 12 meses y qué seguimiento
preventivo conviene priorizar?
""")

st.caption(
    "Proyecto académico | Datos 100% sintéticos | "
    "Maestría en Big Data e Inteligencia Artificial"
)


# ----------------------------------
# CARGAR DATASET
# ----------------------------------

RUTA_DATOS = Path(__file__).resolve().parent / "data" / "credit_risk_customers.csv"
df = pd.read_csv(RUTA_DATOS)

# Crear una copia preparada para el análisis descriptivo
df_limpio = limpiar.preparar(df)

@st.cache_resource(show_spinner="Preparando el análisis predictivo...")
def obtener_resultados_predictivos(datos):
    """Conserva el modelo y el scoring para reutilizarlos entre reruns."""
    resultados = predictiva.entrenar(datos)
    resultados["scoring_cartera"] = predictiva.generar_scoring_cartera(
        datos, resultados
    )
    return resultados


# El modelo final se entrena una vez; sus métricas y scoring se comparten.
resultados_predictivos = obtener_resultados_predictivos(df)

# ----------------------------------
# CREAR LAS SEIS PESTAÑAS
# ----------------------------------

tab_entender, tab_limpiar, tab_descriptiva, \
tab_predictiva, tab_prescriptiva, tab_decisiones = st.tabs([
    "1. Entender",
    "2. Limpiar",
    "3. Descriptiva",
    "4. Predictiva",
    "5. Prescriptiva",
    "6. Decisiones"
])


# ----------------------------------
# 1. ENTENDER LOS DATOS
# ----------------------------------

with tab_entender:
    entender.mostrar(df)


# ----------------------------------
# PESTAÑAS PENDIENTES
# ----------------------------------

with tab_limpiar:
    limpiar.mostrar(df, df_limpio)


with tab_descriptiva:
    descriptiva.mostrar(df_limpio)


with tab_predictiva:
    predictiva.mostrar(df, resultados=resultados_predictivos)


with tab_prescriptiva:
    prescriptiva.mostrar(resultados_predictivos)


with tab_decisiones:
    decisiones.mostrar(resultados_predictivos)

