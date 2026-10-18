# Credit Risk Analytics

### Behavioral Model Personal Loans Portfolio | Streamlit

**F05 Toma de Decisiones con Datos — Ciencia de Datos**  
Maestría en Big Data e Inteligencia Artificial  
Universidad Anáhuac Puebla

## Descripción del proyecto

Credit Risk Analytics es una aplicación interactiva desarrollada en Python y Streamlit para analizar una cartera sintética de préstamos personales activos, estimar la probabilidad de incumplimiento crediticio durante los siguientes 12 meses y proponer acciones preventivas de gestión de cartera.

El proyecto utiliza un enfoque de **Behavioral Probability of Default (PD)**, centrado en clientes con créditos existentes y no en la evaluación de nuevas solicitudes.

## Pregunta guía

¿Qué clientes con préstamos personales activos podrían alcanzar 90 días o más de atraso durante los próximos 12 meses y qué seguimiento preventivo conviene priorizar?

## Dataset

El proyecto utiliza datos 100% sintéticos, generados mediante `generate_dataset.py`.

- 1,000 registros originales.
- 17 variables financieras, contractuales y crediticias.
- 10 registros duplicados incorporados para el ejercicio de limpieza.
- 29 valores faltantes.
- 990 clientes únicos después de la depuración.
- Variable objetivo: `default_12m` (0 = sin incumplimiento; 1 = incumplimiento).

El dataset está disponible en `data/credit_risk_customers.csv` y no contiene información personal real.

## Metodología

La aplicación sigue el framework de Ciencia de Datos utilizado en clase:

**Entender → Limpiar → Describir → Predecir → Prescribir → Decidir**

### 1. Entender los datos
Exploración del dataset, tipos de datos, diccionario de variables y diagnóstico inicial de calidad.

### 2. Limpiar los datos
Identificación y eliminación de duplicados, tratamiento de valores faltantes y validaciones de consistencia.

### 3. Estadística descriptiva
Análisis de indicadores de cartera, Bad Rate y comportamiento del incumplimiento por segmentos.

### 4. Estadística predictiva
Desarrollo de un modelo de regresión logística para estimar la probabilidad de incumplimiento a 12 meses.

Se comparan modelos de 15 y 13 variables mediante validación cruzada estratificada. La evaluación final se realiza con una muestra de prueba independiente del entrenamiento y la selección.

Las métricas incluyen AUC-ROC, Gini, KS, Precision, Recall, F1 y Brier Score.

### 5. Estadística prescriptiva
Clasificación de los 990 clientes en tres segmentos relativos mediante percentiles de las probabilidades estimadas:

- Riesgo Bajo.
- Riesgo Medio.
- Riesgo Alto.

Se proponen acciones diferenciadas de monitoreo habitual, seguimiento preventivo y revisión individual, respectivamente.

### 6. Toma de decisiones
Interfaz interactiva que permite consultar recomendaciones por segmento o cliente, incluyendo:

- Probabilidad estimada de incumplimiento.
- Segmento asignado.
- Acción interna recomendada.
- Objetivo de seguimiento.
- Mensaje sugerido al cliente.
- Justificación de la recomendación.

## Instalación y ejecución

**1. Clonar el repositorio**

```bash
git clone https://github.com/TU_USUARIO/credit-risk-streamlit.git
cd credit-risk-streamlit
```

Sustituir `TU_USUARIO` por el usuario correspondiente de GitHub.

**2. Crear el entorno virtual**

```bash
python -m venv credit_risk_env
```

**3. Activar el entorno**

En macOS o Linux:

```bash
source credit_risk_env/bin/activate
```

En Windows (PowerShell):

```powershell
credit_risk_env\Scripts\Activate.ps1
```

**4. Instalar dependencias**

```bash
python -m pip install -r requirements.txt
```

**5. Ejecutar la aplicación**

```bash
streamlit run app.py
```

La aplicación estará disponible normalmente en:

`http://localhost:8501`

## Archivos principales

| Archivo o carpeta | Descripción |
|---|---|
| `app.py` | Archivo principal de la aplicación Streamlit |
| `tabs/` | Módulos correspondientes a las seis etapas del framework |
| `data/credit_risk_customers.csv` | Dataset sintético |
| `generate_dataset.py` | Código para generar el dataset |
| `project.md` | Contexto y especificaciones del proyecto |
| `requirements.txt` | Dependencias de Python |
| `.gitignore` | Archivos y carpetas excluidos del repositorio |

## Limitaciones

Este ejercicio tiene fines exclusivamente académicos y utiliza datos sintéticos.

La evaluación del modelo emplea una división aleatoria de entrenamiento y prueba; no incluye una validación fuera de tiempo (OOT).

El análisis prescriptivo de la cartera completa incluye predicciones realizadas sobre registros utilizados en el entrenamiento (*in-sample*) y registros de prueba (*out-of-sample*).

Los segmentos representan niveles relativos de riesgo dentro de la cartera simulada y no deben interpretarse como clasificaciones regulatorias.

Antes de cualquier aplicación real serían necesarias validaciones adicionales de calidad de datos, estabilidad, calibración, sesgos y privacidad.

Las acciones propuestas requieren revisión humana y no deben utilizarse para tomar decisiones crediticias adversas de manera automática.

## Autor

Miguel Morales

**Actividad F05 — Streamlit: Toma de decisiones con datos**