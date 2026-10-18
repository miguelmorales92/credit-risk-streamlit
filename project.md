# Guía del proyecto: Credit Risk Analytics con Streamlit

## 1. Propósito y contexto académico

Este proyecto corresponde a la actividad **F05 – Streamlit: Toma de decisiones con datos** de la materia Ciencia de Datos. Retoma como referencia la nota de clase «Marketing Analytics with Streamlit», pero adapta el problema al **riesgo de una cartera vigente de préstamos personales** mediante un **Behavioral Probability of Default (PD) Model**.

Se desarrollará una app educativa con seis pestañas para examinar datos, evaluar calidad, describir segmentos, estimar riesgo y recomendar acciones preventivas. **El objetivo es argumentar decisiones a partir de datos, no solo mostrar gráficas ni conseguir un modelo con buenas métricas.**

**Pregunta guía:** ¿Qué clientes con un préstamo personal activo tienen mayor probabilidad de presentar un incumplimiento crediticio en los próximos 12 meses y qué acciones preventivas de gestión de cartera conviene priorizar?

### Definición del Behavioral PD Model

- **Tipo:** Behavioral Probability of Default (PD) Model para monitoreo de cartera vigente.
- **Población:** clientes con préstamos personales activos en una fecha de observación.
- **Unidad de análisis:** cliente, con un registro por cliente.
- **Objetivo:** estimar la probabilidad de que un cliente alcance 90 días o más de atraso durante los 12 meses siguientes a la fecha de observación.
- **Variable objetivo:** `default_12m` (`1` = alcanzó 90 días o más de atraso en los 12 meses posteriores a la observación; `0` = no). Es una definición simplificada para fines académicos, no una definición regulatoria.
- **Predictores:** información financiera, contractual y de comportamiento crediticio que se supone conocida en la fecha de observación.
- **Uso de negocio:** segmentar clientes y priorizar monitoreo habitual, comunicación preventiva y revisión individual de créditos existentes. Los resultados apoyan la gestión de cartera y no automatizan decisiones crediticias adversas.
- **Supuesto académico temporal:** se supone que los predictores están disponibles en la fecha de observación y que `default_12m` refleja los 12 meses posteriores. El dataset es sintético y no tiene estructura longitudinal ni marcas de tiempo que permitan comprobar este orden temporal; por ello, no se puede descartar formalmente una fuga temporal ni confirmar el comportamiento en producción.

El conjunto de datos es **100 % sintético**, no procede de clientes reales ni de sistemas bancarios.

## 2. Principios de trabajo (adaptados de la nota del profesor)

1. Respetar el recorrido **Entender → Limpiar → Describir → Predecir → Prescribir → Decidir**.
2. Construir una pestaña a la vez. Separar el código en módulos pequeños dentro de `tabs/` y mantener `app.py` como coordinador.
3. Examinar antes de limpiar: documentar por qué un valor es nulo, duplicado o aparentemente extremo. Preservar el CSV original.
4. Distinguir estadísticas descriptivas, asociaciones y predicciones de las conclusiones operativas. Una asociación no implica causalidad.
5. Mantener legibles los comentarios y las etiquetas de la interfaz en **español**; los nombres de columnas continúan en inglés.
6. Utilizar los puntajes como **apoyo** a decisiones sujetas a revisión humana. No automatizar rechazos, restricciones o decisiones crediticias adversas.
7. Documentar suposiciones, limitaciones, fuentes de incertidumbre, privacidad y el carácter sintético de los resultados.
8. Si se usa IA para escribir código, identificarla en la evidencia 4 y explicar qué comprobó o corrigió el estudiante.

## 3. Estado actual y arquitectura deseada

La app Streamlit está implementada con `app.py` como coordinador y módulos en `tabs/`. Las seis pestañas tienen módulos propios. La pestaña 6 consulta recomendaciones por segmento y cliente usando las reglas y el scoring existentes. El CSV sintético existe y se han revisado sus dimensiones, distribución del objetivo y problemas de calidad.

```text
credit_risk_streamlit/
├── credit_risk_env/              # entorno local; NO subir a GitHub
├── data/
│   └── credit_risk_customers.csv # dataset sintético original
├── tabs/
│   ├── __init__.py
│   ├── entender.py
│   ├── limpiar.py
│   ├── descriptiva.py
│   ├── predictiva.py
│   ├── prescriptiva.py
│   └── decisiones.py              # consulta de acciones y mensajes por segmento/cliente
├── app.py                       # configuración, lectura y pestañas
├── requirements.txt
├── .gitignore
└── project.md                    # esta guía
```

Los módulos de pestaña exponen una función `mostrar(...)` que recibe los datos o resultados mediante argumentos explícitos. La pestaña 6 se integrará como módulo cuando se desarrolle. Por ejemplo:

```python
from tabs.entender import mostrar as mostrar_entender
# ...
with tab_entender:
    mostrar_entender(df)
```

Se evitará cargar el CSV repetidamente en cada pestaña. Es conveniente resolver la ruta del CSV respecto al archivo `app.py` usando `pathlib.Path(__file__).resolve().parent` para evitar fallos por directorio de ejecución.

## 4. Dataset verificado

**Ubicación:** `data/credit_risk_customers.csv`.

**Estado observado:**
- **1,000 filas** y **17 columnas**.
- **10 filas duplicadas exactas**; **990 identificadores de cliente únicos**.
- **154 filas con `default_12m = 1`** y **846 con `default_12m = 0`**: proporción de incumplimiento **15.4 % antes de la limpieza**. La tasa puede cambiar al eliminar duplicados.
- **12 nulos** en `monthly_income`, **9** en `employment_type` y **8** en `credit_utilization`. Los conteos incluyen las filas duplicadas; volver a medir después de depurar.
- Los datos son simulados y tienen relaciones artificiales entre variables: los resultados del modelo no acreditan capacidad predictiva sobre clientes reales.

### Diccionario de las 17 columnas

| Columna | Papel y significado propuesto |
|---|---|
| `customer_id` | Identificador ficticio del cliente; nunca incluirlo como predictor. |
| `age` | Edad en años. |
| `monthly_income` | Ingreso mensual simulado, MXN. |
| `employment_type` | Condición laboral (`asalariado`, `independiente`, `otro`). |
| `loan_amount` | Importe original del préstamo, MXN. |
| `loan_term` | Plazo contractual en meses. |
| `interest_rate` | Tasa de interés anual expresada en puntos porcentuales, p. ej. 24.51 = 24.51 %. |
| `dti_ratio` | Razón de pagos de deuda / ingreso; proporción de 0 a 1. |
| `credit_utilization` | Porcentaje de utilización del crédito revolvente, almacenado como proporción de 0 a 1. |
| `previous_delinquencies` | Conteo simulado de atrasos históricos anteriores a la observación. |
| `months_on_book` | Antigüedad del préstamo en meses al momento de observación. |
| `bureau_inquiries_3m` | Consultas al Buró en 3 meses previos a la observación. |
| `bureau_inquiries_6m` | Consultas al Buró en 6 meses previos a la observación. |
| `revolving_accounts` | Número de cuentas revolventes registradas. |
| `credit_history_months` | Meses desde el primer crédito hasta la fecha de observación. |
| `worst_mop_12m` | Severidad histórica simplificada de morosidad en 12 meses previos, escala simulada 0–4; NO es la codificación oficial MOP de Buró. |
| `default_12m` | Variable objetivo binaria: incumplimiento en 12 meses posteriores. |

**Reglas de plausibilidad importantes:**
- `bureau_inquiries_3m <= bureau_inquiries_6m`.
- `months_on_book <= credit_history_months` cuando ambas variables están definidas.
- Mantener consistencia de unidades y rangos: tasas porcentuales frente a proporciones.
- Todos los predictores deben medirse **antes** del periodo de resultado; en especial `worst_mop_12m`, `previous_delinquencies` y las consultas al Buró.
- Los nulos deliberados no equivalen automáticamente a ceros.
- La duplicación exacta debe investigarse y resolverse antes de crear los conjuntos de entrenamiento y prueba para evitar que el mismo cliente aparezca en ambos.

## 5. Preguntas del negocio y seis pestañas

### Pestaña 1. Entender los datos
**Pregunta:** ¿Quiénes son los clientes y qué información se tiene sobre su perfil financiero e historial crediticio?

**Contenido mínimo:** número de registros y variables con `st.metric`, primeras diez filas, diccionario/tipos, valores únicos y faltantes, número de duplicados y selector para explorar una variable. Para variables numéricas usar `describe()`; para categóricas, `value_counts(dropna=False)`. Explicar que el dataset es sintético y señalar dudas sobre unidades, definiciones y la muestra.

**En esta fase no limpiar ni concluir qué clientes son más riesgosos.**

### Pestaña 2. Limpiar los datos
**Pregunta:** ¿Qué tan confiable es la información que utilizaremos?

**Contenido:** conteo de nulos por columna, duplicados exactos, claves duplicadas, tipos, validación de reglas y revisión de extremos. Mostrar una tabla de hallazgos y justificaciones. Mantener `df_original` por separado; producir `df_limpio` sin mutar la fuente. Primero eliminar duplicados exactos; para nulos, considerar imputación ajustada solo en entrenamiento si la variable se utiliza en el modelo, sin fingir que se conocen los valores. Distinguir limpieza general de preprocesamiento específico del modelo.

### Pestaña 3. Estadística descriptiva
**Pregunta:** ¿Cómo se distribuye el incumplimiento y qué perfiles muestran diferencias observables?

**KPIs sugeridos:** clientes únicos, número y porcentaje con incumplimiento, ingreso mediano, DTI mediano. Visualizar tasa de incumplimiento por grupos de `worst_mop_12m`, consultas al Buró y rangos de DTI; informar cantidad de clientes y denominador en cada segmento. No interpretar diferencias como causalidad. Los segmentos descriptivos son propuestos para fines educativos y deben justificarse.

### Pestaña 4. Estadística predictiva
**Pregunta:** ¿Podemos estimar qué clientes tienen más probabilidad de incumplir dentro de 12 meses?

**Modelo previsto:** regresión logística de scikit-learn. Excluir `customer_id` y `default_12m` de los predictores. Separar entrenamiento/prueba (70/30, estratificado, semilla fija) **después** de resolver duplicados. Usar `ColumnTransformer`/`Pipeline` para imputación, escalamiento de numéricas y codificación de `employment_type` sin entrenar transformaciones con información de prueba. Reportar matriz de confusión, precision, recall, F1 y ROC-AUC cuando sean aplicables. Mostrar coeficientes con cuidado: el signo y la magnitud dependen de la codificación y escalamiento. Revisar correlación entre `bureau_inquiries_3m` y `bureau_inquiries_6m` y comprobar posibles fugas temporales.

**Alcance:** todo desempeño proviene de datos sintéticos. No afirmar validez bancaria ni desplegar el modelo para tomar decisiones reales.

### Pestaña 5. Estadística prescriptiva
**Pregunta:** ¿Qué acciones preventivas sugieren los resultados para cada segmento?

**Contenido:** tabla con segmento, rango de probabilidad, volumen, riesgo observado cuando proceda, acción, responsable y justificación. Ejemplo inicial **ilustrativo, no calibrado**: bajo <10 % → monitoreo habitual; medio 10–<25 % → comunicación preventiva; alto >=25 % → revisión individual prioritaria y, cuando corresponda, orientación de apoyo. Revisar umbrales después de observar puntuaciones, resultados y capacidad operativa. No asumir automáticamente que intervenir evita un incumplimiento.

### Pestaña 6. Toma de decisiones
**Pregunta:** ¿Qué acción preventiva conviene considerar para cada segmento o cliente y qué mensaje puede utilizar el equipo de gestión de cartera?

**Contenido implementado:** selectores para consultar cada segmento y cada cliente sintético; KPIs de cartera, acción interna, objetivo, mensaje sugerido y justificación cuantitativa. Los mensajes son ejemplos para revisión del equipo, no comunicaciones automáticas. No revelan al cliente la clasificación ni la probabilidad interna. Reconocer incertidumbre y necesidad de revisión humana.

### Cómo se conecta el recorrido de seis etapas con el caso Behavioral

| Etapa | Aplicación al Behavioral PD Model |
|---|---|
| **Entender** | Reconocer la población de clientes con préstamos activos, la unidad cliente y qué campos describen su situación en la fecha de observación. |
| **Limpiar** | Revisar la calidad de los registros de cartera y preparar una observación por cliente sin alterar el CSV fuente. |
| **Describir** | Examinar patrones históricos de `default_12m` y perfiles de comportamiento; las diferencias observadas no implican causalidad. |
| **Predecir** | Estimar PD a 12 meses con información financiera, contractual y de comportamiento disponible en la fecha de observación, y evaluar discriminación en la muestra de prueba. |
| **Prescribir** | Usar la distribución de PD para organizar segmentos de seguimiento y proponer monitoreo, comunicación preventiva o revisión humana de préstamos existentes. |
| **Decidir** | Orientar a responsables de cartera para priorizar y revisar acciones preventivas, documentando evidencia, incertidumbre y límites del supuesto temporal. |

En todas las etapas se asume, con fines académicos, que los predictores se conocen al momento de observación y que el resultado ocurre en los 12 meses posteriores. La estructura sintética disponible no permite verificarlo longitudinalmente.

## 6. Instalación y ejecución en macOS

Desde la terminal integrada de VS Code, en la ubicación donde se creará el proyecto:

```bash
mkdir credit_risk_streamlit
cd credit_risk_streamlit
python3 -m venv credit_risk_env
source credit_risk_env/bin/activate
python -m pip install streamlit pandas scikit-learn
python -m pip freeze > requirements.txt
streamlit run app.py
```

**Importante:** el último comando funcionará una vez que se haya creado `app.py`. Si ya existe la carpeta, no volver a crearla ni sobrescribir archivos existentes. Guardar el CSV en `data/` y trabajar siempre desde la raíz del proyecto. El prefijo `(credit_risk_env)` en terminal es una evidencia de entorno activo.

`.gitignore` mínimo:

```gitignore
credit_risk_env/
.env
.streamlit/secrets.toml
__pycache__/
*.pyc
.DS_Store
```

Nunca publicar datos de clientes reales, credenciales, tokens o archivos con secretos; si se publicara una clave accidentalmente hay que revocarla.

## 7. Siete evidencias exigidas por la actividad

Cada evidencia requiere **captura verificable + explicación de 80–120 palabras**. Entrega final: un solo PDF con portada (nombre completo), contexto en una línea y enlace al repositorio público de GitHub con `app.py` y CSV anonimizado/sintético.

1. **Entorno y estructura (10 pts):** terminal activada y árbol de carpetas/datos; explicar aislamiento de librerías.
2. **Primera ejecución (10 pts):** app con título propio, vista previa y URL local; justificar contexto, origen y pregunta.
3. **Entender (15 pts):** métricas, tipos y exploración; destacar variables clave y al menos una duda.
4. **Error o decisión técnica (15 pts):** antes/después, diagnóstico, solución y aprendizaje reutilizable; declarar y verificar ayuda de IA, si la hubo.
5. **Limpiar y describir (15 pts):** tratamiento de duplicados/nulos, KPI y gráfica por segmento; interpretar hallazgo.
6. **De los datos a la decisión (25 pts):** variables predictoras, segmentación, acción y mensaje con evidencia y límites.
7. **Reflexión y transferencia (10 pts):** al menos tres límites concretos: calidad, sesgo y privacidad; mitigación con datos reales. Incluir una modificación basada en la opinión real de un compañero que revise la pestaña 6.

Las evidencias deben obtenerse **durante** la construcción. No fabricar errores o capturas. El intercambio con un compañero se realizará realmente antes de la entrega; no declarar retroalimentación que aún no se ha recibido.

## 8. Recomendaciones para un agente de IA

1. Leer `project.md` antes de codificar y confirmar etapa y archivos existentes.
2. Inspeccionar el árbol y el contenido real antes de modificar archivos; la estructura prevista es una guía, no evidencia de implementación.
3. Proponer el **cambio mínimo útil** para la etapa actual. No crear las seis pestañas completas de golpe.
4. Entregar código completo por archivo cuando facilite copiar/pegar, señalando rutas exactas.
5. Conservar los datos originales y evitar editar el CSV para esconder problemas de calidad.
6. Priorizar imports explícitos, funciones como `mostrar(df)` y claves únicas en widgets interactivos.
7. Verificar sintaxis, importaciones y, cuando sea posible, ejecutar comprobaciones locales. No afirmar que Streamlit está funcionando sin haberlo probado.
8. Documentar decisiones para la rúbrica y explicar al estudiante lo que hace cada operación.
9. Alertar ante fuga de información, solapamiento entrenamiento/prueba, mal uso de unidades, interpretaciones causales o afirmaciones de exactitud excesivas.
10. Respetar el contexto: enfoque académico, préstamos personales, datos 100 % sintéticos y recomendaciones de apoyo sujetas a revisión humana.

## 9. Siguiente paso

Crear el entorno virtual y la estructura básica en VS Code; construir `app.py` y `tabs/entender.py` siguiendo la nota de clase. Obtener evidencias 1 y 2, verificar la pestaña Entender y guardar evidencia 3 **antes** de implementar limpieza.

---

*Adaptado de la guía de clase «Project guide: Marketing Analytics with Streamlit», suministrada por el profesor. Las definiciones de riesgo, las columnas y el enfoque de modelación son adaptaciones específicas del presente proyecto.*
