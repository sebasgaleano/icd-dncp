# Pipeline Integral de Ciencia de Datos
## Análisis, Modelado y Evaluación Aplicados
### Adjudicaciones y Contratos — DNCP Paraguay (2021–2022)

---

**Universidad Comunera — Maestría en Ciencia de Datos**  
**Dataset:** Portal de Datos Abiertos DNCP — https://www.contrataciones.gov.py/datos   
**Período analizado:** 2021–2022  
**Herramientas:** Python 3.14 | pandas | scikit-learn | matplotlib | seaborn

---

## Introducción

Este trabajo implementa un pipeline completo de ciencia de datos sobre el dataset
de adjudicaciones y contratos del Estado paraguayo, publicado por la Dirección
Nacional de Contrataciones Públicas (DNCP). El objetivo es demostrar dominio
integral del flujo de trabajo — desde la exploración y limpieza de datos crudos
hasta la construcción, evaluación y comunicación de modelos predictivos y descriptivos.

El dataset cubre todos los procesos de contratación pública formalizados entre
2021 y 2022, incluyendo información sobre instituciones convocantes, montos
adjudicados, tipos de procedimiento, proveedores y categorías de bienes y servicios.

---

## Etapa 1 — Carga, Exploración Inicial y Limpieza de Datos

### Fuentes y estructura

Se trabajó con cuatro archivos CSV descargados del portal DNCP:

| Archivo | Filas | Columnas | Descripción |
|---|---|---|---|
| adjudicaciones_2021.csv | 8.528 | 26 | Procesos resueltos 2021 |
| adjudicaciones_2022.csv | 9.165 | 26 | Procesos resueltos 2022 |
| contratos_2021.csv | 12.468 | 28 | Contratos firmados 2021 |
| contratos_2022.csv | 13.284 | 28 | Contratos firmados 2022 |

La tabla de adjudicaciones registra la *decisión* — a quién se adjudica un proceso
de compra. La tabla de contratos registra la *formalización* — el contrato firmado
con cada proveedor. La relación entre ambas es **1:N**: un proceso puede generar
múltiples contratos con distintos proveedores.

### Consolidación

Los archivos por año se apilaron verticalmente con `pd.concat` previa verificación
de columnas idénticas. Luego se realizó un **left join** por `id_llamado` con
adjudicaciones como tabla principal, conservando todos los procesos aunque no
tuvieran contrato firmado.

Antes del join se verificó la cardinalidad de la llave:

- id_llamado únicos en adjudicaciones: 17.681
- id_llamado únicos en contratos: 17.688
- Llamados sin contrato firmado: 343 (1.9%)

El dataset consolidado resultó en **25.590 filas × 31 columnas**.

### Diagnóstico de calidad

| Columna | Nulos | % |
|---|---|---|
| organismo_financiador | 24.916 | 97.4% |
| organismo_financiador_id | 24.916 | 97.4% |
| vigencia_contrato | 22.841 | 89.3% |
| observaciones | 17.174 | 67.1% |
| restricciones | 1.809 | 7.1% |
| fecha_firma_contrato | 881 | 3.4% |
| monto_adjudicado | 343 | 1.3% |

Se identificaron además:
- **59 filas exactamente duplicadas** — errores de carga
- **84 registros con monto_total_adjudicado = 0** — distribuidos sin patrón
  en todos los tipos de procedimiento

### Decisiones de limpieza

| Decisión | Criterio | Impacto |
|---|---|---|
| Drop 4 columnas | >50% nulos, no recuperables | -4 columnas |
| Drop duplicados exactos | Errores de carga | -59 filas |
| Drop montos en 0 | Inmodelables como target | -84 filas |
| Conversión a datetime | Operaciones temporales en EDA | 2 columnas |
| Exclusión moneda USD | 91 registros (0.4%), evita conversión histórica | -91 filas |

**Dataset final: 25.360 filas × 27 columnas**

### Feature engineering

La variable `num_oferentes` no existe en el dataset original. Se construyó
contando el número de RUC distintos adjudicados por `id_llamado`:

```python
oferentes_por_llamado = df.groupby("id_llamado")["ruc"].nunique()
```

**Limitación metodológica:** esta variable mide el número de *adjudicados*,
no de *oferentes* reales. En una licitación pueden haberse presentado más
empresas de las que finalmente se adjudicaron.

A partir de `num_oferentes` se construyó el target de clasificación:

- `competencia = 0` — único proveedor adjudicado (57.4%)
- `competencia = 1` — múltiples proveedores adjudicados (42.6%)

---

## Etapa 2 — Análisis Exploratorio de Datos (EDA)

### Distribución por tipo de procedimiento

| Tipo | Cantidad | % |
|---|---|---|
| Contratación Directa | 11.899 | 46.9% |
| Licitación Pública Nacional | 5.611 | 22.1% |
| Concurso de Ofertas | 4.877 | 19.2% |
| Contratación por Excepción | 1.413 | 5.6% |
| Locación de Inmuebles | 936 | 3.7% |
| Otros | 624 | 2.5% |

La **Contratación Directa** representa casi la mitad de todos los procesos —
el mecanismo que por definición evita la competencia es el más utilizado
por el Estado paraguayo.

### Instituciones con mayor actividad contractual

Las 5 instituciones más activas concentran el 20% de todos los procesos:

1. Ministerio de Salud Pública y Bienestar Social — 2.246 procesos
2. Instituto de Previsión Social — 1.034 procesos
3. Administración Nacional de Electricidad — 610 procesos
4. Policía Nacional — 585 procesos
5. Facultad de Ciencias Médicas / UNA — 462 procesos

El dataset cubre 408 instituciones convocantes distintas, con una distribución
muy concentrada — las 20 primeras explican más del 50% del total de procesos.

### Distribución de montos por tipo de procedimiento

| Tipo | Mediana (Gs) | Media (Gs) |
|---|---|---|
| Contratación Directa | 75.000.000 | 82.708.602 |
| Concurso de Ofertas | 359.158.070 | 407.403.247 |
| Locación de Inmuebles | 378.000.000 | 1.259.191.660 |
| Licitación Pública Nacional | 4.566.939.200 | 44.409.842.901 |
| Contratación por Excepción | 772.140.000 | 10.205.527.503 |

La brecha entre media y mediana en todos los tipos confirma distribuciones
fuertemente sesgadas. La **Licitación Pública Nacional** opera en rangos de
monto 60 veces superiores a la Contratación Directa — son instrumentos
diseñados para escalas completamente distintas.

La **Contratación por Excepción** presenta la mayor dispersión relativa —
casos que van desde 3M hasta 142.000M Gs en un procedimiento que evita
la competencia. Señal relevante para análisis de transparencia.

### Evolución temporal

El volumen de contrataciones sigue un patrón estacional consistente:

- **Noviembre** es el mes pico en ambos años (2.480 en 2021, 3.198 en 2022)
- **Enero** registra el volumen mínimo (166 en 2021, 111 en 2022)
- 2022 supera consistentemente a 2021 en los meses centrales

La concentración de noviembre responde al cierre del ejercicio presupuestario —
las instituciones ejecutan el presupuesto restante antes de fin de año. Las
contrataciones de noviembre tienen menor tiempo de revisión y control, lo
que representa un riesgo desde la perspectiva de transparencia pública.

---

## Etapa 3 — Clasificación Supervisada

**Pregunta:** ¿El proceso tuvo un único oferente (sin competencia) o múltiples?  
**Target:** `competencia` — variable binaria (0/1)

### Preparación

Se excluyeron los 342 registros con `num_oferentes == 0` (llamados sin match
en contratos). Dataset de modelado: **25.018 registros**.

Split 80/20 con estratificación:
- Train: 20.014 registros
- Test: 5.004 registros
- Proporción preservada en ambos sets: 57.4% / 42.6%

**Features utilizadas:**

| Feature | Tipo |
|---|---|
| tipo_proc_agrupado | Categórica (LabelEncoder) |
| categoria_corta | Categórica (LabelEncoder) |
| monto_total_adjudicado | Numérica |
| monto_periodo | Numérica |

### Resultados

| Modelo | Accuracy | F1 (clase 0) | F1 (clase 1) |
|---|---|---|---|
| Regresión Logística | 0.43 | 0.00 | 0.60 |
| Random Forest | 0.90 | 0.91 | 0.89 |

La Regresión Logística falló completamente — predijo siempre clase 1,
incapaz de separar las clases con relaciones lineales. El Random Forest
capturó las relaciones no lineales y alcanzó 90% de accuracy con F1
balanceado entre clases.

### Matriz de confusión — Random Forest

|  | Predicho: Sin competencia | Predicho: Con competencia |
|---|---|---|
| **Real: Sin competencia** | 2.493 ✓ | 377 ✗ |
| **Real: Con competencia** | 114 ✗ | 2.020 ✓ |

El modelo comete más errores clasificando procesos con competencia como
sin competencia (114 Falsos Negativos) que al revés (377 Falsos Positivos).
Tiende a subestimar la competencia real, no a sobreestimarla.

### Importancia de variables

| Variable | Importancia |
|---|---|
| monto_total_adjudicado | 0.390 |
| monto_periodo | 0.331 |
| categoria_corta | 0.211 |
| tipo_proc_agrupado | 0.067 |

El monto explica el 72% del poder predictivo combinado con `monto_periodo`.
El tipo de procedimiento tiene el menor peso — la presencia o ausencia de
competencia está determinada principalmente por el valor del contrato y
la categoría del bien o servicio, no por la modalidad legal elegida.

---

## Etapa 4 — Regresión Supervisada

**Pregunta:** ¿Cuánto se adjudica en un contrato dado?  
**Target:** `monto_total_adjudicado`

### Transformación del target

La distribución original presentaba skewness de 9.12 — extremadamente sesgada.
Se aplicó transformación logarítmica (`log1p`) que redujo el skewness a 0.89,
produciendo una distribución aproximadamente normal apta para modelado.

**Features utilizadas:**

| Feature | Tipo |
|---|---|
| tipo_proc_agrupado | Categórica (LabelEncoder) |
| categoria_corta | Categórica (LabelEncoder) |
| monto_periodo | Numérica |

*Nota: `monto_total_adjudicado` excluida por ser el target.*

### Resultados

| Modelo | RMSE (log) | R² |
|---|---|---|
| Regresión Lineal | 1.8907 | 0.3409 |
| Random Forest | 0.5223 | 0.9497 |

El Random Forest explica el 95% de la varianza del target transformado.
La Regresión Lineal alcanza solo el 34% — las relaciones entre features
y monto son fundamentalmente no lineales.

### Interpretación

El modelo predice con mayor precisión en el rango intercuartil del dataset
(75M — 1.051M Gs). Los contratos de valor extremo presentan mayor error
de predicción, lo cual es esperable dado que representan casos atípicos
con características únicas.

**Limitación:** el R² de 0.95 se reporta en escala logarítmica. En escala
original el error absoluto puede ser considerable dado el rango de valores
(124.950 Gs a 880.093M Gs).

---

## Etapa 5 — Clustering

**Pregunta:** ¿Qué perfiles de instituciones compradoras existen?  
**Unidad de análisis:** institución convocante (398 instituciones)

### Variables de segmentación

| Variable | Descripción |
|---|---|
| monto_promedio | Media del monto adjudicado por institución |
| frecuencia | Cantidad de procesos en el período |
| prop_directa | Proporción de contrataciones directas |
| diversidad_proveedores | Cantidad de RUC distintos |

Se aplicó `StandardScaler` antes del clustering — obligatorio dado los
rangos muy distintos entre variables.

### Determinación de k

El método del codo indicó un quiebre en k=5. El coeficiente de silhouette
para k=5 fue de 0.46 — aceptable considerando la heterogeneidad del dataset.
Se descartó k=2 (silhouette 0.87) por producir solo dos grupos, insuficientes
para caracterizar perfiles institucionales diversos.

### Perfiles identificados

| Cluster | n | Monto prom. (M Gs) | Frecuencia | Prop. directa | Diversidad |
|---|---|---|---|---|---|
| 0 — Competitivos medios | 144 | 1.131 | 39 | 34% | 25 |
| 1 — Grandes compradores | 28 | 4.858 | 248 | 33% | 136 |
| 2 — Pequeños directos | 222 | 351 | 38 | 81% | 21 |
| 3 — Outlier: Salud | 1 | 39.714 | 2.206 | 22% | 497 |
| 4 — Mega licitadores | 3 | 77.266 | 599 | 9% | 255 |

**Cluster 0 — Competitivos medios (144 instituciones)**  
Volumen e frecuencia moderados, proporción de contratación directa baja (34%).
Instituciones que operan con procesos competitivos a escala intermedia.

**Cluster 1 — Grandes compradores (28 instituciones)**  
Alta frecuencia, montos altos, baja contratación directa (33%), alta diversidad
de proveedores. El grupo más activo en licitaciones competitivas de alto volumen.

**Cluster 2 — Pequeños directos (222 instituciones)**  
El cluster más numeroso — 56% de todas las instituciones. Montos bajos y
altísima proporción de contratación directa (81%). Patrón que merece atención
desde la perspectiva de transparencia pública.

**Cluster 3 — Outlier: Ministerio de Salud (1 institución)**  
Caso único en el dataset. 2.206 procesos, 497 proveedores distintos, monto
promedio de 39.714M Gs. Su volumen lo separa completamente del resto del
Estado — cualquier análisis agregado debe considerarlo por separado.

**Cluster 4 — Mega licitadores: ANDE, IPS, MOPC (3 instituciones)**  
Contratos de valor extremo (77.266M Gs promedio), alta diversidad de
proveedores y la proporción más baja de contratación directa del dataset (9%).
Las instituciones más competitivas y transparentes en términos de modalidad
de contratación.

---

## Conclusiones

El análisis del dataset de contrataciones públicas de Paraguay 2021–2022
revela patrones estructurales relevantes:

**Sobre el sistema de contrataciones:**
- El 47% de los procesos son Contrataciones Directas — el mecanismo que
  evita la competencia es el más utilizado por el Estado.
- El 56% de las instituciones operan con más del 80% de contratación directa,
  sugiriendo una preferencia sistemática por eludir los procesos competitivos.
- La concentración de contrataciones en noviembre representa un riesgo de
  control — el 12% de todos los procesos anuales se concentran en un solo mes.

**Sobre los modelos:**
- El Random Forest superó a los modelos lineales en clasificación (F1=0.90)
  y regresión (R²=0.95), confirmando que las relaciones en datos de
  contratación pública son fundamentalmente no lineales.
- El monto del contrato es el predictor más importante tanto para predecir
  competencia como para predecir el monto mismo — el tamaño del contrato
  determina su naturaleza.

**Limitaciones:**
- `num_oferentes` es un proxy construido desde adjudicados, no oferentes reales
- Los montos están en guaraníes corrientes sin ajuste por inflación
- El período de dos años limita el análisis de tendencias de largo plazo
- El clustering sobre promedios institucionales puede distorsionar el perfil
  de instituciones con pocos contratos

---

