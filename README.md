# Pipeline Integral de Ciencia de Datos
## Adjudicaciones y Contratos — DNCP Paraguay (2021–2022)

![Python](https://img.shields.io/badge/Python-3.14-blue)
![pandas](https://img.shields.io/badge/pandas-2.x-green)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.x-orange)

---

## Descripción

Pipeline completo de ciencia de datos sobre el dataset de contrataciones
públicas del Estado paraguayo, publicado por la Dirección Nacional de
Contrataciones Públicas (DNCP). Cubre desde la limpieza de datos crudos
hasta la construcción y evaluación de modelos predictivos y descriptivos.

**Trabajo Práctico  — Maestría en Ciencia de Datos**  
Universidad Comunera 

---

## Estructura del repositorio
    .
    ├── datos/
    │   ├── adjudicaciones/
    │   │   ├── 2021.csv
    │   │   └── 2022.csv
    │   ├── contratos/
    │   │   ├── 2021.csv
    │   │   └── 2022.csv
    │   ├── df_clean.csv
    │   ├── df_clusters.csv
    │   ├── df_eda.csv
    │   └── df_model.csv
    ├── notebooks/
    │   ├── 01-data_cleaning.ipynb
    │   ├── 02-EDA.ipynb
    │   ├── 03-clasificacion.ipynb
    │   ├── 04-regresion.ipynb
    │   └── 05-clustering.ipynb
    ├── figs/
    ├── reports/
    ├── requirements.txt
    └── README.md

---

## Dataset

| Fuente | DNCP — Portal de Datos Abiertos |
|---|---|
| URL | https://www.contrataciones.gov.py/datos |
| Período | 2021–2022 |
| Volumen final | 25.360 registros × 27 columnas |
| Tablas | adjudicaciones, contratos |

---

## Pipeline

### Etapa 1 — Limpieza
- Carga de 4 CSV (adjudicaciones + contratos × 2 años)
- Left join por `id_llamado`
- Drop de columnas con >50% nulos
- Eliminación de duplicados exactos y montos en 0
- Construcción de `num_oferentes` y `competencia`

### Etapa 2 — EDA
- Distribución por tipo de procedimiento
- Top 15 instituciones convocantes
- Distribución de montos por tipo (escala log)
- Evolución temporal mensual 2021–2022

### Etapa 3 — Clasificación
- **Target:** `competencia` (0/1)
- **Modelos:** Regresión Logística vs Random Forest
- **Resultado:** Random Forest — Accuracy 0.90, F1 0.90

### Etapa 4 — Regresión
- **Target:** `monto_total_adjudicado` (transformación log1p)
- **Modelos:** Regresión Lineal vs Random Forest
- **Resultado:** Random Forest — R² 0.95, RMSE 0.52 (log)

### Etapa 5 — Clustering
- **Unidad:** institución convocante (398 instituciones)
- **Método:** K-Means, k=5 (método del codo + silhouette)
- **Perfiles:** Competitivos medios, Grandes compradores, Pequeños directos, Outlier Salud, Mega licitadores

---

## Instalación

```bash
git clone <repo-url>
cd <repo>
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

---

## Uso

Ejecutar los notebooks en orden:

```bash
jupyter notebook notebooks/01-data_cleaning.ipynb
```

Cada notebook carga el CSV producido por el anterior. El punto de entrada
es siempre `../datos/`.

---

## Resultados principales

- El **47%** de los procesos son Contrataciones Directas — el mecanismo
  que evita la competencia es el más utilizado por el Estado paraguayo
- **Noviembre** concentra el pico de contrataciones en ambos años —
  patrón de cierre de ejercicio presupuestario
- El **Random Forest** superó a los modelos lineales en clasificación
  y regresión — las relaciones en datos de contratación pública son
  fundamentalmente no lineales
- El **56%** de las instituciones operan con más del 80% de contratación
  directa

---

## Nota sobre IA generativa

Partes del código fueron desarrolladas con asistencia de herramientas de
IA generativa. Todo el código asistido está identificado con el comentario
`# asistido por IA` en el notebook correspondiente, junto con una descripción
de lo solicitado, conforme a la política de integridad académica del trabajo.

---

## Licencia del código

MIT