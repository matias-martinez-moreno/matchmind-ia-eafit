# MatchMind: Predicción de Resultados de Fútbol Internacional con XGBoost y Análisis Narrativo con LLM

**Proyecto Final - Inteligencia Artificial - EAFIT 2026-1**

Predicción de resultados de partidos internacionales de fútbol (victoria local, empate, victoria visitante) utilizando XGBoost, explicabilidad con SHAP y análisis narrativos generados por LLM (Groq Llama-3.1-8b).

---

## Descripción

MatchMind es un sistema de predicción de partidos de fútbol que combina:

1. **Machine Learning**: XGBoost entrenado con 14 features de ingeniería de características (rachas recientes, head-to-head histórico, ranking FIFA, diferencia de puntos, neutralidad del campo)
2. **Explicabilidad**: SHAP values para interpretar qué factores influyen en cada predicción
3. **Generación de Texto**: Groq API con Llama-3.1-8b para generar análisis tácticos narrativos en español

**Métricas finales** (test set 5,876 partidos):
- Accuracy: 0.62
- F1-macro: 0.452
- AUC: 0.68

---

## Requisitos

- Python 3.9+
- `pip install -r requirements.txt`

Para usar la integración con Groq LLM:
- Crear cuenta en [Groq Cloud](https://console.groq.com)
- Obtener `GROQ_API_KEY`
- Crear archivo `.env` en la raíz del proyecto con:
  ```
  GROQ_API_KEY=gsk_...
  ```

---

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/matias-martinez-moreno/matchmind-ia-eafit.git
cd matchmind-ia-eafit

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar API key de Groq
cp .env.example .env
# Editar .env y añadir GROQ_API_KEY
```

---

## Estructura del Proyecto

```
matchmind-ia-eafit/
├── notebooks/                          # Análisis y modelos
│   ├── 01_eda.ipynb                   # Análisis exploratorio de datos
│   ├── 02_preprocessing.ipynb          # Feature engineering y split temporal
│   ├── 03_modeling.ipynb               # Entrenamiento de 4 modelos + SHAP
│   └── 04_llm_integration.ipynb        # Integración Groq + evaluación cuantitativa
│
├── src/                                # Código reutilizable
│   ├── data/
│   │   ├── load.py                    # Carga de datasets Kaggle
│   │   └── features.py                # Feature engineering vectorizado
│   ├── models/
│   │   └── train.py                   # Entrenamiento de modelos
│   ├── evaluation/
│   │   ├── metrics.py                 # Métricas: Accuracy, F1, AUC, Confusion Matrix
│   │   └── llm_eval.py                # Evaluación cuantitativa del LLM
│   └── llm/
│       ├── client.py                  # Cliente Groq API
│       ├── prompts.py                 # Prompts documentados
│       └── reporter.py                # Generación de reportes narrativos
│
├── app/
│   └── main.py                        # Demo interactiva Streamlit
│
├── scripts/                           # Scripts ejecutables
│   ├── download_data.py               # Descargar datasets de Kaggle
│   ├── run_eda.py
│   ├── run_preprocessing.py
│   ├── run_modeling.py
│   └── run_llm_integration.py
│
├── data/
│   ├── raw/                           # Datasets originales (descargados)
│   └── processed/                     # Features y splits
│
├── models/
│   └── checkpoints/
│       └── xgboost_model.pkl          # Modelo entrenado
│
├── docs/
│   ├── informe_final.pdf              # Informe PDF (7 páginas)
│   ├── informe_final.tex              # Fuente LaTeX
│   ├── figures/                       # 13 figuras de EDA, SHAP, matrices
│   └── INSTRUCCIONES_DIA10.md         # Guía de compilación
│
├── tests/
│   └── test_app_logic.py              # Tests E2E del app
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Uso

### 1. Descargar datos (primera ejecución)

```bash
python scripts/download_data.py
```

Descarga desde Kaggle:
- `results.csv`: 49,287 partidos internacionales (1872-2026)
- `fifa_ranking.csv`: 67,472 filas de ranking FIFA (1992-2024)

Requiere API key de Kaggle en `~/.kaggle/kaggle.json`

### 2. Ejecutar notebooks (análisis educativo)

```bash
jupyter notebook notebooks/
```

- **01_eda.ipynb**: Exploración de datos, 8 visualizaciones
- **02_preprocessing.ipynb**: Feature engineering vectorizado, split temporal
- **03_modeling.ipynb**: 4 modelos comparados (Dummy, LogReg, RandomForest, XGBoost), SHAP global
- **04_llm_integration.ipynb**: Demo LLM, evaluación cuantitativa sobre 20 casos

### 3. Demo Streamlit

```bash
streamlit run app/main.py
```

Interfaz interactiva para:
- Seleccionar equipo local, visitante y fecha
- Ver predicción (clase + probabilidades)
- Visualizar top-5 features SHAP
- Leer análisis táctico narrativo generado por LLM

---

## Resultados Clave

### Comparativa de Modelos

| Modelo | Accuracy | F1-macro | AUC | Entrenamiento |
|--------|----------|----------|-----|----------------|
| Dummy (clase mayoritaria) | 0.489 | 0.215 | 0.500 | instantáneo |
| Logistic Regression | 0.524 | 0.446 | 0.568 | 0.1s |
| Random Forest (200 árboles) | 0.581 | 0.425 | 0.621 | 8.2s |
| **XGBoost (tuneado)** | **0.620** | **0.452** | **0.680** | 23.4s |

### Feature Importance (SHAP global)

Top-5 features por impacto SHAP en predicción de victoria local:

1. **rank_diff** (diferencia de ranking FIFA): +0.385
2. **h2h_away_wins** (victorias históricas visitante): +0.128
3. **points_diff** (diferencia de puntos FIFA): +0.115
4. **h2h_home_wins** (victorias históricas local): +0.084
5. **home_recent_wins** (racha local 5 últimos): +0.070

Conclusión: **El ranking FIFA es el factor dominante**. Cuando el local tiene mucho mejor ranking, gana el 80%.

### Limitación Documentada

El modelo **subpredice empates**:
- Empates reales: 1,360 (23.1% del test)
- Empates predichos: 578 (9.8%)
- F1 para clase "draw": 0.10

Esto es un problema estructural del fútbol (los empates son estadísticamente raros como frontera) y se documenta explícitamente en el informe.

---

## Componente LLM

### Funcionalidad

Tras cada predicción XGBoost, se envía al LLM:
- Nombres de equipos, fecha, torneo
- Top-5 features SHAP del partido
- Probabilidades y clase predicha

El LLM genera un análisis narrativo en español, max 150 palabras, que incluye:
- Equipo favorito predicho y por qué
- Factor táctico clave (SHAP)
- Una predicción interpretable

### Ejemplo

**Entrada**: Brasil vs Argentina, Copa América 2024, predicción: local gana (77%)

**Salida LLM**:
> Brasil llega a este clásico sudamericano con una clara ventaja de ranking FIFA (+68 puntos), lo que favorece la victoria local. Además, el equipo brasileño ha ganado 3 de los últimos 5 partidos, demostrando buena forma reciente. El factor decisivo es la diferencia de ranking: históricamente, cuando Brasil tiene esta ventaja, gana el partido el 80% de las veces. Argentina, aunque es un rival de élite, llega en condición de visitante en un torneo complicado.

### Evaluación Cuantitativa

Evaluados 20 casos del test set:

| Criterio | Resultado |
|----------|-----------|
| Menciona factor SHAP relevante | 100% |
| Respuesta en español | 100% |
| Cumple límite de palabras (≤200) | 100% |
| Menciona equipo predicho | 55% |

Nota: El 55% en "menciona equipo" incluye variantes lingüísticas válidas ("el local", "el equipo de casa") que no pasan el chequeo de texto exacto.

---

## Metodología

### Split Temporal (No Aleatorio)

Dato crucial: en series temporales, **nunca** usar `train_test_split` aleatorio.

- **Train**: 1993-01-01 a 2015-12-31 (20,715 partidos)
- **Validation**: 2016-01-01 a 2019-12-31 (3,920 partidos)
- **Test**: 2020-01-01 a 2026-05-13 (5,876 partidos)

Esto simula el escenario real: entrenar con el pasado, predecir el futuro.

### Feature Engineering

14 features finales, construidas sin data leakage usando `groupby().rolling().shift(1)`:

1. `home_recent_wins`: victorias locales últimos 5 partidos
2. `away_recent_wins`: victorias visitante últimos 5 partidos
3. `home_goals_for`: promedio goles anotados (ventana 10)
4. `home_goals_against`: promedio goles recibidos (ventana 10)
5. `away_goals_for`: análogo visitante
6. `away_goals_against`: análogo visitante
7. `h2h_home_wins`: victorias previas locales vs visitante
8. `h2h_draws`: empates previos
9. `h2h_away_wins`: victorias previas visitante vs local
10. `home_team_rank`: ranking FIFA local
11. `away_team_rank`: ranking FIFA visitante
12. `home_team_points`: puntos FIFA local
13. `away_team_points`: puntos FIFA visitante
14. `neutral`: 1 si campo neutral, 0 si field advantage

### Métrica Principal: F1-macro

Se eligió **F1-macro** (promedio de F1 de cada clase) porque:
- Target está moderadamente desbalanceado: 48.9% home / 23% draw / 28.4% away
- Accuracy engañaría (un modelo que siempre predice "home" tendría 48.9% accuracy)
- F1-macro penaliza ignorar clases minoritarias

---

## Decisiones Técnicas Documentadas

| Decisión | Justificación |
|----------|---------------|
| Filtrar a post-1993 | Ranking FIFA confiable solo desde 1992; evita sesgo histórico fuerte |
| Mantener outliers (287 partidos >10 goles) | Son partidos reales (ej: Australia 31-0 Samoa); no afectan |
| Eliminar 72 partidos sin score | Solo 0.15% del dataset |
| Imputar ranking faltante con mediana | Equipos sin ranking previo o recién creados |
| Feature engineering con `shift(1)` | Garantiza que cada partido NO usa su propio resultado |
| XGBoost over Random Forest | Pequeña mejora en F1 (0.425 vs 0.452) pero algoritmo más robusto |

---

## Reproducibilidad

Todos los modelos y datos procesados están guardados:

```bash
# Cargar modelo entrenado (sin re-entrenar)
import pickle
with open('models/checkpoints/xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Cargar features procesadas
import pandas as pd
features = pd.read_csv('data/processed/features.csv')
```

---

## Autores

| Nombre | ID EAFIT | Rol |
|--------|----------|-----|
| Matías Martínez Moreno | 1000117104 | Datos, EDA, Modelos ML |
| Samuel Herrera | 1000358613 | LLM, Streamlit, Informe |

Proyecto Final - Inteligencia Artificial - EAFIT 2026-1  
Profesor: [Nombre del Profesor]  
Entrega: 19-22 de mayo de 2026

---

## Referencias

- Kaggle Dataset: [International Football Results 1872–2023](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
- FIFA World Rankings: [Kaggle Dataset](https://www.kaggle.com/datasets/cashncarry/fifaworldranking)
- SHAP: Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions"
- XGBoost: Chen & Guestrin (2016). "XGBoost: A Scalable Tree Boosting System"
- Groq API: [https://console.groq.com](https://console.groq.com)

---

## Licencia

Proyecto académico. Libre para uso educativo.
