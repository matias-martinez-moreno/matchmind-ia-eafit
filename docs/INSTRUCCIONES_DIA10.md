# Instrucciones del Día 10 - Entrega final

Pasos manuales que faltan para entregar el proyecto. Hacer en orden.

---

## 1. Compilar el informe LaTeX en Overleaf

1. Ir a https://overleaf.com e iniciar sesión.
2. New Project -> Upload Project.
3. Subir el archivo `docs/informe_final.tex`.
4. En Overleaf, crear una carpeta `figures/` y subir TODAS las PNG de `docs/figures/`:
   - eda_distribucion.png
   - eda_localia.png
   - eda_ranking_resultado.png
   - eda_heatmap.png
   - comparativa_modelos.png
   - confusion_matrix.png
   - shap_importance.png
   - (opcionalmente: eda_goles_torneo.png, eda_temporal.png, eda_top_equipos.png, eda_h2h_clasicos.png, shap_summary_home.png, shap_individuales.png)
5. Menu -> Compiler -> pdfLaTeX -> Recompile.
6. Verificar que el PDF tenga máximo 8 páginas (sin contar portada ni referencias).
7. Menu -> Download PDF -> guardar como `informe_final.pdf`.
8. Reemplazar `docs/informe_final.pdf` en el repo local.

Si Overleaf da error de compilación: revisar el log buscando líneas en rojo y corregir el `.tex`. Errores comunes:
- Caracteres especiales sin escapar (`%`, `&`, `$`, `_`, `#`).
- Falta `\begin{}` o `\end{}` correspondiente.
- Imagen no encontrada: verificar que el nombre coincide exactamente con el archivo subido.

---

## 2. Hacer git push del proyecto al repositorio

Desde la carpeta `matchmind-ia-eafit/`:

```bash
git init
git add .
git status
```

Verificar que NO aparezcan estos archivos en la lista (deben estar ignorados por `.gitignore`):
- `.env`
- `venv/`
- `__pycache__/`
- `.claude/`
- `CLAUDE.md`
- `data/raw/*.csv` (excepto si quieres incluirlos, ver más abajo)
- `models/checkpoints/*.pkl` (idem)

Si todo bien, hacer el primer commit:

```bash
git commit -m "feat: proyecto MatchMind completo - prediccion de futbol con XGBoost y LLM"
git branch -M main
git remote add origin https://github.com/matias-martinez-moreno/matchmind-ia-eafit.git
git push -u origin main
```

**Importante**: el commit message NO debe mencionar a Claude, ChatGPT u otra IA generativa. Ya está la regla en CLAUDE.md.

### ¿Subir los datos crudos al repo?

Los CSVs originales (`data/raw/results.csv`, etc.) están en `.gitignore` por defecto. Hay dos opciones:

**Opción A (recomendada)**: dejarlos fuera del repo. El usuario clona y ejecuta `python scripts/download_data.py` para descargarlos. Más limpio.

**Opción B**: incluirlos. Quitar las líneas correspondientes del `.gitignore`. Solo pesan ~12 MB en total, está dentro del límite de GitHub.

### ¿Subir el modelo entrenado?

`models/checkpoints/xgboost_model.pkl` (~5 MB) también está ignorado. Para que cualquier persona pueda usar la demo Streamlit sin reentrenar, recomendable incluirlo. Quitar la línea `models/checkpoints/*.pkl` del `.gitignore` si quieres subirlo.

---

## 3. Enviar correo individual al profesor

**Cada integrante** envía un correo individual. Plantilla sugerida:

**Asunto**: `Entrega Proyecto Final IA - [Nombre Apellido] - Grupo Jueves`

**Cuerpo**:

```
Profesor,

Adjunto la entrega del Proyecto Final de Inteligencia Artificial.

Integrante: [Nombre completo]
ID EAFIT: [ID]
Grupo: Jueves

Equipo: Matias Martinez Moreno (1000117104) y Samuel Herrera (1000358613).

Proyecto: MatchMind - Prediccion de Resultados de Partidos
Internacionales de Futbol con XGBoost, SHAP y LLM.

Repositorio: https://github.com/matias-martinez-moreno/matchmind-ia-eafit

El informe PDF esta en docs/informe_final.pdf dentro del repo.

Saludos cordiales,
[Nombre]
```

**Crítico**: si solo un integrante envía correo, el otro pierde 10 puntos automáticos por integrante faltante.

---

## 4. Checklist final antes de cerrar

Verificar uno por uno antes de considerar la entrega lista:

- [ ] `docs/informe_final.pdf` existe en el repo y abre correctamente.
- [ ] El PDF tiene máximo 8 páginas (sin portada ni referencias).
- [ ] `README.md` no tiene emojis y nombra a los integrantes correctos con sus IDs.
- [ ] `requirements.txt` lista todas las dependencias usadas.
- [ ] `scripts/` tiene los 5 scripts ejecutables en orden.
- [ ] Repositorio público en GitHub y clonable.
- [ ] `.env` NO está en el repo (verificar con `git ls-files .env`).
- [ ] `CLAUDE.md` NO está en el repo (verificar con `git ls-files CLAUDE.md`).
- [ ] Ambos integrantes enviaron el correo individual al profesor.
- [ ] Fecha de entrega: entre el 19 y 22 de mayo de 2026.

---

## 5. Probar que el repo es reproducible (opcional pero recomendado)

Para asegurar que NO perdamos los -20 puntos por no-reproducibilidad:

```bash
# Clonar el repo en una carpeta temporal
cd /tmp
git clone https://github.com/matias-martinez-moreno/matchmind-ia-eafit.git
cd matchmind-ia-eafit

# Crear venv fresco e instalar
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Descargar datasets
python scripts/download_data.py

# Configurar API key
copy .env.example .env
# (editar .env y poner GROQ_API_KEY=...)

# Ejecutar pipeline
python scripts/run_preprocessing.py
python scripts/run_modeling.py
python scripts/run_tuning_shap.py
python scripts/run_llm_integration.py

# Probar demo
streamlit run app/main.py
```

Si todo corre sin errores, el repo es reproducible.
