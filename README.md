# Sistema Predictivo de Mantenimiento Industrial y Órdenes de Trabajo con IA

Un Random Forest clasifica la maquinaria en 3 estados (0 Normal, 1 Preventivo, 2 Fallo Crítico)
a partir de 4 sensores, y la API de Groq genera la orden de trabajo.

## Ejecución local

```powershell
python -m pip install -r requirements.txt
```

```powershell
python entrenar.py
```

Para la clave de Groq, copia `.streamlit/secrets.toml.example` como `.streamlit/secrets.toml`
y pega tu clave. Ese archivo está en `.gitignore`.

```powershell
python -m streamlit run app.py
```

## Despliegue

1. Sube a un repositorio de GitHub: `sensores_planta.csv`, `modelo_mantenimiento.pkl`,
   `entrenar.py`, `app.py`, `requirements.txt`, `.gitignore` y `README.md`.
2. En [Streamlit Community Cloud](https://share.streamlit.io) crea una app con ese repositorio y `app.py`.
3. En **Advanced settings** elige **Python 3.13** y pega en **Secrets**:
   `GROQ_API_KEY = "tu_clave"`

Las versiones de `requirements.txt` están fijadas para que el `.pkl` cargue igual en la nube.

## Matriz de pruebas

| # | Escenario | Temp / Vib / Horas / Presión | Resultado esperado |
|---|-----------|------------------------------|--------------------|
| 1 | Estable | 60 / 1.5 / 300 / 92 | Verde y dictamen de rutina |
| 2 | Preventivo | 80 / 4.5 / 2000 / 116 | Amarillo y orden preventiva |
| 3 | Crítico | 98 / 8.8 / 4000 / 146 | Rojo y orden urgente con paro y EPP |
| 4 | Sin CSV | renombrar `sensores_planta.csv` | La app sigue funcionando |
| 5 | Falla de API | clave inválida | Mensaje de error y diagnóstico de ML visible |
