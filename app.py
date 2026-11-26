# ==============================================================================
# DASHBOARD DE MANTENIMIENTO PREDICTIVO: MACHINE LEARNING + GROQ API
# Clase: ISOF CLASS | Docente: Dra. Aiblis Susej Herrera Palafox
# ==============================================================================

from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from groq import Groq

# --- 1. Configuración de la página ---
st.set_page_config(page_title="Mantenimiento Predictivo", page_icon="🏭", layout="wide")

ARCHIVO_MODELO = Path(__file__).parent / "modelo_mantenimiento.pkl"

# Cada estado define su tipo de alerta, nombre y prioridad
ESTADOS = {
    0: {"alerta": st.success, "icono": "✅", "nombre": "Operación Normal", "prioridad": "Rutina"},
    1: {"alerta": st.warning, "icono": "⚠️", "nombre": "Mantenimiento Preventivo", "prioridad": "Programada"},
    2: {"alerta": st.error, "icono": "🚨", "nombre": "Fallo Crítico", "prioridad": "URGENTE"},
}


# --- 2. Carga del modelo (la app NO depende del CSV) ---
@st.cache_resource
def cargar_modelo():
    return joblib.load(ARCHIVO_MODELO)


try:
    modelo = cargar_modelo()
except Exception as e:
    st.error(f"❌ No se pudo cargar '{ARCHIVO_MODELO.name}'. Ejecuta primero `python entrenar.py`.")
    st.caption(f"Detalle: {e}")
    st.stop()


# --- 3. Funciones de Groq ---
def construir_prompt(estado, lecturas, confianza, folio):
    datos = f"""Lecturas de la maquinaria:
- Temperatura: {lecturas['temperatura_c']} °C
- Vibración: {lecturas['vibracion_mms']} mm/s
- Horas de operación: {lecturas['horas_operacion']} h
- Presión: {lecturas['presion_psi']} PSI

Diagnóstico del modelo de Machine Learning: estado {estado} ({ESTADOS[estado]['nombre']}), probabilidad {confianza:.1f}%.
"""
    if estado == 0:
        return datos + """
Emite un dictamen BREVE de monitoreo preventivo (máximo 120 palabras), en Markdown.
No generes una orden de trabajo."""

    urgencia = (
        "Es una orden URGENTE: la primera acción técnica debe ser el protocolo de paro inmediato "
        "de la máquina, incluyendo bloqueo y etiquetado (LOTO)."
        if estado == 2
        else "Es una orden preventiva: las acciones deben poder programarse sin paro de emergencia."
    )
    return datos + f"""
Genera una Orden de Trabajo Técnico en Markdown. {urgencia}
Usa exactamente estas secciones:

## Orden de Trabajo
**Folio:** {folio}
**Prioridad:** {ESTADOS[estado]['prioridad']}

### Diagnóstico
(explica qué lecturas justifican el estado)

### Acciones Técnicas
(exactamente 3 acciones numeradas, concretas y verificables)

### Equipo de Protección Personal (EPP)
(lista del equipo de seguridad requerido)"""


def generar_dictamen(prompt_usuario):
    """Consulta Groq y devuelve (modelo_usado, texto). Lanza excepción si falla."""
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        raise RuntimeError("No se encontró GROQ_API_KEY en los secretos de Streamlit.")

    client = Groq(api_key=api_key, timeout=30.0, max_retries=1)

    # Detección dinámica de modelos de chat activos (evita errores 404)
    excluidos = ["whisper", "guard", "tts", "playai", "embed", "orpheus"]
    modelos_activos = [
        m.id for m in client.models.list().data
        if not any(x in m.id.lower() for x in excluidos)
    ]
    if not modelos_activos:
        raise RuntimeError("No hay modelos de chat activos en la cuenta de Groq.")
    modelo_groq = next((m for m in modelos_activos if "llama" in m.lower()), modelos_activos[0])

    respuesta = client.chat.completions.create(
        model=modelo_groq,
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres el Jefe de Mantenimiento Industrial de una planta de manufactura "
                    "automatizada. Respondes en español, de forma técnica, clara y priorizando "
                    "la seguridad del personal. No contradigas el diagnóstico del modelo ni "
                    "inventes lecturas que no se te proporcionaron."
                ),
            },
            {"role": "user", "content": prompt_usuario},
        ],
        temperature=0.3,
        max_tokens=2048,
    )
    texto = respuesta.choices[0].message.content
    if not texto:
        raise RuntimeError(f"El modelo '{modelo_groq}' devolvió una respuesta vacía.")
    return modelo_groq, texto


# --- 4. Controles de los sensores (barra lateral) ---
st.sidebar.header("🎛️ Lecturas de Sensores")
temperatura = st.sidebar.slider("Temperatura (°C)", 40.0, 120.0, 70.0, step=0.5)
vibracion = st.sidebar.slider("Vibración (mm/s)", 0.0, 12.0, 3.0, step=0.1)
horas = st.sidebar.slider("Horas de operación", 0, 5000, 1200, step=10)
presion = st.sidebar.slider("Presión (PSI)", 70.0, 180.0, 105.0, step=0.5)

# --- 5. Encabezado y tarjetas de métricas ---
st.title("🏭 Sistema Predictivo de Mantenimiento Industrial")
st.write("Sistema híbrido: **Random Forest** clasifica el estado y **Groq** genera la orden de trabajo.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Temperatura", f"{temperatura} °C")
col2.metric("Vibración", f"{vibracion} mm/s")
col3.metric("Operación", f"{horas} hrs")
col4.metric("Presión", f"{presion} PSI")

# --- 6. Predicción ---
lecturas = {
    "temperatura_c": temperatura,
    "vibracion_mms": vibracion,
    "horas_operacion": horas,
    "presion_psi": presion,
}
entrada = pd.DataFrame([lecturas])
probabilidades = {
    int(clase): float(p) * 100
    for clase, p in zip(modelo.classes_, modelo.predict_proba(entrada)[0])
}
estado = int(modelo.predict(entrada)[0])
confianza = probabilidades[estado]
info = ESTADOS[estado]

st.subheader("📊 Diagnóstico del Modelo Predictivo")
info["alerta"](
    f"**Estado {estado}: {info['nombre']}** (Probabilidad: {confianza:.1f}%)",
    icon=info["icono"],
)

for columna, (clase, datos_estado) in zip(st.columns(3), ESTADOS.items()):
    prob = probabilidades.get(clase, 0.0)
    columna.progress(int(round(prob)), text=f"{datos_estado['nombre']}: {prob:.1f}%")

# --- 7. Orden de trabajo con Groq ---
st.subheader("🤖 Orden de Trabajo (Groq API)")
etiqueta_boton = "Generar dictamen de rutina" if estado == 0 else "Generar Orden de Trabajo"

# Se usa un botón para no consumir peticiones cada vez que se mueve un control
if st.button(etiqueta_boton, type="primary"):
    folio = f"OT-{datetime.now():%Y%m%d-%H%M%S}-E{estado}"
    try:
        with st.spinner("Consultando al Jefe de Mantenimiento..."):
            modelo_usado, texto = generar_dictamen(
                construir_prompt(estado, lecturas, confianza, folio)
            )
        st.caption(f"⚡ Respuesta generada con el modelo: `{modelo_usado}`")
        st.markdown(texto)
        st.download_button("⬇️ Descargar (.md)", texto, file_name=f"{folio}.md", mime="text/markdown")
    except Exception as e:
        # Resiliencia: el diagnóstico de ML permanece visible aunque falle la API
        st.error("❌ El servicio de Groq no está disponible en este momento.")
        st.info(
            f"El diagnóstico de Machine Learning sigue siendo válido: **{info['nombre']}** "
            f"({confianza:.1f}%). Prioridad: **{info['prioridad']}**."
        )
        st.caption(f"Detalle técnico: {type(e).__name__}: {e}")
