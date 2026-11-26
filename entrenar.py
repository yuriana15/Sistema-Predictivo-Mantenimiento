# ==============================================================================
# ENTRENAMIENTO DEL MODELO DE MANTENIMIENTO PREDICTIVO
# Clase: ISOF CLASS | Docente: Dra. Aiblis Susej Herrera Palafox
# ==============================================================================

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Evita errores al imprimir emojis en la consola de Windows
sys.stdout.reconfigure(encoding="utf-8")

CARPETA = Path(__file__).parent
ARCHIVO_CSV = CARPETA / "sensores_planta.csv"
ARCHIVO_MODELO = CARPETA / "modelo_mantenimiento.pkl"
FEATURES = ["temperatura_c", "vibracion_mms", "horas_operacion", "presion_psi"]
OBJETIVO = "estado"

# 1. Cargar el dataset
print(f"📂 1. Cargando '{ARCHIVO_CSV.name}'...")
try:
    df = pd.read_csv(ARCHIVO_CSV)
except FileNotFoundError:
    sys.exit(f"❌ Error: no se encontró '{ARCHIVO_CSV.name}'.")

df.columns = df.columns.str.strip().str.lower()
faltantes = [c for c in FEATURES + [OBJETIVO] if c not in df.columns]
if faltantes:
    sys.exit(f"❌ Error: faltan columnas en el CSV: {faltantes}")

# 2. Limpiar valores nulos
filas_originales = len(df)
df = df.dropna()
print(f"🧹 2. Limpieza: {filas_originales - len(df)} filas con nulos eliminadas. Registros: {len(df)}")

X = df[FEATURES]
y = df[OBJETIVO].astype(int)

# 3. División 80% entrenamiento / 20% prueba
# stratify=y garantiza que los 3 estados aparezcan en ambos conjuntos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✂️  3. División: {len(X_train)} para entrenamiento y {len(X_test)} para prueba")

# 4. Entrenar el Random Forest
print("🧠 4. Entrenando RandomForestClassifier...")
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# 5. Evaluar la precisión
y_pred = modelo.predict(X_test)
precision = accuracy_score(y_test, y_pred)
print(f"🎯 Precisión del modelo (Accuracy): {precision * 100:.2f}%")

if precision < 0.80:
    sys.exit("❌ La precisión es menor a 80%. Revisa el dataset; no se exportó el modelo.")

# 6. Exportar el modelo
joblib.dump(modelo, ARCHIVO_MODELO)
print(f"✅ 5. Modelo exportado como '{ARCHIVO_MODELO.name}'.")
