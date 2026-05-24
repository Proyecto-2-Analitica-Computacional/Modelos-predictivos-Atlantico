import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os


DATA_PATH = 'Tarea 2 - Limpieza_Datos/saber11_limpio.csv'
MODEL_PATH = 'models/modelo_general_regresion.h5'
COLUMNS_PATH = 'models/modelo_general_columns.pkl'
METRICS_PATH = 'models/metrics_general.json'


df = pd.read_csv(DATA_PATH)

TARGET = 'punt_global' if 'punt_global' in df.columns else 'puntaje_global'
variables_general_categoricas = [
    'cole_jornada', 'cole_calendario', 'cole_bilingue', 'cole_naturaleza',
    'fami_tieneinternet', 'fami_tienecomputador', 'fami_educacionmadre',
    'fami_educacionpadre', 'fami_estratovivienda', 'fami_tieneautomovil'
]

# Asegurar que 'edad_presentacion' exista
if 'edad_presentacion' not in df.columns:
    df['edad_presentacion'] = 17

selected = [TARGET, 'edad_presentacion'] + [c for c in variables_general_categoricas if c in df.columns]
data = df[selected].copy()

for c in variables_general_categoricas:
    if c in data.columns:
        data[c] = data[c].astype(str).str.lower().str.strip().replace(['', ' ', 'nan', 'none', 'null'], 'sin_informacion').fillna('sin_informacion')

data = pd.get_dummies(data, columns=[c for c in variables_general_categoricas if c in data.columns], drop_first=False)
encoded_cols = [c for c in data.columns if c != TARGET]
data[encoded_cols] = data[encoded_cols].astype(float)

X = data[encoded_cols].astype('float32')
y = pd.to_numeric(data[TARGET], errors='coerce').astype('float32')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

input_shape = X_train.shape[1]
model = keras.Sequential([
    keras.layers.Input(shape=(input_shape,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train.to_numpy(), y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early], verbose=2)

pred = model.predict(X_test.to_numpy(), verbose=0).ravel()
print('RMSE:', np.sqrt(mean_squared_error(y_test, pred)))
print('MAE:', mean_absolute_error(y_test, pred))
print('R2:', r2_score(y_test, pred))

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Guardar el modelo en formato .h5 para máxima compatibilidad
model.save(MODEL_PATH)
# Eliminar el .keras si existe para evitar confusión
keras_path = 'models/modelo_general_regresion.keras'
if os.path.exists(keras_path):
    try:
        os.remove(keras_path)
    except Exception as e:
        print(f"[DEBUG] No se pudo eliminar el archivo .keras: {e}")
joblib.dump(encoded_cols, COLUMNS_PATH)

# Guardar métricas en JSON
import json
metrics = {
    "RMSE": round(float(np.sqrt(mean_squared_error(y_test, pred))), 4),
    "MAE": round(float(mean_absolute_error(y_test, pred)), 4),
    "R2": round(float(r2_score(y_test, pred)), 4)
}
with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
