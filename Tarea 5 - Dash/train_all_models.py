

import os
import re
import unicodedata
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Ruta robusta: siempre busca los archivos desde la raíz del proyecto
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "Tarea 2 - Limpieza_Datos", "saber11_limpio.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.keras.utils.set_random_seed(RANDOM_STATE)

variables_familiares = [
    "fami_estratovivienda", "fami_educacionmadre", "fami_educacionpadre",
    "fami_personashogar", "fami_cuartoshogar", "fami_tieneautomovil",
    "fami_tienelavadora"
]

variables_cole = [
    "cole_naturaleza", "cole_bilingue", "cole_jornada", "cole_calendario",
    "cole_area_ubicacion", "cole_caracter", "cole_genero"
]

variables_general_categoricas = [
    "cole_jornada", "cole_calendario", "cole_bilingue", "cole_naturaleza",
    "fami_tieneinternet", "fami_tienecomputador", "fami_educacionmadre",
    "fami_educacionpadre", "fami_estratovivienda", "fami_tieneautomovil"
]

def preprocess_cole_columns(df):
    data = df.copy()
    # Normalizar y limpiar variables de colegio
    if "cole_naturaleza" in data.columns:
        data["cole_naturaleza"] = clean_text_series(data["cole_naturaleza"]).astype("category")
    if "cole_bilingue" in data.columns:
        data["cole_bilingue"] = data["cole_bilingue"].apply(normalizar_si_no).astype("category")
    if "cole_jornada" in data.columns:
        data["cole_jornada"] = clean_text_series(data["cole_jornada"]).astype("category")
    if "cole_calendario" in data.columns:
        data["cole_calendario"] = clean_text_series(data["cole_calendario"]).astype("category")
    if "cole_area_ubicacion" in data.columns:
        data["cole_area_ubicacion"] = clean_text_series(data["cole_area_ubicacion"]).astype("category")
    if "cole_caracter" in data.columns:
        data["cole_caracter"] = clean_text_series(data["cole_caracter"]).astype("category")
    if "cole_genero" in data.columns:
        data["cole_genero"] = clean_text_series(data["cole_genero"]).astype("category")
    return data

def clean_columns(df):
    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df

def clean_text_series(s):
    return (
        s.astype(str)
        .str.lower()
        .str.strip()
        .replace(["", " ", "nan", "none", "null"], "sin_informacion")
        .fillna("sin_informacion")
    )


def quitar_tildes(texto):
    texto = unicodedata.normalize("NFKD", str(texto))
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return texto


def texto_base(valor):
    if pd.isna(valor):
        return "sin_informacion"

    texto = quitar_tildes(valor).lower().strip()
    texto = texto.replace("mã¡s", "mas").replace("mã", "mas")
    texto = re.sub(r"\s+", " ", texto)

    if texto in {"", " ", "nan", "none", "null", "n/a", "na", "no reporta", "no registra"}:
        return "sin_informacion"

    if "sin informacion" in texto or "sin_informacion" in texto:
        return "sin_informacion"

    return texto


def normalizar_si_no(valor):
    texto = texto_base(valor)

    if texto in {"si", "s", "sí", "1", "true"}:
        return "si"

    if texto in {"no", "n", "0", "false"}:
        return "no"

    return "sin_informacion"


def normalizar_estrato(valor):
    texto = texto_base(valor)

    if texto == "sin_informacion":
        return texto

    m = re.search(r"([1-6])", texto)
    if m:
        return f"estrato {m.group(1)}"

    return "sin_informacion"


def normalizar_personas_hogar(valor):
    texto = texto_base(valor)

    if texto == "sin_informacion":
        return texto

    mapa = {
        "una": "1", "uno": "1", "dos": "2", "tres": "3", "cuatro": "4",
        "cinco": "5", "seis": "6", "siete": "7", "ocho": "8",
        "nueve": "9", "diez": "10", "once": "11", "doce": "12"
    }

    for palabra, numero in mapa.items():
        texto = re.sub(rf"\b{palabra}\b", numero, texto)

    m = re.search(r"(\d{1,2})", texto)
    if not m:
        return "sin_informacion"

    n = int(m.group(1))

    if n <= 2:
        return "1 a 2"
    elif n <= 4:
        return "3 a 4"
    elif n <= 6:
        return "5 a 6"
    elif n <= 8:
        return "7 a 8"
    else:
        return "9 o mas"


def normalizar_cuartos_hogar(valor):
    texto = texto_base(valor)

    if texto == "sin_informacion":
        return texto

    mapa = {
        "uno": "1", "una": "1",
        "dos": "2",
        "tres": "3",
        "cuatro": "4",
        "cinco": "5",
        "seis": "6",
        "siete": "7",
        "ocho": "8",
        "nueve": "9",
        "diez": "10"
    }

    for palabra, numero in mapa.items():
        texto = re.sub(rf"\b{palabra}\b", numero, texto)

    m = re.search(r"(\d{1,2})", texto)
    if not m:
        return "sin_informacion"

    n = int(m.group(1))

    if n == 1:
        return "uno"
    elif n == 2:
        return "dos"
    elif n == 3:
        return "tres"
    elif n == 4:
        return "cuatro"
    else:
        return "cinco_o_mas"


def normalizar_educacion(valor):
    texto = texto_base(valor)

    if texto == "sin_informacion":
        return texto

    texto = texto.replace("_", " ")
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto

def one_hot(df, cols):
    use_cols = [c for c in cols if c in df.columns]
    for c in use_cols:
        df[c] = clean_text_series(df[c])
    df = pd.get_dummies(df, columns=use_cols, drop_first=False)
    encoded_cols = [c for c in df.columns if c.startswith(tuple([v + "_" for v in cols]))]
    return df, encoded_cols


def preprocess_family_columns(df):
    data = df.copy()

    if "fami_estratovivienda" in data.columns:
        data["fami_estratovivienda"] = data["fami_estratovivienda"].apply(normalizar_estrato).astype("category")

    if "fami_educacionmadre" in data.columns:
        data["fami_educacionmadre"] = data["fami_educacionmadre"].apply(normalizar_educacion).astype("category")

    if "fami_educacionpadre" in data.columns:
        data["fami_educacionpadre"] = data["fami_educacionpadre"].apply(normalizar_educacion).astype("category")

    if "fami_personashogar" in data.columns:
        data["fami_personashogar"] = data["fami_personashogar"].apply(normalizar_personas_hogar).astype("category")

    if "fami_cuartoshogar" in data.columns:
        data["fami_cuartoshogar"] = data["fami_cuartoshogar"].apply(normalizar_cuartos_hogar).astype("category")

    if "fami_tieneautomovil" in data.columns:
        data["fami_tieneautomovil"] = data["fami_tieneautomovil"].apply(normalizar_si_no).astype("category")

    if "fami_tienelavadora" in data.columns:
        data["fami_tienelavadora"] = data["fami_tienelavadora"].apply(normalizar_si_no).astype("category")

    return data

def build_classifier(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.30),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.20),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return model

def build_regressor(input_dim, n1=32, n2=16):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(n1, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(n2, activation="relu"),
        layers.Dropout(0.15),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def build_general_regressor(X_train_np):
    input_dim = X_train_np.shape[1]
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def train_familiar(df):
    print("\nEntrenando modelo familiar...")
    data = df.copy()
    data["punt_global"] = pd.to_numeric(data["punt_global"], errors="coerce")
    data = data.dropna(subset=["punt_global"])
    data["alto_desempeno"] = (data["punt_global"] >= 300).astype(int)

    data = preprocess_family_columns(data)
    data, encoded_cols = one_hot(data, variables_familiares)
    X = data[encoded_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32")
    y = data["alto_desempeno"].astype("float32")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)

    model = build_classifier(X_train.shape[1])
    early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_split=.2, epochs=30, batch_size=128, callbacks=[early], verbose=1)

    prob = model.predict(X_test, verbose=0).ravel()
    pred = (prob >= .5).astype(int)
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    auc = roc_auc_score(y_test, prob)
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)
    print("ROC-AUC:", auc)

    # Guardar métricas
    metrics = {
        "AUC": round(float(auc), 4),
        "Accuracy": round(float(acc), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "F1": round(float(f1), 4)
    }
    with open(os.path.join(MODELS_DIR, "metrics_familiar.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    model.save(os.path.join(MODELS_DIR, "modelo_familiar_alto_desempeno.h5"))
    joblib.dump(encoded_cols, os.path.join(MODELS_DIR, "columnas_familiares_encoded.pkl"))

def train_cole(df):
    print("\nEntrenando modelo colegio")
    data = preprocess_cole_columns(df)
    data["punt_global"] = pd.to_numeric(data["punt_global"], errors="coerce")
    data = data.dropna(subset=["punt_global"])
    data["alto_desempeno"] = (data["punt_global"] >= 300).astype(int)



    # One-hot encoding sobre las variables de colegio (igual que familia)
    data, encoded_cols = one_hot(data, variables_cole)
    X = data[encoded_cols].astype(int)
    y = data['alto_desempeno']

    # Guardar columnas para Dash
    joblib.dump(encoded_cols, os.path.join(MODELS_DIR, "columnas_cole_encoded.pkl"))

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Escalado y guardado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler_cole.pkl"))

    # Hiperparámetros óptimos del notebook
    params = {
        'capa_1': 96,
        'capa_2': 48,
        'capa_3': 24,
        'dropout_1': 0.15,
        'dropout_2': 0.05,
        'dropout_3': 0.03,
        'learning_rate': 0.001,
        'batch_size': 128,
        'epochs': 30,  # Más epochs para mejor entrenamiento
        'class_weight_1': 1.5,
        'umbral': 0.35
    }

    def construir_modelo(input_dim, capa_1, capa_2, capa_3, dropout_1, dropout_2, dropout_3, learning_rate):
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dense(capa_1, activation=keras.layers.LeakyReLU(alpha=0.1)),
            layers.Dropout(dropout_1),
            layers.Dense(capa_2, activation=keras.layers.LeakyReLU(alpha=0.1)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_2),
            layers.Dense(capa_3, activation=keras.layers.LeakyReLU(alpha=0.1)),
            layers.Dropout(dropout_3),
            layers.Dense(1, activation="sigmoid")
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.AUC(name="auc")]
        )
        return model

    model = construir_modelo(
        input_dim=X_train_scaled.shape[1],
        capa_1=params['capa_1'],
        capa_2=params['capa_2'],
        capa_3=params['capa_3'],
        dropout_1=params['dropout_1'],
        dropout_2=params['dropout_2'],
        dropout_3=params['dropout_3'],
        learning_rate=params['learning_rate']
    )

    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=0)
    class_weight = {0: 1.0, 1: params['class_weight_1']}

    # Entrenamiento
    model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluación
    prob = model.predict(X_test_scaled, verbose=0).ravel()
    pred = (prob >= params['umbral']).astype(int)
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    auc = roc_auc_score(y_test, prob)
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)
    print("ROC-AUC:", auc)

    # Guardar métricas
    metrics = {
        "AUC": round(float(auc), 4),
        "Accuracy": round(float(acc), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "F1": round(float(f1), 4)
    }
    with open(os.path.join(MODELS_DIR, "metrics_colegio.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Guardar modelo, columnas y scaler
    model.save(os.path.join(MODELS_DIR, "modelo_cole_clasificacion.h5"))
    joblib.dump(encoded_cols, os.path.join(MODELS_DIR, "columnas_cole_encoded.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler_cole.pkl"))

def train_general(df):
    print("\nEntrenando modelo general...")

    data = df.copy()

    TARGET = "punt_global" if "punt_global" in data.columns else "puntaje_global"

    # Asegurar que edad_presentacion exista
    if "edad_presentacion" not in data.columns:
        data["edad_presentacion"] = 17

    selected = [TARGET, "edad_presentacion"] + [
        c for c in variables_general_categoricas
        if c in data.columns
    ]

    data = data[selected].copy()

    # Limpieza EXACTA del archivo train_general_model.py
    for c in variables_general_categoricas:
        if c in data.columns:
            data[c] = (
                data[c]
                .astype(str)
                .str.lower()
                .str.strip()
                .replace(
                    ["", " ", "nan", "none", "null"],
                    "sin_informacion"
                )
                .fillna("sin_informacion")
            )

    # One-hot encoding
    data = pd.get_dummies(
        data,
        columns=[
            c for c in variables_general_categoricas
            if c in data.columns
        ],
        drop_first=False
    )

    encoded_cols = [c for c in data.columns if c != TARGET]

    data[encoded_cols] = data[encoded_cols].astype(float)

    X = data[encoded_cols].astype("float32")

    y = pd.to_numeric(
        data[TARGET],
        errors="coerce"
    ).astype("float32")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # MODELO EXACTAMENTE IGUAL
    input_shape = X_train.shape[1]

    model = keras.Sequential([
        keras.layers.Input(shape=(input_shape,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    early = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X_train.to_numpy(),
        y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early],
        verbose=2
    )

    pred = model.predict(
        X_test.to_numpy(),
        verbose=0
    ).ravel()

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2:", r2)

    # Guardar modelo
    model.save(
        os.path.join(
            MODELS_DIR,
            "modelo_general_regresion.h5"
        )
    )

    # Eliminar .keras si existe
    keras_path = os.path.join(
        MODELS_DIR,
        "modelo_general_regresion.keras"
    )

    if os.path.exists(keras_path):
        try:
            os.remove(keras_path)
        except Exception as e:
            print(f"[DEBUG] No se pudo eliminar el archivo .keras: {e}")

    # IMPORTANTE:
    # usar el mismo nombre que usa el dashboard
    joblib.dump(
        encoded_cols,
        os.path.join(
            MODELS_DIR,
            "columnas_general_encoded.pkl"
        )
    )

    # Guardar métricas
    import json

    metrics = {
        "RMSE": round(float(rmse), 4),
        "MAE": round(float(mae), 4),
        "R2": round(float(r2), 4)
    }

    with open(
        os.path.join(MODELS_DIR, "metrics_general.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            indent=2
        )

        
if __name__ == "__main__":
    import os
    # Ruta robusta: siempre busca el archivo desde la raíz del proyecto
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(PROJECT_ROOT, "Tarea 2 - Limpieza_Datos", "saber11_limpio.csv")
    df = pd.read_csv(DATA_PATH)
    df = clean_columns(df)
    train_familiar(df)
    train_cole(df)
    train_general(df)
    print("\nListo. Modelos guardados en carpeta models.")
