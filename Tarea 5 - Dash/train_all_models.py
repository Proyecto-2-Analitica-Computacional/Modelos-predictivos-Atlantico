
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

DATA_PATH = os.environ.get(
    "DATA_PATH",
    r"C:\Users\Sofia Toro\Documents\GitHub\Modelos-predictivos-Atlantico\Tarea 2 - Limpieza_Datos\saber11_limpio.csv",
)
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
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
    normalizer = layers.Normalization()
    normalizer.adapt(X_train_np)
    model = keras.Sequential([
        normalizer,
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
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred, zero_division=0))
    print("Recall:", recall_score(y_test, pred, zero_division=0))
    print("F1:", f1_score(y_test, pred, zero_division=0))
    print("ROC-AUC:", roc_auc_score(y_test, prob))

    model.save(os.path.join(MODELS_DIR, "modelo_familiar_alto_desempeno.keras"))
    joblib.dump(encoded_cols, os.path.join(MODELS_DIR, "columnas_familiares_encoded.pkl"))

def train_cole(df):
    print("\nEntrenando modelo colegio...")
    data = df.copy()
    data["punt_global"] = pd.to_numeric(data["punt_global"], errors="coerce")
    data = data.dropna(subset=["punt_global"])

    data, encoded_cols = one_hot(data, variables_cole)
    X = data[encoded_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32")
    y = pd.to_numeric(data["punt_global"], errors="coerce").astype("float32")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = build_regressor(X_train_s.shape[1], 32, 16)
    early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X_train_s, y_train, validation_split=.2, epochs=40, batch_size=64, callbacks=[early], verbose=1)

    pred = model.predict(X_test_s, verbose=0).ravel()
    print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
    print("MAE:", mean_absolute_error(y_test, pred))
    print("R2:", r2_score(y_test, pred))

    model.save(os.path.join(MODELS_DIR, "modelo_cole_regresion.keras"))
    joblib.dump(encoded_cols, os.path.join(MODELS_DIR, "columnas_cole_encoded.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler_cole.pkl"))

def train_general(df):
    print("\nEntrenando modelo general...")
    data = df.copy()
    data["punt_global"] = pd.to_numeric(data["punt_global"], errors="coerce")
    data = data.dropna(subset=["punt_global"])
    data = data[data["punt_global"] > 0].copy()

    if "estu_fechanacimiento" in data.columns and "periodo" in data.columns:
        data["estu_fechanacimiento"] = pd.to_datetime(data["estu_fechanacimiento"], errors="coerce", dayfirst=True)
        data["anio_nacimiento"] = data["estu_fechanacimiento"].dt.year
        data["anio_presentacion"] = (pd.to_numeric(data["periodo"], errors="coerce") // 10).astype("Int64")
        data["edad_presentacion"] = data["anio_presentacion"] - data["anio_nacimiento"]
        mask = (data["edad_presentacion"] < 13) | (data["edad_presentacion"] > 60) | data["edad_presentacion"].isna()
        med = data.loc[~mask, "edad_presentacion"].median()
        data.loc[mask, "edad_presentacion"] = med
    elif "edad_presentacion" not in data.columns:
        data["edad_presentacion"] = 17

    selected = ["punt_global", "edad_presentacion"] + [c for c in variables_general_categoricas if c in data.columns]
    data = data[selected].copy()

    for c in variables_general_categoricas:
        if c in data.columns:
            data[c] = clean_text_series(data[c])

    data = pd.get_dummies(data, columns=[c for c in variables_general_categoricas if c in data.columns], drop_first=False)
    encoded_cols = [c for c in data.columns if c != "punt_global"]
    data[encoded_cols] = data[encoded_cols].astype(float)

    X = data[encoded_cols].astype("float32")
    y = pd.to_numeric(data["punt_global"], errors="coerce").astype("float32")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

    model = build_general_regressor(X_train.to_numpy())
    early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X_train.to_numpy(), y_train, validation_split=.2, epochs=100, batch_size=32, callbacks=[early], verbose=1)

    pred = model.predict(X_test.to_numpy(), verbose=0).ravel()
    print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
    print("MAE:", mean_absolute_error(y_test, pred))
    print("R2:", r2_score(y_test, pred))

    model.save(os.path.join(MODELS_DIR, "modelo_general_regresion.keras"))
    joblib.dump(encoded_cols, os.path.join(MODELS_DIR, "columnas_general_encoded.pkl"))

if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\Sofia Toro\Documents\GitHub\Modelos-predictivos-Atlantico\Tarea 2 - Limpieza_Datos\saber11_limpio.csv")
    df = clean_columns(df)
    train_familiar(df)
    train_cole(df)
    train_general(df)
    print("\nListo. Modelos guardados en carpeta models.")
