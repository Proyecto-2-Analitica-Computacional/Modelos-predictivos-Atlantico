import collections
import joblib
import numpy as np
import pandas as pd
import re
import unicodedata
import tensorflow as tf

cols = joblib.load(r"models/columnas_familiares_encoded.pkl")
print("artifact", len(cols), len(set(cols)))
print("duplicates", [k for k, v in collections.Counter(cols).items() if v > 1][:20])


def quitar_tildes(texto):
    texto = unicodedata.normalize("NFKD", str(texto))
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return texto


def texto_base(valor):
    if pd.isna(valor):
        return "sin_informacion"
    texto = quitar_tildes(valor).lower().strip()
    texto = re.sub(r"\s+", " ", texto)
    if texto in {"", "nan", "none", "null"} or "sin informacion" in texto:
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
    m = re.search(r"([1-6])", texto)
    return f"estrato {m.group(1)}" if m else "sin_informacion"


def normalizar_personas_hogar(valor):
    texto = texto_base(valor)
    mapa = {
        "una": "1", "uno": "1", "dos": "2", "tres": "3", "cuatro": "4",
        "cinco": "5", "seis": "6", "siete": "7", "ocho": "8",
        "nueve": "9", "diez": "10", "once": "11", "doce": "12",
    }
    for palabra, numero in mapa.items():
        texto = re.sub(rf"\b{palabra}\b", numero, texto)
    m = re.search(r"(\d{1,2})", texto)
    if not m:
        return "sin_informacion"
    n = int(m.group(1))
    if n <= 2:
        return "1 a 2"
    if n <= 4:
        return "3 a 4"
    if n <= 6:
        return "5 a 6"
    if n <= 8:
        return "7 a 8"
    return "9 o mas"


def normalizar_cuartos_hogar(valor):
    texto = texto_base(valor)
    mapa = {
        "uno": "1", "una": "1", "dos": "2", "tres": "3", "cuatro": "4",
        "cinco": "5", "seis": "6", "siete": "7", "ocho": "8",
        "nueve": "9", "diez": "10",
    }
    for palabra, numero in mapa.items():
        texto = re.sub(rf"\b{palabra}\b", numero, texto)
    m = re.search(r"(\d{1,2})", texto)
    if not m:
        return "sin_informacion"
    n = int(m.group(1))
    return "uno" if n == 1 else "dos" if n == 2 else "tres" if n == 3 else "cuatro" if n == 4 else "cinco_o_mas"


def normalizar_educacion(valor):
    texto = texto_base(valor)
    return re.sub(r"\s+", " ", texto.replace("_", " ")).strip() if texto != "sin_informacion" else texto

vals = {
    "fami_estratovivienda": "estrato 1",
    "fami_educacionmadre": "ninguno",
    "fami_educacionpadre": "ninguno",
    "fami_personashogar": "1 a 2",
    "fami_cuartoshogar": "uno",
    "fami_tieneautomovil": "no",
    "fami_tienelavadora": "no",
}
row = pd.DataFrame([vals])
row["fami_estratovivienda"] = row["fami_estratovivienda"].apply(normalizar_estrato)
row["fami_educacionmadre"] = row["fami_educacionmadre"].apply(normalizar_educacion)
row["fami_educacionpadre"] = row["fami_educacionpadre"].apply(normalizar_educacion)
row["fami_personashogar"] = row["fami_personashogar"].apply(normalizar_personas_hogar)
row["fami_cuartoshogar"] = row["fami_cuartoshogar"].apply(normalizar_cuartos_hogar)
row["fami_tieneautomovil"] = row["fami_tieneautomovil"].apply(normalizar_si_no)
row["fami_tienelavadora"] = row["fami_tienelavadora"].apply(normalizar_si_no)
enc = pd.get_dummies(row, columns=list(vals), drop_first=False)
print("enc shape", enc.shape)
print("enc cols", list(enc.columns)[:30])
mat = np.column_stack([
    enc[c].to_numpy(dtype="float32") if c in enc.columns else np.zeros(len(enc), dtype="float32")
    for c in cols
])
print("mat shape", mat.shape, mat.dtype)
print("row sum", mat.sum())
model = tf.keras.models.load_model(r"models/modelo_familiar_alto_desempeno.keras")
print("model input", model.input_shape)
expected = model.input_shape[-1]
if mat.shape[1] < expected:
    mat = np.hstack([mat, np.zeros((mat.shape[0], expected - mat.shape[1]), dtype=mat.dtype)])
elif mat.shape[1] > expected:
    mat = mat[:, :expected]
print("aligned shape", mat.shape)
pred = model.predict(mat, verbose=0)
print("pred", pred)
