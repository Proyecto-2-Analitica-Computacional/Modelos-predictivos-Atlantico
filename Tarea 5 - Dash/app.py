import os
import json

# --- MÉTRICAS: funciones para cargar métricas de cada modelo ---
def get_metrics_general():
    metrics_path = os.path.join(MODELS_DIR, "metrics_general.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"RMSE": "-", "MAE": "-", "R2": "-"}

def get_metrics_familiar():
    metrics_path = os.path.join(MODELS_DIR, "metrics_familiar.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"AUC": "-", "Accuracy": "-", "Precision": "-", "Recall": "-", "F1": "-"}

def get_metrics_colegio():
    metrics_path = os.path.join(MODELS_DIR, "metrics_colegio.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"AUC": "-", "Accuracy": "-", "Precision": "-", "Recall": "-", "F1": "-"}
import joblib
import numpy as np
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import json
import importlib

try:
    tf = importlib.import_module("tensorflow")
except Exception:
    tf = None
import re
import unicodedata

import joblib
import numpy as np
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import json
import importlib

# CONFIGURACION
# -------------------------

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server

import os
# Ruta robusta: siempre busca el archivo desde la raíz del proyecto
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "Tarea 2 - Limpieza_Datos", "saber11_limpio.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# -------------------------
# VARIABLES
# -------------------------

variables_familiares = [
    "fami_estratovivienda",
    "fami_educacionmadre",
    "fami_educacionpadre",
    "fami_personashogar",
    "fami_cuartoshogar",
    "fami_tieneautomovil",
    "fami_tienelavadora",
]

variables_cole = [
    "cole_naturaleza",
    "cole_bilingue",
    "cole_jornada",
    "cole_calendario",
    "cole_area_ubicacion",
    "cole_caracter",
    "cole_genero",
]

variables_general_categoricas = [
    "cole_jornada",
    "cole_calendario",
    "cole_bilingue",
    "cole_naturaleza",
    "fami_tieneinternet",
    "fami_tienecomputador",
    "fami_educacionmadre",
    "fami_educacionpadre",
    "fami_estratovivienda",
    "fami_tieneautomovil",
]

# -------------------------
# FUNCIONES AUXILIARES
# -------------------------

def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df.columns = (
            df.columns
            .str.lower()
            .str.strip()
            .str.replace(" ", "_")
            .str.replace("-", "_")
        )
        return df
    return pd.DataFrame()

# Mover la función radar_metrics_figure fuera de load_data
def clasificacion_metrics_figure():
    fam = get_metrics_familiar()
    col = get_metrics_colegio()

    metricas = ["AUC", "Precision", "Recall", "F1"]

    familiar = [
        float(fam.get("AUC", 0) or 0),
        float(fam.get("Precision", 0) or 0),
        float(fam.get("Recall", 0) or 0),
        float(fam.get("F1", 0) or 0),
    ]

    colegio = [
        float(col.get("AUC", 0) or 0),
        float(col.get("Precision", 0) or 0),
        float(col.get("Recall", 0) or 0),
        float(col.get("F1", 0) or 0),
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=metricas,
        y=familiar,
        name="Familiar",
        marker_color="#6C63FF"
    ))

    fig.add_trace(go.Bar(
        x=metricas,
        y=colegio,
        name="Colegio",
        marker_color="#00BFAE"
    ))

    fig.update_layout(
        title="Métricas de clasificación",
        yaxis_title="Valor",
        xaxis_title="Métrica",
        barmode="group",
        height=400,
        margin=dict(l=30, r=30, t=50, b=30)
    )

    return fig


def regresion_metrics_figure():
    gen = get_metrics_general()

    metricas = ["RMSE", "MAE", "R²"]

    valores = [
        float(gen.get("RMSE", 0) or 0),
        float(gen.get("MAE", 0) or 0),
        float(gen.get("R2", 0) or 0),
    ]

    fig = go.Figure([
        go.Bar(
            x=metricas,
            y=valores,
            marker_color=["#888888", "#AAAAAA", "#6C63FF"]
        )
    ])

    fig.update_layout(
        title="Métricas del modelo de regresión",
        yaxis_title="Valor",
        xaxis_title="Métrica",
        height=400,
        margin=dict(l=30, r=30, t=50, b=30)
    )

    return fig

def radar_metrics_figure():
    metrics_fam = get_metrics_familiar()
    metrics_col = get_metrics_colegio()
    categories = ["Precision", "Recall", "F1"]
    fam_values = [float(metrics_fam.get(cat, 0) or 0) for cat in categories]
    col_values = [float(metrics_col.get(cat, 0) or 0) for cat in categories]
    # Cerrar el polígono
    fam_values += [fam_values[0]]
    col_values += [col_values[0]]
    cats = categories + [categories[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=fam_values, theta=cats, fill='toself', name='Familiar', line_color='#6C63FF'))
    fig.add_trace(go.Scatterpolar(r=col_values, theta=cats, fill='toself', name='Colegio', line_color='#00BFAE'))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Precisión, Recall y F1 (modelos de clasificación)",
        margin=dict(l=30, r=30, t=50, b=30)
    )
    return fig

df = load_data()

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
        "uno": "1", "una": "1", "dos": "2", "tres": "3", "cuatro": "4",
        "cinco": "5", "seis": "6", "siete": "7", "ocho": "8",
        "nueve": "9", "diez": "10"
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


NORMALIZADORES = {
    "fami_estratovivienda": normalizar_estrato,
    "fami_educacionmadre": normalizar_educacion,
    "fami_educacionpadre": normalizar_educacion,
    "fami_personashogar": normalizar_personas_hogar,
    "fami_cuartoshogar": normalizar_cuartos_hogar,
    "fami_tieneautomovil": normalizar_si_no,
    "fami_tienelavadora": normalizar_si_no,
    "fami_tieneinternet": normalizar_si_no,
    "fami_tienecomputador": normalizar_si_no,
    "cole_bilingue": normalizar_si_no,
}

def options_for(col, default_values=None):
    if df.empty:
        if default_values is None:
            default_values = ["sin_informacion"]
        return [{"label": v, "value": v} for v in default_values]

    # 1) intentar valores directos de la columna
    if col in df.columns:
        series = df[col].dropna()
        if col in NORMALIZADORES:
            series = series.apply(NORMALIZADORES[col])
        raw_vals = series.astype(str).str.strip().tolist()
        seen = {}
        for v in raw_vals:
            key = str(v).strip().lower()
            key = re.sub(r"\s+", " ", key)
            key = re.sub(r"[^0-9a-záéíóúüñ ]", " ", key)
            key = key.strip()
            if key not in seen:
                seen[key] = v

        exclude_patterns = ("sin", "sin_informacion", "sin informacion", "sin-informacion", "sininformacion", "no registra", "no_registra", "desconocido", "nan", "none", "null", "n/a")
        cleaned_keys = [k for k in seen.keys() if not any(p in k for p in exclude_patterns)]
        use_keys = cleaned_keys if cleaned_keys else list(seen.keys())
        options = []
        for k in sorted(use_keys):
            v = seen[k]
            label = str(v).upper() if isinstance(v, str) and not str(v).replace('.', '', 1).isdigit() else str(v)
            options.append({"label": label, "value": v})
        if options:
            return options

    # 2) intentar derivar categorías desde columnas one-hot (más robusto)
    one_hot_cols = []
    for c in df.columns:
        if c.startswith(col + "_"):
            one_hot_cols.append(c)
        elif c.startswith(col) and len(c) > len(col) and c[len(col)] in ("_", ".", "-"):
            one_hot_cols.append(c)
        elif ("_" + col + "_") in c:
            one_hot_cols.append(c)

    if one_hot_cols:
        # Mapear sufijo raw -> cleaned label, pero usar el sufijo raw como `value`
        mapping = {}
        # función para normalizar sufijos numéricos y rangos (p. ej. '1 a 2', 'una', 'cinco')
        def normalize_count_suffix(s):
            s0 = s.lower()
            s0 = s0.replace('-', ' ').replace('.', ' ').strip()
            # map written numbers to digits
            words_to_digits = {
                'una': '1', 'uno': '1', 'dos': '2', 'tres': '3', 'cuatro': '4', 'cinco': '5',
                'seis': '6', 'siete': '7', 'ocho': '8', 'nueve': '9', 'diez': '10', 'once': '11', 'doce': '12'
            }
            for w, d in words_to_digits.items():
                s0 = re.sub(r'\b'+re.escape(w)+r'\b', d, s0)

            # ranges like '1 a 2' or '1 a 2 personas'
            m = re.search(r'(\d+)\s*(a|al|año|a|a\s|a\s+|a\s*)\s*(\d+)', s0)
            if m:
                a = int(m.group(1)); b = int(m.group(3))
                return f"{a}-{b}", a

            # '9 o mas' or '9 o más' -> '9+'
            m2 = re.search(r'(\d+)\s*(o|y)?\s*(mas|más|m)\b', s0)
            if m2:
                a = int(m2.group(1))
                return f"{a}+", a

            # single digit present
            m3 = re.search(r'\b(\d{1,2})\b', s0)
            if m3:
                a = int(m3.group(1))
                return str(a), a

            return s.strip(), 999
        for c in one_hot_cols:
            # intentar obtener el sufijo tal cual aparece tras el primer '_'
            raw_suffix = None
            if (col + "_") in c:
                raw_suffix = c.split(col + "_", 1)[1]
            else:
                raw_suffix = c[len(col):]
                if raw_suffix.startswith("_"):
                    raw_suffix = raw_suffix[1:]
            raw_suffix = raw_suffix or c

            # limpiar para label
            lab = raw_suffix.replace("_", " ")
            lab = lab.strip()
            # normalizar unicode y espacios
            try:
                import unicodedata
                lab = unicodedata.normalize("NFKC", lab)
            except Exception:
                pass
            lab = re.sub(r"\s+", " ", lab)
            lab_up = lab.upper()

            # excluir tokens de relleno
            low = lab.lower()
            if any(tok in low for tok in ("sin", "no reporta", "no registra", "desconocido", "n/a", "nan", "none", "null")):
                mapping.setdefault("__missing__", []).append((raw_suffix, lab_up))
            else:
                # normalizar casos como '1 a 2' vs 'una' agrupándolos
                norm_label, sort_key = normalize_count_suffix(raw_suffix)
                canon_label = norm_label.replace('_', ' ').upper()
                # si ya existe, no sobrescribir (conservar primer raw_suffix encontrado)
                if canon_label not in mapping:
                    mapping[canon_label] = (raw_suffix, canon_label, sort_key)

        # si hay opciones reales, devolverlas (ordenadas por label)
        real_items = [v for k, v in mapping.items() if k != "__missing__"]
        if real_items:
            # ordenar por sort_key si está presente
            real_items_sorted = sorted(real_items, key=lambda x: x[2] if len(x) > 2 else 999)
            opts = [{"label": v[1], "value": v[0]} for v in real_items_sorted]
            return opts
        # fallback a missing
        miss = mapping.get("__missing__", [])
        if miss:
            opts = [{"label": m[1], "value": m[0]} for m in miss]
            return opts

    # 3) buscar columnas similares por tokens (p. ej. fami_personashogar -> fami_personas)
    def tokens(x):
        return [t for t in re.split(r"[_.\- ]+", x.lower()) if t]

    target_tokens = set(tokens(col))
    candidates = []
    for c in df.columns:
        c_tokens = set(tokens(c))
        # puntuación simple: número de tokens en común
        score = len(target_tokens & c_tokens)
        if score > 0:
            candidates.append((score, c))
    candidates.sort(reverse=True)
    if candidates:
        # elegir la mejor candidata
        best = candidates[0][1]
        vals = df[best].dropna().astype(str).str.strip()
        if vals.empty:
            return []
        # si es numérica (todos son dígitos/float), devolver como opciones ordenadas
        if vals.str.replace('.', '', 1).str.isnumeric().all():
            uniq = sorted({float(v) for v in vals.unique()})
            return [{"label": str(int(u)) if float(u).is_integer() else str(u), "value": u} for u in uniq]
        else:
            uniq = sorted({v for v in vals.unique()})
            return [{"label": str(v).upper(), "value": v} for v in uniq]

    # no se encontraron valores reales: devolver lista vacía para evitar mostrar 'sin_informacion'
    return []

def default_for(col, default_value=None):
    opts = options_for(col)
    if not opts:
        return default_value
    return opts[0]["value"]

def load_keras_model(path):
    if tf is None or not os.path.exists(path):
        return None
    try:
        # compile=False evita errores de deserialización de métricas
        return tf.keras.models.load_model(path, custom_objects={"LeakyReLU": tf.keras.layers.LeakyReLU}, compile=False)
    except Exception as e:
        print(f"Error cargando modelo {path}: {e}")
        return None

def load_joblib(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def align_to_model_input(matrix, model):
    if model is None:
        return matrix

    expected = model.input_shape[-1]
    current = matrix.shape[1]

    if current == expected:
        return matrix

    if current < expected:
        padding = np.zeros((matrix.shape[0], expected - current), dtype=matrix.dtype)
        return np.hstack([matrix, padding])

    return matrix[:, :expected]

modelo_familiar = load_keras_model(os.path.join(MODELS_DIR, "modelo_familiar_alto_desempeno.h5"))
cols_familiar = load_joblib(os.path.join(MODELS_DIR, "columnas_familiares_encoded.pkl"))

modelo_cole = load_keras_model(os.path.join(MODELS_DIR, "modelo_cole_clasificacion.h5"))
cols_cole = load_joblib(os.path.join(MODELS_DIR, "columnas_cole_encoded.pkl"))
scaler_cole = load_joblib(os.path.join(MODELS_DIR, "scaler_cole.pkl"))


modelo_general = load_keras_model(os.path.join(MODELS_DIR, "modelo_general_regresion.h5"))
cols_general = load_joblib(os.path.join(MODELS_DIR, "columnas_general_encoded.pkl"))
print("[DEBUG] Columnas del modelo general al cargar:", len(cols_general) if cols_general is not None else None, cols_general)
if modelo_general is not None:
    try:
        print("[DEBUG] Modelo general input_shape:", modelo_general.input_shape)
    except Exception as e:
        print("[DEBUG] Error obteniendo input_shape del modelo general:", e)

# --- DEBUG: Verifica existencia de modelo general y columnas al iniciar el servidor ---
if modelo_general is None:
    print("[DEBUG] No se pudo cargar el modelo general de regresión (modelo_general_regresion.h5)")
else:
    print("[DEBUG] Modelo general de regresión cargado correctamente")
if cols_general is None:
    print("[DEBUG] No se pudo cargar las columnas del modelo general (columnas_general_encoded.pkl)")
else:
    print("[DEBUG] Columnas del modelo general cargadas correctamente")
# --- FIN DEBUG ---

def make_family_vector(values):
    row = pd.DataFrame([values])
    for col in variables_familiares:
        if col in NORMALIZADORES:
            row[col] = row[col].apply(NORMALIZADORES[col])
        else:
            row[col] = clean_text_series(row[col])
    encoded = pd.get_dummies(row, columns=variables_familiares, drop_first=False)
    matrix = np.column_stack([
        encoded[col].to_numpy(dtype="float32") if col in encoded.columns else np.zeros(len(encoded), dtype="float32")
        for col in cols_familiar
    ])
    return align_to_model_input(matrix, modelo_familiar)

def make_cole_vector(values):
    row = pd.DataFrame([values])
    for col in variables_cole:
        if col in NORMALIZADORES:
            row[col] = row[col].apply(NORMALIZADORES[col])
        else:
            row[col] = clean_text_series(row[col])
    encoded = pd.get_dummies(row, columns=variables_cole, drop_first=False)
    encoded = encoded.reindex(columns=cols_cole, fill_value=0).astype(int)
    if scaler_cole is not None:
        X_scaled = scaler_cole.transform(encoded)
        return X_scaled
    return encoded

def make_general_vector(values):
    row = pd.DataFrame([values])
    edad = pd.to_numeric(row.loc[:, "edad_presentacion"], errors="coerce").fillna(17).iloc[0]
    for col in variables_general_categoricas:
        if col in NORMALIZADORES:
            row[col] = row[col].apply(NORMALIZADORES[col])
        else:
            row[col] = clean_text_series(row[col])
    encoded = pd.get_dummies(row, columns=variables_general_categoricas, drop_first=False)
    encoded["edad_presentacion"] = edad
    encoded = encoded.reindex(columns=cols_general, fill_value=0)
    matrix = encoded.astype(float).to_numpy(dtype="float32")
    return align_to_model_input(matrix, modelo_general)


def gauge_probability(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%"},
        title={"text": "Probabilidad de alto desempeño"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#6C63FF"},
            "steps": [
                {"range": [0, 40], "color": "#FDECEC"},
                {"range": [40, 70], "color": "#FFF4D6"},
                {"range": [70, 100], "color": "#E9F7EF"},
            ],
        }
    ))
    fig.update_layout(template="simple_white", height=320, margin=dict(l=30, r=30, t=50, b=20))
    return fig

def gauge_score(score, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"valueformat": ".1f", "font": {"size": 48}},
        title={"text": title},
        gauge={
            "axis": {"range": [0, 500]},
            "bar": {"color": "#6C63FF"},
            "steps": [
                {"range": [0, 250], "color": "#FDECEC"},
                {"range": [250, 350], "color": "#FFF4D6"},
                {"range": [350, 500], "color": "#E9F7EF"},
            ],
        }
    ))
    fig.update_layout(template="simple_white", height=320, margin=dict(l=30, r=30, t=50, b=20))
    return fig

def card(children, border="#6C63FF"):
    return html.Div(
        children,
        style={
            "backgroundColor": "#F5F7FA",
            "padding": "25px",
            "borderRadius": "10px",
            "borderLeft": f"6px solid {border}",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.05)",
            "marginBottom": "30px",
        },
    )

def warning_missing(model_name):
    return html.Div([
        html.H5("Modelo no encontrado"),
        html.P(
            f"No se encontró el artefacto del {model_name}. "
            "Ejecuta primero train_all_models.py o guarda los modelos en la carpeta models."
        )
    ], style={"color": "#C92A2A"})

metrics_general = get_metrics_general()
metrics_familiar = get_metrics_familiar()
metrics_colegio = get_metrics_colegio()

# -------------------------

# -------------------------
# CALLBACKS
# -------------------------
from dash import callback

@app.callback(
    Output("resultado-familiar", "children"),
    Input("btn-familiar", "n_clicks"),
    State("fam_estrato", "value"),
    State("fam_edu_madre", "value"),
    State("fam_edu_padre", "value"),
    State("fam_personas", "value"),
    State("fam_cuartos", "value"),
    State("fam_auto", "value"),
    State("fam_lavadora", "value"),
)
def predict_familiar(n, estrato, madre, padre, personas, cuartos, auto, lavadora):
    if not n:
        return ""
    if modelo_familiar is None or cols_familiar is None:
        return warning_missing("modelo familiar")
    values = {
        "fami_estratovivienda": estrato,
        "fami_educacionmadre": madre,
        "fami_educacionpadre": padre,
        "fami_personashogar": personas,
        "fami_cuartoshogar": cuartos,
        "fami_tieneautomovil": auto,
        "fami_tienelavadora": lavadora,
    }
    try:
        X = make_family_vector(values)
        prob = float(modelo_familiar.predict(X, verbose=0).ravel()[0])
        clasificacion = "Alto desempeño" if prob >= 0.5 else "No alto desempeño"
        return html.Div([
            dcc.Graph(figure=gauge_probability(prob)),
            html.H4(f"Clasificación estimada: {clasificacion}"),
            html.P(f"Probabilidad estimada de alto desempeño: {prob*100:.1f}%"),
            html.P("Interpretación: el resultado se basa únicamente en condiciones familiares, por lo que debe usarse como apoyo analítico y no como decisión definitiva."),
        ], style={"marginTop": "25px"})
    except Exception as e:
        return html.Div(["Error en la predicción familiar: ", str(e)], style={"color": "#C92A2A"})

@app.callback(
    Output("resultado-colegio", "children"),
    Input("btn-colegio", "n_clicks"),
    State("cole_naturaleza", "value"),
    State("cole_bilingue", "value"),
    State("cole_jornada", "value"),
    State("cole_calendario", "value"),
    State("cole_area", "value"),
    State("cole_caracter", "value"),
    State("cole_genero", "value"),
)
def predict_colegio(n, naturaleza, bilingue, jornada, calendario, area, caracter, genero):
    if not n:
        return ""
    if modelo_cole is None or cols_cole is None:
        return warning_missing("modelo colegio")
    values = {
        "cole_naturaleza": naturaleza,
        "cole_bilingue": bilingue,
        "cole_jornada": jornada,
        "cole_calendario": calendario,
        "cole_area_ubicacion": area,
        "cole_caracter": caracter,
        "cole_genero": genero,
    }
    try:
        X = make_cole_vector(values)
        prob = float(modelo_cole.predict(X, verbose=0).ravel()[0])
        prob = max(0, min(1, prob))
        clasificacion = "Alto desempeño" if prob >= 0.5 else "No alto desempeño"
        return html.Div([
            dcc.Graph(figure=gauge_probability(prob)),
            html.H4(f"Clasificación estimada: {clasificacion}"),
            html.P(f"Probabilidad estimada de alto desempeño: {prob*100:.1f}%"),
            html.P("Interpretación: este modelo estima la probabilidad de alto desempeño con base en características institucionales del colegio."),
        ], style={"marginTop": "25px"})
    except Exception as e:
        return html.Div(["Error en la predicción colegio: ", str(e)], style={"color": "#C92A2A"})

@app.callback(
    Output("resultado-general", "children"),
    Input("btn-general", "n_clicks"),
    State("gen_edad", "value"),
    State("gen_jornada", "value"),
    State("gen_calendario", "value"),
    State("gen_bilingue", "value"),
    State("gen_naturaleza", "value"),
    State("gen_internet", "value"),
    State("gen_computador", "value"),
    State("gen_madre", "value"),
    State("gen_padre", "value"),
    State("gen_estrato", "value"),
    State("gen_auto", "value"),
)
def predict_general(n, edad, jornada, calendario, bilingue, naturaleza, internet, computador, madre, padre, estrato, auto):
    if not n:
        return ""
    if modelo_general is None or cols_general is None:
        return warning_missing("modelo general")
    values = {
        "edad_presentacion": edad,
        "cole_jornada": jornada,
        "cole_calendario": calendario,
        "cole_bilingue": bilingue,
        "cole_naturaleza": naturaleza,
        "fami_tieneinternet": internet,
        "fami_tienecomputador": computador,
        "fami_educacionmadre": madre,
        "fami_educacionpadre": padre,
        "fami_estratovivienda": estrato,
        "fami_tieneautomovil": auto,
    }
    try:
        X = make_general_vector(values)
        score = float(modelo_general.predict(X, verbose=0).ravel()[0])
        score = max(0, min(500, score))
        return html.Div([
            dcc.Graph(figure=gauge_score(score, "Puntaje global estimado")),
            html.H4(f"Puntaje global estimado: {score:.1f}"),
            html.P("Interpretación: este modelo estima el puntaje global usando variables mixtas (familiares, tecnológicas e institucionales)."),
        ], style={"marginTop": "25px"})
    except Exception as e:
        return html.Div(["Error en la predicción general: ", str(e)], style={"color": "#C92A2A"})

# -------------------------
# LAYOUT
# -------------------------

app.layout = html.Div([
    html.H2("Dashboard Predictivo Saber 11 - Atlántico"),
    html.P("Producto analítico para la Secretaría de Educación del Atlántico."),
    dcc.Tabs(id="tabs", value="tab-familiar", children=[
        dcc.Tab(label="Modelo Familiar", value="tab-familiar", children=[
            html.Br(),
            card([
                html.H3("1. Predicción de alto desempeño según condiciones familiares"),
                html.P("Este modelo estima la probabilidad de que un estudiante obtenga puntaje global alto, usando variables familiares y socioeconómicas."),
                html.Div([
                    html.Div([
                        html.Label("Estrato de vivienda"),
                        dcc.Dropdown(id="fam_estrato", options=options_for("fami_estratovivienda"), value=default_for("fami_estratovivienda"), clearable=False),
                        html.Label("Educación madre"),
                        dcc.Dropdown(id="fam_edu_madre", options=options_for("fami_educacionmadre"), value=default_for("fami_educacionmadre"), clearable=False),
                        html.Label("Educación padre"),
                        dcc.Dropdown(id="fam_edu_padre", options=options_for("fami_educacionpadre"), value=default_for("fami_educacionpadre"), clearable=False),
                    ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),
                    html.Div([
                        html.Label("Personas en el hogar"),
                        dcc.Dropdown(id="fam_personas", options=options_for("fami_personashogar"), value=default_for("fami_personashogar"), clearable=False),
                        html.Label("Cuartos del hogar"),
                        dcc.Dropdown(id="fam_cuartos", options=options_for("fami_cuartoshogar"), value=default_for("fami_cuartoshogar"), clearable=False),
                        html.Label("Tiene automóvil"),
                        dcc.Dropdown(id="fam_auto", options=options_for("fami_tieneautomovil"), value=default_for("fami_tieneautomovil"), clearable=False),
                        html.Label("Tiene lavadora"),
                        dcc.Dropdown(id="fam_lavadora", options=options_for("fami_tienelavadora"), value=default_for("fami_tienelavadora"), clearable=False),
                    ], style={"width": "48%", "display": "inline-block", "marginLeft": "4%", "verticalAlign": "top"}),
                ]),
                html.Br(),
                html.Button("Calcular predicción familiar", id="btn-familiar", n_clicks=0),
                dcc.Loading(type="circle", children=html.Div(id="resultado-familiar")),
            ])
        ]),
        dcc.Tab(label="Modelo Colegio", value="tab-colegio", children=[
            html.Br(),
            card([
                html.H3("2. Predicción del puntaje global según características del colegio"),
                html.P("Este modelo de regresión estima el puntaje global a partir de variables institucionales."),
                html.Div([
                    html.Div([
                        html.Label("Naturaleza"),
                        dcc.Dropdown(id="cole_naturaleza", options=options_for("cole_naturaleza"), value=default_for("cole_naturaleza"), clearable=False),
                        html.Label("Bilingüe"),
                        dcc.Dropdown(id="cole_bilingue", options=options_for("cole_bilingue"), value=default_for("cole_bilingue"), clearable=False),
                        html.Label("Jornada"),
                        dcc.Dropdown(id="cole_jornada", options=options_for("cole_jornada"), value=default_for("cole_jornada"), clearable=False),
                        html.Label("Calendario"),
                        dcc.Dropdown(id="cole_calendario", options=options_for("cole_calendario"), value=default_for("cole_calendario"), clearable=False),
                    ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),
                    html.Div([
                        html.Label("Área de ubicación"),
                        dcc.Dropdown(id="cole_area", options=options_for("cole_area_ubicacion"), value=default_for("cole_area_ubicacion"), clearable=False),
                        html.Label("Carácter"),
                        dcc.Dropdown(id="cole_caracter", options=options_for("cole_caracter"), value=default_for("cole_caracter"), clearable=False),
                        html.Label("Género del colegio"),
                        dcc.Dropdown(id="cole_genero", options=options_for("cole_genero"), value=default_for("cole_genero"), clearable=False),
                    ], style={"width": "48%", "display": "inline-block", "marginLeft": "4%", "verticalAlign": "top"}),
                ]),
                html.Br(),
                html.Button("Calcular probabilidad alto desempeño colegio", id="btn-colegio", n_clicks=0),
                dcc.Loading(type="circle", children=html.Div(id="resultado-colegio")),
            ])
        ]),
        dcc.Tab(label="Modelo General", value="tab-general", children=[
            html.Br(),
            card([
                html.H3("3. Predicción del puntaje global con variables generales"),
                html.P("Este modelo estima el puntaje global usando edad, recursos tecnológicos, variables familiares y algunas características institucionales."),
                html.Div([
                    html.Div([
                        html.Label("Edad de presentación"),
                        dcc.Input(id="gen_edad", type="number", value=17, min=13, max=60, step=1, style={"width": "100%"}),
                        html.Label("Jornada"),
                        dcc.Dropdown(id="gen_jornada", options=options_for("cole_jornada"), value=default_for("cole_jornada"), clearable=False),
                        html.Label("Calendario"),
                        dcc.Dropdown(id="gen_calendario", options=options_for("cole_calendario"), value=default_for("cole_calendario"), clearable=False),
                        html.Label("Bilingüe"),
                        dcc.Dropdown(id="gen_bilingue", options=options_for("cole_bilingue"), value=default_for("cole_bilingue"), clearable=False),
                        html.Label("Naturaleza"),
                        dcc.Dropdown(id="gen_naturaleza", options=options_for("cole_naturaleza"), value=default_for("cole_naturaleza"), clearable=False),
                    ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),
                    html.Div([
                        html.Label("Internet"),
                        dcc.Dropdown(id="gen_internet", options=options_for("fami_tieneinternet"), value=default_for("fami_tieneinternet"), clearable=False),
                        html.Label("Computador"),
                        dcc.Dropdown(id="gen_computador", options=options_for("fami_tienecomputador"), value=default_for("fami_tienecomputador"), clearable=False),
                        html.Label("Educación madre"),
                        dcc.Dropdown(id="gen_madre", options=options_for("fami_educacionmadre"), value=default_for("fami_educacionmadre"), clearable=False),
                        html.Label("Educación padre"),
                        dcc.Dropdown(id="gen_padre", options=options_for("fami_educacionpadre"), value=default_for("fami_educacionpadre"), clearable=False),
                        html.Label("Estrato"),
                        dcc.Dropdown(id="gen_estrato", options=options_for("fami_estratovivienda"), value=default_for("fami_estratovivienda"), clearable=False),
                        html.Label("Automóvil"),
                        dcc.Dropdown(id="gen_auto", options=options_for("fami_tieneautomovil"), value=default_for("fami_tieneautomovil"), clearable=False),
                    ], style={"width": "48%", "display": "inline-block", "marginLeft": "4%", "verticalAlign": "top"}),
                ]),
                html.Br(),
                html.Button("Calcular puntaje estimado general", id="btn-general", n_clicks=0),
                dcc.Loading(type="circle", children=html.Div(id="resultado-general")),
            ])
        ]),
        dcc.Tab(label="Comparar modelos", value="tab-comparar", children=[
            html.Br(),
            card([
                html.H2("Comparación de modelos y métricas"),
                html.P("Compara el desempeño de los tres modelos principales usando métricas de validación. Pasa el mouse sobre las métricas para ver su significado."),
                # Tabla de métricas
                html.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("Modelo"), html.Th("Tipo"), html.Th("AUC ROC"), html.Th("Precisión"), html.Th("Recall"), html.Th("F1"), html.Th("RMSE"), html.Th("MAE"), html.Th("R²")
                        ])
                    ]),
                    html.Tbody([
                        html.Tr([
                            html.Td("Familiar"), html.Td("Clasificación"),
                            html.Td(html.B(f"{get_metrics_familiar().get('AUC', '-')}", style={"color": "#2ecc40"})),
                            html.Td(get_metrics_familiar().get("Precision", "-")),
                            html.Td(get_metrics_familiar().get("Recall", "-")),
                            html.Td(get_metrics_familiar().get("F1", "-")),
                            html.Td("-"), html.Td("-"), html.Td("-")
                        ]),
                        html.Tr([
                            html.Td("Colegio"), html.Td("Clasificación"),
                            html.Td(get_metrics_colegio().get("AUC", "-")),
                            html.Td(get_metrics_colegio().get("Precision", "-")),
                            html.Td(get_metrics_colegio().get("Recall", "-")),
                            html.Td(get_metrics_colegio().get("F1", "-")),
                            html.Td("-"), html.Td("-"), html.Td("-")
                        ]),
                        html.Tr([
                            html.Td("General"), html.Td("Regresión"),
                            html.Td("-"), html.Td("-"), html.Td("-"), html.Td("-"),
                            html.Td(get_metrics_general().get("RMSE", "-")),
                            html.Td(get_metrics_general().get("MAE", "-")),
                            html.Td(get_metrics_general().get("R2", "-")),
                        ]),
                    ])
                ], style={"width": "100%", "marginBottom": "30px", "fontSize": "16px"}),
                html.Div([
                    html.Div([
                        dcc.Graph(figure=clasificacion_metrics_figure()),
                    ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),
                    html.Div([
                        dcc.Graph(figure=regresion_metrics_figure()),
                    ], style={"width": "48%", "display": "inline-block", "marginLeft": "4%", "verticalAlign": "top"}),
                ], style={"width": "100%", "display": "flex", "flexWrap": "wrap"}),
                html.Div([
                    dcc.Graph(figure=radar_metrics_figure()),
                ], style={"width": "100%", "marginTop": "30px"}),
                html.P("* Las métricas son de validación y pueden actualizarse según los resultados más recientes.", style={"fontSize": "13px", "color": "#888"}),
            ])
        ])
    ])
])

if __name__ == "__main__":
    app.run(debug=True, port=8051, host="0.0.0.0")