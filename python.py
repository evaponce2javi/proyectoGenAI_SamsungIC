import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

sns.set_style("whitegrid")
pd.set_option("display.max_columns", None)

versions = pd.DataFrame(
    {"Version": [np.__version__, pd.__version__, sns.__version__]},
    index=["numpy", "pandas", "seaborn"],
)
display(Markdown("### Entorno"))
display(versions)
print("Entorno listo.")

DATA_PATH = "./sic_mobile_spec.xlsx"

df = pd.read_excel(DATA_PATH, sheet_name=0)
print(f"Filas: {len(df)} | Columnas: {df.shape[1]}")
display(df.head(3))
df.info()

import re

def parse_price(x):
    """'$1,000' / '1200' -> float; vacío/erróneo -> np.nan."""
    if pd.isna(x):
        return np.nan
    m = re.search(r"[\d,]+(\.\d+)?", str(x))
    if not m:
        return np.nan
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return np.nan

def parse_storage(x):
    """'1TB'->1024, '256GB'->256, '128'->128 (GB)."""
    if pd.isna(x):
        return np.nan
    t = str(x).upper().strip()
    m_tb = re.search(r"(\d+\.?\d*)\s*TB", t)
    if m_tb:
        return float(m_tb.group(1)) * 1024.0
    m_gb = re.search(r"(\d+\.?\d*)\s*GB", t)
    if m_gb:
        return float(m_gb.group(1))
    return pd.to_numeric(t, errors="coerce")

def parse_camera(x):
    """MEJORA: 0 MP es imposible -> se trata como dato faltante (NaN)."""
    v = pd.to_numeric(x, errors="coerce")
    if pd.isna(v) or v <= 0:
        return np.nan
    return float(v)

def min_max_scale(col):
    """MEJORA: protección contra división por cero en columnas constantes."""
    lo, hi = col.min(), col.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.0, index=col.index)
    return (col - lo) / (hi - lo)

print("Helpers definidos: parse_price, parse_storage, parse_camera, min_max_scale")

df_clean = df.copy()
df_clean["price_clean"]   = df_clean["price"].apply(parse_price)
df_clean["storage_clean"] = df_clean["storage"].apply(parse_storage)
df_clean["camera"]        = df_clean["camera"].apply(parse_camera)  # 0 -> NaN

n_cam_missing = df_clean["camera"].isna().sum()
print(f"Cámaras marcadas como faltantes (eran 0): {n_cam_missing}")
display(df_clean[["name", "price", "price_clean", "storage", "storage_clean", "camera"]].head(10))

features = ["price_clean", "storage_clean", "ram", "display_size", "battery", "weight", "camera"]
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, col in zip(axes.ravel(), features):
    sns.histplot(df_clean[col].dropna(), kde=True, bins=30, ax=ax)
    ax.set_title(f"{col} distribution")
    ax.set_ylabel("Frequency")
axes.ravel()[-1].axis("off")
plt.tight_layout(); plt.show()

# mapa de calor en lugar de una Series cruda
num = df_clean.select_dtypes(include=[np.number])
plt.figure(figsize=(9, 7))
sns.heatmap(num.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Matriz de correlación"); plt.tight_layout(); plt.show()

print("Correlación de cada feature con price_clean:")
print(num.corr(numeric_only=True)["price_clean"].sort_values(ascending=False).round(3))

df_std = df_clean.copy()

# Flags de negocio
df_std["has_5g"]  = df_std["name"].str.contains("5g", case=False, na=False).astype(float)
df_std["amoled"]  = df_std["display_type"].str.contains("amoled", case=False, na=False).astype(float)

# Min-max sobre todas las columnas numéricas -> {col}_norm
numeric_cols = df_std.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df_std[f"{col}_norm"] = min_max_scale(df_std[col])

# Precio invertido: barato = mejor
df_std["price_inv_norm"] = 1.0 - df_std["price_clean_norm"]

# MEJORA: cámaras faltantes -> 0.0 en el score (peor), de forma explícita y reportada
df_std["camera_norm"] = df_std["camera_norm"].fillna(0.0)
display(df_std.head(3))

required_columns = [
    "name", "rating", "price_clean",
    "rating_norm", "price_inv_norm", "camera_norm", "display_size_norm",
    "battery_norm", "storage_clean_norm", "ram_norm", "weight_norm",
    "has_5g_norm", "amoled_norm",
]
missing = [c for c in required_columns if c not in df_std.columns]
if missing:
    raise ValueError(f"Faltan columnas requeridas: {missing}")

combined_df = df_std[required_columns].copy()
# Seguridad: ningún NaN debe propagarse al scoring
norm_cols = [c for c in required_columns if c.endswith("_norm")]
combined_df[norm_cols] = combined_df[norm_cols].fillna(0.0)

print(f"combined_df listo. Shape: {combined_df.shape}")
display(combined_df.head(5).round(3))

PERSONAS = ["Student", "Creator", "Gaming", "Business_Professional", "Senior"]

PERSONA_WEIGHTS = {
    "Student": {"price_inv_norm": 0.40, "battery_norm": 0.20, "storage_clean_norm": 0.20,
                "ram_norm": 0.10, "rating_norm": 0.10},
    "Creator": {"camera_norm": 0.40, "storage_clean_norm": 0.25, "rating_norm": 0.15,
                "amoled_norm": 0.10, "ram_norm": 0.10},
    "Gaming": {"ram_norm": 0.40, "battery_norm": 0.25, "weight_norm": 0.15, "price_inv_norm": 0.20},
    "Business_Professional": {"battery_norm": 0.30, "storage_clean_norm": 0.25, "price_inv_norm": 0.15,
                              "amoled_norm": 0.10, "weight_norm": 0.10, "has_5g_norm": 0.10},
    "Senior": {"price_inv_norm": 0.40, "weight_norm": 0.30, "battery_norm": 0.30},
}

def validate_weights(weights_dict, available_cols, tol=1e-6):
    """MEJORA: valida suma≈1.0 (renormaliza) y existencia de columnas."""
    clean = {}
    for persona, w in weights_dict.items():
        bad = [c for c in w if c not in available_cols]
        if bad:
            print(f"[AVISO] {persona}: columnas inexistentes ignoradas -> {bad}")
        w = {c: v for c, v in w.items() if c in available_cols}
        total = sum(w.values())
        if abs(total - 1.0) > tol:
            print(f"[AVISO] {persona}: los pesos suman {total:.3f}, se renormalizan a 1.0")
            w = {c: v / total for c, v in w.items()} if total else w
        clean[persona] = w
    return clean

PERSONA_WEIGHTS = validate_weights(PERSONA_WEIGHTS, set(combined_df.columns))
print("\nPesos validados para:", list(PERSONA_WEIGHTS))

def score_persona_from_norms(t: pd.DataFrame, persona: str) -> pd.Series:
    """Suma ponderada de columnas normalizadas, escalada a 0–100."""
    if persona not in PERSONA_WEIGHTS:
        print(f"Aviso: persona '{persona}' desconocida.")
        return pd.Series(0.0, index=t.index)
    score = pd.Series(0.0, index=t.index)
    for col, w in PERSONA_WEIGHTS[persona].items():
        if col in t.columns:
            score += t[col].fillna(0.0) * w
    return score * 100.0

def explain_score(t: pd.DataFrame, persona: str, idx) -> pd.DataFrame:
    """MEJORA: desglosa la contribución de cada feature al score de un teléfono."""
    rows = []
    for col, w in PERSONA_WEIGHTS[persona].items():
        val = float(t.loc[idx, col]) if col in t.columns else 0.0
        rows.append({"feature": col, "valor_norm": round(val, 3),
                     "peso": w, "contribución(0-100)": round(val * w * 100, 2)})
    out = pd.DataFrame(rows).sort_values("contribución(0-100)", ascending=False)
    return out.reset_index(drop=True)

for persona in PERSONAS:
    tmp = combined_df.copy()
    tmp["score"] = score_persona_from_norms(tmp, persona)
    top5 = tmp.sort_values("score", ascending=False).head(5)
    show = top5[["name", "price_clean", "rating", "score"]].reset_index(drop=True)
    show.index += 1
    show.columns = ["Name", "Price ($)", "Rating", "Score"]
    show["Price ($)"] = show["Price ($)"].map(lambda v: f"${v:,.0f}")
    show["Rating"] = show["Rating"].map(lambda v: f"{v:.1f}")
    show["Score"] = show["Score"].map(lambda v: f"{v:.1f}")
    display(Markdown(f"### Top-5 para **{persona}**"))
    display(show)

# Ejemplo de explicabilidad para el #1 de Creator
demo = combined_df.copy()
demo["score"] = score_persona_from_norms(demo, "Creator")
best_idx = demo["score"].idxmax()
display(Markdown(f"#### ¿Por qué **{combined_df.loc[best_idx, 'name']}** es el #1 de *Creator*?"))
display(explain_score(combined_df, "Creator", best_idx))

score_matrix = pd.DataFrame(
    {p: score_persona_from_norms(combined_df, p) for p in PERSONAS},
    index=combined_df.index,
)
results = []
for persona in PERSONAS:
    top_idx = score_matrix[persona].idxmax()
    own = score_matrix.loc[top_idx, persona]
    best_other = score_matrix.loc[top_idx, [c for c in PERSONAS if c != persona]].max()
    results.append({
        "persona": persona,
        "top_phone": combined_df.loc[top_idx, "name"],
        "score_propio": round(own, 1),
        "mejor_otro": round(best_other, 1),
        "coherente": "✅" if own >= best_other else "⚠️",
    })
display(pd.DataFrame(results))
print("Coherente = el top de la persona puntúa al menos tan alto bajo su propia persona.")

import ipywidgets as widgets
from IPython.display import HTML

def build_persona_recommender(data: pd.DataFrame):
    df_p = data.copy()
    price_max = int(np.ceil(df_p["price_clean"].max() / 100.0) * 100)

    price_slider = widgets.IntRangeSlider(value=[0, price_max], min=0, max=price_max, step=10,
                                          description="Precio ($):", continuous_update=False,
                                          layout=widgets.Layout(width="95%"))
    persona_dd = widgets.Dropdown(options=PERSONAS, value=PERSONAS[0], description="Persona:",
                                  layout=widgets.Layout(width="95%"))
    topk = widgets.IntSlider(value=5, min=1, max=15, step=1, description="Top K:",
                             layout=widgets.Layout(width="95%"))
    run = widgets.Button(description="Recomendar", button_style="success", icon="search")
    out = widgets.Output()

    def on_click(_):
        with out:
            out.clear_output(wait=True)
            lo, hi = price_slider.value
            persona = persona_dd.value
            sub = df_p[(df_p["price_clean"] >= lo) & (df_p["price_clean"] <= hi)].copy()
            if sub.empty:
                display(HTML("<b style='color:red'>Sin dispositivos en ese rango.</b>")); return
            sub["score"] = score_persona_from_norms(sub, persona)
            top = sub.sort_values("score", ascending=False).head(topk.value)
            tbl = top[["name", "price_clean", "rating", "score"]].reset_index(drop=True)
            tbl.index += 1
            tbl.columns = ["Name", "Price ($)", "Rating", "Score"]
            display(HTML(f"<h3>Top {persona}</h3>"))
            display(tbl.style.format({"Price ($)": "${:,.0f}", "Rating": "{:.1f}", "Score": "{:.1f}"}))
            display(Markdown(f"**Desglose del #1 ({top.iloc[0]['name']}):**"))
            display(explain_score(sub, persona, top.index[0]))

    run.on_click(on_click)
    ui = widgets.VBox([widgets.HBox([price_slider]), persona_dd, topk, run, out],
                      layout=widgets.Layout(border="1px solid #ddd", padding="12px", max_width="820px"))
    display(Markdown("## 📱 Recomendador por estilo de vida"))
    display(ui)
    on_click(None)

build_persona_recommender(combined_df)

