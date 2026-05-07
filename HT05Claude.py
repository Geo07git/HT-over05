import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, roc_curve
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════
st.set_page_config(page_title="HT Goal Predictor Pro", layout="wide", initial_sidebar_state="expanded")

LEAGUES ={
    "ENG-Premier League": "https://www.football-data.co.uk/mmz4281/{s}/E0.csv",
    "ENG-Championship": "https://www.football-data.co.uk/mmz4281/{s}/E1.csv",
    "ENG-League One": "https://www.football-data.co.uk/mmz4281/{s}/E2.csv",
    "ENG-League Two": "https://www.football-data.co.uk/mmz4281/{s}/E3.csv",
    "ENG-League Conf": "https://www.football-data.co.uk/mmz4281/{s}/EC.csv",
    "ESP-La Liga": "https://www.football-data.co.uk/mmz4281/{s}/SP1.csv",
    "ESP-La Liga 2": "https://www.football-data.co.uk/mmz4281/{s}/SP2.csv",
    "GER-Bundesliga": "https://www.football-data.co.uk/mmz4281/{s}/D1.csv",
    "GER-Bundesliga 2": "https://www.football-data.co.uk/mmz4281/{s}/D2.csv",
    "ITA-Serie A": "https://www.football-data.co.uk/mmz4281/{s}/I1.csv",
    "ITA-Serie B": "https://www.football-data.co.uk/mmz4281/{s}/I2.csv",
    "FRA-Ligue 1": "https://www.football-data.co.uk/mmz4281/{s}/F1.csv",
    "FRA-Ligue 2": "https://www.football-data.co.uk/mmz4281/{s}/F2.csv",
    "NED-Eredivisie": "https://www.football-data.co.uk/mmz4281/{s}/N1.csv",
    "POR-Primeira Liga": "https://www.football-data.co.uk/mmz4281/{s}/P1.csv",
    "BEL-Jupiler Pro": "https://www.football-data.co.uk/mmz4281/{s}/B1.csv",
    "TUR-Super Lig": "https://www.football-data.co.uk/mmz4281/{s}/T1.csv",
    "GRE-Super League": "https://www.football-data.co.uk/mmz4281/{s}/G1.csv",
    "SCO-Scottish Prem": "https://www.football-data.co.uk/mmz4281/{s}/SC0.csv",
    "SCO-Scottish Champ": "https://www.football-data.co.uk/mmz4281/{s}/SC1.csv",
    "SCO-Scottish Div1": "https://www.football-data.co.uk/mmz4281/{s}/SC2.csv",
    "SCO-Scottish Div2": "https://www.football-data.co.uk/mmz4281/{s}/SC3.csv",
}

SEASONS = ["1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"]

COLORS = {
    "LightGBM":          "#63b3ed",
    "XGBoost":           "#68d391",
    "RandomForest":      "#b794f4",
    "LogisticRegression":"#fbd38d",
    "Ensemble":          "#f6e05e",
}
BADGES = {
    "LightGBM":          ("badge-lgb",  "#1a365d", "#63b3ed", "#2b6cb0"),
    "XGBoost":           ("badge-xgb",  "#1c4532", "#68d391", "#276749"),
    "RandomForest":      ("badge-rf",   "#322659", "#b794f4", "#553c9a"),
    "LogisticRegression":("badge-lr",   "#7b341e", "#fbd38d", "#c05621"),
    "Ensemble":          ("badge-ens",  "#2d3748", "#f6e05e", "#b7791f"),
}

MODEL_ORDER = ["LightGBM", "XGBoost", "RandomForest", "LogisticRegression", "Ensemble"]

# ═══════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════
badge_css = "\n".join([
    f'.{cls} {{ background:{bg}; color:{fg}; border: 1px solid {br}; }}'
    for name, (cls, bg, fg, br) in BADGES.items()
])

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
  .main {{ background: #0e1117; }}

  .metric-card {{
    background: #1c1f2e; border: 1px solid #2a2d3e; border-radius: 12px;
    padding: 20px 24px; text-align: center; transition: box-shadow 0.2s;
  }}
  .metric-card:hover {{ box-shadow: 0 4px 20px rgba(99,179,237,0.15); }}
  .metric-label {{ color: #718096; font-size: 12px; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 8px; }}
  .metric-value {{ font-size: 32px; font-weight: 700; color: #e2e8f0; }}
  .metric-sub {{ font-size: 12px; color: #4a5568; margin-top: 4px; }}

  .model-badge {{ display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; margin-right: 6px; }}
  {badge_css}

  .prob-bar-container {{ background: #2d3748; border-radius: 8px; height: 10px; overflow: hidden; margin-top: 6px; position: relative; }}
  .prob-bar-fill {{ height: 100%; border-radius: 8px; transition: width 0.5s ease; }}

  .prediction-box {{ border-radius: 16px; padding: 32px; text-align: center; margin-top: 16px; }}
  .pred-high {{ background: linear-gradient(135deg,#1a3a2a,#1c4532); border: 2px solid #38a169; }}
  .pred-med  {{ background: linear-gradient(135deg,#2d2a1a,#3d3119); border: 2px solid #d69e2e; }}
  .pred-low  {{ background: linear-gradient(135deg,#2d1a1a,#3d1919); border: 2px solid #e53e3e; }}

  .section-title {{
    font-size: 18px; font-weight: 600; color: #e2e8f0;
    margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid #2a2d3e;
  }}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    all_data = []
    for liga, url in LEAGUES.items():
        for s in SEASONS:
            try:
                r = requests.get(url.format(s=s), timeout=10)
                if r.status_code == 200 and len(r.text) > 500:
                    df = pd.read_csv(StringIO(r.text), low_memory=False)
                    df["Liga"] = liga
                    df["Sezon"] = s
                    all_data.append(df)
            except Exception:
                pass
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


# ═══════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame):
    cols = ["Date","HomeTeam","AwayTeam","HTHG","HTAG","FTHG","FTAG","HST","AST","HC","AC","Liga","Sezon"]
    df = df[[c for c in cols if c in df.columns]].dropna(subset=["HomeTeam","AwayTeam","HTHG","HTAG"]).copy()
    df["target"] = ((df["HTHG"] + df["HTAG"]) > 0).astype(int)
    df["ht_total"] = df["HTHG"] + df["HTAG"]

    def _roll(data, team_col, val_col, window=10, min_p=3, agg="mean"):
        """Vectorized rolling — sort once per group, return Series aligned to original index."""
        result = pd.Series(np.nan, index=data.index)
        for team, idx in data.groupby(team_col).groups.items():
            sub = data.loc[idx].sort_values("Date")[val_col].shift(1)
            rolled = sub.rolling(window, min_periods=min_p)
            result.loc[idx] = (rolled.mean() if agg == "mean" else rolled.std())
        return result

    df["home_rate"]     = _roll(df, "HomeTeam", "target")
    df["away_rate"]     = _roll(df, "AwayTeam", "target")
    df["home_htg"]      = _roll(df, "HomeTeam", "HTHG")
    df["away_htg"]      = _roll(df, "AwayTeam", "HTAG")
    df["home_conc"]     = _roll(df, "HomeTeam", "HTAG")
    df["away_conc"]     = _roll(df, "AwayTeam", "HTHG")
    df["home_ht_tot"]   = _roll(df, "HomeTeam", "ht_total")
    df["away_ht_tot"]   = _roll(df, "AwayTeam", "ht_total")
    df["home_sot"]      = _roll(df, "HomeTeam", "HST") if "HST" in df.columns else np.nan
    df["away_sot"]      = _roll(df, "AwayTeam", "AST") if "AST" in df.columns else np.nan
    df["home_std_htg"]  = _roll(df, "HomeTeam", "HTHG", agg="std")
    df["away_std_htg"]  = _roll(df, "AwayTeam", "HTAG", agg="std")
    df["home_std_sot"]  = _roll(df, "HomeTeam", "HST",  agg="std") if "HST" in df.columns else np.nan
    df["away_std_sot"]  = _roll(df, "AwayTeam", "AST",  agg="std") if "AST" in df.columns else np.nan

    #le = LabelEncoder()
    #df["liga_enc"] = le.fit_transform(df["Liga"].astype(str))

    base_cols = ["home_rate","away_rate","home_htg","away_htg","home_conc","away_conc",
                 "home_ht_tot","away_ht_tot","home_std_htg","away_std_htg"]
    df = df.dropna(subset=base_cols)

    for c in ["home_sot","away_sot","home_std_sot","away_std_sot"]:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    # Interaction features
    df["home_attack_vs_away_def"] = df["home_htg"]  - df["away_conc"]
    df["away_attack_vs_home_def"] = df["away_htg"]  - df["home_conc"]
    df["pressure"] = np.log1p(df["home_sot"] + df["away_sot"])
    df["imbalance"] = (df["home_rate"] - df["away_rate"]) * 2
    df["conv_home"] = df["home_htg"] / (df["home_sot"] + 1e-3)
    df["conv_away"] = df["away_htg"] / (df["away_sot"] + 1e-3)
    df["volatility"]      = df["home_std_htg"] + df["away_std_htg"]
    df["combined_attack"] = df["home_htg"] + df["away_htg"]

    features = [
        "home_rate","away_rate","home_htg","away_htg","home_conc","away_conc",
        "home_ht_tot","away_ht_tot","home_sot","away_sot",
        "home_std_htg","away_std_htg","home_std_sot","away_std_sot",
        "home_attack_vs_away_def","away_attack_vs_home_def",
        "pressure","imbalance","conv_home","conv_away","volatility","combined_attack",
    ]
    #features = [f for f in features if f != "liga_enc"]
    return df, features


# ═══════════════════════════════════════════════════════
# BUILD INPUT VECTOR
# ═══════════════════════════════════════════════════════
def build_input_vector(df, features, home, away, liga_sel):
    data = df if liga_sel == "All" else df[df["Liga"] == liga_sel]
    h_rows = data[data["HomeTeam"] == home]
    a_rows = data[data["AwayTeam"] == away]
    if h_rows.empty or a_rows.empty:
        return None, None, None

    h = h_rows.iloc[-1]
    a = a_rows.iloc[-1]

    home_feats   = [f for f in features if f.startswith("home_")]
    away_feats   = [f for f in features if f.startswith("away_")]
    global_feats = [f for f in features if f not in home_feats + away_feats]

    x = np.concatenate([
        h[home_feats].values,
        a[away_feats].values,
        h[global_feats].values if global_feats else []
    ]).reshape(1, -1)
    return x, h_rows, a_rows


# ═══════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def train_all_models(df: pd.DataFrame, features: list):
    df = df.sort_values("Date").copy()

    home_feats   = [f for f in features if f.startswith("home_")]
    away_feats   = [f for f in features if f.startswith("away_")]
    global_feats = [f for f in features if f not in home_feats + away_feats]

    X = np.concatenate([
        df[home_feats].values,
        df[away_feats].values,
        *([ df[global_feats].values ] if global_feats else [])
    ], axis=1)
    y = df["target"].values

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    cv = TimeSeriesSplit(n_splits=5)

    # ── Modele de bază (n_estimators redus pentru viteză) ──────────────────
    base_models = {
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            random_state=42, verbose=-1, n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            eval_metric="logloss", random_state=42, verbosity=0, n_jobs=-1
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500, max_depth=4, random_state=42, n_jobs=-1
        ),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    }

    trained, metrics_out = {}, {}

    for name, model in base_models.items():
        cal = CalibratedClassifierCV(model, cv=cv, method="isotonic")
        cal.fit(X_train, y_train)
        proba = cal.predict_proba(X_test)[:, 1]

        # CV AUC (light — pe train set)
        cv_scores = []
        for tr_idx, val_idx in cv.split(X_train):
            model.fit(X_train[tr_idx], y_train[tr_idx])
            cv_scores.append(roc_auc_score(y_train[val_idx], model.predict_proba(X_train[val_idx])[:, 1]))

        trained[name] = cal
        metrics_out[name] = {
            "AUC":    roc_auc_score(y_test, proba),
            "Brier":  brier_score_loss(y_test, proba),
            "LogLoss":log_loss(y_test, proba),
            "CV_AUC": float(np.mean(cv_scores)),
        }

    # ── Ensemble ────────────────────────────────────────────────────────────
    ens = VotingClassifier([(n, m) for n, m in base_models.items()], voting="soft", n_jobs=-1)
    cal_ens = CalibratedClassifierCV(ens, cv=cv, method="isotonic")
    cal_ens.fit(X_train, y_train)
    ens_proba = cal_ens.predict_proba(X_test)[:, 1]
    trained["Ensemble"] = cal_ens
    metrics_out["Ensemble"] = {
        "AUC":    roc_auc_score(y_test, ens_proba),
        "Brier":  brier_score_loss(y_test, ens_proba),
        "LogLoss":log_loss(y_test, ens_proba),
        "CV_AUC": float(np.mean([metrics_out[n]["CV_AUC"] for n in base_models])),
    }

    # ── ROC & Calibration curves ─────────────────────────────────────────────
    roc_data, cal_data = {}, {}
    for name, model in trained.items():
        p = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, p)
        roc_data[name] = (fpr, tpr)
        frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=10)
        cal_data[name] = (mean_pred, frac_pos)

    # ── Feature importance (LightGBM — nativ) ───────────────────────────────
    fi = None
    try:
        fi = trained["LightGBM"].calibrated_classifiers_[0].estimator.feature_importances_
    except Exception:
        pass

    return trained, metrics_out, roc_data, cal_data, fi, X_test, y_test


# ═══════════════════════════════════════════════════════
# BOOTSTRAP CI90 — optimizat (batch predict)
# ═══════════════════════════════════════════════════════
def bootstrap_ci90(models_dict, x_input, n_bootstrap=100, ci=0.90):
    alpha = (1 - ci) / 2
    rng = np.random.default_rng()
    # Generăm toate perturbările deodată
    noise_matrix = rng.normal(0, 0.08, (n_bootstrap, x_input.shape[1]))
    x_noisy_batch = np.clip(x_input + noise_matrix, None, None)  # shape (n_bootstrap, n_feats)

    results = {}
    for name, model in models_dict.items():
        preds = model.predict_proba(x_noisy_batch)[:, 1]
        preds = np.clip(preds, 1e-4, 1-1e-4)
        results[name] = {
            "mean":  float(np.mean(preds)),
            "lower": float(np.percentile(preds, alpha * 100)),
            "upper": float(np.percentile(preds, (1 - alpha) * 100)),
            "std":   float(np.std(preds)),
        }
    return results

# ═══════════════════════════════════════════════════════
# CACHED BACKTEST
# ═══════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def compute_backtest(_trained, df: pd.DataFrame, features: list, n: int = 100):
    last_n = df.tail(n).copy()
    med = last_n[features].median().values
    X = np.where(np.isnan(last_n[features].values), med, last_n[features].values)
    last_n["pred_prob"]      = _trained["Ensemble"].predict_proba(X)[:, 1]
    last_n["pred_bin"]       = (last_n["pred_prob"] >= 0.55).astype(int)
    last_n["correct"]        = (last_n["pred_bin"] == last_n["target"]).astype(int)
    last_n["cumulative_acc"] = last_n["correct"].expanding().mean()
    return last_n


# ═══════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚽ HT Goal Predictor Pro")
    st.markdown("---")
    st.markdown("**Modele antrenate:**")
    for name in MODEL_ORDER:
        cls = BADGES[name][0]
        st.markdown(f'<span class="model-badge {cls}">{name}</span>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Caracteristici:**")
    for feat in ["Home/Away HT goal rate", "HT goals scored/conceded",
                 "Half-time total rate", "Shots on target",
                 "Interaction features", "League encoding"]:
        st.markdown(f"- {feat}")
    st.markdown("---")
    #with st.sidebar:
    if st.button("🗑️ Reset Cache & Reantrenează"):
        st.cache_resource.clear()
        #st.cache_data.clear()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    st.markdown("---")
    st.caption("Date: football-data.co.uk | Sezoane: 16/17–25/26 | 22 ligi")


# ═══════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════
st.markdown("## ⚽ HT ≥ 1 Goal Predictor — Professional Edition")
st.markdown("Predicție probabilistică multi-model cu calibrare și evaluare statistică completă.")


# ═══════════════════════════════════════════════════════
# LOAD DATA & PREPROCESS
# ═══════════════════════════════════════════════════════
with st.spinner("📥 Se descarcă datele..."):
    raw = load_data()
    if raw.empty:
        st.error("Nu s-au putut descărca datele. Verifică conexiunea la internet.")
        st.stop()
    df, features = preprocess(raw)

# ═══════════════════════════════════════════════════════
# TRAINING TRIGGER
# ═══════════════════════════════════════════════════════
if "models_ready" not in st.session_state:
    st.session_state.models_ready = False

# Inițializare chei lipsă (protecție la rerun)
for key in ["trained", "metrics", "roc_data", "cal_data", "fi", "X_test", "y_test"]:
    if key not in st.session_state:
        st.session_state[key] = None

if not st.session_state.models_ready:
    st.warning("⚠️ Modelele nu sunt încă antrenate.")
    if st.button("🚀 Antrenează modelele"):
        with st.spinner("Se antrenează modelele... (o singură dată per sesiune)"):
            result = train_all_models(df, features)
        (st.session_state.trained, st.session_state.metrics,
         st.session_state.roc_data, st.session_state.cal_data,
         st.session_state.fi, st.session_state.X_test, st.session_state.y_test) = result
        st.session_state.models_ready = True
        st.rerun()
    st.stop()

# Guard dublu — verifică că toate cheile sunt populate
if not all(st.session_state.get(k) is not None for k in ["trained", "metrics", "roc_data", "cal_data"]):
    st.error("❌ Sesiunea a expirat sau a fost resetată. Apasă 'Antrenează modelele' din nou.")
    st.session_state.models_ready = False
    st.stop()

trained  = st.session_state.trained
metrics  = st.session_state.metrics
roc_data = st.session_state.roc_data
cal_data = st.session_state.cal_data
fi       = st.session_state.fi

st.success(f"✅ Modele gata! Dataset: **{len(df):,} meciuri** din 22 ligi și 10 sezoane.")


# ═══════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Predicție",
    "📊 Performanță Modele",
    "📈 Evaluare Statistică",
    "🔍 Analiză Avansată"
])


# ──────────────────────────────────────────────────────
# TAB 1 — PREDICȚIE
# ──────────────────────────────────────────────────────
with tab1:
    col1, col2, col3 = st.columns(3)

    # 🔹 Liga prima (important)
    liga_options = ["All"] + sorted(df["Liga"].dropna().unique())

    with col3:
        liga_sel = st.selectbox("🏆 Ligă", liga_options)

    # 🔹 Filtrare dataframe după ligă
    if liga_sel == "All":
        df_filtered = df
    else:
        df_filtered = df[df["Liga"] == liga_sel]

    # 🔹 Echipe doar din liga selectată
    home_options = sorted(df_filtered["HomeTeam"].dropna().unique())
    away_options = sorted(df_filtered["AwayTeam"].dropna().unique())

    with col1:
        home = st.selectbox("🏠 Echipa Gazdă", home_options)

    with col2:
        away = st.selectbox("✈️ Echipa Oaspete", away_options)

    # Model
    chosen_model = st.selectbox(
        "Model Principal",
        ["Ensemble (recomandat)", "LightGBM", "XGBoost", "RandomForest", "LogisticRegression"],
        index=0
    )
    model_key = "Ensemble" if "Ensemble" in chosen_model else chosen_model

    if st.button("🔮 Calculează Probabilitate", type="primary", use_container_width=True):
        x, h_rows, a_rows = build_input_vector(df, features, home, away, liga_sel)

        if x is None:
            st.warning("Nu există date suficiente pentru această combinație.")
        else:
            with st.spinner("Se calculează CI90..."):
                ci90 = bootstrap_ci90(trained, x, n_bootstrap=50)

            main_prob = ci90[model_key]["mean"]
            ci_lower  = ci90[model_key]["lower"]
            ci_upper  = ci90[model_key]["upper"]
            ci_std    = ci90[model_key]["std"]
            fair_odds = 1 / max(main_prob, 1e-6)

            if main_prob >= 0.65:
                box_cls, verdict, verdict_color = "pred-high", "🔥 SPOT PUTERNIC",   "#68d391"
            elif main_prob >= 0.52:
                box_cls, verdict, verdict_color = "pred-med",  "⚠️ EDGE MARGINAL",   "#f6ad55"
            else:
                box_cls, verdict, verdict_color = "pred-low",  "❌ FĂRĂ EDGE CLAR",  "#fc8181"

            fair_lo = 1 / max(ci_upper, 1e-6)
            fair_hi = 1 / max(ci_lower, 1e-6)

            st.markdown(f"""
            <div class="prediction-box {box_cls}">
              <div style="font-size:14px;color:#718096;margin-bottom:8px;">{home} vs {away}</div>
              <div style="font-size:52px;font-weight:700;color:{verdict_color}">{main_prob:.1%}</div>
              <div style="font-size:13px;color:#a0aec0;margin-top:4px;">Probabilitate HT ≥ 1 gol ({model_key})</div>
              <div style="margin-top:10px;">
                <span style="background:#2d3748;border-radius:8px;padding:6px 14px;font-size:15px;color:#a0aec0;">
                  CI 90%: <strong style="color:#e2e8f0">{ci_lower:.1%}</strong>
                  &nbsp;―&nbsp;
                  <strong style="color:#e2e8f0">{ci_upper:.1%}</strong>
                </span>
              </div>
              <div style="font-size:22px;font-weight:600;color:#e2e8f0;margin-top:12px;">Cote corecte: {fair_odds:.2f}</div>
              <div style="font-size:13px;color:#718096;margin-top:4px;">Range cote CI90: {fair_hi:.2f} — {fair_lo:.2f}</div>
              <div style="font-size:16px;font-weight:700;color:{verdict_color};margin-top:10px;">{verdict}</div>
              <div style="font-size:12px;color:#4a5568;margin-top:4px;">
                σ = {ci_std:.3f} | {'⚠️ Incertitudine mare' if ci_std > 0.05 else '✓ Estimare stabilă'}
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">Comparație toate modelele</div>', unsafe_allow_html=True)

            for mname in sorted(ci90, key=lambda n: -ci90[n]["mean"]):
                mprob  = ci90[mname]["mean"]
                mlow   = ci90[mname]["lower"]
                mhigh  = ci90[mname]["upper"]
                color  = COLORS.get(mname, "#a0aec0")
                pct    = int(mprob * 100)
                pct_lo = int(mlow  * 100)
                pct_hi = int(mhigh * 100)
                st.markdown(f"""
                <div style="margin-bottom:14px;">
                  <div style="display:flex;justify-content:space-between;font-size:13px;color:#e2e8f0;">
                    <span>{mname}</span>
                    <span style="font-weight:600;color:{color}">{mprob:.1%}
                      <span style="color:#718096;font-weight:400;font-size:11px;">CI90: [{mlow:.1%} – {mhigh:.1%}]</span>
                      &nbsp;cote: {1/max(mprob,1e-6):.2f}
                    </span>
                  </div>
                  <div class="prob-bar-container">
                    <div class="prob-bar-fill" style="width:{pct}%;background:{color};opacity:0.35;"></div>
                    <div style="position:absolute;top:0;left:{pct_lo}%;width:{pct_hi-pct_lo}%;height:100%;background:{color};opacity:0.7;border-radius:8px;"></div>
                    <div style="position:absolute;top:0;left:{pct-1}%;width:2px;height:100%;background:{color};border-radius:4px;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            probs_raw = {n: ci90[n]["mean"] for n in ci90}
            spread = max(probs_raw.values()) - min(probs_raw.values())
            consensus = np.mean(list(probs_raw.values()))
            ci90_w = ci90["Ensemble"]["upper"] - ci90["Ensemble"]["lower"]
            st.info(
                f"📌 **Consens:** {consensus:.1%} | **Spread:** {spread:.1%} "
                f"{'— divergente ⚠️' if spread > 0.10 else '— convergente ✓'} | "
                f"**CI90 Ensemble:** [{ci90['Ensemble']['lower']:.1%} – {ci90['Ensemble']['upper']:.1%}] "
                f"(lățime: {ci90_w:.1%})"
            )

            # Form recent
            st.markdown('<div class="section-title">Form recent (ultimele 5 meciuri HT)</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**{home} (acasă)**")
                last5_h = h_rows.tail(5)[["AwayTeam","HTHG","HTAG","target"]].copy()
                last5_h.columns = ["Oponent","HT_H","HT_A","HT≥1"]
                last5_h["HT≥1"] = last5_h["HT≥1"].map({1:"✅", 0:"❌"})
                st.dataframe(last5_h, hide_index=True, use_container_width=True)
            with c2:
                st.markdown(f"**{away} (deplasare)**")
                last5_a = a_rows.tail(5)[["HomeTeam","HTHG","HTAG","target"]].copy()
                last5_a.columns = ["Oponent","HT_H","HT_A","HT≥1"]
                last5_a["HT≥1"] = last5_a["HT≥1"].map({1:"✅", 0:"❌"})
                st.dataframe(last5_a, hide_index=True, use_container_width=True)


# ──────────────────────────────────────────────────────
# TAB 2 — PERFORMANȚĂ MODELE
# ──────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">Metrici de Performanță (Test Set 20%)</div>', unsafe_allow_html=True)

    cols = st.columns(5)
    for i, name in enumerate(MODEL_ORDER):
        m = metrics[name]
        cls = BADGES[name][0]
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label"><span class="model-badge {cls}">{name}</span></div>
              <div class="metric-value">{m['AUC']:.4f}</div>
              <div class="metric-sub">AUC-ROC</div>
              <div style="margin-top:10px;font-size:12px;color:#718096">
                Brier: {m['Brier']:.4f}<br>LogLoss: {m['LogLoss']:.4f}<br>CV AUC: {m['CV_AUC']:.4f}
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    rows = [{"Model": n, "AUC-ROC": f"{metrics[n]['AUC']:.4f}", "Brier": f"{metrics[n]['Brier']:.4f}",
             "Log Loss": f"{metrics[n]['LogLoss']:.4f}", "CV AUC": f"{metrics[n]['CV_AUC']:.4f}"}
            for n in MODEL_ORDER]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown("""
    > **AUC-ROC**: Aproape de 1.0 = bun (0.5 = random).
    > **Brier Score**: Aproape de 0 = probabilități precise.
    > **Log Loss**: Penalizează estimările greșite cu mare încredere.
    """)
    st.info(f"📊 **Rata de bază HT ≥ 1 gol:** {df['target'].mean():.1%} din {len(df):,} meciuri")


# ──────────────────────────────────────────────────────
# TAB 3 — EVALUARE STATISTICĂ
# ──────────────────────────────────────────────────────
with tab3:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">Curbe ROC</div>', unsafe_allow_html=True)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                      line=dict(dash="dash", color="#4a5568", width=1), name="Random"))
        for name in MODEL_ORDER:
            fpr, tpr = roc_data[name]
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                          name=f"{name} ({metrics[name]['AUC']:.3f})",
                                          line=dict(color=COLORS[name], width=2)))
        fig_roc.update_layout(
            xaxis_title="FPR", yaxis_title="TPR",
            paper_bgcolor="#1c1f2e", plot_bgcolor="#1c1f2e",
            font=dict(color="#e2e8f0", size=12),
            legend=dict(bgcolor="#1c1f2e", bordercolor="#2a2d3e", borderwidth=1),
            margin=dict(l=10,r=10,t=10,b=10), height=350
        )
        fig_roc.update_xaxes(gridcolor="#2a2d3e")
        fig_roc.update_yaxes(gridcolor="#2a2d3e")
        st.plotly_chart(fig_roc, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Calibrare Probabilistică</div>', unsafe_allow_html=True)
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                      line=dict(dash="dash", color="#4a5568", width=1), name="Perfect"))
        for name in MODEL_ORDER:
            mean_pred, frac_pos = cal_data[name]
            fig_cal.add_trace(go.Scatter(x=mean_pred, y=frac_pos, mode="lines+markers",
                                          name=name, line=dict(color=COLORS[name], width=2),
                                          marker=dict(size=5)))
        fig_cal.update_layout(
            xaxis_title="Probabilitate prezisă", yaxis_title="Fracție pozitivă reală",
            paper_bgcolor="#1c1f2e", plot_bgcolor="#1c1f2e",
            font=dict(color="#e2e8f0", size=12),
            legend=dict(bgcolor="#1c1f2e", bordercolor="#2a2d3e", borderwidth=1),
            margin=dict(l=10,r=10,t=10,b=10), height=350
        )
        fig_cal.update_xaxes(gridcolor="#2a2d3e")
        fig_cal.update_yaxes(gridcolor="#2a2d3e")
        st.plotly_chart(fig_cal, use_container_width=True)

    st.markdown("""
    > **Calibrare**: Curbă apropiată de diagonală = probabilitățile reflectă realitatea.
    """)

    st.markdown('<div class="section-title">Comparație AUC-ROC</div>', unsafe_allow_html=True)
    fig_bar = go.Figure(go.Bar(
        x=[metrics[n]["AUC"] for n in MODEL_ORDER], y=MODEL_ORDER, orientation="h",
        marker_color=[COLORS[n] for n in MODEL_ORDER],
        text=[f"{metrics[n]['AUC']:.4f}" for n in MODEL_ORDER], textposition="outside"
    ))
    fig_bar.update_layout(
        xaxis=dict(range=[0.5, 1.0], gridcolor="#2a2d3e"),
        yaxis=dict(gridcolor="#2a2d3e"),
        paper_bgcolor="#1c1f2e", plot_bgcolor="#1c1f2e",
        font=dict(color="#e2e8f0", size=12),
        margin=dict(l=10,r=80,t=10,b=10), height=280
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# ──────────────────────────────────────────────────────
# TAB 4 — ANALIZĂ AVANSATĂ
# ──────────────────────────────────────────────────────
with tab4:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">Importanța Caracteristicilor (LightGBM)</div>', unsafe_allow_html=True)
        if fi is not None:
            feat_labels = [
                "home_rate","away_rate","home_htg","away_htg","home_conc","away_conc",
                "home_ht_tot","away_ht_tot","home_sot","away_sot",
                "home_std_htg","away_std_htg","home_std_sot","away_std_sot",
                "home_atk_vs_away_def","away_atk_vs_home_def",
                "pressure","imbalance","conv_home","conv_away","volatility","combined_attack"
            ]
            fi_df = pd.DataFrame({
                "Feature":    feat_labels[:len(fi)],
                "Importance": fi
            }).sort_values("Importance", ascending=False).reset_index(drop=True)

            fi_df["Rank"] = fi_df.index + 1
            fi_df["Importance"] = fi_df["Importance"].round(4)

            # Bara vizuală text
            max_imp = fi_df["Importance"].max()
            fi_df["Bar"] = fi_df["Importance"].apply(
                lambda v: "█" * int((v / max_imp) * 20)
            )

            fi_df = fi_df[["Rank", "Feature", "Importance", "Bar"]]
            st.dataframe(fi_df, hide_index=True, use_container_width=True)
        else:
            st.info("Feature importance indisponibilă.")

    with c2:
        st.markdown('<div class="section-title">Distribuție Probabilități (Ensemble)</div>', unsafe_allow_html=True)
        med_all = df[features].median().values
        X_all   = np.where(np.isnan(df[features].values), med_all, df[features].values)
        sample_probs = trained["Ensemble"].predict_proba(X_all)[:, 1]
        tgt = df["target"].values
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=sample_probs[tgt == 1], name="HT ≥1 gol", nbinsx=30,
            marker_color="rgba(104,211,145,0.6)", marker_line=dict(color="#68d391", width=1)
        ))
        fig_hist.add_trace(go.Histogram(
            x=sample_probs[tgt == 0], name="0 goluri HT", nbinsx=30,
            marker_color="rgba(252,129,129,0.6)", marker_line=dict(color="#fc8181", width=1)
        ))
        fig_hist.update_layout(
            barmode="overlay", xaxis_title="Probabilitate prezisă", yaxis_title="Frecvență",
            paper_bgcolor="#1c1f2e", plot_bgcolor="#1c1f2e",
            font=dict(color="#e2e8f0", size=12),
            legend=dict(bgcolor="#1c1f2e"),
            margin=dict(l=10,r=10,t=10,b=10), height=320
        )
        fig_hist.update_xaxes(gridcolor="#2a2d3e")
        fig_hist.update_yaxes(gridcolor="#2a2d3e")
        st.plotly_chart(fig_hist, use_container_width=True)

    # Backtest — cached
    st.markdown('<div class="section-title">Backtest — Ultimele 100 Meciuri</div>', unsafe_allow_html=True)
    last_n = compute_backtest(trained, df, features, n=100)

    fig_bt = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.6, 0.4], vertical_spacing=0.08)
    fig_bt.add_trace(go.Scatter(
        x=list(range(len(last_n))), y=last_n["pred_prob"],
        mode="lines", name="Prob prezisă", line=dict(color="#63b3ed", width=1.5)
    ), row=1, col=1)
    fig_bt.add_trace(go.Scatter(
        x=list(range(len(last_n))), y=last_n["target"],
        mode="markers", name="Rezultat real",
        marker=dict(color=last_n["target"].map({1:"#68d391",0:"#fc8181"}), size=6)
    ), row=1, col=1)
    fig_bt.add_hline(y=0.55, line_dash="dash", line_color="#4a5568", row=1, col=1)
    fig_bt.add_trace(go.Scatter(
        x=list(range(len(last_n))), y=last_n["cumulative_acc"],
        mode="lines", name="Acuratețe cumulativă", line=dict(color="#f6e05e", width=2)
    ), row=2, col=1)
    fig_bt.update_layout(
        paper_bgcolor="#1c1f2e", plot_bgcolor="#1c1f2e",
        font=dict(color="#e2e8f0", size=11),
        legend=dict(bgcolor="#1c1f2e"),
        margin=dict(l=10,r=10,t=10,b=10), height=420
    )
    for r in [1, 2]:
        fig_bt.update_xaxes(gridcolor="#2a2d3e", row=r, col=1)
        fig_bt.update_yaxes(gridcolor="#2a2d3e", row=r, col=1)
    st.plotly_chart(fig_bt, use_container_width=True)

    total, correct = len(last_n), last_n["correct"].sum()
    hc = last_n[last_n["pred_prob"] >= 0.65]
    hc_acc = hc["correct"].mean() if len(hc) > 0 else 0
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Acuratețe globală (prag 0.55)", f"{correct/total:.1%}", f"{correct}/{total} meciuri")
    col_b.metric("Acuratețe High Confidence (≥0.65)", f"{hc_acc:.1%}", f"{len(hc)} meciuri")
    col_c.metric("Brier Score (Ensemble)", f"{metrics['Ensemble']['Brier']:.4f}", "mai mic = mai bun")

st.markdown("---")
st.caption("⚠️ Acest instrument este educațional. Pariurile implică riscuri financiare. Folosiți responsabil.")