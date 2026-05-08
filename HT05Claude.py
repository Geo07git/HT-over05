import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import (
    load_df, load_artifact, build_input_vector, bootstrap_ci90, compute_backtest,
    COLORS, BADGES, MODEL_ORDER, FEATHER_OK
)

st.set_page_config(page_title="HT Goal Predictor Pro", layout="wide", initial_sidebar_state="expanded")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
DATA_PATH = os.path.join(ARTIFACT_DIR, "processed_data.feather" if FEATHER_OK else "processed_data.pkl")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "ht_models.joblib")

badge_css = "\n".join([
    f'.{cls} {{ background:{bg}; color:{fg}; border: 1px solid {br}; }}'
    for _, (cls, bg, fg, br) in BADGES.items()
])

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
  .main {{ background: #0e1117; }}
  .metric-card {{ background: #1c1f2e; border: 1px solid #2a2d3e; border-radius: 12px; padding: 20px 24px; text-align: center; transition: box-shadow 0.2s; }}
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
  .section-title {{ font-size: 18px; font-weight: 600; color: #e2e8f0; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid #2a2d3e; }}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_processed_data():
    return load_df(DATA_PATH)

@st.cache_resource(show_spinner=False)
def load_models():
    return load_artifact(MODEL_PATH)

if not os.path.exists(DATA_PATH) or not os.path.exists(MODEL_PATH):
    st.error("Lipsesc artefactele din artifacts/. Rulează train_models.py local și urcă fișierele în repo.")
    st.stop()

with st.spinner("📦 Se încarcă artefactele din repo..."):
    df = load_processed_data()
    artifact = load_models()

trained = artifact["trained"]
metrics = artifact["metrics"]
roc_data = artifact["roc_data"]
cal_data = artifact["cal_data"]
fi = artifact["fi"]
features = artifact["features"]

def safe_sample_df(frame: pd.DataFrame, n: int = 50000):
    return frame.sample(n=min(n, len(frame)), random_state=42).copy()

with st.sidebar:
    st.markdown("## ⚽ HT Goal Predictor Pro")
    st.markdown("---")
    st.markdown("**Modele încărcate:**")
    for name in MODEL_ORDER:
        cls = BADGES[name][0]
        st.markdown(f'<span class="model-badge {cls}">{name}</span>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Sursă deploy:**")
    st.markdown(f"- Data: `{os.path.basename(DATA_PATH)}`")
    st.markdown(f"- Models: `{os.path.basename(MODEL_PATH)}`")
    st.markdown("---")
    st.caption("Deploy mode: inference only | Community Cloud ready")

st.markdown("## ⚽ HT ≥ 1 Goal Predictor — Professional Edition")
st.markdown("Predicție probabilistică multi-model cu artefacte încărcate din repo, fără retraining în cloud.")
st.success(f"✅ Artefacte încărcate! Dataset: **{len(df):,} meciuri**.")

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Predicție",
    "📊 Performanță Modele",
    "📈 Evaluare Statistică",
    "🔍 Analiză Avansată"
])

with tab1:
    col1, col2, col3 = st.columns(3)
    liga_options = ["All"] + sorted(df["Liga"].dropna().unique())
    with col3:
        liga_sel = st.selectbox("🏆 Ligă", liga_options)
    df_filtered = df if liga_sel == "All" else df[df["Liga"] == liga_sel]
    home_options = sorted(df_filtered["HomeTeam"].dropna().unique())
    away_options = sorted(df_filtered["AwayTeam"].dropna().unique())
    with col1:
        home = st.selectbox("🏠 Echipa Gazdă", home_options)
    with col2:
        away = st.selectbox("✈️ Echipa Oaspete", away_options)

    chosen_model = st.selectbox("Model Principal", ["Ensemble (recomandat)", "LightGBM", "XGBoost", "RandomForest", "LogisticRegression"], index=0)
    model_key = "Ensemble" if "Ensemble" in chosen_model else chosen_model

    if st.button("🔮 Calculează Probabilitate", type="primary", use_container_width=True):
        x, h_rows, a_rows = build_input_vector(df, features, home, away, liga_sel)
        if x is None:
            st.warning("Nu există date suficiente pentru această combinație.")
        else:
            ci90 = bootstrap_ci90(trained, x, n_bootstrap=50)
            main_prob = ci90[model_key]["mean"]
            ci_lower = ci90[model_key]["lower"]
            ci_upper = ci90[model_key]["upper"]
            ci_std = ci90[model_key]["std"]
            fair_odds = 1 / max(main_prob, 1e-6)
            if main_prob >= 0.65:
                box_cls, verdict, verdict_color = "pred-high", "🔥 SPOT PUTERNIC", "#68d391"
            elif main_prob >= 0.52:
                box_cls, verdict, verdict_color = "pred-med", "⚠️ EDGE MARGINAL", "#f6ad55"
            else:
                box_cls, verdict, verdict_color = "pred-low", "❌ FĂRĂ EDGE CLAR", "#fc8181"
            fair_lo = 1 / max(ci_upper, 1e-6)
            fair_hi = 1 / max(ci_lower, 1e-6)
            st.markdown(f"""
            <div class="prediction-box {box_cls}">
              <div style="font-size:14px;color:#718096;margin-bottom:8px;">{home} vs {away}</div>
              <div style="font-size:52px;font-weight:700;color:{verdict_color}">{main_prob:.1%}</div>
              <div style="font-size:13px;color:#a0aec0;margin-top:4px;">Probabilitate HT ≥ 1 gol ({model_key})</div>
              <div style="margin-top:10px;"><span style="background:#2d3748;border-radius:8px;padding:6px 14px;font-size:15px;color:#a0aec0;">CI 90%: <strong style="color:#e2e8f0">{ci_lower:.1%}</strong> &nbsp;―&nbsp; <strong style="color:#e2e8f0">{ci_upper:.1%}</strong></span></div>
              <div style="font-size:22px;font-weight:600;color:#e2e8f0;margin-top:12px;">Cote corecte: {fair_odds:.2f}</div>
              <div style="font-size:13px;color:#718096;margin-top:4px;">Range cote CI90: {fair_hi:.2f} — {fair_lo:.2f}</div>
              <div style="font-size:16px;font-weight:700;color:{verdict_color};margin-top:10px;">{verdict}</div>
              <div style="font-size:12px;color:#4a5568;margin-top:4px;">σ = {ci_std:.3f} | {'⚠️ Incertitudine mare' if ci_std > 0.05 else '✓ Estimare stabilă'}</div>
            </div>
            """, unsafe_allow_html=True)

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
              <div style="margin-top:10px;font-size:12px;color:#718096">Brier: {m['Brier']:.4f}<br>LogLoss: {m['LogLoss']:.4f}<br>CV AUC: {m['CV_AUC']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
    rows = [{"Model": n, "AUC-ROC": f"{metrics[n]['AUC']:.4f}", "Brier": f"{metrics[n]['Brier']:.4f}", "Log Loss": f"{metrics[n]['LogLoss']:.4f}", "CV AUC": f"{metrics[n]['CV_AUC']:.4f}"} for n in MODEL_ORDER]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.info(f"📊 **Rata de bază HT ≥ 1 gol:** {artifact['base_rate']:.1%} din {artifact['dataset_size']:,} meciuri")

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Curbe ROC</div>', unsafe_allow_html=True)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash", color="#4a5568", width=1), name="Random"))
        for name in MODEL_ORDER:
            fpr, tpr = roc_data[name]
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} ({metrics[name]['AUC']:.3f})", line=dict(color=COLORS[name], width=2)))
        fig_roc.update_layout(xaxis_title="FPR", yaxis_title="TPR", paper_bgcolor="#1c1f2e", plot_bgcolor="#1c1f2e", font=dict(color="#e2e8f0", size=12), legend=dict(bgcolor="#1c1f2e", bordercolor="#2a2d3e", borderwidth=1), margin=dict(l=10,r=10,t=10,b=10), height=350)
        fig_roc.update_xaxes(gridcolor="#2a2d3e")
        fig_roc.update_yaxes(gridcolor="#2a2d3e")
        st.plotly_chart(fig_roc, use_container_width=True)
    with c2:
        st.markdown('<div class="section-title">Calibrare Probabilistică</div>', unsafe_allow_html=True)
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash", color="#4a5568", width=1), name="Perfect"))
        for name in MODEL_ORDER:
            mean_pred, frac_pos = cal_data[name]
            fig_cal.add_trace(go.Scatter(x=mean_pred, y=frac_pos, mode="lines+markers", name=name, line=dict(color=COLORS[name], width=2), marker=dict(size=5)))
        fig_cal.update_layout(xaxis_title="Probabilitate prezisă", yaxis_title="Fracție pozitivă reală", paper_bgcolor="#1c1f2e", plot_bgcolor="#1c1f2e", font=dict(color="#e2e8f0", size=12), legend=dict(bgcolor="#1c1f2e", bordercolor="#2a2d3e", borderwidth=1), margin=dict(l=10,r=10,t=10,b=10), height=350)
        fig_cal.update_xaxes(gridcolor="#2a2d3e")
        fig_cal.update_yaxes(gridcolor="#2a2d3e")
        st.plotly_chart(fig_cal, use_container_width=True)

with tab4:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Importanța Caracteristicilor (LightGBM)</div>', unsafe_allow_html=True)
        if fi is not None:
            feat_labels = ["home_rate","away_rate","home_htg","away_htg","home_conc","away_conc","home_ht_tot","away_ht_tot","home_sot","away_sot","home_std_htg","away_std_htg","home_std_sot","away_std_sot","home_atk_vs_away_def","away_atk_vs_home_def","pressure","imbalance","conv_home","conv_away","volatility","combined_attack"]
            fi_df = pd.DataFrame({"Feature": feat_labels[:len(fi)], "Importance": fi}).sort_values("Importance", ascending=False).reset_index(drop=True)
            fi_df["Rank"] = fi_df.index + 1
            fi_df["Importance"] = fi_df["Importance"].round(4)
            max_imp = fi_df["Importance"].max()
            fi_df["Bar"] = fi_df["Importance"].apply(lambda v: "█" * int((v / max_imp) * 20) if max_imp > 0 else "")
            st.dataframe(fi_df[["Rank", "Feature", "Importance", "Bar"]], hide_index=True, use_container_width=True)
        else:
            st.info("Feature importance indisponibilă.")
    with c2:
        st.markdown('<div class="section-title">Distribuție Probabilități (Ensemble)</div>', unsafe_allow_html=True)
        sample_df = safe_sample_df(df)
        home_feats = [f for f in features if f.startswith("home_")]
        away_feats = [f for f in features if f.startswith("away_")]
        global_feats = [f for f in features if f not in home_feats + away_feats]
        X_all = np.concatenate([sample_df[home_feats].values, sample_df[away_feats].values, *([sample_df[global_feats].values] if global_feats else [])], axis=1)
        sample_probs = trained["Ensemble"].predict_proba(X_all)[:, 1]
        tgt = sample_df["target"].values
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=sample_probs[tgt == 1], name="HT ≥1 gol", nbinsx=30, marker_color="rgba(104,211,145,0.6)", marker_line=dict(color="#68d391", width=1)))
        fig_hist.add_trace(go.Histogram(x=sample_probs[tgt == 0], name="0 goluri HT", nbinsx=30, marker_color="rgba(252,129,129,0.6)", marker_line=dict(color="#fc8181", width=1)))
        fig_hist.update_layout(barmode="overlay", xaxis_title="Probabilitate prezisă", yaxis_title="Frecvență", paper_bgcolor="#1c1f2e", plot_bgcolor="#1c1f2e", font=dict(color="#e2e8f0", size=12), legend=dict(bgcolor="#1c1f2e"), margin=dict(l=10,r=10,t=10,b=10), height=320)
        fig_hist.update_xaxes(gridcolor="#2a2d3e")
        fig_hist.update_yaxes(gridcolor="#2a2d3e")
        st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown('<div class="section-title">Backtest — Ultimele 100 Meciuri</div>', unsafe_allow_html=True)
    last_n = compute_backtest(trained, df, features, n=100)
    fig_bt = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.08)
    fig_bt.add_trace(go.Scatter(x=list(range(len(last_n))), y=last_n["pred_prob"], mode="lines", name="Prob prezisă", line=dict(color="#63b3ed", width=1.5)), row=1, col=1)
    fig_bt.add_trace(go.Scatter(x=list(range(len(last_n))), y=last_n["target"], mode="markers", name="Rezultat real", marker=dict(color=last_n["target"].map({1:"#68d391",0:"#fc8181"}), size=6)), row=1, col=1)
    fig_bt.add_hline(y=0.55, line_dash="dash", line_color="#4a5568", row=1, col=1)
    fig_bt.add_trace(go.Scatter(x=list(range(len(last_n))), y=last_n["cumulative_acc"], mode="lines", name="Acuratețe cumulativă", line=dict(color="#f6e05e", width=2)), row=2, col=1)
    fig_bt.update_layout(paper_bgcolor="#1c1f2e", plot_bgcolor="#1c1f2e", font=dict(color="#e2e8f0", size=11), legend=dict(bgcolor="#1c1f2e"), margin=dict(l=10,r=10,t=10,b=10), height=420)
    for r in [1, 2]:
        fig_bt.update_xaxes(gridcolor="#2a2d3e", row=r, col=1)
        fig_bt.update_yaxes(gridcolor="#2a2d3e", row=r, col=1)
    st.plotly_chart(fig_bt, use_container_width=True)

st.markdown("---")
st.caption("⚠️ Acest instrument este educațional. Pariurile implică riscuri financiare. Folosiți responsabil.")
