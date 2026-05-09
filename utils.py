import gc
import os
import joblib
import pandas as pd
import numpy as np
import requests
from io import StringIO
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, roc_curve
import lightgbm as lgb
import xgboost as xgb

try:
    import pyarrow.feather as feather
    FEATHER_OK = True
except Exception:
    FEATHER_OK = False

LEAGUES = {
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

FEATURES = [
    "home_rate","away_rate","home_htg","away_htg","home_conc","away_conc",
    "home_ht_tot","away_ht_tot","home_sot","away_sot",
    "home_std_htg","away_std_htg","home_std_sot","away_std_sot",
    "home_attack_vs_away_def","away_attack_vs_home_def",
    "pressure","imbalance","conv_home","conv_away","volatility","combined_attack",
]

COLORS = {
    "LightGBM": "#63b3ed",
    "XGBoost": "#68d391",
    "RandomForest": "#b794f4",
    "LogisticRegression": "#fbd38d",
    "Ensemble": "#f6e05e",
}

BADGES = {
    "LightGBM": ("badge-lgb", "#1a365d", "#63b3ed", "#2b6cb0"),
    "XGBoost": ("badge-xgb", "#1c4532", "#68d391", "#276749"),
    "RandomForest": ("badge-rf", "#322659", "#b794f4", "#553c9a"),
    "LogisticRegression": ("badge-lr", "#7b341e", "#fbd38d", "#c05621"),
    "Ensemble": ("badge-ens", "#2d3748", "#f6e05e", "#b7791f"),
}

MODEL_ORDER = ["LightGBM", "XGBoost", "RandomForest", "LogisticRegression", "Ensemble"]


def save_df(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if FEATHER_OK and path.endswith(".feather"):
        feather.write_feather(df.reset_index(drop=True), path)
    else:
        df.to_pickle(path)


def load_df(path: str) -> pd.DataFrame:
    if FEATHER_OK and path.endswith(".feather"):
        return feather.read_feather(path)
    return pd.read_pickle(path)


def cleanup_memory(*objs):
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()


def fetch_raw_data() -> pd.DataFrame:
    all_data = []
    for liga, url in LEAGUES.items():
        for s in SEASONS:
            try:
                r = requests.get(url.format(s=s), timeout=15)
                if r.status_code == 200 and len(r.text) > 500:
                    df = pd.read_csv(StringIO(r.text), low_memory=False)
                    df["Liga"] = liga
                    df["Sezon"] = s
                    all_data.append(df)
            except Exception:
                pass
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def preprocess(df: pd.DataFrame):
    cols = ["Date","HomeTeam","AwayTeam","HTHG","HTAG","FTHG","FTAG","HST","AST","HC","AC","Liga","Sezon"]
    df = df[[c for c in cols if c in df.columns]].dropna(subset=["HomeTeam","AwayTeam","HTHG","HTAG"]).copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df["target"] = ((df["HTHG"] + df["HTAG"]) > 0).astype(int)
    df["ht_total"] = df["HTHG"] + df["HTAG"]

    def _roll(data, team_col, val_col, window=10, min_p=3, agg="mean"):
        result = pd.Series(np.nan, index=data.index)
        for _, idx in data.groupby(team_col).groups.items():
            sub = data.loc[idx].sort_values("Date")[val_col].shift(1)
            rolled = sub.rolling(window, min_periods=min_p)
            result.loc[idx] = rolled.mean() if agg == "mean" else rolled.std()
        return result

    df["home_rate"] = _roll(df, "HomeTeam", "target")
    df["away_rate"] = _roll(df, "AwayTeam", "target")
    df["home_htg"] = _roll(df, "HomeTeam", "HTHG")
    df["away_htg"] = _roll(df, "AwayTeam", "HTAG")
    df["home_conc"] = _roll(df, "HomeTeam", "HTAG")
    df["away_conc"] = _roll(df, "AwayTeam", "HTHG")
    df["home_ht_tot"] = _roll(df, "HomeTeam", "ht_total")
    df["away_ht_tot"] = _roll(df, "AwayTeam", "ht_total")
    df["home_sot"] = _roll(df, "HomeTeam", "HST") if "HST" in df.columns else np.nan
    df["away_sot"] = _roll(df, "AwayTeam", "AST") if "AST" in df.columns else np.nan
    df["home_std_htg"] = _roll(df, "HomeTeam", "HTHG", agg="std")
    df["away_std_htg"] = _roll(df, "AwayTeam", "HTAG", agg="std")
    df["home_std_sot"] = _roll(df, "HomeTeam", "HST", agg="std") if "HST" in df.columns else np.nan
    df["away_std_sot"] = _roll(df, "AwayTeam", "AST", agg="std") if "AST" in df.columns else np.nan

    base_cols = [
        "home_rate","away_rate","home_htg","away_htg","home_conc","away_conc",
        "home_ht_tot","away_ht_tot","home_std_htg","away_std_htg"
    ]
    df = df.dropna(subset=base_cols)

    for c in ["home_sot","away_sot","home_std_sot","away_std_sot"]:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    df["home_attack_vs_away_def"] = df["home_htg"] - df["away_conc"]
    df["away_attack_vs_home_def"] = df["away_htg"] - df["home_conc"]
    df["pressure"] = np.log1p(df["home_sot"] + df["away_sot"])
    df["imbalance"] = (df["home_rate"] - df["away_rate"]) * 2
    df["conv_home"] = df["home_htg"] / (df["home_sot"] + 1e-3)
    df["conv_away"] = df["away_htg"] / (df["away_sot"] + 1e-3)
    df["volatility"] = df["home_std_htg"] + df["away_std_htg"]
    df["combined_attack"] = df["home_htg"] + df["away_htg"]
    return df, FEATURES


def build_matrix(df: pd.DataFrame, features: list):
    home_feats = [f for f in features if f.startswith("home_")]
    away_feats = [f for f in features if f.startswith("away_")]
    global_feats = [f for f in features if f not in home_feats + away_feats]
    X = np.concatenate([
        df[home_feats].values,
        df[away_feats].values,
        *([df[global_feats].values] if global_feats else [])
    ], axis=1)
    return X


def build_input_vector(df, features, home, away, liga_sel):
    data = df if liga_sel == "All" else df[df["Liga"] == liga_sel]
    h_rows = data[data["HomeTeam"] == home]
    a_rows = data[data["AwayTeam"] == away]
    if h_rows.empty or a_rows.empty:
        return None, None, None

    h = h_rows.iloc[-1]
    a = a_rows.iloc[-1]
    home_feats = [f for f in features if f.startswith("home_")]
    away_feats = [f for f in features if f.startswith("away_")]
    global_feats = [f for f in features if f not in home_feats + away_feats]

    x = np.concatenate([
        h[home_feats].values,
        a[away_feats].values,
        h[global_feats].values if global_feats else []
    ]).reshape(1, -1)
    return x, h_rows, a_rows

def train_all_models(df: pd.DataFrame, features: list):
    df = df.sort_values("Date").copy()
    X = build_matrix(df, features)
    y = df["target"].values

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    cv = TimeSeriesSplit(n_splits=5)

    base_models = {
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
        ),
    }

    trained, metrics_out = {}, {}

    for name, model in base_models.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        cv_scores = []
        for tr_idx, val_idx in cv.split(X_train):
            model_cv = base_models[name].__class__(**base_models[name].get_params())
            model_cv.fit(X_train[tr_idx], y_train[tr_idx])
            cv_pred = model_cv.predict_proba(X_train[val_idx])[:, 1]
            cv_scores.append(roc_auc_score(y_train[val_idx], cv_pred))

        trained[name] = model
        metrics_out[name] = {
            "AUC": roc_auc_score(y_test, proba),
            "Brier": brier_score_loss(y_test, proba),
            "LogLoss": log_loss(y_test, proba),
            "CV_AUC": float(np.mean(cv_scores)),
        }

    ens = VotingClassifier(
        [(n, m) for n, m in base_models.items()],
        voting="soft",
        n_jobs=-1
    )

    cal_ens = CalibratedClassifierCV(
        ens,
        cv=cv,
        method="sigmoid"
    )
    cal_ens.fit(X_train, y_train)
    ens_proba = cal_ens.predict_proba(X_test)[:, 1]

    trained["Ensemble"] = cal_ens
    metrics_out["Ensemble"] = {
        "AUC": roc_auc_score(y_test, ens_proba),
        "Brier": brier_score_loss(y_test, ens_proba),
        "LogLoss": log_loss(y_test, ens_proba),
        "CV_AUC": float(np.mean([metrics_out[n]["CV_AUC"] for n in base_models])),
    }

    roc_data, cal_data = {}, {}
    for name, model in trained.items():
        p = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, p)
        roc_data[name] = (fpr, tpr)
        frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=10)
        cal_data[name] = (mean_pred, frac_pos)

    fi = None
    try:
        fi = trained["LightGBM"].feature_importances_
    except Exception:
        pass

    return {
        "trained": trained,
        "metrics": metrics_out,
        "roc_data": roc_data,
        "cal_data": cal_data,
        "fi": fi,
        "features": features,
        "dataset_size": len(df),
        "base_rate": float(df["target"].mean()),
    }

def bootstrap_ci90(models_dict, x_input, n_bootstrap=100, ci=0.90):
    alpha = (1 - ci) / 2
    rng = np.random.default_rng(42)
    noise_matrix = rng.normal(0, 0.08, (n_bootstrap, x_input.shape[1]))
    x_noisy_batch = np.clip(x_input + noise_matrix, None, None)

    results = {}
    for name, model in models_dict.items():
        preds = model.predict_proba(x_noisy_batch)[:, 1]
        preds = np.clip(preds, 1e-4, 1 - 1e-4)
        results[name] = {
            "mean": float(np.mean(preds)),
            "lower": float(np.percentile(preds, alpha * 100)),
            "upper": float(np.percentile(preds, (1 - alpha) * 100)),
            "std": float(np.std(preds)),
        }
    return results


def compute_backtest(trained, df: pd.DataFrame, features: list, n: int = 100):
    last_n = df.tail(n).copy()
    X = build_matrix(last_n, features)
    last_n["pred_prob"] = trained["Ensemble"].predict_proba(X)[:, 1]
    last_n["pred_bin"] = (last_n["pred_prob"] >= 0.55).astype(int)
    last_n["correct"] = (last_n["pred_bin"] == last_n["target"]).astype(int)
    last_n["cumulative_acc"] = last_n["correct"].expanding().mean()
    return last_n


def save_artifact(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path, compress=3)


def load_artifact(path: str):
    return joblib.load(path)
