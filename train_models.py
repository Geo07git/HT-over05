import os
from datetime import datetime
from utils import fetch_raw_data, preprocess, train_all_models, save_df, save_artifact, FEATHER_OK

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

DATA_PATH = os.path.join(ARTIFACT_DIR, "processed_data.feather" if FEATHER_OK else "processed_data.pkl")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "ht_models.joblib")
META_PATH = os.path.join(ARTIFACT_DIR, "build_info.txt")

print("[1/3] Download raw data...")
raw = fetch_raw_data()
if raw.empty:
    raise RuntimeError("Nu s-au putut descărca datele brute.")

print("[2/3] Preprocess data...")
df, features = preprocess(raw)
if df.empty:
    raise RuntimeError("Preprocess a returnat dataframe gol.")
save_df(df, DATA_PATH)

print("[3/3] Train models...")
artifact = train_all_models(df, features)
save_artifact(artifact, MODEL_PATH)

with open(META_PATH, "w", encoding="utf-8") as f:
    f.write(f"Build time: {datetime.utcnow().isoformat()}Z\n")
    f.write(f"Rows: {len(df)}\n")
    f.write(f"Features: {len(features)}\n")
    f.write(f"Data file: {os.path.basename(DATA_PATH)}\n")
    f.write(f"Model file: {os.path.basename(MODEL_PATH)}\n")

print("Done.")
print(f"Saved data to: {DATA_PATH}")
print(f"Saved models to: {MODEL_PATH}")
