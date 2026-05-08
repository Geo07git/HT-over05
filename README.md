# HT Goal Predictor Pro - Streamlit Cloud Deploy

## Fisiere
- `train_models.py` -> rulezi local pentru a genera artefactele
- `app.py` -> aplicatia pentru Streamlit Community Cloud
- `utils.py` -> functii comune
- `artifacts/` -> aici ajung fisierele generate local si apoi urcate in repo

## Pasii corecti
1. Rulezi local:
   ```bash
   python train_models.py
   ```
2. Verifici ca in `artifacts/` exista:
   - `processed_data.feather` (sau `.pkl`)
   - `ht_models.joblib`
3. Faci commit si push in GitHub.
4. In Streamlit Community Cloud deployezi `app.py`.

## Observatii
- Aplicatia din cloud NU antreneaza modele.
- Aplicatia doar incarca artefactele deja salvate in repo.
- Daca updatezi datele, rerulezi local `train_models.py`, apoi faci commit/push din nou.
