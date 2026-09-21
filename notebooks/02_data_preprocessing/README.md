# Notebook 2 — Data Preprocessing (Criterion 2, 10 marks)

Place the executed notebook here, with outputs attached:

- `Notebook2_Data_Preprocessing.ipynb`

Covers: `preprocess_fundus_image()` — dark-border crop → resize to 224×224 → CLAHE contrast
enhancement → Ben Graham local-average subtraction → pixel normalisation. This exact pipeline
(minus the final normalisation step — see `app/backend/preprocessing.py`) is reused unchanged by the
local application in `../../app/`.

Matching Kaggle outputs (preprocessed split CSVs, before/after comparison figures, metadata JSON) go
in `../../kaggle_artifacts/notebook_2_preprocessing/`.
