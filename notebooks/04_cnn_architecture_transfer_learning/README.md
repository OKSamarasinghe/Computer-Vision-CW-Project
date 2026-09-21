# Notebook 4 — CNN Architecture & Transfer Learning (Criterion 4, 20 marks)

Place the executed notebook here, with outputs attached:

- `Notebook4_CNN_Architecture_TransferLearning.ipynb`

Covers: EfficientNetB0 backbone selection and justification, staged transfer learning (frozen
head-only training, then fine-tuning the last ~20 backbone layers), and learning-rate tuning.

Matching Kaggle outputs go in `../../kaggle_artifacts/notebook_4_architecture/`, including
`diabetic_retinopathy_efficientnetb0_final_model.keras` (this notebook's own best checkpoint —
distinct from, and superseded by, the Notebook 5 model the app actually uses) and
`notebook_4_metadata.json`.
