# Notebook 5 — Training Strategy & Experimental Design (Criterion 5, 10 marks)

Place the executed notebook here, with outputs attached:

- `Notebook5_Training_Strategy_Experimental_Design.ipynb`

Covers: rebuilding the Notebook 4 architecture with added L2 regularisation, EarlyStopping, LR
scheduling, and a compared pair of Stage-2 fine-tuning configurations (with/without LR warm-up).

**This notebook produces the final model the application in `../../app/` actually loads.** Matching
Kaggle outputs go in `../../kaggle_artifacts/notebook_5_training_strategy/`, including
`diabetic_retinopathy_efficientnetb0_notebook5_final_model.keras`,
`notebook_5_combined_training_history.pkl`, and `notebook_5_metadata.json`. Copy the `.keras` file
from there into `../../app/models/` before running the app.
