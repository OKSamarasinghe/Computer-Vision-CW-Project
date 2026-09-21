# app/models/

Put the final trained model here before running the application:

**`diabetic_retinopathy_efficientnetb0_notebook5_final_model.keras`**

Copy it from `../../kaggle_artifacts/notebook_5_training_strategy/` (the output of Notebook 5 —
this is the model Notebook 6 also loads for evaluation and Grad-CAM, so it is the correct one for
the app to use too).

This file is excluded from plain git tracking in `.gitignore` — set up Git LFS to track it properly
(`git lfs track "*.keras"`) rather than committing it as a normal blob. See the top-level README.
