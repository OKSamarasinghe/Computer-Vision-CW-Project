# Kaggle Artifacts

Everything downloaded from each Kaggle notebook session's **Output** tab (the `/kaggle/working/`
directory at the end of a run) goes here, one subfolder per notebook. These are the raw evidence
files the report and video pull figures/numbers from, and — for Notebook 5 specifically — the trained
model the local application actually runs on.

Expected contents per folder, based on this project's own notebooks (exact filenames confirmed from
each notebook's findings summary):

## notebook_1_eda/
Split CSVs (train/val/test), class-distribution chart, any dataset cross-check outputs.

## notebook_2_preprocessing/
`train_split_labels.csv`, `val_split_labels.csv`, `test_split_labels.csv`, rebuilt/preprocessed image
folders, before/after preprocessing comparison figure, `notebook_2_metadata.json`.

## notebook_3_augmentation/
Augmented training manifest CSV, `class_weights.pkl`, before/after class-distribution comparison chart.

## notebook_4_architecture/
`diabetic_retinopathy_efficientnetb0_final_model.keras`, `combined_training_history.pkl`,
`notebook_4_metadata.json`, Stage 1 / Stage 2 training-curve figures.

## notebook_5_training_strategy/  ← the model the app uses comes from here
`diabetic_retinopathy_efficientnetb0_notebook5_final_model.keras`,
`notebook_5_combined_training_history.pkl`, `notebook_5_metadata.json`,
`notebook_5_combined_training_curves.png`.
**Copy the `.keras` file from here into `../../app/models/`.**

## notebook_6_evaluation_gradcam/
`notebook_6_test_set_metrics_table.csv`, `notebook_6_confusion_matrix.png`,
`notebook_6_reproduced_training_curves.png`, `notebook_6_error_analysis_confused_pairs.csv`,
`notebook_6_misclassified_examples_grid.png`, `notebook_6_gradcam_correct_examples_grid.png`,
`notebook_6_gradcam_misclassified_examples_grid.png`, individual `gradcam_correct_*.png` /
`gradcam_confused_row*.png` overlays, `notebook_6_metadata.json`.

---
Large `.keras`/`.h5` files should be tracked with Git LFS rather than committed as plain blobs — see
the top-level `README.md` and `.gitignore`.
