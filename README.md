# Diabetic Retinopathy Stage Detection — Coursework Repository

CNN-based (EfficientNetB0, transfer learning) classifier that grades diabetic retinopathy severity
(No_DR / Mild / Moderate / Severe / Proliferative_DR) from retinal fundus images, trained on the
APTOS 2019 dataset. Built for a Computer Vision coursework project: dataset justification →
preprocessing → augmentation & balancing → model → training strategy → evaluation → an innovation
feature (Grad-CAM + an uncertainty-aware clinical decision-support web app).

## Repository structure

```
diabetic-retinopathy-dr-grading/
├── notebooks/              Kaggle notebooks, WITH their printed cell outputs, one folder per notebook
├── kaggle_artifacts/        Everything downloaded from each Kaggle session's "Output" tab
│                            (models, CSVs, figures, metadata JSON) — the trained model the app/ uses lives here
├── app/                     The local innovation application (Streamlit UI + backend + preprocessing)
├── report/                  Report source, required diagrams, and screenshot evidence
├── video/                   Video script/storyboard and the hosted demo URL
└── docs/                    Supporting design notes (e.g. the app's innovation rationale + citations)
```

See the `README.md` inside each top-level folder for exactly what goes where.

## Suggested workflow

1. **Kaggle** — run Notebooks 1–6 (already built; see `notebooks/`) end to end.
2. After each notebook finishes, download its **Output** files from Kaggle and drop them into the
   matching `kaggle_artifacts/notebook_N_.../` folder (see that folder's README for the exact filenames
   this project's notebooks are known to produce).
3. Export each notebook **with its outputs still attached** (`File → Download → .ipynb`, or
   `jupyter nbconvert --to notebook --execute` if re-running locally) into `notebooks/0N_.../`.
4. Copy the **final trained model** —
   `diabetic_retinopathy_efficientnetb0_notebook5_final_model.keras` from
   `kaggle_artifacts/notebook_5_training_strategy/` — into `app/models/`.
5. Set up and run the local application (`app/README.md`) to confirm it grades a real image end to end.
6. Build the report and record the video using the evidence gathered above (`report/README.md`,
   `video/README.md`).
7. `git init`, commit, push to GitHub. **Use Git LFS for the `.keras` model file(s)** — see `.gitignore`
   and the note below.

## Git LFS note (large model files)

`.keras`/`.h5` model files are excluded from plain git tracking in `.gitignore` because GitHub blocks
files over 100 MB and this project's final model may approach that. Before your first commit:

```bash
git lfs install
git lfs track "*.keras" "*.h5"
git add .gitattributes
```

Then commit and push as normal — the model files in `app/models/` and `kaggle_artifacts/notebook_4.../`
and `notebook_5.../` will be tracked via LFS instead of being ignored. If you'd rather not use LFS at
all, keep those specific `.keras` files out of git entirely and note in your README that they're
available via the linked Kaggle notebook outputs instead.

## The innovation feature, in one paragraph

Grad-CAM (Notebook 6's mandatory innovation feature) is extended into a small local web application
that does four things beyond a bare classifier, **none of which require any additional training**:
(1) Monte Carlo Dropout at inference time turns the model's existing Dropout layers into an uncertainty
estimator, flagging low-confidence grades for manual review; (2) the Grad-CAM heatmap is summarised into
a plain-language region description; (3) the predicted grade + confidence is mapped onto a referral-
urgency recommendation aligned with NHS diabetic eye screening guidance; (4) all of the above is bundled
into a downloadable one-page PDF screening report. Full rationale and citations: `docs/innovation_design_notes.md`.
