# Diabetic Retinopathy Stage Detection — Coursework Repository

CNN-based (EfficientNetB0, transfer learning) classifier that grades diabetic retinopathy severity
(No_DR / Mild / Moderate / Severe / Proliferative_DR) from retinal fundus images, trained on the
APTOS 2019 dataset. Built for a Computer Vision coursework project: dataset justification →
preprocessing → augmentation & balancing → model → training strategy → evaluation → an innovation
feature (a multi-page, uncertainty-aware clinical decision-support web application built around
Grad-CAM).

**Developed by [Oshadha Samarasinghe](https://github.com/OKSamarasinghe)**

## 🟢 Live demo

**[dr-grading-assistant.streamlit.app](https://dr-grading-assistant.streamlit.app/)**

Hosted free on Streamlit Community Cloud. Note: the app sleeps after ~12 hours with no visitors
and takes about 30 seconds to wake on the next visit.

## Headline results (real, measured — see Notebook 6 / `app`'s Model Performance page)

| Metric | Value |
|---|---|
| Test accuracy | 75.82% |
| Quadratic weighted kappa | 0.8256 |
| Validation accuracy | 79.27% |
| Test set size | 550 images (held out, untouched until final evaluation) |

Minority-class recall (Mild 0.32, Severe 0.38, Proliferative_DR 0.27) is honestly weak — this is
shown prominently in the app and report rather than hidden, and is the main motivation behind the
uncertainty-aware design described below.

## Repository structure

```
Computer-Vision-CW-Project/
├── .streamlit/config.toml   Root copy of the app's theme (required here for Streamlit Cloud —
│                             see "Deployment" below)
├── requirements.txt          Root copy of Python dependencies, used only for Streamlit Cloud
├── runtime.txt                Pins the Python version for Streamlit Cloud (3.11)
├── packages.txt               Linux system packages Streamlit Cloud must apt-install (libgl1,
│                             libglib2.0-0 — required for OpenCV to import on a headless server)
├── .lfsconfig                Forces Git LFS to fetch over HTTPS, for reliable cloud deployment
├── notebooks/                Kaggle notebooks, WITH their printed cell outputs, one folder per notebook
├── kaggle_artifacts/         Evidence downloaded from each Kaggle session's Output tab (CSVs,
│                             figures, metadata JSON, and the two trained .keras models via Git
│                             LFS) — NOT the bulk preprocessed/augmented image folders; those are
│                             regenerable and stay on Kaggle only (see the live notebook links below)
├── app/                      The innovation application — Streamlit UI + backend + preprocessing
│                             (this is the actual source of truth; the root-level config files
│                             above are deployment copies of what's inside app/)
├── report/                   Report source, required diagrams, and screenshot evidence
├── video/                    Video script/storyboard and the hosted demo URL
└── docs/                     Supporting design notes (the app's innovation rationale + citations)
```

See the `README.md` inside each top-level folder for exactly what goes where.

## Live notebooks on Kaggle

The fully executed versions of these notebooks (including the bulk preprocessed/augmented image
folders not duplicated in this repo) are permanently hosted on Kaggle:

- [Notebook 1 — EDA & Dataset Justification](https://www.kaggle.com/code/oshadhaksamarasinghe/notebook-1-eda-dataset-justification)
- [Notebook 2 — Data Preprocessing](https://www.kaggle.com/code/oshadhaksamarasinghe/notebook-2-data-preprocessing)
- [Notebook 3 — Augmentation & Balancing](https://www.kaggle.com/code/oshadhaksamarasinghe/notebook-3-augmentation-and-balancing)
- [Notebook 4 — CNN Architecture & Transfer Learning](https://www.kaggle.com/code/oshadhaksamarasinghe/notebook-4-cnn-architecture-transfer-learning)
- [Notebook 5 — Training Strategy & Experimental Design](https://www.kaggle.com/code/oshadhaksamarasinghe/notebook-5-training-strategy-exp-design)
- [Notebook 6 — Model Evaluation & Grad-CAM](https://www.kaggle.com/code/oshadhaksamarasinghe/notebook-6-model-evaluation-gradcam)

## The innovation feature

A five-page Streamlit application wraps the trained model in an uncertainty-aware, clinically
framed decision-support tool — **with zero additional model training anywhere**:

- **Dashboard** — model status, session activity, and a plain-English walkthrough of the pipeline.
- **New Scan** — upload → staged progress indicator → grade with confidence badges → full
  per-class probability chart → plain-language explanation → downloadable PDF.
- **Scan History** — every scan in the session, stored and redisplayed (not recomputed), so
  figures never drift between what was shown and what's in a downloaded report.
- **Model Performance** — the project's real, measured evaluation results across five tabs
  (per-class metrics, error structure, dataset composition, architecture, limitations) — including
  the model's genuine weaknesses, shown deliberately rather than hidden.
- **About** — purpose, an icon-card pipeline diagram, safety/ethics, and citations.

Four techniques make this more than a bare classifier demo, none requiring retraining:
1. **Monte Carlo Dropout** — the model's existing Dropout layers are kept active at inference (30
   stochastic passes), turning them into a genuine uncertainty estimator that flags low-confidence
   grades for manual review.
2. **Grad-CAM** (ported from Notebook 6) plus a geometric region-description helper.
3. **A clinical referral-urgency mapping**, illustrative and explicitly labelled as such, inspired
   by NHS England (2023) diabetic eye screening guidance — a low-confidence prediction always
   escalates caution, never reduces it.
4. **A detailed, multi-page PDF screening report** (ReportLab), bundling the images, the full
   probability breakdown, the interpretation, and a methodology appendix.

Full rationale and citations: `docs/innovation_design_notes.md`.

## Deployment (Streamlit Community Cloud, free tier)

The app is hosted from this repo directly. A few things were required beyond the app itself:

- `requirements.txt`, `.streamlit/config.toml` and `runtime.txt` all had to be duplicated at the
  **repository root** — Streamlit Cloud only looks for them there when the entrypoint
  (`app/frontend/streamlit_app.py`) lives in a subdirectory.
- `packages.txt` installs `libgl1` and `libglib2.0-0`, Linux libraries OpenCV needs that aren't
  present on Streamlit Cloud's base image by default.
- `runtime.txt` pins Python to `3.11`, since Streamlit Cloud's default (3.14 at time of writing)
  has no stable TensorFlow wheels yet.
- `.lfsconfig` forces Git LFS to fetch the trained model over HTTPS for reliable cloud cloning.

## Suggested workflow (for reproducing this project from scratch)

1. **Kaggle** — run Notebooks 1–6 end to end (already built; see `notebooks/`).
2. Download each notebook's **Output** files from Kaggle and drop the small evidence files (not
   the bulk image folders) into the matching `kaggle_artifacts/notebook_N_.../` folder.
3. Export each notebook **with its outputs still attached** into `notebooks/0N_.../`.
4. Copy the final trained model —
   `diabetic_retinopathy_efficientnetb0_notebook5_final_model.keras` from
   `kaggle_artifacts/notebook_5_training_strategy/` — into `app/models/`.
5. Set up and run the local application (`app/README.md`):
```powershell
   cd app
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   streamlit run frontend/streamlit_app.py
```
6. Build the report and record the video using the evidence gathered above (`report/README.md`,
   `video/README.md`).
7. To deploy your own copy: fork/clone, set up Git LFS (below), then deploy via
   [share.streamlit.io](https://share.streamlit.io) with main file path
   `app/frontend/streamlit_app.py` — the root-level `requirements.txt`, `runtime.txt` and
   `packages.txt` in this repo are already configured for a clean deploy.

## Git LFS note (large model files)

`.keras`/`.h5` model files are tracked with Git LFS rather than committed as plain blobs, since
GitHub blocks files over 100MB and the free LFS quota is limited. Already configured in this repo
(`.gitattributes`); to replicate elsewhere:

```bash
git lfs install
git lfs track "*.keras" "*.h5"
git add .gitattributes
```

Then commit and push as normal — model files in `app/models/` and `kaggle_artifacts/notebook_4.../`
and `notebook_5.../` will be tracked via LFS.