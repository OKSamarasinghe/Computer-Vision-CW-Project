# Diabetic Retinopathy Grading Assistant (local application)

The coursework's innovation deliverable (Criterion 9): a Streamlit application that grades an
uploaded fundus image with the trained model, and turns that grade into confidence-aware, clinically
framed decision support — without any additional model training. See
`../docs/innovation_design_notes.md` for the full design rationale and citations.

## What it does

1. Runs the uploaded image through the **exact Notebook 2 preprocessing pipeline**
   (`backend/preprocessing.py`).
2. Grades it with the **trained Notebook 5 model**, using **Monte Carlo Dropout** (30 stochastic
   forward passes) to get both a predicted grade and an uncertainty estimate — `backend/inference.py`.
3. Generates a **Grad-CAM overlay** (ported from Notebook 6) showing which region drove the
   prediction, plus a plain-language description of that region — `backend/gradcam.py`.
4. Maps the grade + confidence onto a **referral-urgency recommendation**, escalating low-confidence
   predictions — `backend/clinical_triage.py`.
5. Writes a short **plain-language explanation** of the result — `backend/explanation.py`.
6. Bundles all of the above into a **downloadable one-page PDF screening report** —
   `backend/report_generator.py`.

## Setup

```bash
cd app
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy the trained model into place (see `models/README.md`):

```
app/models/diabetic_retinopathy_efficientnetb0_notebook5_final_model.keras
```

## Run

```bash
streamlit run frontend/streamlit_app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`), upload a fundus image,
and the app will show the preprocessed image, the Grad-CAM overlay, the predicted grade with
confidence, the referral recommendation, the plain-language explanation, and a PDF download button.

## Test (no model required)

```bash
pytest tests/
```

This checks the preprocessing pipeline in isolation on a synthetic image, so you can confirm the
pipeline is wired correctly even before the (potentially large) model file is in place.

## Notes on hardware

Everything here runs on CPU if needed, but will use your RTX 3050 automatically if you have a
GPU-enabled `tensorflow` build installed and CUDA/cuDNN set up — a single MC-Dropout inference pass
(30 forward passes on one 224x224 image) is lightweight compared to training, so a 4GB GPU (or even
CPU-only) is comfortably enough for this application.
