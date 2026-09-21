# Video

Max 20 minutes, face visible throughout — not voiceover-only.

- `script_and_storyboard.md` — your talking points, in the required structure: steps taken → what
  was innovated → backend algorithms/architecture → what was specifically done at each stage, with
  every decision justified out loud (including any synthetic/augmented data use).
- `hosted_video_url.txt` — the final hosted video link, once recorded and uploaded (e.g. YouTube
  unlisted, university video platform). This URL must also be embedded in the report.

Suggested flow, using this repo as your evidence source:

1. Dataset & problem (`../notebooks/01_.../`, `../report/diagrams/pipeline_diagram.png` on screen).
2. Preprocessing, live on a sample image (`../notebooks/02_.../`, or the app's "show preprocessing
   steps" panel for a faster live demo).
3. Augmentation & balancing (`../notebooks/03_.../`).
4. Architecture & transfer learning (`../notebooks/04_.../`, `../report/diagrams/architecture_diagram.png`).
5. Training strategy (`../notebooks/05_.../`).
6. Evaluation: metrics, confusion matrix, error analysis (`../notebooks/06_.../`,
   `../kaggle_artifacts/notebook_6_evaluation_gradcam/`).
7. **Innovation, live**: run `../app/` on camera — upload an image, show the grade, the Grad-CAM
   overlay, the confidence/uncertainty numbers, the referral recommendation, the explanation, and
   download the PDF report. Explicitly explain the MC-Dropout mechanism and why no retraining was
   needed (see `../docs/innovation_design_notes.md`).
8. Limitations, honestly: minority-class recall, the optic-disc Grad-CAM confound, the illustrative
   (non-clinical) nature of the referral mapping.
