# Diagrams

Required:
- `architecture_diagram.png` — the CNN/transfer-learning model structure (EfficientNetB0 backbone →
  GlobalAveragePooling2D → Dropout(0.3) → Dense(128, relu) → Dropout(0.2) → Dense(5, softmax)).
- `pipeline_diagram.png` — end-to-end: raw data → preprocessing → augmentation → training → evaluation
  → (this project's addition) the app's MC-Dropout/Grad-CAM/triage inference pipeline.

Add any other diagram that makes a non-obvious step easier to follow (e.g. the staged transfer-
learning freeze/unfreeze schedule, or the MC-Dropout uncertainty flow).
