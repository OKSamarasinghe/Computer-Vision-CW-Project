# Innovation Design Notes — Uncertainty-Aware Clinical Decision-Support Interface

*(Working notes for Criterion 9, "Innovation, Practical Impact & Critical Discussion" — 5 marks.
Content here is written to be liftable into the report's innovation section, with full citations.)*

## 1. What this is, in one sentence

A local web application that wraps the trained EfficientNetB0 model in an uncertainty-aware,
clinically-framed decision-support interface — combining two of the brief's three listed innovation
examples (Grad-CAM explanation, a small web UI) plus a template-based plain-language explanation
layer (the brief's third example) into one coherent tool, **using no additional training anywhere**.

## 2. Why this combination, not just "a UI with an uploader"

Notebook 6's evaluation already surfaced two honest, load-bearing findings that a bare "upload →
grade" demo would ignore:

- **Minority-class recall is weak** (0.27–0.38 for Mild, Severe, Proliferative_DR) despite the
  Notebook 3 balancing strategy — meaning the model itself is genuinely more likely to be wrong on
  exactly the classes where being wrong matters most clinically (Severe/Proliferative_DR are the
  sight-threatening stages).
- **Grad-CAM is sometimes confounded by the optic disc's high visual salience**, a documented
  failure mode of gradient-based saliency on fundus images, not a code defect.

Rather than hiding these limitations behind a confident-looking demo, the application is designed
*around* them: it treats every prediction as provisional until backed by a quantified confidence
signal, and it never claims more certainty than the model actually has.

## 3. The four components, and why each needs no additional training

### 3.1 Monte Carlo Dropout uncertainty estimation

The trained model already contains `Dropout(0.3)` and `Dropout(0.2)` layers in its head (added for
overfitting prevention in Notebooks 4–5). At normal inference time, Keras disables Dropout
(`training=False`). Setting `training=True` at inference instead keeps Dropout stochastic: running
the same image through the network `T` times gives `T` slightly different softmax outputs, and their
spread is a usable approximation of the model's own predictive uncertainty (epistemic uncertainty in
particular). This is the Monte Carlo Dropout method (Gal and Ghahramani, 2016), and it requires
**zero changes to trained weights** — only a different flag at inference time.

This is not a hypothetical fit for this exact task: Siebert, Grasshoff and Rostalski (2023) apply
uncertainty analysis methods directly to diabetic retinopathy grading models, motivated by the same
concern this project found empirically — that a network "typically does not know when they do not
know," and screening automation should be able to refer difficult decisions to specialists rather
than grade everything with false confidence. Separately, recent DR-specific work (e.g. a 2025
DRetNet-style framework) explicitly uses MC Dropout to separate aleatoric uncertainty (image-quality
noise) from epistemic uncertainty (limited training data for a class) and uses the result to flag
cases for manual review — precisely this application's design.

**Design choice:** `HIGH_UNCERTAINTY_STANDARD_DEVIATION_THRESHOLD` (app/config.py) is set
conservatively, so the confidence flag errs toward requesting manual review rather than toward false
reassurance — appropriate given Notebook 6's own minority-class recall findings.

### 3.2 Grad-CAM region description

Ported unchanged from Notebook 6 (`generate_gradcam_heatmap`, `overlay_gradcam_heatmap_on_image`).
The one addition, `describe_heatmap_activation_region()`, is pure geometry on the already-computed
heatmap (an intensity-weighted centroid and a concentration ratio) — no new trained component, just
post-processing that turns a visual overlay into a sentence a non-specialist can read. The
application's explanation panel explicitly carries forward Notebook 6's own honest caveat about the
optic-disc confound, rather than presenting Grad-CAM output as unconditionally reliable.

### 3.3 Clinical referral-urgency mapping

Maps the predicted grade (and confidence flag) onto a referral urgency level, in spirit aligned with
NHS England's diabetic eye screening grading/referral pathway — screening results are graded (R0–R3,
M0–M1) and routed to routine annual recall, a surveillance clinic, or hospital eye services depending
on grade (NHS England, 2023). This project's 5-class APTOS labels don't map one-to-one onto the
NHS's R/M grading system, so `REFERRAL_URGENCY_BY_CLASS_NAME` (app/config.py) is an explicitly
**simplified, illustrative** mapping built for this prototype, not a reproduction of the NHS pathway
— every recommendation the app produces carries a disclaimer saying so and stating this is not a
diagnostic device.

**Design choice:** a Low-confidence prediction always escalates the recommended action by at least
one step, so uncertainty can only increase caution, never decrease it.

### 3.4 Plain-language explanation + downloadable PDF report

Template-based natural-language generation (string formatting over already-computed values — no LLM
call, no training) satisfies the brief's third innovation example while keeping the application fully
offline and reproducible for the video demo. The PDF report (`backend/report_generator.py`) is the
concrete "practical impact" artefact: a one-page document bundling the image, the Grad-CAM overlay,
the grade, the confidence, and the recommendation — the kind of thing an actual screening workflow
would need to pass along, not just a chat-style demo.

## 4. What this deliberately does NOT do

- No new model is trained, fine-tuned, or has its weights modified anywhere in `app/`.
- No external LLM/chatbot API is called (keeps the demo self-contained and reliable on camera).
- The clinical mapping is explicitly labelled illustrative, not validated guidance — avoiding an
  overclaim the coursework's ethical-concerns section would otherwise have to walk back.

## 5. References (Harvard APA 7th, 2022+ where the report requires it)

- NHS England (2023) *Diabetic eye screening: Patient grading, referral, surveillance*. Available
  at: https://assets.publishing.service.gov.uk/media/65046832fc63f60014957368/20230725_Diabetic_eye_screening_pathways_-_patient_grading_referral_surveillance.pdf
  (Accessed: 21 September 2026).
- Siebert, M., Grasshoff, J. and Rostalski, P. (2023) 'Uncertainty Analysis of Deep Kernel Learning
  Methods on Diabetic Retinopathy Grading', *IEEE Access*, 11, pp. 146173–146184. doi:
  10.1109/ACCESS.2023.3343642.
- Alghamdi, H. (2025) [Grad-CAM/ophthalmologist-agreement study — already cited in
  `Report_Section_Criterion6_Model_Evaluation.md`; carry the same full citation into the reference
  list, since it supports both the Notebook 6 discussion and this application's Grad-CAM component].

Note: Gal and Ghahramani's (2016) original Monte Carlo Dropout paper predates the report's 2022+
citation requirement — it is included above for technical completeness/attribution, but should be
cited in the report itself only alongside a 2022+ source that applies the same method to DR grading
(e.g. Siebert, Grasshoff and Rostalski, 2023, above), so the report's citation-recency requirement is
met by the DR-specific evidence, with the original method paper as supporting background if your
citation style allows it. Check your module's exact rule on background/foundational citations before
submission.
