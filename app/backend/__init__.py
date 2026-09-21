"""Backend package for the Diabetic Retinopathy Grading application.

Modules:
    preprocessing     -- ported unchanged from Notebook 2
    inference          -- model loading + Monte Carlo Dropout prediction
    gradcam            -- ported from Notebook 6, plus a quantitative region-description helper
    uncertainty        -- interprets MC Dropout statistics into a plain confidence flag
    clinical_triage    -- maps grade + confidence onto a referral-urgency recommendation
    explanation        -- template-based plain-language explanation generator
    report_generator   -- builds the downloadable PDF screening report
"""
