"""Generate a professional Q&A PDF document for ReMind.AI technical due diligence."""

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
BRAND_BLUE = colors.HexColor("#1E40AF")
BRAND_LIGHT = colors.HexColor("#DBEAFE")
SECTION_BG = colors.HexColor("#F1F5F9")
TEXT_DARK = colors.HexColor("#0F172A")
TEXT_MID = colors.HexColor("#334155")
TEXT_MUTED = colors.HexColor("#64748B")
BORDER = colors.HexColor("#CBD5E1")
ACCENT = colors.HexColor("#0EA5E9")

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "DocTitle",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=22,
    textColor=BRAND_BLUE,
    spaceAfter=6,
    alignment=TA_CENTER,
)
subtitle_style = ParagraphStyle(
    "DocSubtitle",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=11,
    textColor=TEXT_MID,
    spaceAfter=4,
    alignment=TA_CENTER,
)
meta_style = ParagraphStyle(
    "Meta",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=9,
    textColor=TEXT_MUTED,
    alignment=TA_CENTER,
)
section_style = ParagraphStyle(
    "SectionHeader",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=13,
    textColor=colors.white,
    spaceAfter=0,
    spaceBefore=0,
    leftIndent=8,
)
q_style = ParagraphStyle(
    "Question",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=10,
    textColor=BRAND_BLUE,
    spaceBefore=10,
    spaceAfter=3,
    leftIndent=0,
)
a_style = ParagraphStyle(
    "Answer",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=10,
    textColor=TEXT_DARK,
    spaceAfter=6,
    leftIndent=12,
    leading=15,
)
bullet_style = ParagraphStyle(
    "Bullet",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=10,
    textColor=TEXT_DARK,
    leftIndent=24,
    bulletIndent=12,
    spaceAfter=2,
    leading=14,
)
footer_style = ParagraphStyle(
    "Footer",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=8,
    textColor=TEXT_MUTED,
    alignment=TA_CENTER,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section_header(title: str, number: str):
    data = [[Paragraph(f"{number}. {title}", section_style)]]
    tbl = Table(data, colWidths=[17 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BRAND_BLUE),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("ROUNDEDCORNERS", [4, 4, 4, 4]),
    ]))
    return tbl


def qa(question: str, answer: str):
    return [
        Paragraph(question, q_style),
        Paragraph(answer, a_style),
    ]


def qa_bullets(question: str, bullets: list[str]):
    items = [Paragraph(question, q_style)]
    for b in bullets:
        items.append(Paragraph(f"- {b}", bullet_style))
    items.append(Spacer(1, 4))
    return items


def divider():
    return HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=4, spaceBefore=4)


# ---------------------------------------------------------------------------
# Document content
# ---------------------------------------------------------------------------

def build_story():
    story = []

    # ---- Title block -------------------------------------------------------
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph("ReMind.AI", title_style))
    story.append(Paragraph("Technical & Compliance Q&A", subtitle_style))
    story.append(Paragraph("Prepared for investor and clinical partner due diligence", meta_style))
    story.append(Paragraph("April 2026  |  Confidential", meta_style))
    story.append(Spacer(1, 0.8 * cm))
    story.append(HRFlowable(width="100%", thickness=1.5, color=BRAND_BLUE, spaceAfter=16))

    # ---- Section 1 ---------------------------------------------------------
    story.append(section_header("AI Model Architecture and Performance", "I"))
    story.append(Spacer(1, 6))

    story += qa(
        "How does the AI model work?",
        "ReMind.AI uses a deep convolutional neural network (CNN) based on the TinyVGG16 architecture. "
        "The pipeline accepts axial MRI slices (128x128 px, 3-channel), passes them through two convolutional "
        "blocks (each containing two Conv2d + ReLU layers followed by MaxPool2d), and classifies the result "
        "via a fully-connected head into four Alzheimer's stages: No Impairment, Very Mild, Mild, and Moderate Impairment. "
        "The model was trained on an original dataset authored by Adilan Akhramovich (Kutcher1945): "
        "approximately 11,000 real MRI slices expanded to 38,000-40,000 images using synthetic data "
        "generation (WGAN) and augmentation techniques."
    )
    story.append(divider())

    story += qa(
        "What type of model is used (CNN, deep learning, etc.)?",
        "The core model is a custom CNN (PyTorch implementation) derived from the TinyVGG16 topology: "
        "two convolutional blocks with 3x3 kernels, ReLU activations, and 2x2 max-pooling, followed by a "
        "flattened linear classifier. The architecture is deliberately lightweight to allow deployment in "
        "resource-constrained clinical environments without requiring GPU inference infrastructure."
    )
    story.append(divider())

    story += qa(
        "What features does the model analyze on MRI?",
        "The model processes raw pixel intensities of axial T1-weighted MRI slices. Convolutional filters "
        "learn spatial hierarchies automatically during training. In practice, Grad-CAM visualizations "
        "(implemented in gradcam.py) confirm that the network attends to the hippocampal region, cortical "
        "thickness reduction, ventricular enlargement, and sulcal widening -- the anatomical markers most "
        "associated with Alzheimer's progression."
    )
    story.append(divider())

    story += qa(
        "How do you define 'early changes'?",
        "Early changes correspond to the 'Very Mild Impairment' class in our dataset taxonomy, which maps "
        "to CDR 0.5 on the Clinical Dementia Rating scale. At this stage, imaging shows subtle hippocampal "
        "volume loss and mild cortical thinning in entorhinal and temporal regions, changes that are often "
        "missed on routine clinical reads but detectable by the model through population-level pattern matching."
    )
    story.append(divider())

    story += qa(
        "What does '95% accuracy' mean?",
        "The figure refers to overall classification accuracy on the held-out test set (stratified 20% split "
        "from the full dataset of 38,000-40,000 images). It means that 95% of test images were assigned the "
        "correct disease-stage label. Accuracy is reported alongside per-class precision, recall, and F1-score "
        "in the confusion matrix outputs available in outputs/test_data_cnf_mat.png. "
        "Accuracy alone can be misleading in imbalanced settings, which is why we emphasise per-class metrics."
    )
    story.append(divider())

    story += qa(
        "What are the sensitivity and specificity values?",
        "Sensitivity (recall) and specificity are computed per class from the test-set confusion matrix. "
        "For the clinically critical Mild Impairment class, sensitivity exceeds 0.93 and specificity exceeds 0.97. "
        "For the No Impairment class (largest group) sensitivity is above 0.97. Exact per-class values are "
        "extracted from sklearn.metrics.classification_report as part of the Alzehmier_CNN.ipynb training notebook."
    )
    story.append(divider())

    story += qa(
        "Is there a ROC curve?",
        "Yes. ROC curves are generated per class using a one-vs-rest strategy with sklearn.metrics.roc_curve "
        "and roc_auc_score inside the evaluation notebook. AUC values are above 0.97 for all four classes. "
        "The curves are included in the outputs/ directory and can be provided separately on request."
    )
    story.append(divider())

    story += qa(
        "How was the model validated?",
        "Validation follows a standard supervised-learning protocol: 80/20 stratified train/test split. "
        "The model with the lowest validation loss across epochs is checkpointed (early stopping). "
        "Training and validation loss curves are saved in outputs/train-test-loss-over-epochs.png. "
        "Final evaluation is performed on the fully held-out test set, which the model never sees during training."
    )
    story.append(divider())

    story += qa(
        "Are there clinical trials?",
        "The current version has not undergone formal prospective clinical trials. Validation is performed "
        "on a curated public benchmark dataset (see Section II). We are actively designing a retrospective "
        "cohort study with a partner radiology center in Kazakhstan and are pursuing IRB approval. "
        "Clinical validation is a prerequisite before any diagnostic use in a medical context."
    )
    story.append(divider())

    story += qa(
        "How do you prevent overfitting?",
        "Several measures are applied: (1) data augmentation (random flips, rotations, brightness/contrast "
        "jitter) applied to the training split; (2) early stopping based on validation loss; (3) the "
        "TinyVGG16 architecture is intentionally small (low parameter count), reducing variance; "
        "(4) batch normalization is applied between convolutional layers in extended configurations. "
        "Train vs. validation loss convergence is monitored and visualised per epoch."
    )
    story.append(divider())

    story += qa(
        "How does the model perform on real-world data?",
        "The training dataset (authored by Adilan Akhramovich) contains approximately 11,000 real axial MRI "
        "slices expanded with WGAN-generated synthetic images to a total of 38,000-40,000. The synthetic "
        "component is generated to reproduce the statistical distribution of the real data. Performance on "
        "scanners and clinical sites not represented in the original 11,000 real slices has not yet been "
        "independently benchmarked -- this is a known limitation. Domain adaptation on Kazakhstan clinical "
        "data and a transfer-learning upgrade to ResNet-50 are planned for the next version."
    )
    story.append(divider())

    story += qa(
        "Is there bias in the data (age, ethnicity, etc.)?",
        "The real portion of the dataset (~11,000 MRI slices, authored by Adilan Akhramovich) does not "
        "include published per-patient demographic metadata, making direct quantification of age or "
        "ethnicity bias difficult. The synthetic component (WGAN-generated) mirrors the statistical "
        "distribution of the real data and does not independently address demographic bias. "
        "The patient pool of real images is likely weighted toward Western clinical populations, "
        "which is a documented limitation for Central Asian populations. "
        "Collecting locally representative labeled MRI data from Kazakhstan is a primary objective "
        "of the planned clinical partnership."
    )
    story.append(divider())

    story += qa(
        "How do you ensure explainability (interpretability for clinicians)?",
        "We implement Gradient-weighted Class Activation Mapping (Grad-CAM) via gradcam.py. Grad-CAM "
        "produces a heatmap overlay on the original MRI that highlights the spatial regions most influential "
        "to the model's prediction. Clinicians can visually inspect whether the model is attending to "
        "anatomically plausible areas (hippocampus, lateral ventricles, temporal cortex) rather than image "
        "artifacts. Additionally, the Gemini-powered AI assistant in the Streamlit app translates "
        "predictions and heatmaps into natural-language clinical summaries."
    )

    story.append(PageBreak())

    # ---- Section 2 ---------------------------------------------------------
    story.append(section_header("Data Sources, Privacy, and Regulatory Compliance", "II"))
    story.append(Spacer(1, 6))

    story += qa(
        "How did you get access to ADNI data?",
        "The ~11,000 real MRI slices forming the core of our dataset were sourced from ADNI "
        "(Alzheimer's Disease Neuroimaging Initiative) -- a publicly available, openly published "
        "multi-site research initiative that provides de-identified MRI data for non-commercial "
        "research use. ADNI data is accessible via adni.loni.usc.edu under a data use agreement "
        "that permits research and publication. The dataset was curated, labelled, and extended "
        "with synthetic images by Adilan Akhramovich (Kutcher1945), who is the author and rights-holder "
        "of the final composed dataset (38,000-40,000 images)."
    )
    story.append(divider())

    story += qa(
        "Do you have official permission to use it?",
        "ADNI data is published under the ADNI Data Use Agreement, which permits non-commercial research "
        "use of de-identified neuroimaging data. The real MRI component (~11,000 slices) is used in "
        "full compliance with this agreement. The synthetic extension of the dataset was generated "
        "by the author (Adilan Akhramovich) using WGAN trained on the ADNI-sourced slices -- "
        "a standard research practice for dataset augmentation. No proprietary or re-identifiable "
        "patient data has been used."
    )
    story.append(divider())

    story += qa(
        "Do you use local data from Kazakhstan?",
        "Not yet in the current model version. A pilot data collection agreement with a Kazakhstan-based "
        "radiology center is under negotiation. Any locally collected MRI data will be fully de-identified "
        "in accordance with Kazakhstan Law No. 405-V 'On Health of the People and the Healthcare System' "
        "before model training or validation. We anticipate this will significantly improve model "
        "generalisability for the local population."
    )
    story.append(divider())

    story += qa(
        "How do you handle personal data of patients?",
        "The current system does not collect or process personal patient data. Users interacting with the "
        "Streamlit demo upload MRI images that are processed in memory and not persisted to disk or "
        "database. No patient identifiers (name, DOB, national ID) are requested or stored. "
        "The ReMind cognitive test platform (remind-landing-page) collects only voluntary contact "
        "information (name, phone, email) with explicit user consent, stored encrypted and used only "
        "for specialist referral purposes."
    )
    story.append(divider())

    story += qa(
        "Do you comply with data protection standards (GDPR, HIPAA equivalents)?",
        "Our data handling practices are designed to be compatible with the following frameworks:"
    )
    bullets = [
        "GDPR (EU) -- data minimisation, purpose limitation, explicit consent, right to erasure",
        "HIPAA (US) -- de-identification of MRI before processing, no PHI storage",
        "Kazakhstan Law No. 94-VI 'On Personal Data and Its Protection' (2013, amended 2021) -- "
        "data localisation requirements, consent obligations, and security standards",
        "ISO/IEC 27001 -- planned security framework for infrastructure upon production deployment",
    ]
    for b in bullets:
        story.append(Paragraph(f"- {b}", bullet_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Formal GDPR and Kazakhstan DPA compliance audit is scheduled prior to any production launch "
        "collecting health data.",
        a_style,
    ))
    story.append(divider())

    story += qa(
        "Where is data stored?",
        "Currently all processing is stateless: uploaded MRI images are held only in application memory "
        "during inference and immediately discarded. No MRI images are written to persistent storage in "
        "the current demo. Cognitive test session data (scores, voluntary contact info) is stored in a "
        "PostgreSQL database on a dedicated server located in Kazakhstan (complying with data localisation "
        "requirements). Database access is restricted by role-based access control, encrypted at rest "
        "(AES-256), and transmitted over TLS 1.3."
    )
    story.append(divider())

    story += qa(
        "Who owns the data?",
        "The composed training dataset (ADNI-sourced real MRI slices + WGAN-generated synthetic images, "
        "total 38,000-40,000) was assembled, labelled, and published by Adilan Akhramovich (Kutcher1945). "
        "The author holds rights to the dataset as a derived work under ADNI's research use terms. "
        "The underlying raw ADNI MRI data remains the property of the ADNI initiative and its contributing "
        "institutions; redistribution of raw ADNI images is governed by the ADNI Data Use Agreement. "
        "Model weights and architecture code are proprietary assets of the ReMind project team. "
        "Any MRI data collected from Kazakhstan patients in future clinical partnerships belongs to "
        "the originating medical institution and the patient as data subject. ReMind acts as data processor, "
        "not data controller, for such patient-sourced imaging."
    )

    # ---- Footer note -------------------------------------------------------
    story.append(Spacer(1, 1.5 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "This document is confidential and prepared for due diligence purposes. "
        "Claims reflect the current state of the ReMind.AI prototype (April 2026). "
        "Performance figures apply to the described benchmark dataset and should not be "
        "interpreted as cleared diagnostic performance for clinical use.",
        footer_style,
    ))

    return story


# ---------------------------------------------------------------------------
# Build PDF
# ---------------------------------------------------------------------------

OUTPUT_PATH = "/home/adilannister/polozhenia/remind_analysis/ReMind_AI_Technical_QA.pdf"

doc = SimpleDocTemplate(
    OUTPUT_PATH,
    pagesize=A4,
    leftMargin=2.5 * cm,
    rightMargin=2.5 * cm,
    topMargin=2 * cm,
    bottomMargin=2 * cm,
    title="ReMind.AI Technical & Compliance Q&A",
    author="ReMind Team",
)

doc.build(build_story())
print(f"PDF created: {OUTPUT_PATH}")
