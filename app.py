import os
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from model_arch import AlzheimerDetector
from google import genai
from dotenv import load_dotenv
import base64
import io
import requests
from gradcam import generate_gradcam_visualization, create_comparison_image
import numpy as np
import re

# Markdown to HTML converter for better text rendering
def md_to_html(text):
    """Convert markdown text to properly formatted HTML."""
    if not text:
        return ""

    # Convert headers (do this before other formatting)
    text = re.sub(r'^### (.+)$', r'<h3 style="color: #000000; font-size: 1.2rem; margin: 1.5rem 0 0.8rem 0; font-weight: 700;">\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$', r'<h2 style="color: #000000; font-size: 1.5rem; margin: 1.8rem 0 1rem 0; font-weight: 700;">\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$', r'<h1 style="color: #000000; font-size: 1.8rem; margin: 2rem 0 1rem 0; font-weight: 700;">\1</h1>', text, flags=re.MULTILINE)

    # Convert bold text (**text**)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong style="color: #000000; font-weight: 700;">\1</strong>', text)

    # Convert italic (*text*) - but not if it's part of **
    text = re.sub(r'(?<!\*)\*(?!\*)([^*]+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', text)

    # Convert horizontal rules (thick decorative lines)
    text = re.sub(r'^━{10,}$', '<hr style="border: none; border-top: 3px solid #000000; margin: 1.5rem 0; opacity: 0.8;">', text, flags=re.MULTILINE)
    text = re.sub(r'^-{3,}$', '<hr style="border: none; border-top: 1px solid #cccccc; margin: 1.5rem 0;">', text, flags=re.MULTILINE)

    # Convert bullet points (•, -, *)
    text = re.sub(r'^[•\-\*] (.+)$', r'<div style="margin-left: 1.5rem; padding: 0.4rem 0; color: #2d3748; line-height: 1.6;"><span style="color: #000000; font-weight: 600;">•</span> \1</div>', text, flags=re.MULTILINE)

    # Convert numbered lists (1., 2., etc.) with Roman numerals support
    text = re.sub(r'^([IVX]+)\. (.+)$', r'<div style="margin: 1rem 0 0.5rem 0; padding: 0.5rem 0; color: #000000; font-size: 1.1rem;"><strong>\1.</strong> \2</div>', text, flags=re.MULTILINE)
    text = re.sub(r'^(\d+)\. (.+)$', r'<div style="margin-left: 1.5rem; padding: 0.4rem 0; color: #2d3748;"><strong style="color: #000000;">\1.</strong> \2</div>', text, flags=re.MULTILINE)

    # Convert line breaks
    text = text.replace('\n', '<br>')

    # Clean up excessive breaks
    text = re.sub(r'(<br>){4,}', '<br><br>', text)

    return text

# Load environment variables
load_dotenv()

# Configure Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("API ключ Gemini не настроен. Проверьте ваш файл .env или определите ключ в коде.")
    st.stop()

# Configure Gemini
_genai_client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = 'gemini-3.1-flash-lite-preview'

# Pixtral API Configuration
PIXTRAL_API_KEY = os.getenv("PIXTRAL_API_KEY", "556m2SLKNQBicaqmK7V7OOB2Seld7TvY")
PIXTRAL_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"

# Page configuration
st.set_page_config(
    page_title="ReMind.AI",
    page_icon="",
    layout="wide",  # page width
    # initial_sidebar_state="expanded",
)

# Custom CSS styles - Premium Dark Design
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

    /* ── App background ── */
    .stApp { background: #050a14 !important; }
    .main  { background: #050a14 !important; padding: 2rem; }
    [data-testid="stAppViewContainer"] { background: #050a14 !important; }
    [data-testid="block-container"] { background: transparent !important; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #0d1426 !important;
        border-right: 1px solid rgba(255,255,255,0.07) !important;
    }
    [data-testid="stSidebar"] * { color: rgba(255,255,255,0.75) !important; }
    [data-testid="stSidebar"] .row-widget.stRadio > div {
        background: rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 8px;
        border: 1px solid rgba(255,255,255,0.07);
    }
    [data-testid="stSidebar"] .row-widget.stRadio > div label {
        color: rgba(255,255,255,0.65) !important;
        padding: 10px 16px;
        border-radius: 8px;
        transition: all 0.2s;
    }
    [data-testid="stSidebar"] .row-widget.stRadio > div label:hover {
        background: rgba(255,255,255,0.08) !important;
        color: rgba(255,255,255,0.95) !important;
    }

    /* ── Global text ── */
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: rgba(255,255,255,0.85) !important;
    }
    h2 { font-size: 2.2rem !important; font-weight: 700 !important; letter-spacing: -0.5px !important; }
    h4 { color: rgba(255,255,255,0.45) !important; font-weight: 400 !important; font-size: 1rem !important; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #3b5bdb, #22d3ee) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 2rem !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        transition: all 0.2s !important;
        box-shadow: 0 0 20px rgba(59,91,219,0.35) !important;
        letter-spacing: 0.3px !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 0 32px rgba(59,91,219,0.55) !important;
        opacity: 0.92 !important;
    }
    .stButton > button:disabled {
        background: rgba(255,255,255,0.07) !important;
        color: rgba(255,255,255,0.25) !important;
        box-shadow: none !important;
        transform: none !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }

    /* ── File Uploader ── */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.03) !important;
        border: 1.5px dashed rgba(59,91,219,0.4) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        transition: all 0.2s !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(59,91,219,0.7) !important;
        background: rgba(59,91,219,0.05) !important;
    }
    [data-testid="stFileUploader"] section { border: none !important; background: transparent !important; }
    [data-testid="stFileUploader"] * { color: rgba(255,255,255,0.6) !important; }
    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #3b5bdb, #22d3ee) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileUploader"] button * { color: white !important; }
    [data-testid="stFileUploader"] button svg,
    [data-testid="stFileUploader"] button svg path,
    [data-testid="stFileUploader"] button svg line { stroke: white !important; fill: none !important; }
    [data-testid="stFileUploader"] small { color: rgba(255,255,255,0.35) !important; }

    /* ── Prediction box ── */
    .prediction-box {
        background: linear-gradient(135deg, rgba(59,91,219,0.15), rgba(34,211,238,0.1));
        border: 1px solid rgba(59,91,219,0.3);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 1.5rem 0;
        text-align: center;
        box-shadow: 0 0 40px rgba(59,91,219,0.2);
        animation: fadeInUp 0.5s ease;
    }
    .prediction-box h3 { color: rgba(255,255,255,0.95) !important; font-size: 1.1rem !important; font-weight: 500 !important; letter-spacing: 0.12em !important; text-transform: uppercase !important; margin-bottom: 0.5rem !important; }
    .prediction-box p  { color: white !important; font-weight: 700 !important; }
    .prediction-box strong { color: rgba(255,255,255,0.9) !important; }

    /* ── Recommendations box ── */
    .recommendations-box {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
    }
    .recommendations-box h2 { color: rgba(255,255,255,0.95) !important; font-size: 1.6rem !important; margin-bottom: 1.5rem !important; }
    .recommendations-box h3 { color: rgba(255,255,255,0.85) !important; font-weight: 600 !important; }
    .recommendations-box strong { color: rgba(255,255,255,0.9) !important; font-weight: 600 !important; }
    .recommendations-box hr { border: none !important; border-top: 1px solid rgba(255,255,255,0.1) !important; margin: 1.2rem 0 !important; }

    /* ── Image container ── */
    .image-container { display: flex; flex-direction: column; align-items: center; margin: 1.5rem 0; }
    .image-container img { border-radius: 16px; box-shadow: 0 0 40px rgba(0,0,0,0.5); border: 1px solid rgba(255,255,255,0.1); }
    .image-label { margin-top: 0.75rem; color: rgba(255,255,255,0.4) !important; font-size: 0.85rem; }

    /* ── Alerts ── */
    .stAlert { border-radius: 12px !important; }
    div[data-testid="stNotification"] { background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 12px !important; }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: #3b5bdb !important; }
    .stSpinner *, [data-testid="stSpinner"] * { color: rgba(255,255,255,0.6) !important; }

    /* ── Form inputs (Данные пациента page) ── */
    input[type="text"], input[type="number"], textarea {
        background: rgba(255,255,255,0.05) !important;
        border: 1.5px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        color: rgba(255,255,255,0.85) !important;
        -webkit-text-fill-color: rgba(255,255,255,0.85) !important;
    }
    input:focus, textarea:focus {
        border-color: rgba(59,91,219,0.6) !important;
        box-shadow: 0 0 0 3px rgba(59,91,219,0.15) !important;
    }
    div[data-baseweb="select"] > div {
        background: rgba(255,255,255,0.05) !important;
        border: 1.5px solid rgba(255,255,255,0.1) !important;
        color: rgba(255,255,255,0.85) !important;
        border-radius: 10px !important;
    }
    ul[role="listbox"] { background: #0d1426 !important; border: 1px solid rgba(255,255,255,0.1) !important; }
    ul[role="listbox"] li { color: rgba(255,255,255,0.75) !important; background: transparent !important; }
    ul[role="listbox"] li:hover { background: rgba(255,255,255,0.08) !important; }

    /* ── Divider lines ── */
    hr { border-color: rgba(255,255,255,0.08) !important; }

    /* ── Column image labels ── */
    .stImage p { color: rgba(255,255,255,0.5) !important; }

    /* ── Animations ── */
    @keyframes fadeIn    { from { opacity:0 } to { opacity:1 } }
    @keyframes fadeInUp  { from { opacity:0; transform:translateY(16px) } to { opacity:1; transform:translateY(0) } }
    </style>
    """,
    unsafe_allow_html=True,
)

# Pixtral Image Validation Function
def validate_mri_image(image_base64):
    """
    Validate if the uploaded image is a brain MRI scan using Pixtral vision AI.

    Args:
        image_base64: Base64 encoded image string

    Returns:
        tuple: (is_valid: bool, message: str, confidence: str)
    """
    validation_prompt = """Проанализируйте это изображение внимательно и определите, является ли оно МРТ снимком головного мозга.

Вы должны ответить ТОЧНО в этом формате:
ВАЛИДНО: [ДА/НЕТ]
УВЕРЕННОСТЬ: [ВЫСОКАЯ/СРЕДНЯЯ/НИЗКАЯ]
ПРИЧИНА: [Краткое объяснение]

Критерии для валидного МРТ головного мозга:
1. Должно быть медицинским изображением (черно-белое или цветное медицинское изображение)
2. Должны быть видны структуры мозга (кора головного мозга, желудочки, белое/серое вещество)
3. Должно быть МРТ снимком (не КТ, рентген, УЗИ или другие типы изображений)
4. Должен быть правильный аксиальный, сагиттальный или корональный вид мозга
5. Не фотография, рисунок или немедицинское изображение

Примеры НЕВАЛИДНЫХ изображений:
- Фотографии людей, животных, объектов, пейзажей
- Снимки других частей тела (колено, грудь, живот МРТ)
- КТ снимки, рентгеновские снимки, УЗИ
- Низкокачественные или полностью размытые изображения
- Рисунки или иллюстрации"""

    payload = {
        "model": "pixtral-12b-2409",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": validation_prompt},
                    {"type": "image_url", "image_url": {"url": image_base64}}
                ]
            }
        ],
        "temperature": 0.3,
        "top_p": 1,
        "stream": False
    }

    try:
        response = requests.post(
            PIXTRAL_ENDPOINT,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {PIXTRAL_API_KEY}"
            },
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            result_text = data.get('choices', [{}])[0].get('message', {}).get('content', '')

            is_valid = "ВАЛИДНО: ДА" in result_text.upper() or "VALID: YES" in result_text.upper()
            lines = result_text.strip().split('\n')
            confidence = "НЕИЗВЕСТНО"
            reason = "Причина не указана"

            for line in lines:
                if "УВЕРЕННОСТЬ:" in line.upper() or "CONFIDENCE:" in line.upper():
                    confidence = line.split(':', 1)[1].strip()
                elif "ПРИЧИНА:" in line.upper() or "REASON:" in line.upper():
                    reason = line.split(':', 1)[1].strip()

            return is_valid, reason, confidence
        else:
            # API unavailable or key issue — skip validation, proceed
            return True, "Проверка недоступна, анализ продолжен", "НИЗКАЯ"

    except Exception as e:
        return True, "Проверка пропущена, анализ продолжен", "НИЗКАЯ"


def analyze_brain_regions(image_base64, predicted_class, confidence_percent):
    """
    Use Pixtral AI to analyze specific brain regions and identify abnormalities.

    Args:
        image_base64: Base64 encoded MRI image
        predicted_class: The predicted Alzheimer's stage
        confidence_percent: Model confidence percentage

    Returns:
        str: Detailed medical analysis of brain regions
    """
    analysis_prompt = f"""Вы эксперт-радиолог, анализирующий МРТ снимок головного мозга. ОТВЕЧАЙТЕ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ.

**Клинический контекст:**
- Прогноз модели ИИ: {predicted_class}
- Уверенность модели: {confidence_percent:.1f}%

**Ваша задача:**
Проанализируйте это МРТ изображение головного мозга и предоставьте детальную оценку следующего:

1. **Гиппокампальная область:** Оценить атрофию, потерю объема или структурные изменения
2. **Желудочковая система:** Оценить размер желудочков и любое расширение
3. **Корковые области:** Искать истончение коры, особенно в височных и теменных долях
4. **Белое вещество:** Определить любые гиперинтенсивности белого вещества или повреждения
5. **Общая структура мозга:** Общие наблюдения об объеме и симметрии мозга

**Форматируйте ваш ответ так:**

РЕГИОНАЛЬНЫЙ АНАЛИЗ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Гиппокамп и медиальная височная доля:
[Ваши детальные находки]

- Желудочковая система:
[Ваши детальные находки]

- Корковые области:
[Ваши детальные находки]

- Белое вещество:
[Ваши детальные находки]

- Общая оценка:
[Резюме ключевых находок]

КОРРЕЛЯЦИЯ С ПРОГНОЗОМ ИИ:
[Как ваши находки подтверждают или противоречат прогнозу ИИ "{predicted_class}"]

**Важно:** Будьте конкретны относительно локализации (левое/правое полушарие, передний/задний отдел и т.д.) и тяжести (легкая/умеренная/тяжелая).
ОТВЕЧАЙТЕ ПОЛНОСТЬЮ НА РУССКОМ ЯЗЫКЕ.
"""

    payload = {
        "model": "pixtral-12b-2409",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": analysis_prompt},
                    {"type": "image_url", "image_url": {"url": image_base64}}
                ]
            }
        ],
        "temperature": 0.4,
        "top_p": 0.9,
        "stream": False
    }

    try:
        response = requests.post(
            PIXTRAL_ENDPOINT,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {PIXTRAL_API_KEY}"
            },
            json=payload,
            timeout=45
        )

        if response.status_code == 200:
            data = response.json()
            analysis_text = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            return analysis_text if analysis_text else "Невозможно сгенерировать детальный анализ."
        else:
            return "Региональный анализ недоступен (сервис Pixtral вернул ошибку). Рекомендации будут основаны только на диагнозе CNN модели."

    except Exception as e:
        return "Региональный анализ недоступен. Рекомендации будут основаны только на диагнозе CNN модели."

# Load Alzheimer's model
@st.cache_resource
def load_model():
    model = AlzheimerDetector(input_shape=3, hidden_units=10, output_shape=4, image_dimension=128).to("cpu")
    model.load_state_dict(torch.load(f"models/alz_CNN.pt", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Class definitions in Russian
class_names = ['Легкое нарушение', 'Умеренное нарушение', 'Нет нарушений', 'Очень легкое нарушение']

# Preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.ToTensor(),
])

# Sidebar
st.sidebar.image('img/logo_3.jpg', width="stretch")
options = st.sidebar.radio('Опции:', ['Данные пациента', 'Диагностика', 'Виртуальный ассистент'], index=1)

# Main page image

# Convert image to Base64
def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Image path
image_path = "img/medical.jpg"
base64_image = get_base64_of_image(image_path)

# Apply background styling
st.markdown(
    f"""
    <style>
    .stApp {{
        background: #050a14;
        background-attachment: fixed;
    }}

    /* Input fields modern styling */
    input[type="text"], input[type="number"], textarea {{
        background-color: #FFFFFF !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 10px !important;
        padding: 12px !important;
        transition: all 0.3s ease !important;
        color: #2d3748 !important;
    }}

    input[type="text"]:focus, input[type="number"]:focus, textarea:focus {{
        border-color: #000000 !important;
        box-shadow: 0 0 0 3px rgba(0, 0, 0, 0.1) !important;
    }}

    /* Input labels */
    label {{
        color: #2d3748 !important;
        font-weight: 500 !important;
    }}

    /* Select dropdown styling */
    div[data-baseweb="select"] {{
        background-color: #FFFFFF !important;
        border-radius: 10px !important;
        border: 2px solid #e2e8f0 !important;
    }}

    div[data-baseweb="select"] > div {{
        color: #2d3748 !important;
        background-color: #FFFFFF !important;
    }}

    div[data-baseweb="select"] input {{
        color: #2d3748 !important;
    }}

    div[data-baseweb="select"] span {{
        color: #2d3748 !important;
    }}

    div[data-baseweb="select"]:hover {{
        border-color: #000000 !important;
    }}

    /* Dropdown menu items */
    ul[role="listbox"] li {{
        color: #2d3748 !important;
        background-color: #FFFFFF !important;
    }}

    ul[role="listbox"] li:hover {{
        background-color: #f7fafc !important;
    }}

    /* Selected option in dropdown */
    div[data-baseweb="select"] [aria-selected="true"] {{
        background-color: #000000 !important;
        color: white !important;
    }}

    /* Placeholder text */
    input::placeholder, textarea::placeholder {{
        color: #a0aec0 !important;
        opacity: 1 !important;
    }}

    /* Number input buttons */
    button[data-testid="stNumberInputStepUp"],
    button[data-testid="stNumberInputStepDown"] {{
        background-color: #f7fafc !important;
        border-radius: 6px !important;
        color: #2d3748 !important;
    }}

    button[data-testid="stNumberInputStepUp"]:hover,
    button[data-testid="stNumberInputStepDown"]:hover {{
        background-color: #edf2f7 !important;
    }}

    /* Number input buttons SVG icons */
    button[data-testid="stNumberInputStepUp"] svg,
    button[data-testid="stNumberInputStepDown"] svg {{
        fill: #2d3748 !important;
        stroke: #2d3748 !important;
    }}

    /* Text area */
    textarea {{
        min-height: 120px !important;
    }}

    /* Streamlit widget labels and text */
    .stTextInput label, .stNumberInput label, .stSelectbox label, .stTextArea label {{
        color: #2d3748 !important;
    }}

    /* Widget help text */
    .stTextInput small, .stNumberInput small, .stSelectbox small, .stTextArea small {{
        color: #718096 !important;
    }}

    /* All paragraph text in main content */
    .main p {{
        color: #2d3748 !important;
    }}

    /* Strong/bold text */
    strong {{
        color: #2d3748 !important;
    }}

    /* Selectbox specific fixes */
    .stSelectbox div[data-baseweb="select"] {{
        background-color: white !important;
    }}

    .stSelectbox div[data-baseweb="select"] > div {{
        background-color: white !important;
    }}

    /* Force all text in selectbox to be dark */
    .stSelectbox * {{
        color: #2d3748 !important;
    }}

    /* Override any conflicting styles */
    [data-baseweb="select"] [data-baseweb="input"] {{
        color: #2d3748 !important;
    }}

    /* Number input value text */
    .stNumberInput input[type="number"] {{
        color: #2d3748 !important;
        -webkit-text-fill-color: #2d3748 !important;
    }}

    /* Comprehensive text visibility fix */
    .stTextInput div[data-baseweb="input"] input,
    .stNumberInput div[data-baseweb="input"] input,
    .stTextArea textarea {{
        color: #2d3748 !important;
        -webkit-text-fill-color: #2d3748 !important;
    }}

    /* Ensure SVG icons in buttons are visible */
    button svg {{
        fill: currentColor !important;
    }}

    button[data-testid="stNumberInputStepUp"] svg path,
    button[data-testid="stNumberInputStepDown"] svg path {{
        stroke: #2d3748 !important;
        fill: #2d3748 !important;
    }}

    /* Markdown content visibility */
    .main div {{
        color: inherit;
    }}

    /* Ensure all divs with inline styles have visible text */
    div[style*="background"] {{
        color: #2d3748;
    }}

    /* Headers in styled divs */
    div[style] h1, div[style] h2, div[style] h3, div[style] h4, div[style] h5, div[style] h6 {{
        color: inherit;
    }}

    /* Strong tags in styled divs */
    div[style] strong {{
        color: #2d3748;
        font-weight: 600;
    }}

    /* Horizontal rules */
    div[style] hr {{
        border-color: rgba(255,255,255,0.08);
    }}

    /* ── Dark inputs (comprehensive override) ── */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {{
        background: rgba(255,255,255,0.05) !important;
        border: 1.5px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        color: rgba(255,255,255,0.85) !important;
        -webkit-text-fill-color: rgba(255,255,255,0.85) !important;
        caret-color: rgba(255,255,255,0.85) !important;
        padding: 0.65rem 0.9rem !important;
        font-size: 0.95rem !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }}
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {{
        border-color: rgba(59,91,219,0.6) !important;
        box-shadow: 0 0 0 3px rgba(59,91,219,0.15) !important;
        outline: none !important;
    }}
    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder {{
        color: rgba(255,255,255,0.25) !important;
        -webkit-text-fill-color: rgba(255,255,255,0.25) !important;
    }}

    /* Number input container & +/- buttons */
    .stNumberInput > div > div {{
        background: rgba(255,255,255,0.05) !important;
        border: 1.5px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
    }}
    button[data-testid="stNumberInputStepUp"],
    button[data-testid="stNumberInputStepDown"] {{
        background: rgba(255,255,255,0.07) !important;
        border: none !important;
        border-left: 1px solid rgba(255,255,255,0.08) !important;
        color: rgba(255,255,255,0.6) !important;
        border-radius: 0 !important;
    }}
    button[data-testid="stNumberInputStepUp"]:hover,
    button[data-testid="stNumberInputStepDown"]:hover {{
        background: rgba(255,255,255,0.12) !important;
    }}
    button[data-testid="stNumberInputStepUp"] svg path,
    button[data-testid="stNumberInputStepDown"] svg path {{
        stroke: rgba(255,255,255,0.6) !important;
        fill: none !important;
    }}

    /* Selectbox */
    .stSelectbox > div > div[data-baseweb="select"] > div {{
        background: rgba(255,255,255,0.05) !important;
        border: 1.5px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        color: rgba(255,255,255,0.85) !important;
    }}
    .stSelectbox > div > div[data-baseweb="select"] > div:focus-within {{
        border-color: rgba(59,91,219,0.6) !important;
        box-shadow: 0 0 0 3px rgba(59,91,219,0.15) !important;
    }}
    .stSelectbox svg {{ fill: rgba(255,255,255,0.4) !important; }}
    [data-baseweb="popover"], [data-baseweb="menu"] {{
        background: #0d1426 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
    }}
    [data-baseweb="option"] {{
        background: transparent !important;
        color: rgba(255,255,255,0.75) !important;
    }}
    [data-baseweb="option"]:hover {{
        background: rgba(59,91,219,0.15) !important;
    }}

    /* Input labels */
    .stTextInput label, .stNumberInput label,
    .stTextArea label, .stSelectbox label {{
        color: rgba(255,255,255,0.45) !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.04em !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# st.image('img/home_page.jpg', width="stretch")

if options == 'Данные пациента':
    exec(open("paciente.py", encoding="utf-8").read())

elif options == 'Диагностика':
    st.markdown("""
        <div style='text-align:center; padding: 2rem 0 2.5rem 0;'>
            <div style='display:inline-flex;align-items:center;gap:8px;background:rgba(59,91,219,0.1);
                        border:1px solid rgba(59,91,219,0.3);border-radius:999px;padding:6px 16px;margin-bottom:1.2rem;'>
                <div style='width:7px;height:7px;border-radius:50%;background:#22d3ee;
                            animation:fadeIn 1s infinite alternate;'></div>
                <span style='font-size:0.72rem;font-weight:600;letter-spacing:0.18em;
                             text-transform:uppercase;color:#22d3ee !important;'>ИИ-диагностика</span>
            </div>
            <h2 style='color:rgba(255,255,255,0.95) !important;font-size:2.2rem !important;
                       margin-bottom:0.6rem !important;'>Диагностика на основе ИИ</h2>
            <p style='color:rgba(255,255,255,0.38) !important;font-size:1rem;max-width:560px;margin:0 auto;line-height:1.6;'>
                Классификация болезни Альцгеймера с использованием глубокого обучения и анализа МРТ.
                Точность модели — <strong style='color:rgba(255,255,255,0.7) !important;'>95.47%</strong>
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <p style='text-align:center;font-size:0.9rem;color:rgba(255,255,255,0.3) !important;
                  margin-bottom:0.5rem;letter-spacing:0.05em;'>
            Загрузите МРТ снимок для начала анализа
        </p>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader(
            "Перетащите изображение МРТ сюда или нажмите для выбора",
            type=["jpg", "jpeg", "png"],
            help="Поддерживаемые форматы: JPG, JPEG, PNG"
        )

    # Variable to store the prediction
    predicted_class = None

    if uploaded_file:
        # Display uploaded image with modern styling
        image = Image.open(uploaded_file)
        image_base64 = f"data:image/{'jpeg' if uploaded_file.type == 'image/jpeg' else 'png'};base64,{base64.b64encode(uploaded_file.getvalue()).decode()}"

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                f"""
                <div class='image-container'>
                    <img src='{image_base64}'
                         style='max-width: 100%; width: 500px; border-radius: 16px; box-shadow: 0 10px 40px rgba(0,0,0,0.15);'>
                    <p class='image-label'>МРТ снимок загружен</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # STAGE 1: Validate image using Pixtral AI
        with st.spinner('Проверка изображения с помощью Pixtral AI...'):
            is_valid, reason, confidence = validate_mri_image(image_base64)

        if not is_valid:
            # Image is NOT a brain MRI - show error
            st.error(f"""
            **Обнаружено недействительное изображение**

            Это не похоже на МРТ снимок головного мозга.

            **Анализ ИИ:** {reason}

            **Пожалуйста, загрузите:**
            - МРТ снимки головного мозга (аксиальные, сагиттальные или корональные виды)
            - Медицинские изображения в стандартных форматах (JPG, PNG)
            - Четкие, правильно ориентированные снимки

            **Не принимаются:**
            - Фотографии, скриншоты или немедицинские изображения
            - КТ снимки, рентгеновские снимки или другие методы визуализации
            - МРТ снимки других частей тела
            """)
            st.stop()  # Stop execution - don't proceed to prediction
        else:
            # Image validated successfully
            st.success(f"**Изображение проверено:** {reason} (Уверенность: {confidence})")

        # STAGE 2: Processing and prediction with loading animation
        with st.spinner('Анализ МРТ снимка с помощью модели ИИ...'):
            input_image = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(input_image)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                confidence, predicted = torch.max(probabilities, 0)
                predicted_class = class_names[predicted.item()]
                confidence_percent = confidence.item() * 100

            # Generate Grad-CAM visualization
            gradcam_results = generate_gradcam_visualization(
                model, input_image, image, class_names
            )

        # Display prediction with enhanced design
        st.markdown(f"""
            <div class='prediction-box'>
                <h3>Результат диагностики</h3>
                <p style='font-size:2.2rem;margin:1rem 0 0.5rem 0;font-weight:800;
                          background:linear-gradient(135deg,rgba(255,255,255,0.95),rgba(255,255,255,0.6));
                          -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
                    {predicted_class}
                </p>
                <div style='display:inline-flex;align-items:center;gap:8px;margin-top:1rem;
                            background:rgba(34,211,238,0.12);border:1px solid rgba(34,211,238,0.3);
                            border-radius:999px;padding:8px 20px;'>
                    <span style='color:rgba(255,255,255,0.5) !important;font-size:0.85rem;'>Уверенность:</span>
                    <strong style='color:#22d3ee !important;font-size:1rem;'>{confidence_percent:.1f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Display Grad-CAM Visualization
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
            <div style='text-align:center;margin:2rem 0 1.2rem 0;'>
                <h3 style='color:rgba(255,255,255,0.9) !important;font-size:1.2rem !important;font-weight:600 !important;'>
                    Визуализация внимания модели ИИ (Grad-CAM)
                </h3>
                <p style='color:rgba(255,255,255,0.35) !important;font-size:0.9rem;margin-top:0.3rem;'>
                    Тепловая карта показывает, на какие области мозга ИИ обратил внимание при прогнозировании
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Display three images side by side
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<p style='text-align:center;font-size:0.8rem;color:rgba(255,255,255,0.4) !important;margin-bottom:0.5rem;'>Оригинальное МРТ</p>", unsafe_allow_html=True)
            st.image(image, width="stretch")
        with col2:
            st.markdown("<p style='text-align:center;font-size:0.8rem;color:rgba(255,255,255,0.4) !important;margin-bottom:0.5rem;'>Тепловая карта Grad-CAM</p>", unsafe_allow_html=True)
            st.image(gradcam_results['heatmap_only'], width="stretch")
        with col3:
            st.markdown("<p style='text-align:center;font-size:0.8rem;color:rgba(255,255,255,0.4) !important;margin-bottom:0.5rem;'>Комбинированный вид</p>", unsafe_allow_html=True)
            st.image(gradcam_results['overlayed'], width="stretch")

        st.markdown("""
            <div style='background:rgba(59,91,219,0.08);border-left:3px solid rgba(59,91,219,0.5);
                        padding:0.9rem 1.2rem;margin:1rem 0;border-radius:0 10px 10px 0;'>
                <p style='margin:0;color:rgba(255,255,255,0.55) !important;font-size:0.85rem;line-height:1.6;'>
                    <strong style='color:rgba(255,255,255,0.8) !important;'>Как читать:</strong>
                    Красные/жёлтые области — на чём сосредоточился ИИ.
                    Горячие цвета (красный) = высокое внимание, холодные (синий) = низкое.
                </p>
            </div>
        """, unsafe_allow_html=True)

        # ===================================================================
        # STEP-BY-STEP PROGRESSIVE ANALYSIS WORKFLOW
        # ===================================================================

        # Initialize session state for tracking workflow progress
        if 'analysis_step' not in st.session_state:
            st.session_state.analysis_step = 0
        if 'brain_analysis_result' not in st.session_state:
            st.session_state.brain_analysis_result = None

        # Display progress indicator
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
            <div style='text-align:center;margin:2rem 0 1.2rem 0;'>
                <h3 style='color:rgba(255,255,255,0.9) !important;font-size:1.2rem !important;font-weight:600 !important;'>
                    Многоэтапный конвейер анализа ИИ
                </h3>
                <p style='color:rgba(255,255,255,0.35) !important;font-size:0.9rem;margin-top:0.3rem;'>
                    Завершите каждый этап для получения комплексных медицинских заключений
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Progress bar
        progress_steps = ["[Done] Диагностика завершена", "[Pending] Региональный анализ", "[Pending] Финальные рекомендации"]
        if st.session_state.analysis_step >= 1:
            progress_steps[1] = "[Done] Региональный анализ завершен"
        if st.session_state.analysis_step >= 2:
            progress_steps[2] = "[Done] Финальные рекомендации завершены"

        cols = st.columns(3)
        for i, (col, step) in enumerate(zip(cols, progress_steps)):
            with col:
                if "[Done]" in step:
                    st.markdown(f"""
                        <div style='background:linear-gradient(135deg,rgba(59,91,219,0.2),rgba(34,211,238,0.15));
                                    border:1px solid rgba(59,91,219,0.4);padding:0.9rem 1rem;
                                    border-radius:12px;text-align:center;
                                    box-shadow:0 0 16px rgba(59,91,219,0.2);'>
                            <span style='color:rgba(255,255,255,0.9) !important;font-weight:600;font-size:0.85rem;'>
                                ✓ {step.replace('[Done] ', '')}
                            </span>
                        </div>
                    """, unsafe_allow_html=True)
                elif "[Pending]" in step:
                    st.markdown(f"""
                        <div style='background:rgba(255,255,255,0.03);border:1.5px dashed rgba(255,255,255,0.12);
                                    padding:0.9rem 1rem;border-radius:12px;text-align:center;'>
                            <span style='color:rgba(255,255,255,0.3) !important;font-weight:500;font-size:0.85rem;'>
                                {step.replace('[Pending] ', '')}
                            </span>
                        </div>
                    """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ===================================================================
        # STEP 2: DETAILED BRAIN REGION ANALYSIS
        # ===================================================================
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            step2_disabled = st.session_state.analysis_step >= 1
            get_detailed_analysis = st.button(
                "Этап 2: Получить детальный анализ областей мозга",
                width="stretch",
                disabled=step2_disabled,
                type="primary" if not step2_disabled else "secondary"
            )

        if get_detailed_analysis or st.session_state.analysis_step >= 1:
            if get_detailed_analysis:
                with st.spinner("Анализ областей мозга с помощью Pixtral AI... Это может занять 5-10 секунд..."):
                    brain_analysis = analyze_brain_regions(image_base64, predicted_class, confidence_percent)
                    st.session_state.brain_analysis_result = brain_analysis
                    st.session_state.analysis_step = 1

            # Display the analysis
            st.markdown(f"""
                <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.1);
                            border-radius:20px;padding:2.5rem;margin:2rem 0;'>
                    <h2 style='color:rgba(255,255,255,0.92) !important;font-size:1.4rem !important;
                               margin-bottom:1.5rem;text-align:center;font-weight:700 !important;
                               border-bottom:1px solid rgba(255,255,255,0.08);padding-bottom:1rem;'>
                        Региональный анализ областей мозга
                    </h2>
                    <div style='color:rgba(255,255,255,0.65);line-height:1.9;font-size:0.95rem;'>
                        {md_to_html(st.session_state.brain_analysis_result)}
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Important disclaimer
            st.info("""
                **Анализ завершен.** Региональные находки зафиксированы.
                Переходите к Этапу 3 для получения комплексных рекомендаций по лечению.
            """)

        # ===================================================================
        # STEP 3: COMPREHENSIVE MEDICAL RECOMMENDATIONS (Uses ALL Data)
        # ===================================================================

        def get_comprehensive_recommendations(diagnosis, confidence, brain_analysis, gradcam_data):
            """
            Generate comprehensive recommendations using ALL collected data:
            - CNN diagnosis + confidence
            - Grad-CAM attention regions
            - Pixtral regional analysis
            """
            prompt = f"""Вы эксперт-невролог, создающий комплексный план лечения и управления. ОТВЕЧАЙТЕ ПОЛНОСТЬЮ НА РУССКОМ ЯЗЫКЕ.

**ДИАГНОСТИЧЕСКИЕ ДАННЫЕ ПАЦИЕНТА:**

1. **Диагноз модели ИИ:** {diagnosis}
   - Уверенность модели: {confidence:.1f}%
   - Обучена на обширном датасете МРТ болезни Альцгеймера (точность 95.47%)

2. **Области фокуса модели ИИ (Grad-CAM анализ):**
   - CNN модель в основном сосредоточилась на: гиппокампальных областях, желудочковой системе и корковых областях
   - Это области, которые больше всего повлияли на решение классификации ИИ

3. **Детальный региональный анализ мозга (Pixtral AI):**
{brain_analysis}

**ВАША ЗАДАЧА:**
На основе ВСЕХ вышеуказанных данных (диагноз, внимание модели и детальные региональные находки), создайте комплексный, персонализированный план медицинских действий.

**ФОРМАТИРУЙТЕ ВАШ ОТВЕТ ТАК:**

КОМПЛЕКСНЫЙ ПЛАН МЕДИЦИНСКИХ ДЕЙСТВИЙ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

I. НЕМЕДЛЕННЫЕ СЛЕДУЮЩИЕ ШАГИ
• [Срочные действия, требуемые в течение 1-2 недель]
• [Необходимые направления к специалистам]
• [Дополнительные диагностические тесты для назначения]

II. РЕКОМЕНДАЦИИ ПО ЛЕЧЕНИЮ
• [Варианты медикаментов на основе тяжести]
• [Соображения по дозировке]
• [Ожидаемые результаты и мониторинг]

III. КОГНИТИВНЫЕ ВМЕШАТЕЛЬСТВА
• [Программы когнитивных тренировок]
• [Упражнения для памяти]
• [Активности для здоровья мозга]

IV. МОДИФИКАЦИИ ОБРАЗА ЖИЗНИ
• [Диетические рекомендации (средиземноморская диета и т.д.)]
• [Режим упражнений (аэробные + силовые тренировки)]
• [Улучшения гигиены сна]
• [Техники управления стрессом]

V. СОЦИАЛЬНЫЕ МЕРЫ И ПОДДЕРЖКА
• [Обучение ухаживающих и группы поддержки]
• [Активности социального взаимодействия]
• [Планирование безопасности дома]

VI. ПЛАН МОНИТОРИНГА
• [График последующих визуализаций (например, МРТ каждые 6-12 месяцев)]
• [Частота когнитивной оценки]
• [Ключевые биомаркеры для отслеживания]

VII. КОРРЕЛЯЦИЯ С НАХОДКАМИ ИИ
• [Как региональные находки мозга коррелируют с рекомендованным лечением]
• [Почему конкретные вмешательства нацелены на пораженные области]
• [Ожидаемое прогрессирование на основе текущих находок]

VIII. ТРЕВОЖНЫЕ ПРИЗНАКИ ДЛЯ НАБЛЮДЕНИЯ
• [Симптомы, требующие немедленной медицинской помощи]
• [Признаки быстрого прогрессирования]
• [Побочные эффекты медикаментов для мониторинга]

IX. ИССЛЕДОВАНИЯ И КЛИНИЧЕСКИЕ ИСПЫТАНИЯ
• [Релевантные текущие испытания для этой стадии]
• [Развивающиеся терапии для обсуждения с неврологом]

**ВАЖНО:** Будьте конкретны, основывайтесь на доказательствах и цитируйте текущие клинические руководства, где применимо. Адаптируйте рекомендации к тяжести, указанной диагнозом ({diagnosis}).
ОТВЕЧАЙТЕ ПОЛНОСТЬЮ НА РУССКОМ ЯЗЫКЕ.
"""

            try:
                response = _genai_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
                return response.text
            except Exception as e:
                st.error(f"Ошибка генерации рекомендаций: {str(e)}")
                return "Невозможно сгенерировать рекомендации. Пожалуйста, проконсультируйтесь с врачом."

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            step3_disabled = st.session_state.analysis_step < 1
            get_recommendations = st.button(
                "Этап 3: Получить комплексные медицинские рекомендации",
                width="stretch",
                disabled=step3_disabled,
                type="primary" if not step3_disabled else "secondary",
                help="Сначала завершите Этап 2" if step3_disabled else "Сгенерировать финальные рекомендации, используя все данные анализа"
            )

        if get_recommendations:
            with st.spinner("Синтез комплексных медицинских рекомендаций из всех источников данных... Это может занять 10-15 секунд..."):
                recommendations = get_comprehensive_recommendations(
                    diagnosis=predicted_class,
                    confidence=confidence_percent,
                    brain_analysis=st.session_state.brain_analysis_result,
                    gradcam_data=gradcam_results
                )
                st.session_state.analysis_step = 2

            # Display comprehensive disclaimer
            st.warning("""
                **КРИТИЧЕСКОЕ МЕДИЦИНСКОЕ ПРЕДУПРЕЖДЕНИЕ**

                Этот комплексный анализ интегрирует:
                - Диагноз CNN модели (точность 95.47% на тестовых данных)
                - Визуализация внимания Grad-CAM (прозрачность решений ИИ)
                - Региональный анализ мозга Pixtral (радиологическая интерпретация ИИ)
                - Медицинские знания Gemini (рекомендации на основе доказательств)

                **ОДНАКО:**
                - Это НЕ клинический диагноз
                - Это НЕ заменяет невролога, радиолога или врача
                - Это НЕ одобрено FDA для клинических решений
                - ИИ может делать ошибки и может галлюцинировать находки

                **ОБЯЗАТЕЛЬНЫЕ ДЕЙСТВИЯ:**
                - Поделитесь этими результатами с квалифицированным медицинским работником
                - Получите профессиональную радиологическую интерпретацию МРТ
                - Пройдите комплексную неврологическую оценку
                - Следуйте рекомендациям вашего врача, а не только предложениям ИИ

                **Этот инструмент предназначен для ПОМОЩИ медицинским специалистам, а не для их замены.**
            """)

            # Display recommendations with modern design
            st.markdown(f"""
                <div class='recommendations-box'>
                    <h2 style='text-align:center;margin-bottom:1.5rem;'>
                        Комплексный план медицинских действий
                    </h2>
                    <div style='background:linear-gradient(135deg,rgba(59,91,219,0.12),rgba(34,211,238,0.08));
                                border:1px solid rgba(59,91,219,0.25);padding:1rem 1.2rem;
                                border-radius:12px;margin-bottom:2rem;'>
                        <p style='margin:0;color:rgba(255,255,255,0.55) !important;font-size:0.85rem;line-height:1.6;'>
                            <strong style='color:rgba(255,255,255,0.8) !important;'>Источники данных:</strong>
                            CNN диагноз ({predicted_class}, {confidence_percent:.1f}%) · Grad-CAM · Региональный анализ · Медицинская литература
                        </p>
                    </div>
                    <div style='line-height:1.9;color:rgba(255,255,255,0.65);font-size:0.95rem;'>
                        {md_to_html(recommendations)}
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Final summary
            st.success("""
                **Анализ завершен.** Все три этапа анализа ИИ завершены.

                Теперь у вас есть:
                1. Начальный диагноз CNN с оценкой уверенности
                2. Визуализация Grad-CAM, показывающая процесс принятия решения ИИ
                3. Детальный анализ областей мозга от Pixtral AI
                4. Комплексные медицинские рекомендации от Gemini AI

                **Следующие шаги:** Распечатайте или сохраните этот отчет.
            """)
    else:
        # Show helpful instructions when no image is uploaded
        st.markdown("""
            <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
                        border-radius:20px;padding:3rem;margin:2rem auto;max-width:600px;text-align:center;'>
                <div style='font-size:2.5rem;margin-bottom:1rem;'>🧠</div>
                <h3 style='color:rgba(255,255,255,0.9) !important;font-size:1.2rem !important;
                           font-weight:600 !important;margin-bottom:0.8rem !important;'>
                    Готов к анализу
                </h3>
                <p style='color:rgba(255,255,255,0.4) !important;font-size:0.95rem;line-height:1.7;'>
                    Загрузите МРТ снимок выше, чтобы начать анализ болезни Альцгеймера с помощью ИИ.
                    Модель классифицирует стадию и предоставит детальные выводы.
                </p>
                <div style='margin-top:1.8rem;padding:1rem 1.5rem;
                            background:rgba(59,91,219,0.08);border:1px solid rgba(59,91,219,0.2);
                            border-radius:12px;display:inline-block;'>
                    <p style='margin:0;color:rgba(255,255,255,0.45) !important;font-size:0.85rem;line-height:1.8;'>
                        <strong style='color:rgba(255,255,255,0.7) !important;'>Поддерживаемые форматы:</strong> JPG, JPEG, PNG<br>
                        <strong style='color:rgba(255,255,255,0.7) !important;'>Точность модели:</strong>
                        <span style='color:#22d3ee !important;font-weight:600;'> 95.47%</span>
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)

elif options == 'Виртуальный ассистент':
    exec(open("chatbot.py", encoding="utf-8").read())

# Footer
st.markdown("""
    <div style='margin-top:4rem;border-top:1px solid rgba(255,255,255,0.07);padding:2rem 1rem;text-align:center;'>
        <div style='max-width:800px;margin:0 auto;'>
            <p style='font-size:1rem;font-weight:700;color:rgba(255,255,255,0.85) !important;margin-bottom:0.5rem;'>
                ReMind.AI
            </p>
            <p style='font-size:0.85rem;color:rgba(255,255,255,0.35) !important;line-height:1.7;margin-bottom:1.2rem;'>
                Обнаружение болезни Альцгеймера с помощью ИИ.<br>
                Архитектура PyTorch TinyVGG16 · Точность 95.47%
            </p>
            <div style='display:inline-block;padding:0.75rem 1.2rem;background:rgba(251,191,36,0.08);
                        border:1px solid rgba(251,191,36,0.25);border-radius:10px;margin-bottom:1.5rem;'>
                <p style='margin:0;font-size:0.8rem;color:rgba(255,255,255,0.5) !important;line-height:1.6;'>
                    <strong style='color:rgba(251,191,36,0.9) !important;'>⚠ Медицинское предупреждение:</strong>
                    Результаты должны быть интерпретированы квалифицированными специалистами.
                    Инструмент предназначен для помощи, а не для замены врача.
                </p>
            </div>
            <p style='font-size:0.75rem;color:rgba(255,255,255,0.2) !important;'>
                © 2025 ReMind.AI · На базе Gemini и Streamlit
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
