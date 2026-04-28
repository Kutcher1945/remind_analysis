import streamlit as st
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

st.markdown("""
    <style>
    /* Chat bubbles */
    .message-user {
        background: linear-gradient(135deg, rgba(59,91,219,0.25), rgba(34,211,238,0.18));
        border: 1px solid rgba(59,91,219,0.35);
        color: rgba(255,255,255,0.92) !important;
        padding: 0.9rem 1.25rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.75rem 0;
        margin-left: 20%;
        animation: slideInRight 0.3s ease;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    .message-assistant {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.09);
        border-left: 3px solid rgba(59,91,219,0.5);
        color: rgba(255,255,255,0.82) !important;
        padding: 0.9rem 1.25rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.75rem 0;
        margin-right: 20%;
        animation: slideInLeft 0.3s ease;
        line-height: 1.7;
        font-size: 0.95rem;
    }
    @keyframes slideInRight { from { opacity:0; transform:translateX(16px) } to { opacity:1; transform:translateX(0) } }
    @keyframes slideInLeft  { from { opacity:0; transform:translateX(-16px) } to { opacity:1; transform:translateX(0) } }

    /* Chat input override */
    .stChatInputContainer { border-top: 1px solid rgba(255,255,255,0.07) !important; padding-top: 0.75rem; }
    [data-testid="stChatInput"] textarea {
        background: rgba(255,255,255,0.05) !important;
        border: 1.5px solid rgba(255,255,255,0.1) !important;
        border-radius: 12px !important;
        color: rgba(255,255,255,0.85) !important;
        -webkit-text-fill-color: rgba(255,255,255,0.85) !important;
    }
    [data-testid="stChatInput"] textarea::placeholder { color: rgba(255,255,255,0.3) !important; }
    </style>
""", unsafe_allow_html=True)

MEDICAL_TEMPLATE = """Вы медицинский ассистент, специализирующийся на болезни Альцгеймера и нейродегенеративных заболеваниях.
ОТВЕЧАЙТЕ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ.
Предоставляйте точные, основанные на доказательствах и легко понятные ответы.
Используйте ясный и доступный язык. Поддерживайте профессиональный, но эмпатичный тон.
ВАЖНО: ВСЕ ОТВЕТЫ ДОЛЖНЫ БЫТЬ ПОЛНОСТЬЮ НА РУССКОМ ЯЗЫКЕ.
Вопрос пользователя: {question}"""

GEMINI_MODEL = 'gemini-3.1-flash-lite-preview'

def init_gemini():
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        st.error('API ключ Gemini не настроен.')
        st.stop()
    return genai.Client(api_key=api_key)

def get_gemini_response(client, question):
    try:
        prompt = MEDICAL_TEMPLATE.format(question=question)
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return response.text
    except Exception as e:
        return f"Ошибка генерации ответа: {str(e)}"

# Header
st.markdown("""
    <div style='text-align:center;padding:2rem 0 2.5rem 0;'>
        <div style='display:inline-flex;align-items:center;gap:8px;background:rgba(59,91,219,0.1);
                    border:1px solid rgba(59,91,219,0.3);border-radius:999px;padding:6px 16px;margin-bottom:1.2rem;'>
            <div style='width:7px;height:7px;border-radius:50%;background:#22d3ee;'></div>
            <span style='font-size:0.72rem;font-weight:600;letter-spacing:0.18em;
                         text-transform:uppercase;color:#22d3ee !important;'>Виртуальный ассистент</span>
        </div>
        <h2 style='color:rgba(255,255,255,0.95) !important;font-size:2rem !important;
                   margin-bottom:0.6rem !important;'>Медицинский ассистент ИИ</h2>
        <p style='color:rgba(255,255,255,0.35) !important;font-size:0.9rem;'>
            Специализированный консультант по болезни Альцгеймера · Gemini 3.1 Flash Lite
        </p>
    </div>
""", unsafe_allow_html=True)

if not os.getenv('GEMINI_API_KEY'):
    st.error('API ключ Gemini не настроен.')
    st.stop()

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'model' not in st.session_state:
    st.session_state.model = init_gemini()

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    for message in st.session_state.messages:
        css_class = "message-user" if message["role"] == "user" else "message-assistant"
        st.markdown(f"<div class='{css_class}'>{message['content']}</div>", unsafe_allow_html=True)

    if prompt := st.chat_input("Введите вопрос о болезни Альцгеймера..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f"<div class='message-user'>{prompt}</div>", unsafe_allow_html=True)
        with st.spinner('Генерация ответа...'):
            response = get_gemini_response(st.session_state.model, prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(f"<div class='message-assistant'>{response}</div>", unsafe_allow_html=True)

with col2:
    # About card
    st.markdown("""
        <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
                    border-radius:16px;padding:1.4rem;margin-bottom:1rem;'>
            <h4 style='color:rgba(255,255,255,0.85) !important;margin-bottom:0.8rem;
                       font-size:0.95rem !important;font-weight:600 !important;'>Об ассистенте</h4>
            <p style='color:rgba(255,255,255,0.4) !important;line-height:1.65;margin:0;font-size:0.85rem;'>
                Предоставляет информацию на основе доказательств о болезни Альцгеймера
                и нейродегенеративных заболеваниях.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Topics card
    topics = ["Симптомы и диагностика", "Варианты лечения", "Факторы риска",
              "Стратегии профилактики", "Поддержка ухаживающих",
              "Последние исследования", "Медицинские тесты", "Медицинские ресурсы"]
    items_html = "".join(
        f"<div style='padding:0.5rem 0;border-bottom:1px solid rgba(255,255,255,0.06);font-size:0.85rem;color:rgba(255,255,255,0.6) !important;'>{t}</div>"
        for t in topics
    )
    st.markdown(f"""
        <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
                    border-radius:16px;padding:1.4rem;margin-bottom:1rem;'>
            <h4 style='color:rgba(255,255,255,0.85) !important;margin-bottom:0.9rem;
                       font-size:0.95rem !important;font-weight:600 !important;'>Предлагаемые темы</h4>
            {items_html}
        </div>
    """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
        <div style='background:rgba(251,191,36,0.07);border:1px solid rgba(251,191,36,0.22);
                    border-radius:12px;padding:1rem 1.1rem;margin-bottom:1rem;'>
            <p style='margin:0;font-size:0.8rem;color:rgba(255,255,255,0.45) !important;line-height:1.65;'>
                <strong style='color:rgba(251,191,36,0.85) !important;'>⚠ Предупреждение:</strong>
                Этот ИИ предоставляет общую информацию. Он не заменяет профессиональную медицинскую консультацию.
            </p>
        </div>
    """, unsafe_allow_html=True)

    if st.button('Очистить беседу', width="stretch"):
        st.session_state.messages = []
        st.rerun()
