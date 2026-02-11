import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Enhanced custom styles
st.markdown(
    """
    <style>
        /* Chat container */
        .chat-container {
            background: white;
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.06);
            margin-bottom: 1rem;
        }

        /* User messages */
        .message-user {
            background: #000000;
            color: white;
            padding: 1rem 1.25rem;
            border-radius: 18px 18px 4px 18px;
            margin: 0.75rem 0;
            margin-left: 20%;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            animation: slideInRight 0.3s ease;
        }

        /* Assistant messages */
        .message-assistant {
            background: #f5f5f5;
            color: #000000;
            padding: 1rem 1.25rem;
            border-radius: 18px 18px 18px 4px;
            margin: 0.75rem 0;
            margin-right: 20%;
            border-left: 4px solid #000000;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            animation: slideInLeft 0.3s ease;
            line-height: 1.6;
        }

        /* Info card styling */
        .info-card {
            background: #000000;
            color: white;
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        }

        .info-card h4 {
            color: white !important;
            margin-bottom: 1rem;
            font-size: 1.2rem;
        }

        .info-card ul {
            list-style: none;
            padding: 0;
        }

        .info-card li {
            padding: 0.5rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.2);
        }

        .info-card li:last-child {
            border-bottom: none;
        }

        /* Animations */
        @keyframes slideInRight {
            from {
                opacity: 0;
                transform: translateX(20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        @keyframes slideInLeft {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        /* Chat input styling */
        .stChatInputContainer {
            border-top: 2px solid #cccccc;
            padding-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Medical instructions template for the model
MEDICAL_TEMPLATE = """Вы медицинский ассистент, специализирующийся на болезни Альцгеймера и нейродегенеративных заболеваниях.
ОТВЕЧАЙТЕ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ.
Предоставляйте точные, основанные на доказательствах и легко понятные ответы.
Ваша цель - помочь людям лучше понять болезнь Альцгеймера и предоставить полезную информацию.

Помните:
1. Используйте ясный и доступный язык
2. Основывайте ответы на научных доказательствах
3. Включайте ссылки на исследования, где это уместно
4. Поддерживайте профессиональный, но эмпатичный тон

ВАЖНО: ВСЕ ОТВЕТЫ ДОЛЖНЫ БЫТЬ ПОЛНОСТЬЮ НА РУССКОМ ЯЗЫКЕ.

Вопрос пользователя: {question}"""

def init_gemini():
    """
    Initialize the Gemini client
    """
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        st.error('API ключ Gemini не настроен. Пожалуйста, установите его как переменную окружения.')
        st.stop()

    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash')

def get_gemini_response(model, question):
    """
    Get a response from the Gemini model for a question about Alzheimer's
    """
    try:
        prompt = MEDICAL_TEMPLATE.format(question=question)
        response = model.generate_content([prompt, question])
        return response.text
    except Exception as e:
        return f"Ошибка генерации ответа: {str(e)}"

# Enhanced title and description
st.markdown(
     """
        <div style='text-align: center; padding: 1rem 0 2rem 0;'>
            <h2 style='color: #000000; margin-bottom: 0.5rem;'>
                Медицинский ассистент ИИ
            </h2>
            <h4 style='color: #555555; font-weight: 400;'>
                Специализированный консультант по болезни Альцгеймера на базе Google Gemini 2.5 Flash
            </h4>
        </div>
    """,
    unsafe_allow_html=True,
)

# API Key verification
if not os.getenv('GEMINI_API_KEY'):
    st.error('API ключ Gemini не настроен. Пожалуйста, установите его как переменную окружения.')
    st.stop()

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Initialize Gemini client
if 'model' not in st.session_state:
    st.session_state.model = init_gemini()

# Application layout
col1, col2 = st.columns([2, 1], gap="large")

with col1:

    # st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    # Display chat history
    for message in st.session_state.messages:
        css_class = "message-user" if message["role"] == "user" else "message-assistant"
        st.markdown(f"<div class='{css_class}'>{message['content']}</div>", unsafe_allow_html=True)

    # User input
    if prompt := st.chat_input("Введите ваш вопрос о болезни Альцгеймера здесь"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f"<div class='message-user'>{prompt}</div>", unsafe_allow_html=True)

        # Get response from model
        with st.spinner('Генерация ответа...'):
            response = get_gemini_response(st.session_state.model, prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(f"<div class='message-assistant'>{response}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    # Info card with modern design
    st.markdown("""
        <div style='background: white; border-radius: 16px; padding: 1.5rem; box-shadow: 0 4px 20px rgba(0,0,0,0.06); margin-bottom: 1rem; border: 1px solid #e0e0e0;'>
            <h4 style='color: #000000; margin-bottom: 1rem; font-size: 1.1rem;'>Об этом ассистенте</h4>
            <p style='color: #333333; line-height: 1.6; margin: 0;'>
                На базе Google Gemini 2.5 Flash, этот ИИ-ассистент предоставляет информацию на основе доказательств
                о болезни Альцгеймера и связанных нейродегенеративных заболеваниях.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Topics card
    st.markdown("""
        <div class='info-card'>
            <h4>Предлагаемые темы</h4>
            <ul style='margin: 0;'>
                <li>Симптомы и диагностика</li>
                <li>Варианты лечения</li>
                <li>Факторы риска</li>
                <li>Стратегии профилактики</li>
                <li>Поддержка ухаживающих</li>
                <li>Последние исследования</li>
                <li>Медицинские тесты</li>
                <li>Медицинские ресурсы</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Disclaimer
    st.info(
        "**Медицинское предупреждение:** Этот ИИ предоставляет общую информацию и образовательный контент. "
        "Он не заменяет профессиональную медицинскую консультацию. Всегда консультируйтесь с квалифицированными медицинскими специалистами."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Clear button with modern styling
    if st.button('Очистить беседу', use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.rerun()
