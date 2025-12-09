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
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem 1.25rem;
            border-radius: 18px 18px 4px 18px;
            margin: 0.75rem 0;
            margin-left: 20%;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
            animation: slideInRight 0.3s ease;
        }

        /* Assistant messages */
        .message-assistant {
            background: #f7fafc;
            color: #2d3748;
            padding: 1rem 1.25rem;
            border-radius: 18px 18px 18px 4px;
            margin: 0.75rem 0;
            margin-right: 20%;
            border-left: 4px solid #667eea;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            animation: slideInLeft 0.3s ease;
            line-height: 1.6;
        }

        /* Info card styling */
        .info-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.25);
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
            border-top: 2px solid #e2e8f0;
            padding-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Medical instructions template for the model
MEDICAL_TEMPLATE = """You are a medical assistant specialized in Alzheimer's and neurodegenerative diseases.
Provide accurate, evidence-based, and easy-to-understand answers.
Your goal is to help people better understand Alzheimer's and provide useful information.

Remember:
1. Use clear and accessible language
2. Base answers on scientific evidence
3. Include references to studies when relevant
4. Maintain a professional yet empathetic tone

User question: {question}"""

def init_gemini():
    """
    Initialize the Gemini client
    """
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        st.error('⚠️ Gemini API key has not been configured. Please set it as an environment variable.')
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
        return f"Error generating response: {str(e)}"

# Enhanced title and description
st.markdown(
     """
        <div style='text-align: center; padding: 1rem 0 2rem 0;'>
            <h2 style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       margin-bottom: 0.5rem;'>
                🤖 AI Medical Assistant
            </h2>
            <h4 style='color: #718096; font-weight: 400;'>
                Specialized Alzheimer's advisor powered by Google Gemini 2.5 Flash
            </h4>
        </div>
    """,
    unsafe_allow_html=True,
)

# API Key verification
if not os.getenv('GEMINI_API_KEY'):
    st.error('⚠️ Gemini API key has not been configured. Please set it as an environment variable.')
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
    if prompt := st.chat_input("Type your question about Alzheimer's here"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f"<div class='message-user'>{prompt}</div>", unsafe_allow_html=True)

        # Get response from model
        with st.spinner('Generating response...'):
            response = get_gemini_response(st.session_state.model, prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(f"<div class='message-assistant'>{response}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    # Info card with modern design
    st.markdown("""
        <div style='background: white; border-radius: 16px; padding: 1.5rem; box-shadow: 0 4px 20px rgba(0,0,0,0.06); margin-bottom: 1rem;'>
            <h4 style='color: #2d3748; margin-bottom: 1rem; font-size: 1.1rem;'>ℹ️ About This Assistant</h4>
            <p style='color: #4a5568; line-height: 1.6; margin: 0;'>
                Powered by Google's Gemini 2.5 Flash, this AI assistant provides evidence-based information
                about Alzheimer's disease and related neurodegenerative conditions.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Topics card
    st.markdown("""
        <div class='info-card'>
            <h4>💡 Suggested Topics</h4>
            <ul style='margin: 0;'>
                <li>🧠 Symptoms & Diagnosis</li>
                <li>💊 Treatment Options</li>
                <li>⚠️ Risk Factors</li>
                <li>🛡️ Prevention Strategies</li>
                <li>💝 Caregiver Support</li>
                <li>🔬 Latest Research</li>
                <li>📋 Medical Tests</li>
                <li>🏥 Healthcare Resources</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Disclaimer
    st.info(
        "⚠️ **Medical Disclaimer:** This AI provides general information and educational content. "
        "It does not replace professional medical advice. Always consult qualified healthcare professionals.",
        icon="⚠️"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Clear button with modern styling
    if st.button('🔄 Clear Conversation', use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.rerun()
