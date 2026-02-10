import streamlit as st
import io

# Enhanced header
st.markdown("""
    <div style='text-align: center; padding: 1rem 0 2rem 0;'>
        <h2 style='background: linear-gradient(135deg, #FF6B35 0%, #E63946 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   margin-bottom: 0.5rem;'>
            📋 Информация о пациенте
        </h2>
        <h4 style='color: #718096; font-weight: 400;'>
            Заполните медицинский профиль пациента для комплексной диагностики
        </h4>
    </div>
""", unsafe_allow_html=True)

# Form card with modern design
st.markdown("""
    <div style='background: white; border-radius: 20px; padding: 2rem; box-shadow: 0 10px 40px rgba(0,0,0,0.08); margin-bottom: 2rem;'>
        <h3 style='color: #2d3748; margin-bottom: 1.5rem; font-size: 1.3rem;'>
            👤 Личная информация
        </h3>
    </div>
""", unsafe_allow_html=True)

# Form to enter patient data with better labels
col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown("<p style='color: #2d3748; font-weight: 600; font-size: 1.1rem; margin-bottom: 1rem;'>👤 Основная информация</p>", unsafe_allow_html=True)
    name = st.text_input("Полное имя", placeholder="Введите полное имя пациента")
    age = st.number_input("Возраст (лет)", min_value=0, max_value=120, value=60, help="Возраст пациента в годах")
    gender = st.selectbox("Пол", ["Мужской", "Женский", "Другой", "Предпочитаю не указывать"])
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p style='color: #2d3748; font-weight: 600; font-size: 1.1rem; margin-bottom: 1rem;'>📝 Медицинская история</p>", unsafe_allow_html=True)
    medical_history = st.text_area(
        "Медицинская история",
        placeholder="Предыдущие диагнозы, операции, хронические заболевания, медикаменты...",
        height=150,
        help="Включите соответствующую медицинскую историю, текущие медикаменты и известные заболевания"
    )

with col2:
    st.markdown("<p style='color: #2d3748; font-weight: 600; font-size: 1.1rem; margin-bottom: 1rem;'>📊 Физические измерения</p>", unsafe_allow_html=True)
    weight = st.number_input("Вес (кг)", min_value=0.0, max_value=300.0, value=70.0, step=0.1)
    height = st.number_input("Рост (см)", min_value=0.0, max_value=250.0, value=170.0, step=0.1)

    # Calculate and display BMI with color coding
    bmi = weight / ((height / 100) ** 2) if height > 0 else 0

    # BMI categories
    if bmi < 18.5:
        bmi_category = "Недостаточный вес"
        bmi_color = "#3b82f6"
    elif 18.5 <= bmi < 25:
        bmi_category = "Нормальный"
        bmi_color = "#10b981"
    elif 25 <= bmi < 30:
        bmi_category = "Избыточный вес"
        bmi_color = "#f59e0b"
    else:
        bmi_category = "Ожирение"
        bmi_color = "#ef4444"

    st.markdown(f"""
        <div style='background: linear-gradient(135deg, {bmi_color}15 0%, {bmi_color}05 100%);
                    border-left: 4px solid {bmi_color};
                    border-radius: 12px;
                    padding: 1.5rem;
                    margin-top: 1.5rem;'>
            <h4 style='color: #2d3748; margin: 0 0 0.5rem 0; font-size: 0.9rem; text-transform: uppercase;'>
                Индекс массы тела
            </h4>
            <p style='font-size: 2rem; font-weight: 700; color: {bmi_color}; margin: 0;'>
                {bmi:.1f}
            </p>
            <p style='margin: 0.5rem 0 0 0; color: #4a5568; font-size: 0.9rem;'>
                Категория: <strong>{bmi_category}</strong>
            </p>
        </div>
    """, unsafe_allow_html=True)

# Migrate old Spanish session state keys to English (for backwards compatibility)
if "reporte_generado" in st.session_state:
    st.session_state.report_generated = st.session_state.reporte_generado
    del st.session_state.reporte_generado

# Initialize report state if it doesn't exist
if "report_generated" not in st.session_state:
    st.session_state.report_generated = False
    st.session_state.report_content = ""

# Spacer
st.markdown("<br>", unsafe_allow_html=True)

# Generate report button with modern styling
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("📄 Сгенерировать отчет пациента", use_container_width=True, type="primary"):
        st.session_state.report_content = f"""
### 📋 Медицинский отчет пациента

**Личная информация:**
- **Полное имя:** {name}
- **Возраст:** {age} лет
- **Пол:** {gender}

**Физические измерения:**
- **Вес:** {weight} кг
- **Рост:** {height} см
- **ИМТ:** {bmi:.1f} ({bmi_category})

**Медицинская история:**
{medical_history if medical_history else "Медицинская история не предоставлена"}

---
*Отчет сгенерирован: {st.session_state.get('report_timestamp', 'Н/Д')}*
*ReMind.AI - Система медицинского анализа*
"""
        st.session_state.report_generated = True
        # Store timestamp
        import datetime
        st.session_state.report_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Display generated report with modern design
if st.session_state.report_generated:
    st.markdown("<br>", unsafe_allow_html=True)

    # Report display card
    st.markdown("""
        <div style='background: white; border-radius: 20px; padding: 2.5rem; box-shadow: 0 10px 40px rgba(0,0,0,0.08); margin: 2rem 0;'>
            <h3 style='color: #2d3748; margin-bottom: 1.5rem; text-align: center;'>
                ✅ Отчет пациента сгенерирован
            </h3>
        </div>
    """, unsafe_allow_html=True)

    # Display report content - convert markdown to styled HTML
    import re

    # Function to convert markdown to HTML
    def markdown_to_html(text):
        # Convert headers
        text = re.sub(r'### (.+)', r'<h3 style="background: linear-gradient(135deg, #FF6B35 0%, #E63946 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 1.3rem; margin: 1rem 0;">\1</h3>', text)
        # Convert bold text
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong style="color: #2d3748;">\1</strong>', text)
        # Convert bullet points
        text = re.sub(r'^- (.+)$', r'<div style="margin-left: 1rem;">• \1</div>', text, flags=re.MULTILINE)
        # Convert line breaks
        text = text.replace('\n', '<br>')
        # Convert horizontal rules
        text = text.replace('---', '<hr style="border: none; border-top: 1px solid #e2e8f0; margin: 1.5rem 0;">')
        return text

    html_content = markdown_to_html(st.session_state.report_content)

    st.markdown(f"""
        <div style='background: #FFF1F2; border-radius: 16px; padding: 2rem; border-left: 4px solid #FF6B35; color: #2d3748; line-height: 1.8; font-size: 1rem;'>
            {html_content}
        </div>
    """, unsafe_allow_html=True)

    # Download button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        report_bytes = st.session_state.report_content.encode("utf-8")
        st.download_button(
            label="⬇️ Скачать отчет как TXT",
            data=report_bytes,
            file_name=f"otchet_pacienta_{name.replace(' ', '_')}_{st.session_state.get('report_timestamp', '').split()[0]}.txt",
            mime="text/plain",
            use_container_width=True
        )

# Next steps info box
st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
    <div style='background: linear-gradient(135deg, #FF6B35 0%, #E63946 100%);
                border-radius: 20px;
                padding: 2rem;
                text-align: center;
                color: white;
                box-shadow: 0 10px 30px rgba(255, 107, 53, 0.4);'>
        <h3 style='color: white; margin-bottom: 1rem;'>🔬 Готовы к диагностике?</h3>
        <p style='margin: 0; font-size: 1.05rem; opacity: 0.95;'>
            После заполнения информации о пациенте, перейдите в раздел <strong>Диагностика</strong>
            для загрузки МРТ снимков и получения анализа на основе ИИ.
        </p>
    </div>
""", unsafe_allow_html=True)