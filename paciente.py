import streamlit as st
import io
import re
import datetime

# Header
st.markdown("""
    <div style='text-align:center;padding:2rem 0 2.5rem 0;'>
        <div style='display:inline-flex;align-items:center;gap:8px;background:rgba(59,91,219,0.1);
                    border:1px solid rgba(59,91,219,0.3);border-radius:999px;padding:6px 16px;margin-bottom:1.2rem;'>
            <div style='width:7px;height:7px;border-radius:50%;background:#22d3ee;'></div>
            <span style='font-size:0.72rem;font-weight:600;letter-spacing:0.18em;
                         text-transform:uppercase;color:#22d3ee !important;'>Данные пациента</span>
        </div>
        <h2 style='color:rgba(255,255,255,0.95) !important;font-size:2rem !important;margin-bottom:0.6rem !important;'>
            Информация о пациенте
        </h2>
        <p style='color:rgba(255,255,255,0.35) !important;font-size:0.9rem;'>
            Заполните медицинский профиль пациента для комплексной диагностики
        </p>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("<p style='color:rgba(255,255,255,0.7) !important;font-weight:600;font-size:0.95rem;margin-bottom:0.8rem;'>Основная информация</p>", unsafe_allow_html=True)
    name = st.text_input("Полное имя", placeholder="Введите полное имя пациента")
    age = st.number_input("Возраст (лет)", min_value=0, max_value=120, value=60)
    gender = st.selectbox("Пол", ["Мужской", "Женский", "Другой", "Предпочитаю не указывать"])
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p style='color:rgba(255,255,255,0.7) !important;font-weight:600;font-size:0.95rem;margin-bottom:0.8rem;'>Медицинская история</p>", unsafe_allow_html=True)
    medical_history = st.text_area(
        "Медицинская история",
        placeholder="Предыдущие диагнозы, операции, хронические заболевания, медикаменты...",
        height=150,
    )

with col2:
    st.markdown("<p style='color:rgba(255,255,255,0.7) !important;font-weight:600;font-size:0.95rem;margin-bottom:0.8rem;'>Физические измерения</p>", unsafe_allow_html=True)
    weight = st.number_input("Вес (кг)", min_value=0.0, max_value=300.0, value=70.0, step=0.1)
    height = st.number_input("Рост (см)", min_value=0.0, max_value=250.0, value=170.0, step=0.1)

    bmi = weight / ((height / 100) ** 2) if height > 0 else 0

    if bmi < 18.5:
        bmi_category, bmi_color = "Недостаточный вес", "#60a5fa"
    elif bmi < 25:
        bmi_category, bmi_color = "Нормальный", "#4ade80"
    elif bmi < 30:
        bmi_category, bmi_color = "Избыточный вес", "#fbbf24"
    else:
        bmi_category, bmi_color = "Ожирение", "#f87171"

    st.markdown(f"""
        <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
                    border-left:3px solid {bmi_color};border-radius:12px;padding:1.4rem;margin-top:1.2rem;'>
            <p style='color:rgba(255,255,255,0.35) !important;font-size:0.75rem;text-transform:uppercase;
                      letter-spacing:0.12em;margin:0 0 0.4rem 0;'>Индекс массы тела</p>
            <p style='font-size:2.2rem;font-weight:800;color:{bmi_color} !important;margin:0;line-height:1;'>
                {bmi:.1f}
            </p>
            <p style='margin:0.5rem 0 0 0;color:rgba(255,255,255,0.45) !important;font-size:0.85rem;'>
                Категория: <strong style='color:{bmi_color} !important;'>{bmi_category}</strong>
            </p>
        </div>
    """, unsafe_allow_html=True)

if "report_generated" not in st.session_state:
    st.session_state.report_generated = False
    st.session_state.report_content = ""

st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("Сгенерировать отчет пациента", width="stretch", type="primary"):
        st.session_state.report_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.report_content = f"""
### Медицинский отчет пациента

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
*Отчет сгенерирован: {st.session_state.report_timestamp}*
*ReMind.AI — Система медицинского анализа*
"""
        st.session_state.report_generated = True

if st.session_state.report_generated:
    st.markdown("<br>", unsafe_allow_html=True)

    def markdown_to_html(text):
        text = re.sub(r'### (.+)', r'<h3 style="color:rgba(255,255,255,0.9);font-size:1.1rem;margin:1rem 0;">\1</h3>', text)
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong style="color:rgba(255,255,255,0.85);">\1</strong>', text)
        text = re.sub(r'^- (.+)$', r'<div style="margin-left:1rem;padding:0.2rem 0;color:rgba(255,255,255,0.6);">• \1</div>', text, flags=re.MULTILINE)
        text = text.replace('\n', '<br>')
        text = text.replace('---', '<hr style="border:none;border-top:1px solid rgba(255,255,255,0.08);margin:1.2rem 0;">')
        return text

    st.markdown(f"""
        <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
                    border-left:3px solid rgba(59,91,219,0.5);border-radius:16px;
                    padding:2rem;color:rgba(255,255,255,0.65);line-height:1.8;font-size:0.95rem;'>
            {markdown_to_html(st.session_state.report_content)}
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.download_button(
            label="Скачать отчет как TXT",
            data=st.session_state.report_content.encode("utf-8"),
            file_name=f"otchet_{name.replace(' ', '_')}_{st.session_state.get('report_timestamp', '').split()[0]}.txt",
            mime="text/plain",
            width="stretch"
        )

# Next steps
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style='background:linear-gradient(135deg,rgba(59,91,219,0.15),rgba(34,211,238,0.1));
                border:1px solid rgba(59,91,219,0.3);border-radius:20px;padding:2rem;text-align:center;
                box-shadow:0 0 40px rgba(59,91,219,0.15);'>
        <h3 style='color:rgba(255,255,255,0.92) !important;margin-bottom:0.8rem;font-size:1.2rem !important;'>
            Готовы к диагностике?
        </h3>
        <p style='margin:0;color:rgba(255,255,255,0.45) !important;font-size:0.9rem;line-height:1.7;'>
            После заполнения информации о пациенте перейдите в раздел
            <strong style='color:rgba(255,255,255,0.75) !important;'>Диагностика</strong>
            для загрузки МРТ снимков и получения анализа на основе ИИ.
        </p>
    </div>
""", unsafe_allow_html=True)
