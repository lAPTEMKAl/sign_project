"""
demo/app.py
Демонстрационный модуль (Streamlit). Показывает: исходное изображение,
результат работы модели (класс знака), уверенность предсказания, top-3,
и краткую статистику по истории запусков.

Запуск:
    streamlit run demo/app.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import streamlit as st
from PIL import Image

from inference import DB_PATH, Predictor, init_db, log_run

st.set_page_config(page_title="Распознавание дорожных знаков", layout="centered")
st.title("🚦 Распознавание дорожных знаков (RTSD)")
st.caption("Демонстрационный модуль — лучшая модель **ResNet18** из сравнения 5 архитектур")

MODEL_PATH = "models/best_model.pt"


@st.cache_resource
def load_predictor():
    return Predictor(MODEL_PATH)


@st.cache_resource
def get_db_conn():
    return init_db(DB_PATH)


try:
    predictor = load_predictor()
except FileNotFoundError:
    st.error(
        f"Не найден чекпойнт {MODEL_PATH}. Сначала запустите обучение: "
        "`python src/run_all.py --data data/processed`"
    )
    st.stop()

conn = get_db_conn()

uploaded_file = st.file_uploader("Загрузите изображение дорожного знака", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    with col1:
        st.subheader("Исходное изображение")
        st.image(image, use_container_width=True)

    result = predictor.predict(image)
    log_run(conn, uploaded_file.name, predictor.arch, result["predicted_class"],
            result["confidence"], result["inference_ms"])

    with col2:
        st.subheader("Результат")
        st.metric("Предсказанный класс", result["predicted_class"])
        st.metric("Уверенность", f"{result['confidence']*100:.1f}%")
        st.metric("Время инференса", f"{result['inference_ms']:.1f} мс")
        st.progress(min(result["confidence"], 1.0))

        st.write("**Топ-3 варианта:**")
        for cls, prob in result["topk"]:
            st.write(f"- {cls}: {prob*100:.1f}%")

        if result["confidence"] < 0.5:
            st.warning("Низкая уверенность модели — изображение может быть мелким, "
                       "размытым или плохо освещённым.")

st.divider()

# --- Статистика + кнопка очистки ---
header_col, btn_col = st.columns([3, 1])
with header_col:
    st.subheader("📊 Статистика по истории запусков")
with btn_col:
    if st.button("🗑️ Очистить историю", type="secondary"):
        conn.execute("DELETE FROM history")
        conn.commit()
        st.rerun()

history_df = pd.read_sql("SELECT * FROM history ORDER BY id DESC", conn)
if len(history_df) == 0:
    st.info("История пуста — загрузите изображение, чтобы появилась статистика.")
else:
    st.write(f"Всего запросов: **{len(history_df)}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Средняя уверенность", f"{history_df['confidence'].mean()*100:.1f}%")
    c2.metric("Среднее время, мс", f"{history_df['inference_ms'].mean():.1f}")
    c3.metric("Уникальных классов", history_df["predicted_class"].nunique())

    st.write("**Последние запросы:**")
    st.dataframe(history_df.head(20), use_container_width=True)

    st.write("**Частота предсказанных классов:**")
    st.bar_chart(history_df["predicted_class"].value_counts())
