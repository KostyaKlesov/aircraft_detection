import streamlit as st
import os
from model import detect_airplanes, detect_airplanes_in_video
from PIL import Image

st.set_page_config(page_title="Airplane Detector", layout="centered")
st.title("✈️ Обнаружение самолётов")

tab1, tab2 = st.tabs(["Фото", "Видео"])

with tab1:
    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"], key="img")
    if uploaded_file:
        image_path = "temp_input.jpg"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(Image.open(image_path), caption="Исходное изображение", use_column_width=True)

        if st.button("sОбработать изображение"):
            output_path = "temp_output.jpg"
            count = detect_airplanes(image_path, save_path=output_path)
            st.success(f"Найдено самолётов: {count}")
            st.image(output_path, caption="Результат", use_column_width=True)

with tab2:
    uploaded_video = st.file_uploader("Загрузите видео", type=["mp4", "avi", "mov"], key="vid")
    if uploaded_video:
        video_path = "temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        if st.button("Обработать видео"):
            st.info("Обработка видео, подождите...")
            output_path = "output_video.avi"
            count = detect_airplanes_in_video(video_path, output_path=output_path)
            st.success(f"Обнаружено самолётов: {count}")
            st.video(output_path)
