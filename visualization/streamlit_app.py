import streamlit as st

# HTML 파일 읽기
html_file_path = "./result/test5.html"  # HTML 파일 경로
with open(html_file_path, "r", encoding="utf-8") as f:
    html_content = f.read()

# Streamlit에서 HTML 렌더링
st.title("완성된 지도 보기")
st.components.v1.html(html_content, height=700)

