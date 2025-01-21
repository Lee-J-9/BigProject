import streamlit as st

# GitHub Pages URL
url = 'https://raw.githubusercontent.com/Lee-J-9/BigProject/refs/heads/main/visualization/result/test5.html'
# HTML 파일을 직접 임베드
st.components.v1.html(f'<iframe src="{url}" width="100%" height="700"></iframe>')
