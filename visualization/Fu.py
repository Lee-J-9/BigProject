import requests
import streamlit as st

# GitHub의 raw 파일 URL
url = "https://raw.githubusercontent.com/<username>/<repository>/<branch>/<path-to-file>.html"

# HTML 파일 가져오기
response = requests.get(url)
if response.status_code == 200:
    html_content = response.text
    st.components.v1.html(html_content, height=700)
else:
    st.error(f"HTML 파일을 불러오지 못했습니다. 상태 코드: {response.status_code}")
