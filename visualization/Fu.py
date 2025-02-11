import requests
import streamlit as st

# 이거 왜 됨?
# main 때문인가? 
# 응 아니야 아니아니야

# GitHub의 raw 파일 URL
url = 'https://raw.githubusercontent.com/Lee-J-9/BigProject/refs/heads/vis_test/visualization/result/trash_bins_map_with_seoul_layer.html'



st.title('안녕하세요 이재구 입니다.')
st.subheader('서울시 쓰레기통 지도')
st.write('추가 쓰레기통 X, 테스트용')

# HTML 파일 가져오기
response = requests.get(url)
if response.status_code == 200:
    html_content = response.text
    st.components.v1.html(html_content, height=700)
else:
    st.error(f"HTML 파일을 불러오지 못했습니다. 상태 코드: {response.status_code}")
