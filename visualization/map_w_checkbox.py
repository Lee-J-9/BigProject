import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster
import geopandas as gpd

# 중심 좌표 설정
center_lat, center_lon = 37.5665, 126.9780

# GeoJSON 데이터 로드
legal_boundary = gpd.read_file("https://raw.githubusercontent.com/Lee-J-9/BigProject/refs/heads/vis_test/data_for_publish/legal_boundary.geojson")
trash_bins_with_districts = gpd.read_file("https://raw.githubusercontent.com/Lee-J-9/BigProject/refs/heads/vis_test/data_for_publish/trash_bins_with_districts.geojson")

# Sidebar에서 사용자 입력 받기
st.sidebar.title("레이어 선택")
show_seoul_boundary = st.sidebar.checkbox("서울시 경계", value=True)
show_all_bins = st.sidebar.checkbox("모든 쓰레기통", value=True)
selected_districts = st.sidebar.multiselect(
    "구별 쓰레기통 보기",
    trash_bins_with_districts['SIG_KOR_NM'].unique(),
    default=[]
)

# Folium 지도 생성
m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

# 서울시 경계 추가
if show_seoul_boundary:
    folium.GeoJson(
        legal_boundary,
        tooltip="서울시 경계"
    ).add_to(m)

# 모든 쓰레기통 추가
if show_all_bins:
    all_marker_cluster = MarkerCluster().add_to(m)
    for _, row in trash_bins_with_districts.iterrows():
        folium.Marker(
            location=[row['geometry'].y, row['geometry'].x],
            tooltip=f"구: {row['SIG_KOR_NM']}"
        ).add_to(all_marker_cluster)

# 구별 쓰레기통 추가
for district_name in selected_districts:
    district_trash_bins = trash_bins_with_districts[trash_bins_with_districts['SIG_KOR_NM'] == district_name]
    marker_cluster = MarkerCluster().add_to(m)
    for _, row in district_trash_bins.iterrows():
        folium.Marker(
            location=[row['geometry'].y, row['geometry'].x],
            tooltip=f"구: {district_name}"
        ).add_to(marker_cluster)

# Streamlit에 지도 표시
st.title("서울시 쓰레기통 지도 🗺️")
st_folium(m, width=800, height=600)
