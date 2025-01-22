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
selected_districts = st.sidebar.multiselect(
    "구 선택",
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

# 구별 경계 및 쓰레기통 통합 레이어 추가
for district_name in selected_districts:
    # 해당 구의 경계 추출
    district_boundary = legal_boundary[legal_boundary['SIG_KOR_NM'] == district_name]
    if not district_boundary.empty:
        folium.GeoJson(
            district_boundary,
            tooltip=district_name,  # 경계 툴팁 추가
            style_function=lambda x: {
                "fillColor": "blue",  # 경계 색상
                "color": "blue",
                "fillOpacity": 0.1,
                "weight": 2,
            },
        ).add_to(m)
    
    # 해당 구의 쓰레기통 데이터 필터링
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
