import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import geopandas as gpd

# 데이터 로드
# 서울시 경계 데이터 및 쓰레기통 데이터 로드
legal_boundary = gpd.read_file("https://raw.githubusercontent.com/Lee-J-9/BigProject/refs/heads/vis_test/data_for_publish/legal_boundary.geojson")
trash_bins_with_districts = gpd.read_file("https://raw.githubusercontent.com/Lee-J-9/BigProject/refs/heads/vis_test/data_for_publish/trash_bins_with_districts.geojson")

# 중심 좌표 설정 (예: 서울시청 좌표)
center_lat, center_lon = 37.5665, 126.9780

# Streamlit 제목
st.title("서울시 쓰레기통 지도")
st.write("서울시 쓰레기통 위치 및 구별 경계를 확인하세요!")

# 사용자 선택: 레이어 옵션
layer_options = ["서울시 통합 경계 및 쓰레기통"] + list(trash_bins_with_districts['SIG_KOR_NM'].unique())
selected_layers = st.multiselect("표시할 레이어를 선택하세요:", layer_options, default=["서울시 통합 경계 및 쓰레기통"])


# 폴리움 지도 생성
m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

# 1. 서울시 통합 경계 및 쓰레기통 레이어
if "서울시 통합 경계 및 쓰레기통" in selected_layers:
    seoul_layer = folium.FeatureGroup(name="서울시 통합 경계 및 쓰레기통", show=True)

    # 서울시 전체 경계 추가
    folium.GeoJson(
        legal_boundary,
        tooltip="서울시 경계"
    ).add_to(seoul_layer)

    # 모든 쓰레기통 마커 추가
    all_marker_cluster = MarkerCluster().add_to(seoul_layer)

    for _, row in trash_bins_with_districts.iterrows():
        folium.Marker(
            location=[row['geometry'].y, row['geometry'].x],
            tooltip=f"구: {row['SIG_KOR_NM']}"
        ).add_to(all_marker_cluster)

    seoul_layer.add_to(m)

# 2. 구별 경계 및 쓰레기통 통합 레이어 (코드 재확인 부분 삽입)
for district_name in trash_bins_with_districts['SIG_KOR_NM'].unique():
    if district_name in selected_layers:
        district_layer = folium.FeatureGroup(name=f"{district_name}", show=False)

        district_boundary = legal_boundary[legal_boundary['SIG_KOR_NM'] == district_name]

        if not district_boundary.empty:
            folium.GeoJson(
                district_boundary,
                tooltip=district_name
            ).add_to(district_layer)

            district_trash_bins = trash_bins_with_districts[trash_bins_with_districts['SIG_KOR_NM'] == district_name]
            marker_cluster = MarkerCluster().add_to(district_layer)

            for _, row in district_trash_bins.iterrows():
                folium.Marker(
                    location=[row['geometry'].y, row['geometry'].x],
                    tooltip=f"구: {district_name}"
                ).add_to(marker_cluster)

        district_layer.add_to(m)
        
# Streamlit에서 Folium 지도 렌더링
st_data = st_folium(m, width=800, height=600)
