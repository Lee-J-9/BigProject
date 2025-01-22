import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# 중심 좌표 설정
center_lat, center_lon = 37.5665, 126.9780

# GeoJSON 데이터 로드
legal_boundary = gpd.read_file("legal_boundary.geojson")
trash_bins_with_districts = gpd.read_file("trash_bins_with_districts.geojson")

# 신규 쓰레기통 데이터 로드
new_trash_bins = pd.read_csv("new_trash_bins.csv")  # 'latitude', 'longitude' 컬럼 필요

# 신규 쓰레기통을 GeoDataFrame으로 변환
new_trash_bins['geometry'] = new_trash_bins.apply(
    lambda row: Point(row['longitude'], row['latitude']), axis=1
)
new_trash_bins_gdf = gpd.GeoDataFrame(new_trash_bins, geometry='geometry', crs=legal_boundary.crs)

# 신규 쓰레기통에 구 이름 매핑 (구 경계와 교차 검사)
new_trash_bins_gdf = gpd.sjoin(new_trash_bins_gdf, legal_boundary[['SIG_KOR_NM', 'geometry']], how="left", op='within')

# MarkerCluster 기본 옵션 설정
default_marker_cluster_options = {
    "zoomToBoundsOnClick": True,
    "showCoverageOnHover": True,
    "maxClusterRadius": 200,
    "disableClusteringAtZoom": 15
}

# Sidebar에서 사용자 입력 받기
st.sidebar.title("레이어 선택")
show_seoul_boundary = st.sidebar.checkbox("서울시 경계", value=True)
selected_districts = st.sidebar.multiselect(
    "구 선택",
    legal_boundary['SIG_KOR_NM'].unique(),
    default=[]
)
show_new_bins = st.sidebar.checkbox("신규 쓰레기통 표시", value=True)

# Folium 지도 생성
m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

# 서울시 경계 추가
if show_seoul_boundary:
    folium.GeoJson(
        legal_boundary,
        tooltip="서울시 경계"
    ).add_to(m)

# 기존 쓰레기통 및 신규 쓰레기통 구별로 표시
for district_name in selected_districts:
    # 구별 경계 추가
    district_boundary = legal_boundary[legal_boundary['SIG_KOR_NM'] == district_name]
    if not district_boundary.empty:
        folium.GeoJson(
            district_boundary,
            tooltip=district_name,
            style_function=lambda x: {
                "fillColor": "blue",
                "color": "blue",
                "fillOpacity": 0.1,
                "weight": 2,
            },
        ).add_to(m)
    
    # 기존 쓰레기통 표시
    district_trash_bins = trash_bins_with_districts[trash_bins_with_districts['SIG_KOR_NM'] == district_name]
    marker_cluster = MarkerCluster(**default_marker_cluster_options).add_to(m)
    for _, row in district_trash_bins.iterrows():
        folium.Marker(
            location=[row['geometry'].y, row['geometry'].x],
            tooltip=f"구: {district_name}",
            icon=folium.Icon(icon="trash", prefix="fa", color="green")
        ).add_to(marker_cluster)
    
    # 신규 쓰레기통 표시 (좌표 기준으로 구별)
    if show_new_bins:
        district_new_bins = new_trash_bins_gdf[new_trash_bins_gdf['SIG_KOR_NM'] == district_name]
        new_bin_cluster = MarkerCluster(**default_marker_cluster_options).add_to(m)
        for _, row in district_new_bins.iterrows():
            folium.Marker(
                location=[row['geometry'].y, row['geometry'].x],
                tooltip=f"신규 쓰레기통 - 구: {district_name}",
                icon=folium.Icon(icon="plus", prefix="fa", color="red")
            ).add_to(new_bin_cluster)

# Streamlit에 지도 표시
st.title("서울시 쓰레기통 지도 🗺️")
st_folium(m, width=800, height=600)
