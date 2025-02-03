import streamlit as st
import folium
import geopandas as gpd
import pandas as pd
from streamlit_folium import folium_static

# 파일 경로 설정
route_geojson = "./testing/cluster_routes.geojson"
csv_path = "./testing/광진구_clusters_route 1.csv"

# 제목
st.title("📍 클러스터별 경로 및 쓰레기통 위치 시각화")

# GeoJSON 데이터 로드
gdf_routes = gpd.read_file(route_geojson)
df_clusters = pd.read_csv(csv_path)

# 📌 클러스터가 없으면 에러 처리
if 'cluster' not in gdf_routes.columns or 'cluster' not in df_clusters.columns:
    st.error("⚠️ `cluster` 컬럼이 데이터에 없습니다. 확인해주세요!")
    st.stop()

# 📌 **사이드바에서 클러스터 선택 옵션 추가**
selected_cluster = st.sidebar.selectbox("📌 클러스터 선택", df_clusters['cluster'].unique())

# 지도 중심 설정 (서울 광진구 기준)
center = [37.54, 127.08]
m = folium.Map(location=center, zoom_start=13)

# 📌 클러스터별 색상 설정 (경로 & 마커 동일하게 적용)
cluster_colors = {0: "blue", 1: "green", 2: "purple"}

# 🚗 **선택한 클러스터의 경로만 지도에 추가**
filtered_routes = gdf_routes[gdf_routes['cluster'] == selected_cluster]
for _, row in filtered_routes.iterrows():
    color = cluster_colors.get(selected_cluster, "gray")  # 예외 처리

    folium.GeoJson(
        data=row['geometry'].__geo_interface__,
        name=f"Cluster {selected_cluster}",
        style_function=lambda feature, color=color: {'color': color, 'weight': 3}
    ).add_to(m)
    
# 🗑️ **선택한 클러스터의 쓰레기통 마커 추가 (수정된 코드)**
filtered_bins = df_clusters[df_clusters['cluster'] == selected_cluster]

for _, row in filtered_bins.iterrows():
    order_number = row['order']  # Order 값 가져오기
    marker_color = cluster_colors.get(selected_cluster, "gray")  # 클러스터별 색상

    folium.Marker(
        location=[row['latitude'], row['longitude']],
        icon=folium.DivIcon(
            icon_size=(30, 30),
            icon_anchor=(15, 15),
            html=f'<div style="font-size: 12pt; color: white; background-color: {marker_color}; '
                 f'border-radius: 50%; padding: 5px; width: 25px; height: 25px; '
                 f'display: flex; justify-content: center; align-items: center;">{order_number}</div>'
        ),
        popup=f"Order: {order_number}"
    ).add_to(m)


# 🌍 **스트림릿에서 지도 표시**
folium_static(m)

