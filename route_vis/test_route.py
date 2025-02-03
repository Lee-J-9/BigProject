import streamlit as st
import folium
import geopandas as gpd
import pandas as pd
import requests
from io import BytesIO
from streamlit_folium import folium_static

# 📌 GitHub RAW 데이터 URL
geojson_url = "https://raw.githubusercontent.com/Lee-J-9/BigProject/main/route_vis/data/cluster_routes.geojson"
csv_url = "https://raw.githubusercontent.com/Lee-J-9/BigProject/main/route_vis/data/광진구_clusters_route%201.csv"

# 🚀 데이터 다운로드 함수 (Streamlit 캐싱 사용)
@st.cache_data
def load_geojson(url):
    response = requests.get(url)
    if response.status_code == 200:
        return gpd.read_file(BytesIO(response.content))
    else:
        st.error(f"❌ GeoJSON 다운로드 실패! 상태 코드: {response.status_code}")
        return None

@st.cache_data
def load_csv(url):
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_csv(BytesIO(response.content))
    else:
        st.error(f"❌ CSV 다운로드 실패! 상태 코드: {response.status_code}")
        return None

# 📌 데이터 불러오기
gdf_routes = load_geojson(geojson_url)
df_clusters = load_csv(csv_url)

# 📌 데이터 체크
if gdf_routes is None or df_clusters is None:
    st.stop()

# 📌 Streamlit UI
st.title("📍 클러스터별 경로 및 쓰레기통 위치 시각화")

# 📌 클러스터 선택 옵션
selected_cluster = st.sidebar.selectbox("📌 클러스터 선택", df_clusters['cluster'].unique())

# 🌍 지도 설정
center = [37.4335, 127.0138]
m = folium.Map(location=center, zoom_start=13)

# 📌 클러스터별 색상 설정
cluster_colors = {0: "blue", 1: "green", 2: "purple"}

# 🚗 선택한 클러스터의 경로 추가
filtered_routes = gdf_routes[gdf_routes['cluster'] == selected_cluster]
for _, row in filtered_routes.iterrows():
    color = cluster_colors.get(selected_cluster, "gray")
    folium.GeoJson(
        data=row['geometry'].__geo_interface__,
        name=f"Cluster {selected_cluster}",
        style_function=lambda feature: {'color': color, 'weight': 3}
    ).add_to(m)

# 🗑️ 선택한 클러스터의 쓰레기통 마커 추가
filtered_bins = df_clusters[df_clusters['cluster'] == selected_cluster]
for _, row in filtered_bins.iterrows():
    order_number = row['order']
    marker_color = cluster_colors.get(selected_cluster, "gray")
    
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

# 🌍 지도 표시
folium_static(m)
