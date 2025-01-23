import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster
import geopandas as gpd

# matplotlib에서 colormap 사용을 위해 import
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ------ 캐싱 함수 -------
@st.cache_data
def load_geojson(url):
    return gpd.read_file(url)

# ------ 데이터 로드 -------
legal_boundary_url = "https://raw.githubusercontent.com/Lee-J-9/BigProject/refs/heads/vis_test/data_for_publish/legal_boundary.geojson"
trash_bins_url = "https://raw.githubusercontent.com/Lee-J-9/BigProject/refs/heads/vis_test/data_for_publish/trash_bins_with_districts.geojson"

legal_boundary = load_geojson(legal_boundary_url)
trash_bins_with_districts = load_geojson(trash_bins_url)

# ------ 전역 변수 설정 ------
center_lat, center_lon = 37.5665, 126.9780

default_marker_cluster_options = {
    "zoomToBoundsOnClick": True,
    "showCoverageOnHover": True,
    "maxClusterRadius": 200,
    "disableClusteringAtZoom": 14
}

# ------ Streamlit UI -------
st.set_page_config(page_title="서울시 쓰레기통 지도", layout="wide")
st.title("서울시 쓰레기통 지도 🗺️")

# Sidebar
with st.sidebar:
    st.header("레이어 선택")
    show_seoul_boundary = st.checkbox("서울시 전체 경계", value=True)
    selected_districts = st.multiselect(
        "구 선택",
        trash_bins_with_districts['SIG_KOR_NM'].unique(),
        default=[]
    )

def create_map(selected_districts, show_boundary):
    # 지도 초기화 (원하는 타일 사용: CartoDB positron)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB positron")
    
    # 서울시 전체 경계 표시
    if show_boundary:
        folium.GeoJson(
            legal_boundary,
            tooltip="서울시 경계",
            style_function=lambda x: {
                "fillColor": "#999999",
                "color": "#999999",
                "fillOpacity": 0.05,
                "weight": 2,
            },
        ).add_to(m)

    # (1) 선택된 구의 수만큼 'summer' 컬러맵에서 색상을 가져오기
    num_districts = len(selected_districts)
    if num_districts > 0:
        colormap = cm.get_cmap('summer', num_districts)  # summer 컬러맵에서 N단계 색

    # (2) 구를 순회하며, 색상 및 경계/마커 표시
    for i, district_name in enumerate(selected_districts):
        # colormap에서 i번째 색상 추출 -> hex 변환
        color = mcolors.to_hex(colormap(i)) if num_districts > 0 else "#00b493"
        
        # 해당 구 경계 데이터
        district_boundary = legal_boundary[legal_boundary['SIG_KOR_NM'] == district_name]
        if not district_boundary.empty:
            folium.GeoJson(
                district_boundary,
                tooltip=district_name,
                style_function=lambda x, col=color: {
                    "fillColor": col,
                    "color": col,
                    "fillOpacity": 0.1,
                    "weight": 2,
                },
            ).add_to(m)
        
        # 해당 구의 쓰레기통 데이터
        district_trash_bins = trash_bins_with_districts[trash_bins_with_districts['SIG_KOR_NM'] == district_name]
        
        # 쓰레기통 MarkerCluster
        marker_cluster = MarkerCluster(**default_marker_cluster_options).add_to(m)

        for _, row in district_trash_bins.iterrows():
            icon = folium.Icon(
                icon="trash",
                prefix="fa",
                color="blue"  # 아이콘 색상(마커)은 여기서 추가로 변경 가능
            )
            folium.Marker(
                location=[row['geometry'].y, row['geometry'].x],
                tooltip=f"{district_name} 쓰레기통",
                icon=icon
            ).add_to(marker_cluster)
    
    return m

# 최종 지도 생성 및 표시
result_map = create_map(selected_districts, show_seoul_boundary)
st_folium(result_map, width=900, height=600)