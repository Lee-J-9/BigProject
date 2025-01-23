import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster
import geopandas as gpd

# --- 데이터 캐싱: 여러 번 실행해도 최초 1회만 다운로드 및 파싱 ---
@st.cache_data
def load_geodata():
    # 1) 서울시 법정 경계
    legal_boundary_data = gpd.read_file(
        "https://raw.githubusercontent.com/Lee-J-9/BigProject/refs/heads/vis_test/data_for_publish/legal_boundary.geojson"
    )
    # 2) 기존 쓰레기통 데이터
    trash_bin_data = gpd.read_file(
        "https://raw.githubusercontent.com/Lee-J-9/BigProject/refs/heads/vis_test/data_for_publish/trash_bins_with_districts.geojson"
    )
    # 3) 신규 쓰레기통 데이터 (배치 점수 포함)
    new_trash_bin_data = gpd.read_file(
        "https://raw.githubusercontent.com/Lee-J-9/BigProject/refs/heads/final_vis/data_for_publish/rc_trash_bins.geojson"
    )
    return legal_boundary_data, trash_bin_data, new_trash_bin_data

# 데이터 로드
legal_boundary, trash_bins_with_districts, new_trash_bins = load_geodata()

# 지도의 초기 중앙좌표(서울시청) 및 기본 줌 레벨
center_lat, center_lon = 37.5665, 126.9780

# MarkerCluster 기본 옵션
default_marker_cluster_options = {
    "zoomToBoundsOnClick": True,
    "showCoverageOnHover": True,
    "maxClusterRadius": 200,
    "disableClusteringAtZoom": 14
}

# 법정 경계 스타일 함수
def district_style_function(_):
    return {
        "fillColor": "#00b493",
        "color": "#00b493",
        "fillOpacity": 0.1,
        "weight": 2,
    }

# Streamlit에서 제목 표시
st.title("서울시 쓰레기통 지도 🗺️")

# --- 사이드바 설정 ---
st.sidebar.title("지도 옵션")
# 1) 서울시 전체 경계 표시 여부
show_seoul_boundary = st.sidebar.checkbox("서울시 전체 경계", value=True)

# 2) 구 선택 (다중 선택 가능)
selected_districts = st.sidebar.multiselect(
    "구 선택",
    sorted(trash_bins_with_districts['SIG_KOR_NM'].unique()),
    default=[]
)

# 3) 신규 쓰레기통 데이터 표시 여부
show_new_bins = st.sidebar.checkbox("신규 쓰레기통(배치 점수) 표시", value=True)

# Folium 지도 생성
m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

# 서울시 전체 경계 표시
if show_seoul_boundary:
    folium.GeoJson(
        legal_boundary,
        tooltip="서울시 경계"
    ).add_to(m)

# 구를 하나도 선택하지 않았을 경우 안내문
if len(selected_districts) == 0:
    st.info("왼쪽 사이드바에서 구를 선택해보세요!")
else:
    for district_name in selected_districts:
        # 1) 구 경계 표시
        district_boundary = legal_boundary[legal_boundary['SIG_KOR_NM'] == district_name]
        if not district_boundary.empty:
            folium.GeoJson(
                district_boundary,
                tooltip=district_name,
                style_function=district_style_function,
            ).add_to(m)

        # 2) 기존 쓰레기통 마커 추가
        district_trash_bins = trash_bins_with_districts[
            trash_bins_with_districts['SIG_KOR_NM'] == district_name
        ]
        marker_cluster_existing = MarkerCluster(**default_marker_cluster_options).add_to(m)

        for _, row in district_trash_bins.iterrows():
            icon_existing = folium.Icon(
                icon="trash",
                prefix="fa",
                color="blue"  # 기존 쓰레기통 -> 파란색
            )
            # Tooltip에는 구 이름만 표시(예시)
            tooltip_existing = f"구: {district_name}"
            folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                tooltip=tooltip_existing,
                icon=icon_existing
            ).add_to(marker_cluster_existing)

        # 3) 신규 쓰레기통 마커 추가 (배치 점수 포함)
        if show_new_bins:
            district_new_trash_bins = new_trash_bins[
                new_trash_bins['SIG_KOR_NM'] == district_name
            ]
            if not district_new_trash_bins.empty:
                marker_cluster_new = MarkerCluster(**default_marker_cluster_options).add_to(m)
                for _, row_new in district_new_trash_bins.iterrows():
                    icon_new = folium.Icon(
                        icon="trash",
                        prefix="fa",
                        color="red"  # 신규 쓰레기통 -> 빨간색
                    )
                    # 배치 점수 포함한 툴팁
                    tooltip_text_new = (
                        f"구: {district_name}<br>"
                        f"배치 점수: {row_new.get('score', '점수 정보 없음')}"
                    )
                    folium.Marker(
                        location=[row_new.geometry.y, row_new.geometry.x],
                        tooltip=tooltip_text_new,
                        icon=icon_new
                    ).add_to(marker_cluster_new)

# 지도 표시
st_folium(m, width=800, height=600)

