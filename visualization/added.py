import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster
import geopandas as gpd

# --- GeoData를 캐싱해서 성능 개선 ---
@st.cache_data
def load_geodata():
    legal_boundary_data = gpd.read_file(
        "https://raw.githubusercontent.com/Lee-J-9/BigProject/refs/heads/vis_test/data_for_publish/legal_boundary.geojson"
    )
    trash_bin_data = gpd.read_file(
        "https://raw.githubusercontent.com/Lee-J-9/BigProject/refs/heads/vis_test/data_for_publish/trash_bins_with_districts.geojson"
    )
    return legal_boundary_data, trash_bin_data

# 데이터 불러오기
legal_boundary, trash_bins_with_districts = load_geodata()

# 중심좌표 설정(서울시청 근처)
center_lat, center_lon = 37.5665, 126.9780

# MarkerCluster 기본 옵션
default_marker_cluster_options = {
    "zoomToBoundsOnClick": True,      # 클러스터 클릭 시 확대
    "showCoverageOnHover": True,      # 마우스 오버 시 클러스터 범위 표시
    "maxClusterRadius": 200,          # 클러스터링 반경(픽셀 단위)
    "disableClusteringAtZoom": 14     # 특정 줌 레벨 이상에서는 클러스터 해제
}

# 구 경계 스타일 함수
def district_style_function(_):
    return {
        "fillColor": "#00b493",
        "color": "#00b493",
        "fillOpacity": 0.1,
        "weight": 2,
    }

# Streamlit 제목
st.title("서울시 쓰레기통 지도 🗺️")

# 사이드바
st.sidebar.title("레이어 선택")
show_seoul_boundary = st.sidebar.checkbox("서울시 전체 경계", value=True)

# 구 선택 (다중 선택 가능)
selected_districts = st.sidebar.multiselect(
    "구 선택",
    trash_bins_with_districts['SIG_KOR_NM'].unique(),
    default=[]
)

# Folium 지도 생성
m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

# 서울시 전체 경계 표시
if show_seoul_boundary:
    folium.GeoJson(
        legal_boundary,
        tooltip="서울시 경계"
    ).add_to(m)

# 구가 선택되지 않았을 경우 안내 메시지
if len(selected_districts) == 0:
    st.info("왼쪽 사이드바에서 구를 선택해보세요!")
else:
    # 선택한 구들에 대해 반복
    for district_name in selected_districts:
        # 해당 구 경계
        district_boundary = legal_boundary[legal_boundary['SIG_KOR_NM'] == district_name]
        if not district_boundary.empty:
            folium.GeoJson(
                district_boundary,
                tooltip=district_name,
                style_function=district_style_function
            ).add_to(m)

        # 해당 구의 쓰레기통 데이터
        district_trash_bins = trash_bins_with_districts[
            trash_bins_with_districts['SIG_KOR_NM'] == district_name
        ]

        # MarkerCluster 추가
        marker_cluster = MarkerCluster(**default_marker_cluster_options).add_to(m)

        for _, row in district_trash_bins.iterrows():
            # Font Awesome 아이콘 (쓰레기통 아이콘)
            icon = folium.Icon(
                icon="trash",
                prefix="fa",
                color="blue"
            )
            # 팝업에 표시할 내용 (데이터프레임에 'road_addr' 컬럼이 있다고 가정)
            popup_text = row.get('address', '주소 정보 없음')

            folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                tooltip=f"구: {district_name}",
                popup=popup_text,
                icon=icon
            ).add_to(marker_cluster)

# 지도 출력
st_folium(m, width=800, height=600)
