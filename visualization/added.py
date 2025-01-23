import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster
import geopandas as gpd

# --- 1) 데이터 로딩 & 캐싱 ---
@st.cache_data
def load_geodata():
    # 서울시 경계
    legal_boundary_data = gpd.read_file(
        "https://raw.githubusercontent.com/Lee-J-9/BigProject/refs/heads/vis_test/data_for_publish/legal_boundary.geojson"
    )
    # 기존 쓰레기통 데이터
    trash_bin_data = gpd.read_file(
        "https://raw.githubusercontent.com/Lee-J-9/BigProject/refs/heads/vis_test/data_for_publish/trash_bins_with_districts.geojson"
    )
    # 신규 쓰레기통 데이터(배치 점수)
    new_trash_bin_data = gpd.read_file(
        "https://raw.githubusercontent.com/Lee-J-9/BigProject/refs/heads/final_vis/data_for_publish/rc_trash_bins.geojson"
    )
    return legal_boundary_data, trash_bin_data, new_trash_bin_data

legal_boundary, trash_bins_with_districts, new_trash_bins = load_geodata()

# --- 2) 세션 스테이트 초기화 ---
# 2-1) 지도 초기 상태(센터, 줌 레벨)
if "map_center" not in st.session_state:
    st.session_state["map_center"] = [37.5665, 126.9780]  # 서울시청
if "map_zoom" not in st.session_state:
    st.session_state["map_zoom"] = 11

# 2-2) "전체 구" 체크박스 상태 & 멀티셀렉트 상태
if "all_districts_checkbox" not in st.session_state:
    st.session_state["all_districts_checkbox"] = False  # 서울시 전체 체크박스 초기값
if "selected_districts" not in st.session_state:
    st.session_state["selected_districts"] = []         # 멀티셀렉트로 선택된 구들
if "previous_selection" not in st.session_state:
    st.session_state["previous_selection"] = []         # "전체 구" 켜기 전 선택 상태 기억용

# --- 3) 사이드바 UI ---
st.sidebar.title("지도 옵션")

# # (a) 서울시 전체 경계 on/off
# show_seoul_boundary = st.sidebar.checkbox("서울시 전체 경계", value=True)

# (d) 구 선택 옵션(전체 선택 + 멀티셀렉트)
all_districts = sorted(trash_bins_with_districts["SIG_KOR_NM"].unique())

# (b) 기존 쓰레기통 표시
show_existing_bins = st.sidebar.checkbox("기존 쓰레기통 표시", value=True)

# (c) 신규 쓰레기통 표시
show_new_bins = st.sidebar.checkbox("신규 쓰레기통(배치 점수) 표시", value=True)

all_selected_check = st.sidebar.checkbox(
    "서울시 전체", 
    value=st.session_state["all_districts_checkbox"]
)

multiselect_districts = st.sidebar.multiselect(
    "구 선택(멀티셀렉트)",
    all_districts,
    default=st.session_state["selected_districts"]
)

# --- 4) "전체 구" 체크박스 & 멀티셀렉트 동기화 로직 ---
if all_selected_check and not st.session_state["all_districts_checkbox"]:
    # 1) (Off -> On)으로 바뀔 때:
    #    현재 멀티셀렉트 상태를 보관해두고,
    st.session_state["previous_selection"] = multiselect_districts
    #    모든 구를 선택하도록 설정
    st.session_state["selected_districts"] = all_districts
    st.session_state["all_districts_checkbox"] = True
    #    멀티셀렉트에도 반영
    multiselect_districts = all_districts

elif not all_selected_check and st.session_state["all_districts_checkbox"]:
    # 2) (On -> Off)로 바뀔 때:
    #    전에 보관했던 부분 선택 상태로 복원
    st.session_state["selected_districts"] = st.session_state["previous_selection"]
    st.session_state["all_districts_checkbox"] = False
    #    멀티셀렉트에도 반영
    multiselect_districts = st.session_state["previous_selection"]

else:
    # 3) 체크박스 상태가 그대로인 경우(변동 없음)
    if not all_selected_check:
        # 체크박스가 원래부터 Off라면, 멀티셀렉트가 곧 사용자가 선택한 "selected"
        st.session_state["selected_districts"] = multiselect_districts
    else:
        # 체크박스가 On 상태를 유지 중이라면, 계속 모든 구
        st.session_state["selected_districts"] = all_districts
        multiselect_districts = all_districts
        
# 최종적으로 사용할 구 목록
selected_districts = st.session_state["selected_districts"]

# --- 5) Folium 지도 생성(세션 상태의 좌표/줌 사용) ---
m = folium.Map(
    location=st.session_state["map_center"],
    zoom_start=st.session_state["map_zoom"]
)

# MarkerCluster 기본 옵션
default_marker_cluster_options = {
    "zoomToBoundsOnClick": True,
    "showCoverageOnHover": True,
    "maxClusterRadius": 200,
    "disableClusteringAtZoom": 14
}

# 구 경계 스타일 함수
def district_style_function(_):
    return {
        "fillColor": "#00b493",
        "color": "#00b493",
        "fillOpacity": 0.1,
        "weight": 2,
    }

# # (a) 서울시 전체 경계 표시
# if show_seoul_boundary:
#     folium.GeoJson(
#         legal_boundary,
#         tooltip="서울시 경계"
#     ).add_to(m)

# (b) 선택된 구가 없다면 안내 메시지
if len(selected_districts) == 0:
    st.info("왼쪽 사이드바에서 '서울시 전체' 또는 구를 선택해보세요!")
else:
    # (c) 선택된 구들의 경계 + 쓰레기통 표시
    for district_name in selected_districts:
        # 1) 구 경계 표시
        district_boundary = legal_boundary[legal_boundary["SIG_KOR_NM"] == district_name]
        if not district_boundary.empty:
            folium.GeoJson(
                district_boundary,
                tooltip=district_name,
                style_function=district_style_function
            ).add_to(m)

        # 2) 기존 쓰레기통 표시
        if show_existing_bins:
            district_trash_bins = trash_bins_with_districts[
                trash_bins_with_districts["SIG_KOR_NM"] == district_name
            ]
            if not district_trash_bins.empty:
                cluster_existing = MarkerCluster(**default_marker_cluster_options).add_to(m)
                for _, row in district_trash_bins.iterrows():
                    icon_existing = folium.Icon(icon="trash", prefix="fa", color="blue")
                    folium.Marker(
                        location=[row.geometry.y, row.geometry.x],
                        tooltip=f"구: {district_name}",
                        icon=icon_existing
                    ).add_to(cluster_existing)

        # 3) 신규 쓰레기통 표시(배치 점수)
        if show_new_bins:
            district_new_bins = new_trash_bins[
                new_trash_bins["SIG_KOR_NM"] == district_name
            ]
            if not district_new_bins.empty:
                cluster_new = MarkerCluster(**default_marker_cluster_options).add_to(m)
                for _, row_new in district_new_bins.iterrows():
                    icon_new = folium.Icon(icon="trash", prefix="fa", color="red")
                    tooltip_text_new = (
                        f"구: {district_name}<br>"
                        f"배치 점수: {row_new.get('score', '점수 정보 없음')}"
                    )
                    folium.Marker(
                        location=[row_new.geometry.y, row_new.geometry.x],
                        tooltip=tooltip_text_new,
                        icon=icon_new
                    ).add_to(cluster_new)

# --- 6) st_folium으로 지도 렌더링 & 마지막 지도 상태 받아 세션에 저장 ---
map_data = st_folium(m, width=800, height=600)

if map_data and "center" in map_data:
    # Folium이 필요한 건 [lat, lng] 형태
    lat = map_data["center"].get("lat", 0)
    lng = map_data["center"].get("lng", 0)
    if lat != 0 and lng != 0:
        st.session_state["map_center"] = [lat, lng]
        st.session_state["map_zoom"] = map_data["zoom"]
