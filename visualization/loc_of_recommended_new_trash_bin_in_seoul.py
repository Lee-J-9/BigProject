import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster
import geopandas as gpd

# --- 페이지 설정 ---
st.set_page_config(layout="wide")  # 화면 전체 너비 사용

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

# (1) 모든 구 목록 (정렬)
all_districts = sorted(trash_bins_with_districts["SIG_KOR_NM"].unique())

# (2) 서울시 전체 체크박스
all_selected_check = st.sidebar.checkbox(
    "서울시 전체", 
    value=st.session_state["all_districts_checkbox"]
)

# (3) 구 멀티셀렉트
multiselect_districts = st.sidebar.multiselect(
    "구 선택(멀티셀렉트)",
    all_districts,
    default=st.session_state["selected_districts"]
)

# (4) 기존 / 신규 쓰레기통 표시 여부
show_existing_bins = st.sidebar.checkbox("기존 쓰레기통 표시", value=True)
show_new_bins = st.sidebar.checkbox("신규 쓰레기통(배치 점수) 표시", value=True)


# --- 4) "전체 구" 체크박스 & 멀티셀렉트 동기화 로직 ---
if all_selected_check and not st.session_state["all_districts_checkbox"]:
    # (Off -> On)으로 바뀔 때
    st.session_state["previous_selection"] = multiselect_districts  # 기존 선택 상태 저장
    st.session_state["selected_districts"] = all_districts          # 모든 구 선택
    st.session_state["all_districts_checkbox"] = True
    multiselect_districts = all_districts

elif not all_selected_check and st.session_state["all_districts_checkbox"]:
    # (On -> Off)로 바뀔 때
    st.session_state["selected_districts"] = st.session_state["previous_selection"]
    st.session_state["all_districts_checkbox"] = False
    multiselect_districts = st.session_state["previous_selection"]

else:
    # 체크박스 상태가 그대로인 경우(변동 없음)
    if not all_selected_check:
        # 체크박스가 Off 상태면 -> 멀티셀렉트 선택을 그대로
        st.session_state["selected_districts"] = multiselect_districts
    else:
        # 체크박스가 On 상태를 유지 -> 계속 모든 구
        st.session_state["selected_districts"] = all_districts
        multiselect_districts = all_districts

# 최종 사용할 구 목록
selected_districts = st.session_state["selected_districts"]

# ----------------------------------------------------------------------------
# (A) 지도와 표를 나란히(옆에) 배치하기 위해 2개의 컬럼을 만든다
col_map, col_table = st.columns([1,1])  # 왼쪽 넓게(2), 오른쪽 좁게(1)
# ----------------------------------------------------------------------------

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

# --- (B) 왼쪽 컬럼: 지도 표시 ---
# --- 지도 렌더링 ---
with col_map:
    st.markdown("### 서울시 쓰레기통 지도")
    m = folium.Map(
        location=st.session_state["map_center"],
        zoom_start=st.session_state["map_zoom"]
    )

    # 지도 데이터 렌더링
    if len(selected_districts) > 0:
        for district_name in selected_districts:
            # 구 경계 표시
            district_boundary = legal_boundary[legal_boundary["SIG_KOR_NM"] == district_name]
            if not district_boundary.empty:
                folium.GeoJson(
                    district_boundary,
                    tooltip=district_name,
                    style_function=lambda _: {"fillColor": "#00b493", "fillOpacity": 0.1, "weight": 2}
                ).add_to(m)

            # 기존 쓰레기통
            if show_existing_bins:
                district_trash_bins = trash_bins_with_districts[
                    trash_bins_with_districts["SIG_KOR_NM"] == district_name
                ]
                if not district_trash_bins.empty:
                    cluster_existing = MarkerCluster(**default_marker_cluster_options).add_to(m)
                    for _, row in district_trash_bins.iterrows():
                        folium.Marker(
                            location=[row.geometry.y, row.geometry.x],
                            tooltip=f"구: {district_name}",
                            icon=folium.Icon(icon="trash", prefix="fa", color="blue")
                        ).add_to(cluster_existing)

            # 신규 쓰레기통
            if show_new_bins:
                district_new_bins = new_trash_bins[
                    new_trash_bins["SIG_KOR_NM"] == district_name
                ]
                if not district_new_bins.empty:
                    cluster_new = MarkerCluster(**default_marker_cluster_options).add_to(m)
                    for _, row_new in district_new_bins.iterrows():
                        folium.Marker(
                            location=[row_new.geometry.y, row_new.geometry.x],
                            tooltip=f"구: {district_name}<br>점수: {row_new.get('score', '없음')}",
                            icon=folium.Icon(icon="star", prefix="fa", color="red")
                        ).add_to(cluster_new)

    map_data = st_folium(m, width=700, height=500)

# --- (C) 오른쪽 컬럼: 선택된 구의 신규 쓰레기통 점수 표 ---
with col_table:
    st.markdown("#### 신규 쓰레기통 점수 정보")
    if len(selected_districts) == 0:
        st.write("선택된 구가 없습니다.")
    else:
        # 선택된 구들에 대해 new_trash_bins를 필터링
        df_filtered = new_trash_bins[new_trash_bins["SIG_KOR_NM"].isin(selected_districts)]
        
        if df_filtered.empty:
            st.write("선택된 구에 신규 쓰레기통 데이터가 없습니다.")
        else:
            # geometry는 테이블에서 빼고, SIG_KOR_NM / score 등만 표시
            df_table = df_filtered[["SIG_KOR_NM", "score"]].reset_index(drop=True)
            st.dataframe(df_table)

# # --- 마지막: 지도 상태를 세션에 업데이트 (지도 이동/확대 정보) ---
# if map_data and "center" in map_data:
#     lat = map_data["center"].get("lat", 0)
#     lng = map_data["center"].get("lng", 0)
#     if lat != 0 and lng != 0:
#         st.session_state["map_center"] = [lat, lng]
#         st.session_state["map_zoom"] = map_data["zoom"]


