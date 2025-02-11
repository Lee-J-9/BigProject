
import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster
import geopandas as gpd

# --- 페이지 설정 ---
st.set_page_config(layout="wide")  # 화면 전체 너비 사용
if "selected_districts" not in st.session_state:
    st.session_state["selected_districts"] = []

# --- 1) 데이터 로딩 & 캐싱 ---
@st.cache_data
def load_geodata():
    # 서울시 경계
    legal_boundary_data = gpd.read_file(
        "https://raw.githubusercontent.com/Lee-J-9/BigProject/refs/heads/main/data_for_publish/legal_boundary.geojson"
    )
    # 기존 쓰레기통 데이터
    trash_bin_data = gpd.read_file(
        "https://raw.githubusercontent.com/Lee-J-9/BigProject/refs/heads/main/data_for_publish/trash_bins_with_districts.geojson"
    )
    new_trash_bin_data = gpd.read_file(
        "https://raw.githubusercontent.com/Lee-J-9/BigProject/refs/heads/rdata/data_for_publish/new_trash_bins.geojson"

    )
    return legal_boundary_data, trash_bin_data, new_trash_bin_data

legal_boundary, trash_bins_with_districts, new_trash_bins = load_geodata()


# --- 2) 세션 스테이트 초기화 ---
# 2-1) 지도 초기 상태(센터, 줌 레벨)
if "map_center" not in st.session_state:
    st.session_state["map_center"] = [37.5665, 126.9780]  # 서울시청
if "map_zoom" not in st.session_state:
    st.session_state["map_zoom"] = 11

# --- 3) 사이드바 UI ---
st.sidebar.title("지도 옵션")

# (1) 모든 구 목록 (정렬)
all_districts = sorted(trash_bins_with_districts["SIG_KOR_NM"].unique())

# (3) 구 멀티셀렉트
multiselect_districts = st.sidebar.multiselect(
    "구 선택(멀티셀렉트)",
    all_districts,
    default=st.session_state["selected_districts"]
)

# (4) 기존 / 신규 쓰레기통 표시 여부
show_existing_bins = st.sidebar.checkbox("기존 쓰레기통 표시", value=True)
show_new_bins = st.sidebar.checkbox("신규 쓰레기통(배치 점수) 표시", value=True)

# 최종 사용할 구 목록
selected_districts = st.session_state["selected_districts"]
# 최종 사용할 구 목록
selected_districts = multiselect_districts if multiselect_districts else []  # 멀티셀렉트 값을 확인 후 설정
# ----------------------------------------------------------------------------
# (A) 지도와 표를 나란히(옆에) 배치하기 위해 2개의 컬럼을 만든다
col_map, col_img = st.columns([2,1])  # 왼쪽 넓게(2), 오른쪽 좁게(1)
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
                            tooltip=f"구: {district_name}<br>점수: {row_new.get('점수', '없음')}",
                            icon=folium.Icon(icon="star", prefix="fa", color="red")
                        ).add_to(cluster_new)

    map_data = st_folium(m, width=700, height=500)
    
with col_img:
    st.markdown("### 🖼️ 점수 산정 방식")
    image_url = "https://raw.githubusercontent.com/Lee-J-9/BigProject/rdata/data_for_publish/score.png"
    st.image(image_url, caption="5분에 한번 쓰레기통을 만날 수 있게 하겠습니다.", use_container_width=True)
    st.markdown("### 📊 신규 쓰레기통 점수 정보")
    if len(multiselect_districts) == 0:
        st.write("선택된 구가 없습니다.")
    else:
        df_filtered = new_trash_bins[new_trash_bins["SIG_KOR_NM"].isin(multiselect_districts)]
        if df_filtered.empty:
            st.write("선택된 구에 신규 쓰레기통 데이터가 없습니다.")
        else:
            df_table = df_filtered[["SIG_KOR_NM", "주소", "점수"]].reset_index(drop=True)
            st.dataframe(df_table)


st.markdown("---")  # 구분선 추가

st.markdown("### 📊 신규 쓰레기통 점수 정보")
if len(multiselect_districts) == 0:
    st.write("선택된 구가 없습니다.")
else:
    df_filtered = new_trash_bins[new_trash_bins["SIG_KOR_NM"].isin(multiselect_districts)]
    if df_filtered.empty:
        st.write("선택된 구에 신규 쓰레기통 데이터가 없습니다.")
    else:
        df_table = df_filtered[["SIG_KOR_NM", "주소", "점수"]].reset_index(drop=True)
        st.dataframe(df_table, height=500)
