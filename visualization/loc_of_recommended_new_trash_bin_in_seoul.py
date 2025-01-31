import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster
import geopandas as gpd

# --- 페이지 설정 ---
st.set_page_config(layout="wide")
if "selected_districts" not in st.session_state:
    st.session_state["selected_districts"] = []

# --- 1) 데이터 로딩 & 캐싱 ---
@st.cache_data(ttl=3600)  # 캐싱 유지 시간 1시간 (3600초)
def load_geodata():
    legal_boundary_data = gpd.read_file(
        "https://raw.githubusercontent.com/Lee-J-9/BigProject/refs/heads/main/data_for_publish/legal_boundary.geojson"
    )
    trash_bin_data = gpd.read_file(
        "https://raw.githubusercontent.com/Lee-J-9/BigProject/refs/heads/main/data_for_publish/trash_bins_with_districts.geojson"
    )
    new_trash_bin_data = gpd.read_file(
        "https://raw.githubusercontent.com/Lee-J-9/BigProject/refs/heads/main/data_for_publish/rc_trash_bins.geojson"
    )
    return legal_boundary_data, trash_bin_data, new_trash_bin_data

legal_boundary, trash_bins_with_districts, new_trash_bins = load_geodata()

# --- 2) 지도 초기 상태 설정 ---
if "map_center" not in st.session_state:
    st.session_state["map_center"] = [37.5665, 126.9780]  # 서울시청 위치
if "map_zoom" not in st.session_state:
    st.session_state["map_zoom"] = 11

# --- 3) 사이드바 UI ---
st.sidebar.title("지도 옵션")
all_districts = sorted(trash_bins_with_districts["SIG_KOR_NM"].unique())
multiselect_districts = st.sidebar.multiselect(
    "구 선택(멀티셀렉트)", all_districts, default=st.session_state["selected_districts"]
)
show_existing_bins = st.sidebar.checkbox("기존 쓰레기통 표시", value=True)
show_new_bins = st.sidebar.checkbox("신규 쓰레기통(배치 점수) 표시", value=True)
selected_districts = multiselect_districts if multiselect_districts else []

# --- 4) 레이아웃 설정 ---
col_map, col_table = st.columns([1, 1])  # 지도와 표를 나란히 배치

# --- 5) Folium 지도 생성 (최적화 반영) ---
with col_map:
    st.markdown("### 서울시 쓰레기통 지도")
    m = folium.Map(
        location=st.session_state["map_center"],
        zoom_start=st.session_state["map_zoom"]
    )
    if selected_districts:
        filtered_boundaries = legal_boundary[legal_boundary["SIG_KOR_NM"].isin(selected_districts)]
        folium.GeoJson(
            filtered_boundaries,
            tooltip=folium.GeoJsonTooltip(fields=["SIG_KOR_NM"]),
            style_function=lambda _: {"fillColor": "#00b493", "fillOpacity": 0.1, "weight": 2}
        ).add_to(m)
        
        if show_existing_bins:
            filtered_trash_bins = trash_bins_with_districts[trash_bins_with_districts["SIG_KOR_NM"].isin(selected_districts)]
            if not filtered_trash_bins.empty:
                cluster_existing = MarkerCluster().add_to(m)
                for _, row in filtered_trash_bins.iterrows():
                    folium.Marker(
                        location=[row.geometry.y, row.geometry.x],
                        tooltip=f"구: {row.SIG_KOR_NM}",
                        icon=folium.Icon(icon="trash", prefix="fa", color="blue")
                    ).add_to(cluster_existing)
        
        if show_new_bins:
            filtered_new_bins = new_trash_bins[new_trash_bins["SIG_KOR_NM"].isin(selected_districts)]
            if not filtered_new_bins.empty:
                cluster_new = MarkerCluster().add_to(m)
                for _, row in filtered_new_bins.iterrows():
                    folium.Marker(
                        location=[row.geometry.y, row.geometry.x],
                        tooltip=f"구: {row.SIG_KOR_NM}<br>점수: {row.get('score', '없음')}",
                        icon=folium.Icon(icon="star", prefix="fa", color="red")
                    ).add_to(cluster_new)
    map_data = st_folium(m, width=700, height=500)

# --- 6) 신규 쓰레기통 점수 표 (최적화) ---
with col_table:
    st.markdown("#### 신규 쓰레기통 점수 정보")
    if selected_districts:
        df_filtered = new_trash_bins[new_trash_bins["SIG_KOR_NM"].isin(selected_districts)]
        if not df_filtered.empty:
            df_table = df_filtered[["SIG_KOR_NM", "score"]].reset_index(drop=True)
            st.table(df_table)  # 최적화: st.dataframe 대신 st.table 사용
        else:
            st.write("선택된 구에 신규 쓰레기통 데이터가 없습니다.")
    else:
        st.write("선택된 구가 없습니다.")
