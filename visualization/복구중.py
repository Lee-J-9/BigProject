# 공통 MarkerCluster 옵션 (기본 줌 제한: 16)
default_marker_cluster_options = {
    "zoomToBoundsOnClick": True,      # 클러스터 클릭 시 확대
    "showCoverageOnHover": True,      # 마우스 오버 시 클러스터 영역 표시
    "maxClusterRadius": 200,          # 클러스터링 반경 (픽셀)
    "disableClusteringAtZoom": 16     # 줌 레벨 16 이상에서 클러스터링 비활성화
}

# 버스 정류장 MarkerCluster 옵션 (줌 제한: 18)
bus_stop_marker_cluster_options = {
    "zoomToBoundsOnClick": True,
    "showCoverageOnHover": True,
    "maxClusterRadius": 200,
    "disableClusteringAtZoom": 18  # 줌 레벨 18 이상에서 클러스터링 비활성화
}

# 1. sports_facility 데이터 처리
g1 = folium.plugins.FeatureGroupSubGroup(fg, name='sports_facility')
map_folium.add_child(g1)
marker_cluster = MarkerCluster(options=default_marker_cluster_options).add_to(g1)
for idx, row in data['sports_facility'].iterrows():
    folium.Marker(
        location=[row.geometry.y, row.geometry.x],
        popup=folium.Popup(f"sports_facility: {row['시설주소']}", max_width=300),
        icon=folium.Icon(color=color_map['sports_facility'], icon=icon_map['sports_facility'])
    ).add_to(marker_cluster)

# 2. trash_bin_2024 데이터 처리
g2 = folium.plugins.FeatureGroupSubGroup(fg, name='trash_bin_2024')
map_folium.add_child(g2)
marker_cluster = MarkerCluster(options=default_marker_cluster_options).add_to(g2)
for idx, row in data['trash_bin_2024'].iterrows():
    folium.Marker(
        location=[row.geometry.y, row.geometry.x],
        popup=folium.Popup(f"trash_bin_2024: {row['address']}", max_width=300),
        icon=folium.Icon(color=color_map['trash_bin_2024'], icon=icon_map['trash_bin_2024'])
    ).add_to(marker_cluster)

# 3. parking_lot 데이터 처리
g3 = folium.plugins.FeatureGroupSubGroup(fg, name='parking_lot')
map_folium.add_child(g3)
marker_cluster = MarkerCluster(options=default_marker_cluster_options).add_to(g3)
for idx, row in data['parking_lot'].iterrows():
    folium.Marker(
        location=[row.geometry.y, row.geometry.x],
        popup=folium.Popup(f"parking_lot: {row['주소']}", max_width=300),
        icon=folium.Icon(color=color_map['parking_lot'], icon=icon_map['parking_lot'])
    ).add_to(marker_cluster)

# 4. school 데이터 처리
g4 = folium.plugins.FeatureGroupSubGroup(fg, name='school')
map_folium.add_child(g4)
marker_cluster = MarkerCluster(options=default_marker_cluster_options).add_to(g4)
for idx, row in data['school'].iterrows():
    folium.Marker(
        location=[row.geometry.y, row.geometry.x],
        popup=folium.Popup(f"school: {row['학교명']}", max_width=300),
        icon=folium.Icon(color=color_map['school'], icon=icon_map['school'])
    ).add_to(marker_cluster)

# 5. bus_stop 데이터 처리 (줌 제한: 18)
g5 = folium.plugins.FeatureGroupSubGroup(fg, name='bus_stops')
map_folium.add_child(g5)
marker_cluster = MarkerCluster(options=bus_stop_marker_cluster_options).add_to(g5)
for idx, row in data['bus_stops'].iterrows():
    folium.Marker(
        location=[row.geometry.y, row.geometry.x],
        popup=folium.Popup(f"bus_stops: {row['정류소명']}", max_width=300),
        icon=folium.Icon(color=color_map['bus_stops'], icon=icon_map['bus_stops'])
    ).add_to(marker_cluster)

# 6. legal_boundary 데이터 (구 경계 데이터) 추가
folium.GeoJson(
    legal_boundary,
    name="구",  # 레이어 이름
    style_function=lambda feature: {
        'fillColor': '#00b493',        # 영역 채우기 색상 (은색)
        'color': '#007c65',             # 경계선 색상
        'weight': 2,                  # 경계선 두께
        'fillOpacity': 0.2            # 영역 투명도
    },
    tooltip=folium.GeoJsonTooltip(fields=['SIG_KOR_NM'], aliases=['구:'])  # 팁으로 구 이름 표시
).add_to(map_folium)

# 레이어 컨트롤 추가
folium.LayerControl(collapsed=False).add_to(map_folium)

# 결과 저장
map_folium.save("map_boundary_layers_with_bus_stop_zoom.html")
print("지도 저장 완료: map_boundary_layers_with_bus_stop_zoom.html")