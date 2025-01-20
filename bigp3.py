#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bigp3.py

이 스크립트는 다음 작업을 수행합니다:
1) '배출량.csv' 데이터를 불러와 구 단위 쓰레기 배출량의 과거 시계열을 구성하고,
   Transformer 기반 모델을 이용해 향후 8년(예: 2023~2030년) 배출량을 예측합니다.
2) 'data/2024_서울시_쓰레기통_좌표.csv' (기존 쓰레기통) 정보를 로드하여,
   리밸런싱/신규 설치가 필요한 쓰레기통 위치(후보 지점)를 선정합니다.
3) '인도4.csv'를 기반으로 보도(인도) 네트워크를 구성하고, 
   네트워크 최단 거리(보행 거리)를 통해 기존 쓰레기통 간 거리를 계산합니다
   (유클리드 거리 대신 네트워크 거리를 활용).
4) 'data/도로명주소 (1).csv'를 활용하여 최종 산출된 쓰레기통 위치에
   근접한 도로명주소를 우선적으로 추출(간이 Reverse Geocoding)합니다.
5) 구별 최소 설치 개수(10개) 포함, 최종적으로 1,500개 지점을 결정합니다.
6) 최종 결과를 CSV와 Folium HTML 지도로 각각 산출하며,
   기존 쓰레기통, 재배치 (rebalanced), 신규 설치 (new)를 구분 표기합니다.

주의:
- PyTorch, NetworkX, Folium, GeoPandas, shapely 등의 라이브러리가 설치되어 있어야 합니다.
- 실제 대규모 데이터에 대해 50m 간격 샘플링을 수행하면 많은 노드가 생길 수 있으니
  성능 고려가 필요합니다.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import networkx as nx
import folium
from scipy.spatial import cKDTree

try:
    import geopandas as gpd
    from shapely.geometry import Point, LineString, MultiLineString
except ImportError:
    print("경고: geopandas나 shapely가 설치되어 있지 않아 공간 연산이 제한될 수 있습니다.")

# --------------------------------------------------------------------------------
# (0) 하이퍼파라미터 & 설정
# --------------------------------------------------------------------------------

# 시계열 예측 범위 (8년)
PREDICT_YEARS = 8  # 2023 ~ 2030

# 최종적으로 설치할 쓰레기통 목표 개수
TARGET_NEW_BINS = 1500

# 구별 최소 설치 개수 (강제)
MIN_BINS_PER_DISTRICT = 10

# 네트워크 거리 계산 시 샘플링 간격 (m)
SAMPLING_INTERVAL = 50

# 이상적 거리 (기존 쓰레기통과 200m 정도가 이상적)
IDEAL_DISTANCE_M = 200

# Folium 지도 초기 위치(서울 시청 기준쯤)
MAP_CENTER_LAT = 37.5665
MAP_CENTER_LNG = 126.9780
MAP_ZOOM_START = 11

# --------------------------------------------------------------------------------
# (1) 배출량.csv 불러와 시계열 예측 (Transformer)
# --------------------------------------------------------------------------------

class TrashDataset(Dataset):
    """
    구 단위 연도별 배출량 시계열을 Transformer에 공급하기 위한 간단한 PyTorch Dataset
    """
    def __init__(self, seq_data, input_size=1, window=5, predict=1):
        self.seq_data = seq_data
        self.window = window
        self.predict = predict
        self.input_size = input_size

    def __len__(self):
        return len(self.seq_data) - self.window - self.predict + 1

    def __getitem__(self, idx):
        x = self.seq_data[idx : idx + self.window]
        y = self.seq_data[idx + self.window : idx + self.window + self.predict]
        return x.astype(np.float32), y.astype(np.float32)

class SimpleTransformer(nn.Module):
    """
    연 단위 시계열 예측을 위한 간단한 Transformer 예시
    """
    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(d_model, 1)
        
        # 포지셔널 인코딩 대신 간단히 연도 index에 대한 Embedding
        self.pos_embed = nn.Embedding(200, d_model)  # 최대 200년 정도로 설정

    def forward(self, x_in):
        # x_in: (batch, seq_len, 1)
        b, s, _ = x_in.shape
        positions = torch.arange(s).unsqueeze(0).repeat(b,1).to(x_in.device)  # (batch, seq_len)
        embed = self.pos_embed(positions)  # (batch, seq_len, d_model)

        # 값(배출량)을 임베딩(단순 확장)
        val_embed = x_in.repeat(1,1,embed.shape[-1])  # (b, s, d_model)
        src = embed + val_embed

        encoded = self.transformer_encoder(src)  # (b, s, d_model)
        out = self.linear(encoded[:, -1, :])      # 마지막 시점만 예측
        return out

def predict_future(model, data_seq, future_steps=8, window=5, device='cpu'):
    """
    마지막 window 구간을 기반으로 future_steps만큼 순차 예측
    """
    model.eval()
    predicted_values = []
    current_seq = data_seq[-window:].copy()  # (window, 1)

    for _ in range(future_steps):
        inp = torch.from_numpy(current_seq).unsqueeze(0).float().to(device)
        with torch.no_grad():
            out = model(inp)
        pred_val = out.item()
        predicted_values.append(pred_val)

        next_seq = np.vstack([current_seq[1:], [[pred_val]]])
        current_seq = next_seq
    return predicted_values

def run_transformer_forecast(df_all, target_district='종로구', device='cpu'):
    """
    df_all: 배출량.csv 전체
    target_district: 특정 구를 예시로 시계열 예측
    """
    df_filtered = df_all.loc[df_all['자치구별(2)'] == target_district].sort_values('시점')
    arr = df_filtered['배출량(C) (톤/일)'].fillna(method='ffill').fillna(method='bfill').values
    if len(arr) < 10:
        return None

    window = 5
    ds = TrashDataset(arr.reshape(-1,1), window=window, predict=1)
    dl = DataLoader(ds, batch_size=4, shuffle=True)

    model = SimpleTransformer()
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # 간단히 50 epoch 학습
    for epoch in range(50):
        model.train()
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y.squeeze(-1))
            loss.backward()
            optim.step()

    # 예측
    future_pred = predict_future(model, arr.reshape(-1,1), future_steps=PREDICT_YEARS,
                                 window=window, device=device)
    last_year = int(df_filtered['시점'].max())  # 예: 2022
    future_years = [y for y in range(last_year+1, last_year+1+PREDICT_YEARS)]
    forecast_df = pd.DataFrame({
        'year': future_years,
        'predicted_ton_per_day': future_pred
    })
    return forecast_df

# --------------------------------------------------------------------------------
# (2) 데이터 불러오기 & 네트워크 구성
# --------------------------------------------------------------------------------

def load_datasets():
    df_trash = pd.read_csv('배출량.csv', encoding='utf-8', na_values=['',' '])
    df_bins = pd.read_csv('data/2024_서울시_쓰레기통_좌표.csv', encoding='utf-8', na_values=['',' '])
    df_addr = pd.read_csv('data/도로명주소 (1).csv', encoding='utf-8', na_values=['',' '])
    return df_trash, df_bins, df_addr

def build_sidewalk_graph(sidewalk_file='data/인도4.csv', sampling_interval=50.0):
    """
    인도(보도) 데이터(GeoJSON, Shapefile, CSV 등)에 담긴 LineString(혹은 MultiLineString)을
    networkx 그래프로 변환한다.
    여기서는 CSV 형식으로 제공된 '인도4.csv'를 불러와 MultiLineString geometry를 파싱한다고 가정.
    (실제 인도4.csv 내부 구조나 좌표계 차이로 인해 별도 보정이 필요할 수 있음)

    - sampling_interval: m 단위로 샘플링 점을 생성
    - returns: G (networkx.Graph)
    """
    try:
        df_side = pd.read_csv(sidewalk_file, encoding='utf-8')
    except:
        print(f"인도4.csv 읽기 오류. '{sidewalk_file}' 경로나 인코딩을 확인하세요.")
        return None

    if 'geometry_type' not in df_side.columns or 'coordinates' not in df_side.columns:
        print("인도4.csv에 geometry_type, coordinates 정보가 없습니다.")
        return None

    # GeoDataFrame으로 변환 시도
    lines_list = []
    for idx, row in df_side.iterrows():
        gtype = row['geometry_type']
        coords_str = row['coordinates']
        # 예: "MultiLineString, [[[x1,y1], [x2,y2], ...]]]"
        # 여기서 실제 파싱 로직은 데이터 구조에 따라 달라짐
        if not isinstance(coords_str, str):
            continue
        # 간단 파싱 예시 (현실에서는 json.loads 등 사용 권장)
        # coords_str가 "[[[128.31,37.96], [128.32,37.96], ...]]" 형태라고 가정
        import ast
        try:
            coords = ast.literal_eval(coords_str)
        except:
            coords = None
        
        if coords is None:
            continue

        # MultiLineString 구조일 수 있으니, 각 LineString 별로 처리
        # coords ~ [[[x1,y1],[x2,y2],...]] or [[...],[...]]
        if gtype == 'MultiLineString':
            for subline in coords:
                # subline: [[x1,y1],[x2,y2],...]
                lines_list.append(LineString(subline))
        elif gtype == 'LineString':
            lines_list.append(LineString(coords))
        else:
            # 다른 geometry_type이면 생략
            pass

    if len(lines_list) == 0:
        print("유효한 인도 라인이 없음.")
        return None

    G = nx.Graph()
    # 샘플링하여 노드/엣지 추가
    for ls in lines_list:
        length_m = ls.length  # shapely length (좌표계가 EPSG:4326이면 단위가 degree임)
        # 주의: 인도4.csv가 경위도 좌표계이면 ls.length는 degree 단위.
        # 정확한 m 단위 계산을 위해선 좌표계를 투영해야 함(예: EPSG:5179).
        # 여기서는 간단히 haversine 기반으로 분절 길이를 계산하겠습니다.

        # ls를 sampling_interval 간격으로 분할
        dist_acc = 0.0
        prev_pt = None
        pts = []
        while dist_acc < ls.length:
            pt = ls.interpolate(dist_acc/ls.length, normalized=True)
            pts.append(pt)
            dist_acc += sampling_interval
        # 마지막 점(라인 끝) 포함
        pts.append(ls.interpolate(1.0, normalized=True))

        # pts를 순회하며 노드 추가, 인접 노드 연결
        # 노드 id는 (lat, lon) tuple 로 간단 표기
        # 실제론 고유 id 필요
        prev_node = None
        for p in pts:
            lat, lon = p.y, p.x  # shapely: (x=경도, y=위도) 식
            node_id = (round(lat, 7), round(lon, 7))  # 반올림
            if node_id not in G:
                G.add_node(node_id)
            if prev_node is not None:
                segment_dist = haversine(prev_node[0], prev_node[1],
                                         node_id[0], node_id[1])
                G.add_edge(prev_node, node_id, weight=segment_dist)
            prev_node = node_id
    return G

def haversine(lat1, lon1, lat2, lon2):
    """
    위도경도(deg) -> m 단위 haversine 거리
    """
    from math import radians, sin, cos, atan2, sqrt
    R = 6371e3
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    c = 2*atan2(sqrt(a), sqrt(1-a))
    return R*c

def connect_point_to_graph(G, lat, lon):
    """
    (lat,lon)에 가장 가까운 노드를 찾아서 연결 에지를 하나 추가하고,
    그 노드 id를 return.
    """
    # 매우 단순하게 모든 노드를 순회 => 느림
    # 실제로는 spatial index(KDTree) 사용 권장
    min_d = 999999999
    closest_node = None
    for node in G.nodes():
        d = haversine(lat, lon, node[0], node[1])
        if d < min_d:
            min_d = d
            closest_node = node
    # (lat,lon)을 새 노드로 추가하고, closest_node와 연결
    if closest_node is None:
        return None

    new_node_id = (round(lat,7), round(lon,7))
    if new_node_id not in G:
        G.add_node(new_node_id)
    G.add_edge(new_node_id, closest_node, weight=min_d)
    return new_node_id

def network_distance(G, lat1, lon1, lat2, lon2):
    """
    (lat1,lon1) ~ (lat2,lon2) 네트워크 최단거리
    """
    node_a = connect_point_to_graph(G, lat1, lon1)
    node_b = connect_point_to_graph(G, lat2, lon2)
    if (node_a is None) or (node_b is None):
        return 9999999
    try:
        dist = nx.shortest_path_length(G, source=node_a, target=node_b, weight='weight')
        return dist
    except nx.NetworkXNoPath:
        return 9999999

# --------------------------------------------------------------------------------
# (3) 구 단위 배출량 예측 & 통합
# --------------------------------------------------------------------------------

def forecast_all_districts(df_trash, device='cpu'):
    districts = df_trash['자치구별(2)'].unique()
    forecast_results = {}
    for dist in districts:
        if pd.isna(dist) or dist == '소계':
            continue
        f_df = run_transformer_forecast(df_trash, dist, device=device)
        forecast_results[dist] = f_df
    return forecast_results

# --------------------------------------------------------------------------------
# (4) 쓰레기통 리밸런싱 & 신규 배치 점수 계산 (네트워크 거리 사용)
# --------------------------------------------------------------------------------

def compute_bin_scores_fast(df_bins, df_trash_forecast, G_sidewalk):
    """
    df_bins: 기존 쓰레기통 위치(혹은 후보 위치)
    df_trash_forecast: 구별 예측 배출량
    G_sidewalk: 보도 네트워크 그래프
    
    개선사항:
    1) KDTree를 사용해 일정 범위(예: 500m) 내 후보끼리만 거리 계산.
    2) 나머지는 거리가 멀어 가점이 거의 없다고 간주 (혹은 0점 처리)
    """
    candidate_bins = []

    # (기존 구 매핑 로직 그대로)
    for idx, row in df_bins.iterrows():
        lat = row['latitude']
        lng = row['longitude']
        address = row['address'] if 'address' in row else ''

        to_gu = None
        for gu_candidate in df_trash_forecast.keys():
            if isinstance(address, str) and (gu_candidate in address):
                to_gu = gu_candidate
                break

        if to_gu is None:
            future_val = 1000.0
            base_score = 1.0
        else:
            fdf = df_trash_forecast[to_gu]
            if fdf is not None:
                future_val = fdf['predicted_ton_per_day'].iloc[-1]
            else:
                future_val = 1000.0
            base_score = float(future_val) / 10.0

        candidate_bins.append({
            'lat': lat,
            'lng': lng,
            'address_text': address,
            'gu': to_gu,
            'base_score': base_score
        })

    df_candidates = pd.DataFrame(candidate_bins)

    # KDTree 구성 (유클리드 근사)
    coords = df_candidates[['lat','lng']].values
    tree = cKDTree(coords)
    
    # 예: 500m 이내 후보끼리만 네트워크 거리 계산
    # 나머지는 거리가 멀어 가점이 거의 없다고 간주 (혹은 0점 처리)
    search_radius = 500.0
    n = len(coords)
    
    distance_scores = np.zeros(n, dtype=np.float32)

    for i in range(n):
        lat_i, lng_i = coords[i]
        # KDTree로 이웃 찾기
        idxes = tree.query_ball_point([lat_i, lng_i], r=search_radius)
        
        dist_sum = 0.0
        dist_count = 0
        
        for j in idxes:
            if i == j:
                continue
            lat_j, lng_j = coords[j]
            # 네트워크 거리
            d_ij = network_distance(G_sidewalk, lat_i, lng_i, lat_j, lng_j)
            
            dist_diff = abs(d_ij - IDEAL_DISTANCE_M)
            local_score = max(0, (300 - dist_diff)/300)  # 단순 예시
            dist_sum += local_score
            dist_count += 1
        
        if dist_count > 0:
            distance_scores[i] = dist_sum / dist_count
        else:
            distance_scores[i] = 0.0

    df_candidates['distance_score'] = distance_scores
    df_candidates['final_score'] = df_candidates['base_score'] + df_candidates['distance_score']
    return df_candidates

# --------------------------------------------------------------------------------
# (5) 최종 1,500개 선정 + 구별 최소 배치 + 재배치/신규 표기
# --------------------------------------------------------------------------------

def allocate_bins(df_candidates, existing_bin_count, target_new_bins=1500):
    # 구별 그룹
    grouped = df_candidates.groupby('gu')
    results = []

    for gu_name, subdf in grouped:
        subdf_sorted = subdf.sort_values('final_score', ascending=False).reset_index(drop=True)
        chunk = subdf_sorted.iloc[:MIN_BINS_PER_DISTRICT].copy()
        chunk['allocation_rank'] = range(1, len(chunk)+1)
        chunk['gu_min_alloc'] = True
        results.append(chunk)

    df_min_alloc = pd.concat(results, ignore_index=True)
    allocated_so_far = len(df_min_alloc)
    df_min_alloc_idx = set(df_min_alloc.index)

    df_remaining = df_candidates.drop(index=df_min_alloc_idx).sort_values('final_score', ascending=False)
    slots_left = target_new_bins - allocated_so_far

    if slots_left < 0:
        # 최소 배치만으로 이미 초과
        df_min_alloc = df_min_alloc.sort_values('final_score', ascending=False)
        df_min_alloc = df_min_alloc.head(target_new_bins)
        df_final = df_min_alloc.copy()
    else:
        df_chosen = df_remaining.iloc[:slots_left].copy()
        df_chosen['allocation_rank'] = range(1, len(df_chosen)+1)
        df_chosen['gu_min_alloc'] = False
        df_final = pd.concat([df_min_alloc, df_chosen], ignore_index=True)

    # rebalanced/new 구분 (간단히 절반씩)
    n_final = len(df_final)
    half_n = n_final // 2
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    df_final.loc[:half_n, 'bin_type'] = 'rebalanced'
    df_final.loc[half_n:, 'bin_type'] = 'new'

    return df_final

# --------------------------------------------------------------------------------
# (6) 도로명주소 매핑 (간이)
# --------------------------------------------------------------------------------

def attach_nearest_road_name(df_bins, df_addr):
    if 'latitude' not in df_addr.columns or 'longitude' not in df_addr.columns:
        df_bins['road_name'] = None
        return df_bins

    bins_with_addr = df_bins.copy()
    addr_coords = df_addr[['latitude','longitude']].values
    bin_coords = bins_with_addr[['lat','lng']].values

    def simple_nearest(bcoord):
        min_d = 999999999
        min_idx = -1
        for idx2, acoord in enumerate(addr_coords):
            d = haversine(bcoord[0], bcoord[1], acoord[0], acoord[1])
            if d<min_d:
                min_d = d
                min_idx = idx2
        return min_idx, min_d

    road_names = []
    for bc in bin_coords:
        n_idx, dist_ = simple_nearest(bc)
        if n_idx<0:
            road_names.append(None)
        else:
            if 'road_address' in df_addr.columns:
                road_val = df_addr.loc[n_idx,'road_address']
            else:
                # 예시: 'address' 컬럼 가정
                road_val = df_addr.loc[n_idx,'address']
            road_names.append(road_val)

    bins_with_addr['road_name'] = road_names
    return bins_with_addr

# --------------------------------------------------------------------------------
# (7) Folium 지도 만들기
# --------------------------------------------------------------------------------

def create_folium_map(df_alloc, output_html='final_map.html'):
    """
    df_alloc에는 lat,lng,bin_type 등이 포함됨.
    bin_type=='rebalanced' -> 주황,
    bin_type=='new' -> 초록,
    기존(existing)은 여기서는 df_alloc에 포함 안 되었으므로 시연용으로 표시하지 않음.
    만약 기존 좌표도 함께 표시하려면 인자로 받아 추가 마커 생성.
    """
    map_ = folium.Map(location=[MAP_CENTER_LAT, MAP_CENTER_LNG],
                      zoom_start=MAP_ZOOM_START)

    colors = {'rebalanced':'orange',
              'new':'green',
              'existing':'blue'}

    for idx, row in df_alloc.iterrows():
        lat = row['lat']
        lng = row['lng']
        btype = row.get('bin_type','new')
        marker_color = colors.get(btype, 'green')
        popup_text = f"Type={btype}, Score={row['final_score']:.2f}, Addr={row.get('road_name','')}"
        folium.CircleMarker(location=[lat,lng],
                            radius=5,
                            color=marker_color,
                            fill=True,
                            fill_opacity=0.8,
                            popup=popup_text).add_to(map_)

    map_.save(output_html)
    print(f"Folium 지도 저장 완료: {output_html}")

# --------------------------------------------------------------------------------
# 메인 실행
# --------------------------------------------------------------------------------

def main():
    device = 'cpu'
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print("1) 데이터 불러오기...")
    df_trash, df_bins_raw, df_addr = load_datasets()

    print("2) 보도(인도) 네트워크 그래프 생성 (인도4.csv 사용) ...")
    G_sidewalk = build_sidewalk_graph(sidewalk_file='data/인도4.csv',
                                      sampling_interval=SAMPLING_INTERVAL)
    if G_sidewalk is None or len(G_sidewalk.nodes)==0:
        print("보도 네트워크 생성 실패. 유클리드 거리로 대체하거나, 스크립트 중단.")
        return

    print("3) 구별 배출량 Transformer 예측 (8년) ...")
    forecast_dict = forecast_all_districts(df_trash, device=device)

    print("4) 후보점 점수 산정 (네트워크 거리 기반, 200m 이상적) ...")
    # 여기서는 '기존 쓰레기통' 위치를 그대로 후보로 가정.
    # 실제로는 인도4.csv를 기반으로 50m 간격 모든 노드에 대해 점수 계산도 가능.
    df_candidates = compute_bin_scores_fast(df_bins_raw, forecast_dict, G_sidewalk)

    print("5) 구별 최소 10개 + 총 1,500개 배치 => rebalanced/new 분류")
    existing_bin_count = {}
    df_alloc = allocate_bins(df_candidates, existing_bin_count,
                            target_new_bins=TARGET_NEW_BINS)

    print("6) 도로명주소 매핑...")
    df_alloc = attach_nearest_road_name(df_alloc, df_addr)

    print("7) CSV 및 지도 시각화 결과 생성...")
    out_csv = 'final_bins_allocation.csv'
    df_alloc.to_csv(out_csv, index=False, encoding='utf-8')
    print(f"저장됨: {out_csv}")

    create_folium_map(df_alloc, output_html='final_map.html')

    print("모든 작업 완료. 'final_bins_allocation.csv', 'final_map.html' 참고 바랍니다.")

if __name__ == '__main__':
    main()
