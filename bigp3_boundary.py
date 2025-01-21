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
from shapely.geometry import Point, LineString, MultiLineString, Polygon
import webbrowser
import json
import geopandas as gpd
from math import radians, sin, cos, sqrt, atan2

try:
    import geopandas as gpd
except ImportError:
    print("경고: geopandas가 설치되어 있지 않아 공간 연산이 제한될 수 있습니다.")

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
# (1) 시계열 예측을 위한 간단한 모델
# --------------------------------------------------------------------------------

class SimpleModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=16):  # hidden_size 축소
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])

def predict_future(model, data_seq, future_steps=8):
    model.eval()
    last_seq = torch.FloatTensor(data_seq[-5:]).reshape(1, -1, 1)
    predicted = []
    
    for _ in range(future_steps):
        with torch.no_grad():
            next_val = model(last_seq).item()
        predicted.append(next_val)
        last_seq = torch.cat([last_seq[:, 1:], torch.FloatTensor([[[next_val]]])], dim=1)
    
    return predicted

# --------------------------------------------------------------------------------
# (2) 거리 계산 함수
# --------------------------------------------------------------------------------

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # 지구의 반경 (km)
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    
    return distance * 1000  # 미터 단위로 변환

# --------------------------------------------------------------------------------
# (3) 쓰레기통 배치 점수 계산
# --------------------------------------------------------------------------------

def calculate_facility_score(target_coords, facility_coords, max_distance=500):
    """
    시설물과의 거리 기반 점수 계산
    max_distance: 최대 고려 거리 (미터)
    """
    if len(facility_coords) == 0:
        return 0
    
    tree = cKDTree(facility_coords)
    distances, _ = tree.query(target_coords, k=3)  # 가장 가까운 3개 시설 고려
    
    # 거리에 따른 점수 계산 (가까울수록 높은 점수)
    scores = np.where(distances < max_distance, 
                     1 - (distances / max_distance), 
                     0)
    return np.mean(scores)  # 평균 점수 반환

def calculate_region_dispersion(target_coord, existing_scores, all_coords):
    """
    지역 분산도 점수 계산
    """
    if len(existing_scores) == 0:
        return 0
    
    # 모든 좌표에 대해 거리 계산
    tree = cKDTree(all_coords)
    distances, _ = tree.query([target_coord], k=2)  # k=2로 자신을 제외한 가장 가까운 위치 찾기
    
    # 가장 가까운 다른 위치와의 거리를 기준으로 점수 계산
    # (첫 번째는 자기 자신이므로 두 번째 값 사용)
    distance = distances[0][1]
    
    # 거리가 멀수록 높은 점수 (최대 1000m까지)
    dispersion_score = min(distance * 1000, 1000) / 1000 * 0.2
    return dispersion_score

def compute_bin_scores(df_bins, facilities_dict, forecast_value):
    """
    각종 시설물과의 거리를 고려한 종합 점수 계산
    """
    scores = []
    bin_coords = df_bins[['latitude', 'longitude']].values
    bin_tree = cKDTree(bin_coords)
    
    for idx, row in df_bins.iterrows():
        target_coords = np.array([[row['latitude'], row['longitude']]])
        
        # 1. 기본 점수 (예측 배출량 기반)
        total_score = forecast_value * 0.3
        
        # 2. 기존 쓰레기통과의 거리 점수 (최소 거리 제한 강화)
        neighbors = bin_tree.query_ball_point(target_coords[0], r=0.005)  # 약 500m로 증가
        if len(neighbors) > 1:  # 자기 자신 제외
            total_score = -np.inf  # 다른 쓰레기통이 너무 가까우면 제외
            scores.append(total_score)
            continue
        
        # 3. 각 시설물과의 거리 점수
        # 주차장 점수 (20% 가중치)
        parking_score = calculate_facility_score(
            target_coords, 
            facilities_dict['parking'][['latitude', 'longitude']].values
        ) * 0.2
        
        # 체육시설 점수 (20% 가중치)
        sports_score = calculate_facility_score(
            target_coords, 
            facilities_dict['sports'][['latitude', 'longitude']].values
        ) * 0.2
        
        # 버스정류소 점수 (30% 가중치)
        bus_score = calculate_facility_score(
            target_coords, 
            facilities_dict['bus_stops'][['Y좌표', 'X좌표']].values
        ) * 0.3
        
        # 4. 지역 분산도 점수 추가
        region_score = calculate_region_dispersion(target_coords[0], scores, bin_coords)
        
        # 종합 점수 계산
        facility_score = parking_score + sports_score + bus_score
        total_score = total_score + facility_score + region_score
        
        scores.append(total_score)
    
    return np.array(scores)

def select_diverse_locations(df_bins, scores, n=10, min_distance=500):
    """
    분산된 위치 선택
    """
    selected_indices = []
    remaining_indices = set(range(len(df_bins)))
    
    while len(selected_indices) < n and remaining_indices:
        # 남은 위치 중 가장 높은 점수를 가진 위치 선택
        best_score = -np.inf
        best_idx = None
        
        for idx in remaining_indices:
            if scores[idx] > best_score:
                # 이미 선택된 위치들과의 거리 확인
                if not selected_indices or all(
                    calculate_distance(
                        df_bins.iloc[idx]['latitude'],
                        df_bins.iloc[idx]['longitude'],
                        df_bins.iloc[sel_idx]['latitude'],
                        df_bins.iloc[sel_idx]['longitude']
                    ) >= min_distance
                    for sel_idx in selected_indices
                ):
                    best_score = scores[idx]
                    best_idx = idx
        
        if best_idx is None:
            break
            
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
    
    return selected_indices

# --------------------------------------------------------------------------------
# (4) 메인 실행 함수
# --------------------------------------------------------------------------------

def load_existing_bins():
    # 기존 쓰레기통 데이터 로드
    bins_df = pd.read_csv('2024_서울시_쓰레기통_좌표.csv', encoding='utf-8')
    # 중복된 좌표 제거 (동일 위치에 여러 개의 쓰레기통이 있는 경우)
    bins_df = bins_df.drop_duplicates(subset=['latitude', 'longitude'])
    # 강동구 데이터만 필터링
    bins_df = bins_df[bins_df['address'].str.contains('강동구', na=False)]
    return bins_df[['latitude', 'longitude']]

def calculate_distance_penalty(point, existing_bins):
    # 기존 쓰레기통과의 거리에 따른 페널티 계산
    distances = []
    for _, bin_location in existing_bins.iterrows():
        dist = calculate_distance(
            point[0], point[1],
            bin_location['latitude'], bin_location['longitude']
        )
        distances.append(dist)
    
    min_distance = min(distances) if distances else float('inf')
    if min_distance < 300:  # 300m 이내에 기존 쓰레기통이 있는 경우
        return float('-inf')  # 매우 낮은 점수 부여
    return 0  # 300m 이상 떨어진 경우 페널티 없음

def evaluate_location(point, population_data, floating_data, existing_bins):
    # ... existing code ...
    
    # 기존 쓰레기통과의 거리 페널티 추가
    distance_penalty = calculate_distance_penalty(point, existing_bins)
    if distance_penalty == float('-inf'):
        return float('-inf')
    
    total_score = population_score + floating_score
    return total_score

def main():
    # 데이터 로드
    parking = pd.read_csv('주차장 좌표.csv')
    sports = pd.read_csv('체육시설 좌표.csv')
    bus_stops = pd.read_csv('서울시 버스정류소 위치정보.csv', encoding='cp949')
    existing_bins = pd.read_csv('2024_서울시_쓰레기통_좌표.csv')
    
    # 강동구 경계 좌표
    GANGDONG_BOUNDS = {
        'min_lat': 37.52,
        'max_lat': 37.57,
        'min_lon': 127.12,
        'max_lon': 127.19
    }
    
    # 더 조밀한 그리드 포인트 생성 (50m 간격)
    lat_steps = np.linspace(GANGDONG_BOUNDS['min_lat'], GANGDONG_BOUNDS['max_lat'], 200)
    lon_steps = np.linspace(GANGDONG_BOUNDS['min_lon'], GANGDONG_BOUNDS['max_lon'], 200)
    grid_points = []
    
    # 강동구 기존 쓰레기통
    gangdong_bins = existing_bins[existing_bins['address'].str.contains('강동구', na=False)]
    existing_coords = gangdong_bins[['latitude', 'longitude']].values
    
    # 기존 쓰레기통과 일정 거리 이상 떨어진 그리드 포인트만 선택
    for lat in lat_steps:
        for lon in lon_steps:
            point = np.array([lat, lon])
            # 기존 쓰레기통과의 최소 거리 확인
            if len(existing_coords) > 0:
                distances = np.sqrt(np.sum((existing_coords - point)**2, axis=1))
                min_distance = np.min(distances)
                if min_distance > 0.002:  # 약 200m
                    grid_points.append([lat, lon])
            else:
                grid_points.append([lat, lon])
    
    # 그리드 포인트를 DataFrame으로 변환
    candidate_locations = pd.DataFrame(grid_points, columns=['latitude', 'longitude'])
    
    # 시설물 데이터 딕셔너리 생성
    facilities_dict = {
        'parking': parking[parking['주소'].str.contains('강동구', na=False)],
        'sports': sports[sports['시설주소'].str.contains('강동구', na=False)],
        'bus_stops': bus_stops[bus_stops['정류소명'].str.contains('강동구', na=False)]
    }
    
    # 각 후보 위치의 점수 계산
    scores = []
    for idx, row in candidate_locations.iterrows():
        score = 0
        point = np.array([[row['latitude'], row['longitude']]])
        
        # 주차장 점수
        parking_score = calculate_facility_score(
            point, 
            facilities_dict['parking'][['latitude', 'longitude']].values
        ) * 0.3
        
        # 체육시설 점수
        sports_score = calculate_facility_score(
            point, 
            facilities_dict['sports'][['latitude', 'longitude']].values
        ) * 0.3
        
        # 버스정류소 점수
        bus_score = calculate_facility_score(
            point, 
            facilities_dict['bus_stops'][['Y좌표', 'X좌표']].values
        ) * 0.4
        
        total_score = parking_score + sports_score + bus_score
        scores.append(total_score)
    
    candidate_locations['score'] = scores
    
    # 점수로 정렬하고 상위 100개 선택
    top_locations = candidate_locations.nlargest(100, 'score')
    
    print("\n100개 추천 위치:")
    for idx, row in top_locations.iterrows():
        print(f"위도: {row['latitude']:.6f}, 경도: {row['longitude']:.6f}, 점수: {row['score']:.2f}")
    
    # 지도에 표시
    m = folium.Map(
        location=[37.545, 127.155],
        zoom_start=13
    )

    # 기존 쓰레기통 표시 (파란색)
    for _, row in gangdong_bins.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=6,
            color='blue',
            popup='기존 쓰레기통',
            fill=True,
            fill_opacity=0.7
        ).add_to(m)

    # 추천 위치 표시 (빨간색)
    for _, row in top_locations.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=6,
            color='red',
            popup=f'추천 위치 (점수: {row["score"]:.2f})',
            fill=True,
            fill_opacity=0.7
        ).add_to(m)

    # 범례 추가
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color:white;
                padding: 10px;
                border-radius: 5px;">
        <p><i class="fa fa-circle" style="color:blue"></i> 기존 쓰레기통</p>
        <p><i class="fa fa-circle" style="color:red"></i> 추천 설치 위치</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # 지도 저장
    m.save('gangdong_trash_bins.html')
    
    print(f"\n총 추천 위치 수: {len(top_locations)}")
    return top_locations

if __name__ == "__main__":
    result_df = main()
