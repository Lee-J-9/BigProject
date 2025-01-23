import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import folium
from shapely.geometry import Point, LineString
import geopandas as gpd
from folium.plugins import HeatMap

def load_and_preprocess_data(params):
    print("1. 데이터 로드 및 전처리 시작...")
    
    # 서울시 경계
    seoul_bounds = {
        'min_lat': 37.42,
        'max_lat': 37.70,
        'min_lng': 126.76,
        'max_lng': 127.18
    }
    
    print("2. 인도 데이터 처리 중...")
    # 인도 데이터 로드 및 전처리
    sidewalks = pd.read_csv('인도5.csv')
    sidewalk_points = []
    total_rows = len(sidewalks)
    
    for idx, row in enumerate(sidewalks.iterrows(), 1):
        if idx % 10000 == 0:
            print(f"   인도 데이터 처리 진행률: {idx}/{total_rows} ({(idx/total_rows*100):.1f}%)")
        coords = eval(row[1]['coordinates'])[0]
        line = LineString(coords)
        points = [line.interpolate(i/params['point_interval'], normalized=True) 
                 for i in range(int(params['point_interval'])+1)]
        sidewalk_points.extend([(float(p.y), float(p.x)) for p in points if 
                              seoul_bounds['min_lat'] <= p.y <= seoul_bounds['max_lat'] and
                              seoul_bounds['min_lng'] <= p.x <= seoul_bounds['max_lng']])
    
    # 후보 위치 개수 제한
    if len(sidewalk_points) > params['max_candidates']:
        print(f"   후보 위치가 너무 많습니다. {params['max_candidates']}개로 제한합니다.")
        np.random.seed(42)  # 재현성을 위한 시드 설정
        indices = np.random.choice(len(sidewalk_points), 
                                 size=params['max_candidates'], 
                                 replace=False)
        sidewalk_points = [sidewalk_points[i] for i in indices]
    
    print(f"   생성된 후보 위치 수: {len(sidewalk_points)}")
    
    print("3. 기타 시설물 데이터 로드 중...")
    # 기타 데이터 로드
    trash_bins = pd.read_csv('2024_서울시_쓰레기통_좌표.csv')
    sports = pd.read_csv('체육시설 좌표.csv')
    parking = pd.read_csv('주차장 좌표.csv')
    schools = pd.read_csv('서울시_학교_좌표.csv')
    
    print("4. 서울시 데이터 필터링 중...")
    # 서울시 경계 내 데이터만 필터링
    def filter_seoul(df, lat_col, lng_col):
        return df[
            (df[lat_col] >= seoul_bounds['min_lat']) &
            (df[lat_col] <= seoul_bounds['max_lat']) &
            (df[lng_col] >= seoul_bounds['min_lng']) &
            (df[lng_col] <= seoul_bounds['max_lng'])
        ]
    
    trash_bins = filter_seoul(trash_bins, 'latitude', 'longitude')
    sports = filter_seoul(sports, 'latitude', 'longitude')
    parking = filter_seoul(parking, 'latitude', 'longitude')
    schools = filter_seoul(schools, 'latitude', 'longitude')
    
    # sidewalk_points를 DataFrame으로 변환하기 전에 모든 좌표가 float 타입인지 확인
    sidewalk_df = pd.DataFrame(sidewalk_points, columns=['latitude', 'longitude'])
    sidewalk_df['latitude'] = sidewalk_df['latitude'].astype(float)
    sidewalk_df['longitude'] = sidewalk_df['longitude'].astype(float)
    
    return sidewalk_df, trash_bins, sports, parking, schools

def calculate_score(point, facilities, weights, params):
    """각 후보 위치의 점수 계산"""
    scores = {}
    
    # 1. 기존 쓰레기통과의 거리 점수
    dist_to_bins = np.min([haversine_distance((point[0], point[1]), 
                         (float(row['latitude']), float(row['longitude'])))
                         for _, row in facilities['trash_bins'].iterrows()])
    
    # 최소 거리 제한 적용
    if dist_to_bins < params['min_distance_to_existing']:
        return -float('inf')  # 최소 거리 미만이면 해당 위치 제외
    
    scores['bin_dist'] = min(dist_to_bins / params['max_distance_to_existing'], 1.0)
    
    # 2. 주요 시설물과의 근접성 점수
    for facility_type in ['schools', 'sports', 'parking']:
        distances = [haversine_distance((point[0], point[1]), 
                   (float(row['latitude']), float(row['longitude'])))
                   for _, row in facilities[facility_type].iterrows()]
        if distances:
            min_dist = min(distances)
            scores[f'{facility_type}_dist'] = max(1 - (min_dist / params['facility_influence_radius']), 0)
    
    # 가중치 적용 및 총점 계산
    total_score = sum(score * weights[key] for key, score in scores.items())
    return total_score

def haversine_distance(point1, point2):
    """두 지점 간의 거리 계산 (km)"""
    lat1, lon1 = float(point1[0]), float(point1[1])
    lat2, lon2 = float(point2[0]), float(point2[1])
    
    R = 6371  # 지구 반경 (km)
    
    # 라디안으로 변환
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def recommend_locations(n_new_bins, sidewalk_points, facilities, weights, params):
    print("5. 위치 추천 알고리즘 시작...")
    print(f"   총 {len(sidewalk_points)}개의 후보 위치에 대해 계산 중...")
    
    total_points = len(sidewalk_points)
    scores = []
    recommended_points = []
    
    for idx, row in sidewalk_points.iterrows():
        if idx % 1000 == 0:
            print(f"   스코어 계산 진행률: {idx}/{total_points} ({(idx/total_points*100):.1f}%)")
        
        point = (float(row['latitude']), float(row['longitude']))
        
        # 이미 추천된 위치들과의 거리 확인
        if recommended_points:
            min_dist_to_recommended = min(haversine_distance(point, p) 
                                        for p in recommended_points)
            if min_dist_to_recommended < params['min_distance_between_new']:
                continue
        
        score = calculate_score(point, facilities, weights, params)
        if score > -float('inf'):  # 유효한 점수인 경우만 추가
            scores.append({
                'latitude': point[0],
                'longitude': point[1],
                'score': score
            })
            recommended_points.append(point)
    
    scores_df = pd.DataFrame(scores)
    recommended = scores_df.nlargest(n_new_bins, 'score')
    
    return recommended

def visualize_results(trash_bins, recommended_locations):
    print("6. 결과 시각화 중...")
    """결과 시각화"""
    m = folium.Map(
        location=[37.545, 127.085],
        zoom_start=14
    )
    
    # 기존 쓰레기통 표시
    for _, bin in trash_bins.iterrows():
        folium.CircleMarker(
            location=[bin['latitude'], bin['longitude']],
            radius=5,
            color='blue',
            fill=True,
            popup='기존 쓰레기통'
        ).add_to(m)
    
    # 추천 위치 표시 (점수와 함께)
    for _, loc in recommended_locations.iterrows():
        folium.CircleMarker(
            location=[loc['latitude'], loc['longitude']],
            radius=5,
            color='red',
            fill=True,
            popup=f'추천 위치 (점수: {loc["score"]:.2f})'
        ).add_to(m)
    
    # 범례 추가
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px;">
        <p><i class="fa fa-circle" style="color:blue"></i> 기존 쓰레기통</p>
        <p><i class="fa fa-circle" style="color:red"></i> 추천 위치</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def main(n_new_bins):
    print(f"\n=== 서울시 쓰레기통 위치 추천 시스템 시작 (요청 개수: {n_new_bins}개) ===\n")
    
    # 하이퍼 파라미터 설정
    params = {
        'point_interval': 300,              # 후보 위치 간격 (미터)
        'max_candidates': 5000,            # 최대 후보 위치 개수 (서울시 전체를 위해 증가)
        'min_distance_to_existing': 0.3,    # 기존 쓰레기통과의 최소 거리 (km)
        'max_distance_to_existing': 0.5,    # 기존 쓰레기통과의 최대 평가 거리 (km)
        'min_distance_between_new': 0.3,    # 새로운 쓰레기통 간의 최소 거리 (km)
        'facility_influence_radius': 0.3    # 주요 시설물의 영향 반경 (km)
    }
    
    print("하이퍼 파라미터 설정:")
    for key, value in params.items():
        print(f"   - {key}: {value}")
    
    # 데이터 로드 및 전처리
    sidewalk_points, trash_bins, sports, parking, schools = load_and_preprocess_data(params)
    
    # 시설물 데이터 딕셔너리
    facilities = {
        'trash_bins': trash_bins,
        'sports': sports,
        'parking': parking,
        'schools': schools
    }
    
    print("\n가중치 정보:")
    # 가중치 설정
    weights = {
        'bin_dist': 0.4,      # 기존 쓰레기통과의 거리
        'schools_dist': 0.3,  # 학교와의 거리
        'sports_dist': 0.2,   # 체육시설과의 거리
        'parking_dist': 0.1   # 주차장과의 거리
    }
    
    for key, value in weights.items():
        print(f"   - {key}: {value*100}%")
    
    # 위치 추천
    recommended_locations = recommend_locations(n_new_bins, sidewalk_points, facilities, weights, params)
    
    # 시각화
    m = visualize_results(trash_bins, recommended_locations)
    
    # 결과 저장
    m.save('seoul_trash_bins_scored.html')
    
    # CSV 파일로 결과 저장
    recommended_locations.to_csv('recommended_trash_bins.csv', index=False, encoding='utf-8-sig')
    
    print("\n7. 처리 완료!")
    print(f"   - 지도 파일: seoul_trash_bins_scored.html")
    print(f"   - 추천 위치 CSV: recommended_trash_bins.csv")
    print(f"   - 추천된 위치 수: {len(recommended_locations)}")
    
    return recommended_locations

# 실행
new_locations = main(1000)  # 서울시 전체를 위해 추천 개수 증가
print("새로운 쓰레기통 위치 추천 완료!")
