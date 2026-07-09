import os
import json
from collections import defaultdict

# 결과 저장할 경로 설정
# output_dir = "visualization6"  # 기존 visualization 폴더와 같은 위치에 저장
# os.makedirs(output_dir, exist_ok=True)

# JSON 파일 경로 설정 - 입력 파일 경로
input_json_path = "/home/lee/research/Research2025/YOLOv11/visualization_number_vehicle_7/received_file_20240822_101524/json/vehicle_data.json"
# 출력 파일 경로
output_json_path = input_json_path.replace(".json", "_updated.json")
# 변경된 object_id 목록 저장할 파일 경로
changed_ids_path = input_json_path.replace(".json", "_changed_ids.txt")

def main():
    print(f"JSON 파일 로딩 중: {input_json_path}")
    
    # JSON 파일 로드
    with open(input_json_path, 'r') as f:
        vehicle_data = json.load(f)
    
    print(f"총 {len(vehicle_data)}개의 데이터 로드 완료")
    
    # 데이터를 object_id별로 그룹화
    vehicles_by_id = defaultdict(list)
    for item in vehicle_data:
        object_id = item["object_id"]
        vehicles_by_id[object_id].append(item)
    
    print(f"총 {len(vehicles_by_id)}개의 차량 ID 발견")
    
    # 각 차량별로 프레임 순서대로 정렬
    for object_id, frames in vehicles_by_id.items():
        frames.sort(key=lambda x: x["frame"])
    
    # 변경된 object_id 목록
    changed_object_ids = set()
    
    # 각 차량별로 차선 변경 시점 찾기
    for object_id, frames in vehicles_by_id.items():
        # 이전 차선 ID 초기화
        prev_lane_id = None
        
        # 프레임을 순회하며 lane_id 변경 시점 찾기
        for i, frame_data in enumerate(frames):
            current_lane_id = frame_data["lane_id"]
            
            # 차선 ID가 변경되었고, 현재와 이전 lane_id가 비어있지 않은 경우
            if prev_lane_id is not None and current_lane_id != "" and prev_lane_id != "" and current_lane_id != prev_lane_id:
                print(f"차량 ID {object_id}: 차선 변경 감지 - {prev_lane_id} -> {current_lane_id} (프레임 {frame_data['frame']})")
                
                # 현재 프레임 인덱스
                current_index = i
                
                # 변경 전 6프레임부터 변경 후 6프레임까지 category 수정
                for j in range(max(0, current_index - 6), min(len(frames), current_index + 7)):
                    # "stop" 카테고리는 변경하지 않음
                    if frames[j]["category"] != "stop":
                        frames[j]["category"] = "lane_change"
                        changed_object_ids.add(object_id)
            
            # 현재 차선 ID를 이전 차선 ID로 업데이트
            prev_lane_id = current_lane_id
    
    # 수정된 데이터를 원래 구조로 다시 변환
    updated_vehicle_data = []
    for frames in vehicles_by_id.values():
        updated_vehicle_data.extend(frames)
    
    # 데이터를 frame 순서대로 정렬 (원래 순서 유지)
    updated_vehicle_data.sort(key=lambda x: x["frame"])
    
    # 수정된 데이터 JSON 파일로 저장
    with open(output_json_path, 'w') as f:
        json.dump(updated_vehicle_data, f, indent=2)
    
    print(f"업데이트된 JSON 저장 완료: {output_json_path}")
    
    # 변경된 object_id 목록 저장
    with open(changed_ids_path, 'w') as f:
        for object_id in sorted(changed_object_ids):
            f.write(f"{object_id}\n")
    
    print(f"변경된 object_id 목록 저장 완료: {changed_ids_path}")
    print(f"총 {len(changed_object_ids)}개의 차량에서 차선 변경 감지")

if __name__ == "__main__":
    main()
