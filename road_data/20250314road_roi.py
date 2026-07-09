import cv2
import json
import numpy as np
import os
import copy

# 전역 변수: 현재 다각형의 점들과 전체 라벨링 정보를 저장합니다.
current_points = []
lane_annotations = []
centerline_points = []  # 중앙선 그리기 위한 점들
selected_lane_index = -1  # 중앙선을 그릴 영역의 인덱스
is_drawing_centerline = False  # 중앙선 그리기 모드 여부
exclusion_zones = []  # 검출 제외 영역 목록
is_drawing_exclusion = False  # 제외 영역 그리기 모드 여부

# 저장 파일 경로 설정
JSON_FILENAME = "lane_annotations.json"
IMAGE_FILENAME = "lane_annotations_result.jpg"

# 색상 생성 함수: 라벨 인덱스에 따라 다른 색상을 반환합니다
def get_color(index):
    colors = [
        (0, 255, 0),    # 녹색
        (255, 0, 0),    # 파란색
        (0, 0, 255),    # 빨간색
        (255, 255, 0),  # 청록색
        (255, 0, 255),  # 자홍색
        (0, 255, 255),  # 노란색
        (128, 0, 0),    # 진한 파란색
        (0, 128, 0),    # 진한 녹색
        (0, 0, 128),    # 진한 빨간색
        (128, 128, 0),  # 올리브색
    ]
    return colors[index % len(colors)]

# 기존 레이블링 파일 불러오기
def load_annotations():
    global exclusion_zones
    if os.path.exists(JSON_FILENAME):
        try:
            with open(JSON_FILENAME, "r") as f:
                data = json.load(f)
                
                # 파일 형식 확인: 새 형식은 dict형태로 "lanes"와 "exclusion_zones"를 포함
                if isinstance(data, dict) and "lanes" in data:
                    exclusion_zones = data.get("exclusion_zones", [])
                    print(f"로드된 제외 영역 수: {len(exclusion_zones)}")
                    # 좌표 디버깅
                    if exclusion_zones and len(exclusion_zones) > 0:
                        first_zone = exclusion_zones[0]
                        print(f"첫 번째 제외 영역: {len(first_zone)}개 점, 첫 점: {first_zone[0]}")
                    return data["lanes"]
                elif isinstance(data, list):
                    if data and isinstance(data[0], dict) and "exclusion_zones" in data[0]:
                        exclusion_zones = data[0]["exclusion_zones"]
                        print(f"기존 형식에서 로드된 제외 영역 수: {len(exclusion_zones)}")
                    else:
                        print("기존 형식에서 제외 영역을 찾을 수 없습니다.")
                    return data
        except Exception as e:
            print(f"파일 불러오기 오류: {e}")
            import traceback
            traceback.print_exc()
    return []

# 모든 영역(차선과 제외영역)을 그려주는 함수
def draw_all_annotations(canvas, annotations, with_fill=True):
    global exclusion_zones
    result = canvas.copy()
    img_height, img_width = canvas.shape[:2]
    
    # 차선 영역 그리기 (정규화된 좌표를 디스플레이 크기로 변환)
    for i, ann in enumerate(annotations):
        pts = np.array(ann["points"], np.int32)
        pts = pts.reshape((-1, 1, 2))
        color = get_color(i)
        
        if with_fill:
            overlay = result.copy()
            cv2.fillPoly(overlay, [pts.reshape(-1, 2)], color)
            alpha = 0.3  # 투명도
            cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
        
        cv2.polylines(result, [pts], isClosed=True, color=color, thickness=2)
        label_pos = (pts[0][0][0] + 10, pts[0][0][1] + 10)
        cv2.putText(result, ann["label"], label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if "centerline" in ann and ann["centerline"] and len(ann["centerline"]) > 1:
            centerline_pts = np.array(ann["centerline"], np.int32)
            for j in range(len(centerline_pts) - 1):
                pt1 = (int(centerline_pts[j][0]), int(centerline_pts[j][1]))
                pt2 = (int(centerline_pts[j+1][0]), int(centerline_pts[j+1][1]))
                cv2.line(result, pt1, pt2, (255, 255, 255), 2)
    
    # 제외 영역 그리기
    for i, zone in enumerate(exclusion_zones):
        # exclusion_zones는 정규화된 좌표로 저장되어 있으므로 변환
        pixel_points = []
        for point in zone:
            if isinstance(point, list) or isinstance(point, tuple):
                x, y = point
                pixel_x = int(x * img_width)
                pixel_y = int(y * img_height)
                pixel_points.append((pixel_x, pixel_y))
        if len(pixel_points) >= 3:
            pts = np.array(pixel_points, np.int32).reshape((-1, 1, 2))
            if with_fill:
                overlay = result.copy()
                cv2.fillPoly(overlay, [pts.reshape(-1, 2)], (0, 0, 255))
                alpha = 0.4
                cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
            cv2.polylines(result, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
            center_x = int(np.mean([p[0] for p in pixel_points]))
            center_y = int(np.mean([p[1] for p in pixel_points]))
            label = f"EZone {i+1}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_pos = (center_x - text_width // 2, center_y + text_height // 2)
            cv2.putText(result, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return result

# 마우스 클릭 이벤트 함수
def click_event(event, x, y, flags, param):
    global current_points, img, original_img, lane_annotations, centerline_points, is_drawing_centerline, selected_lane_index, is_drawing_exclusion, exclusion_zones

    if is_drawing_exclusion:
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
            current_points.append((x, y))
            if len(current_points) > 1:
                cv2.line(img, current_points[-2], current_points[-1], (0, 0, 255), 2)
            if len(current_points) > 2:
                temp_img = img.copy()
                cv2.line(temp_img, current_points[0], current_points[-1], (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow("Image", temp_img)
            else:
                cv2.imshow("Image", img)
    elif is_drawing_centerline:
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img, (x, y), 3, (255, 255, 255), -1)
            centerline_points.append((x, y))
            if len(centerline_points) > 1:
                cv2.line(img, centerline_points[-2], centerline_points[-1], (255, 255, 255), 2)
            cv2.imshow("Image", img)
    else:
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
            current_points.append((x, y))
            if len(current_points) > 1:
                cv2.line(img, current_points[-2], current_points[-1], (0, 0, 255), 2)
            cv2.imshow("Image", img)

# 정규화 여부 확인 함수
def is_coord_normalized(x, y):
    return 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0

# 레이블링 데이터 정규화 함수: 좌표가 정규화되지 않은 경우만 변환
def normalize_annotations(annotations, img_width, img_height):
    normalized = []
    for ann in annotations:
        norm_ann = {"label": ann["label"], "points": []}
        for x, y in ann["points"]:
            if is_coord_normalized(x, y):
                norm_ann["points"].append((x, y))
            else:
                norm_ann["points"].append((x/img_width, y/img_height))
        if "centerline" in ann and ann["centerline"]:
            norm_ann["centerline"] = []
            for x, y in ann["centerline"]:
                if is_coord_normalized(x, y):
                    norm_ann["centerline"].append((x, y))
                else:
                    norm_ann["centerline"].append((x/img_width, y/img_height))
        else:
            norm_ann["centerline"] = []
        normalized.append(norm_ann)
    return normalized

def is_normalized(annotations):
    if not annotations:
        return False
    for ann in annotations:
        for x, y in ann["points"]:
            if x > 1.0 or y > 1.0:
                return False
        if "centerline" in ann and ann["centerline"]:
            for x, y in ann["centerline"]:
                if x > 1.0 or y > 1.0:
                    return False
    return True

# JSON 파일에 레이블링 정보 저장 (항상 새 형식으로 저장)
def save_annotations_json(annotations, filename, exclusion_zones):
    try:
        data = {
            "lanes": annotations,
            "exclusion_zones": exclusion_zones
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        with open(filename, "r") as f:
            saved_data = json.load(f)
            if isinstance(saved_data, dict):
                ez_count = len(saved_data.get("exclusion_zones", []))
                print(f"확인: JSON 파일에 {ez_count}개의 제외 영역이 저장되었습니다.")
            else:
                print("경고: 제외 영역이 저장되지 않았을 수 있습니다.")
        print(f"JSON 파일 저장 완료: {filename}")
    except Exception as e:
        print(f"JSON 파일 저장 오류: {e}")
        import traceback
        traceback.print_exc()

# 중앙선 유효성 확인 함수 추가
def is_centerline_valid(points, centerline):
    """중앙선이 영역 내부 또는 근처에 위치하는지 확인"""
    if not centerline:
        return False
        
    # 영역의 경계 계산
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)
    
    # 경계를 약간 확장 (여유공간 20%)
    width = max_x - min_x
    height = max_y - min_y
    min_x -= width * 0.2
    max_x += width * 0.2
    min_y -= height * 0.2
    max_y += height * 0.2
    
    # 첫 번째 중앙선 좌표가 확장된 영역 내에 있는지 확인
    first_point = centerline[0]
    return min_x <= first_point[0] <= max_x and min_y <= first_point[1] <= max_y

# 화면 업데이트 함수 수정
def update_display(display_img, lane_annotations, exclusion_zones=None, with_fill=True, 
                  highlight_index=-1, centerline_mode=False):
    """현재 상태를 반영하여 화면을 일관되게 업데이트하는 함수"""
    DISPLAY_WIDTH, DISPLAY_HEIGHT = display_img.shape[1], display_img.shape[0]
    
    # 정규화 여부 확인
    is_norm = is_normalized(lane_annotations)
    
    # 표시용 어노테이션 준비
    display_annotations = []
    for i, ann in enumerate(lane_annotations):
        display_ann = {
            "label": ann["label"],
            "points": []
        }
        
        # 좌표 처리
        if is_norm:
            # 정규화된 좌표를 디스플레이 크기로 변환
            display_ann["points"] = [(int(x * DISPLAY_WIDTH), int(y * DISPLAY_HEIGHT)) 
                                    for x, y in ann["points"]]
        else:
            # 픽셀 좌표를 디스플레이 크기에 맞게 스케일링
            img_height, img_width = original_img.shape[:2]
            scale_x = DISPLAY_WIDTH / img_width
            scale_y = DISPLAY_HEIGHT / img_height
            display_ann["points"] = [(int(x * scale_x), int(y * scale_y)) 
                                    for x, y in ann["points"]]
        
        # 중앙선 처리 - 유효성 검사 추가
        if "centerline" in ann and ann["centerline"] and is_centerline_valid(ann["points"], ann["centerline"]):
            if is_norm:
                display_ann["centerline"] = [(int(x * DISPLAY_WIDTH), int(y * DISPLAY_HEIGHT)) 
                                          for x, y in ann["centerline"]]
            else:
                display_ann["centerline"] = [(int(x * scale_x), int(y * scale_y)) 
                                          for x, y in ann["centerline"]]
        else:
            # 유효하지 않은 centerline은 표시하지 않음
            if "centerline" in ann and ann["centerline"]:
                print(f"경고: '{ann['label']}' 영역의 중앙선이 영역을 벗어나 표시되지 않습니다.")
        
        display_annotations.append(display_ann)
    
    # 이미지 복사 후 어노테이션 그리기
    img_copy = display_img.copy()
    result = draw_all_annotations(img_copy, display_annotations, with_fill=with_fill)
    
    # 특정 영역 강조 표시 (선택된 경우)
    if highlight_index >= 0 and highlight_index < len(display_annotations):
        selected_pts = np.array(display_annotations[highlight_index]["points"], np.int32)
        selected_pts = selected_pts.reshape((-1, 1, 2))
        cv2.polylines(result, [selected_pts], isClosed=True, color=(0, 0, 255), thickness=3)
        
        # 선택된 영역의 라벨 강조
        label_pos = (selected_pts[0][0][0] + 10, selected_pts[0][0][1] + 30)
        mode_text = "중앙선 그리기: " if centerline_mode else "선택됨: "
        cv2.putText(result, f"{mode_text}{lane_annotations[highlight_index]['label']}", 
                    label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return result

def main():
    global img, current_points, lane_annotations, original_img, centerline_points, is_drawing_centerline, selected_lane_index, exclusion_zones, is_drawing_exclusion

    # 시각화 해상도 설정
    DISPLAY_WIDTH = 1280
    DISPLAY_HEIGHT = 720

    # 기존 레이블링 불러오기
    lane_annotations = load_annotations()
    
    # 이미지 파일 경로 (적절히 변경)
    image_path = "./visualization6/received_file_20240822_112527/original/00000187.jpg"
    original_img = cv2.imread(image_path)
    if original_img is None:
        print("이미지를 불러올 수 없습니다. 파일 경로를 확인하세요.")
        return
        
    img_height, img_width = original_img.shape[:2]
    print(f"원본 이미지 크기: {img_width}x{img_height}")
    
    # 불러온 레이블링 데이터가 정규화되지 않은 경우 정규화 처리
    if lane_annotations:
        if not is_normalized(lane_annotations):
            lane_annotations = normalize_annotations(lane_annotations, img_width, img_height)
            print("불러온 레이블링 데이터를 정규화했습니다.")
        print(f"기존 레이블링 {len(lane_annotations)}개를 불러왔습니다.")
        print(f"데이터 정규화 상태: {'정규화됨 (0-1)' if is_normalized(lane_annotations) else '정규화되지 않음'}")
    
    # 디스플레이용 이미지 리사이즈
    display_img = cv2.resize(original_img, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    print(f"시각화 해상도: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
    
    # 초기 화면 업데이트
    if lane_annotations:
        img = update_display(display_img, lane_annotations)
    else:
        img = display_img.copy()
    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", click_event)
    
    print(">> 차선 영역을 지정하세요.")
    print("   - 마우스 좌클릭: 다각형의 점 추가")
    print("   - 'n' 키: 현재 영역 저장 후 다음 영역 지정")
    print("   - 'q' 키: 종료 및 JSON 파일 저장")
    print("   - 'd' 키: 마지막으로 추가한 영역 삭제")
    print("   - 'r' 키: 라벨로 영역 삭제")
    print("   - 'l' 키: 선택한 영역에 중앙선 그리기")
    print("   - 'm' 키: 모든 좌표를 0-1 범위로 정규화")
    print("   - 't' 키: 제외 영역 그리기 모드 토글")
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        
        # 'm' 키: 정규화 처리
        if key == ord("m"):
            if not lane_annotations:
                print("정규화할 데이터가 없습니다.")
                continue
            if is_normalized(lane_annotations):
                print("이미 모든 데이터가 정규화되어 있습니다 (0-1 범위).")
                continue
            lane_annotations = normalize_annotations(lane_annotations, img_width, img_height)
            print("정규화되지 않은 좌표가 0-1 범위로 정규화되었습니다.")
            save_annotations_json(lane_annotations, JSON_FILENAME, exclusion_zones)
            img = update_display(display_img, lane_annotations)
            cv2.imshow("Image", img)
        
        # 'n' 키: 새 영역 저장 (디스플레이 좌표를 정규화 후 저장)
        elif key == ord("n"):
            if len(current_points) < 3:
                print("다각형을 만들려면 최소 3개의 점이 필요합니다.")
                continue
            lane_label = input("해당 영역의 라벨을 입력하세요: ")
            normalized_points = [(x / float(DISPLAY_WIDTH), y / float(DISPLAY_HEIGHT)) for x, y in current_points]
            lane_annotations.append({
                "label": lane_label,
                "points": normalized_points,
                "centerline": []
            })
            print(f"라벨 '{lane_label}' 영역이 저장되었습니다.")
            current_points = []
            img = update_display(display_img, lane_annotations)
            cv2.imshow("Image", img)
        
        # 'd' 키: 마지막 영역 삭제
        elif key == ord("d"):
            if lane_annotations:
                removed = lane_annotations.pop()
                print(f"'{removed['label']}' 영역이 삭제되었습니다.")
                img = update_display(display_img, lane_annotations)
                cv2.imshow("Image", img)
                print(f"남은 영역 수: {len(lane_annotations)}")
            else:
                print("삭제할 영역이 없습니다.")
        
        # 'r' 키: 라벨로 영역 삭제
        elif key == ord("r"):
            if not lane_annotations:
                print("삭제할 영역이 없습니다.")
                continue
            existing_labels = [f"{i}: {ann['label']}" for i, ann in enumerate(lane_annotations)]
            print("현재 영역 목록:")
            for label in existing_labels:
                print(f"  {label}")
            label_to_delete = input("삭제할 영역의 라벨을 입력하세요: ")
            deleted_count = 0
            original_count = len(lane_annotations)
            for i in range(len(lane_annotations)-1, -1, -1):
                if lane_annotations[i]["label"] == label_to_delete:
                    del lane_annotations[i]
                    deleted_count += 1
            if deleted_count > 0:
                print(f"'{label_to_delete}' 라벨을 가진 {deleted_count}개 영역이 삭제되었습니다.")
                img = update_display(display_img, lane_annotations)
                cv2.imshow("Image", img)
                print(f"영역 {original_count}개 → {len(lane_annotations)}개")
            else:
                print(f"'{label_to_delete}' 라벨을 가진 영역이 없습니다.")
        
        # 'l' 키: 중앙선 그리기 모드 전환
        elif key == ord("l"):
            if not lane_annotations:
                print("중앙선을 그릴 영역이 없습니다. 먼저 영역을 생성하세요.")
                continue
            existing_labels = [f"{i}: {ann['label']}" for i, ann in enumerate(lane_annotations)]
            print("현재 영역 목록:")
            for label in existing_labels:
                print(f"  {label}")
            try:
                selected_lane_index = int(input("중앙선을 그릴 영역의 번호를 입력하세요: "))
                if selected_lane_index < 0 or selected_lane_index >= len(lane_annotations):
                    print("유효하지 않은 영역 번호입니다.")
                    continue
            except ValueError:
                print("숫자를 입력해야 합니다.")
                continue
            is_drawing_centerline = True
            centerline_points = []
            img = update_display(display_img, lane_annotations, highlight_index=selected_lane_index, centerline_mode=True)
            cv2.imshow("Image", img)
            print(f"'{lane_annotations[selected_lane_index]['label']}' 영역에 중앙선을 그립니다.")
            print("  - 마우스 좌클릭: 중앙선의 점 추가")
            print("  - 'c' 키: 중앙선 그리기 완료")
            print("  - 'x' 키: 중앙선 그리기 취소")
        
        # 'c' 키를 누르면 중앙선 저장
        elif key == ord("c") and is_drawing_centerline:
            if len(centerline_points) < 2:
                print("중앙선은 최소 2개 이상의 점이 필요합니다.")
                continue
            
            # 중앙선 저장 전 유효성 확인
            normalized_centerline = [(x/DISPLAY_WIDTH, y/DISPLAY_HEIGHT) for x, y in centerline_points]
            if is_centerline_valid(lane_annotations[selected_lane_index]["points"], normalized_centerline):
                # 중앙선 저장
                lane_annotations[selected_lane_index]["centerline"] = normalized_centerline
                print(f"'{lane_annotations[selected_lane_index]['label']}' 영역에 중앙선이 저장되었습니다.")
                centerline_points = []
                
                # 중앙선이 모두 그려졌는지 확인
                has_all_centerlines = True
                for ann in lane_annotations:
                    if "centerline" not in ann or not ann["centerline"]:
                        has_all_centerlines = False
                        break
                
                if has_all_centerlines:
                    print("모든 영역에 중앙선을 그렸습니다!")
            else:
                print("경고: 중앙선이 영역 밖에 있습니다. 영역 내부에 중앙선을 그려주세요.")
            continue
        
        # 'x' 키: 중앙선 취소
        elif key == ord("x") and is_drawing_centerline:
            print("중앙선 그리기를 취소했습니다.")
            is_drawing_centerline = False
            selected_lane_index = -1
            centerline_points = []
            img = update_display(display_img, lane_annotations)
            cv2.imshow("Image", img)
        
        # 't' 키: 제외 영역 그리기 모드 토글
        elif key == ord('t'):
            if is_drawing_exclusion:
                if current_points and len(current_points) >= 3:
                    normalized_points = [(x / float(DISPLAY_WIDTH), y / float(DISPLAY_HEIGHT)) for x, y in current_points]
                    exclusion_zones.append(normalized_points)
                    print(f"제외 영역이 추가되었습니다. (총 {len(exclusion_zones)}개)")
                    print(f"제외 영역 점 수: {len(normalized_points)}개")
                current_points.clear()
                is_drawing_exclusion = False
            else:
                is_drawing_exclusion = True
                is_drawing_centerline = False
                current_points.clear()
                print("제외 영역 그리기 모드를 시작합니다. 다각형 점을 찍은 후 다시 't'를 누르면 완료됩니다.")
            img = update_display(display_img, lane_annotations)
            cv2.imshow("Image", img)
        
        # 'q' 키: 종료
        elif key == ord("q"):
            break

    # 최종 레이블링 이미지 저장
    if lane_annotations:
        result_img = draw_all_annotations(original_img, 
                        [ { "label": ann["label"],
                            "points": [(int(x * img_width), int(y * img_height)) for x, y in ann["points"]],
                            "centerline": [(int(x * img_width), int(y * img_height)) for x, y in ann["centerline"]]
                          } for ann in lane_annotations ], with_fill=True)
        cv2.imwrite(IMAGE_FILENAME, result_img)
        print(f"레이블링 이미지가 '{IMAGE_FILENAME}'에 저장되었습니다.")
        normalized_status = "정규화됨 (0-1)" if is_normalized(lane_annotations) else "정규화되지 않음 (픽셀 좌표)"
        save_annotations_json(lane_annotations, JSON_FILENAME, exclusion_zones)
        print(f"모든 라벨링 정보가 '{JSON_FILENAME}'에 저장되었습니다. ({normalized_status})")
        print(f"총 {len(lane_annotations)}개 영역이 저장되었습니다.")
    else:
        print("저장할 레이블링이 없습니다.")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
