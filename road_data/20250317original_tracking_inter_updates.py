import os
import cv2
import numpy as np
import math
import json
from ultralytics import YOLO
import random
from shapely.geometry import Point, Polygon, LineString
from shapely.geometry import box as shapely_box
from scipy.interpolate import interp1d
from scipy.misc import derivative
import traceback

# 18개 클래스에 대한 파스텔톤 색상 (BGR 순서)
# 14: car, 15: bus, 16: truck
CLASS_COLORS = {
    0:  (199, 110, 255), 1:  (20, 255, 57), 2:  (255, 81, 31), 3:  (51, 255, 255),
    4:  (51, 153, 255), 5:  (255, 0, 191), 6:  (255, 255, 0), 7:  (255, 0, 255),
    8:  (0, 255, 204), 9:  (252, 240, 15), 10: (58, 7, 255), 11: (255, 0, 143),
    12: (212, 255, 127), 13: (0, 69, 255), 14: (47, 255, 173), 15: (147, 20, 255),
    16: (80, 127, 255), 17: (0, 215, 255)
}

# 카테고리 별 색상 및 아이콘 정의 추가
CATEGORY_COLORS = {
    "stop": (0, 0, 255),         # 정지 - 빨간색
    "lane_change": (0, 255, 255), # 차선 변경 - 노란색
    # "turn_right": (0, 165, 255),  # 우회전 - 주황색
    # "turn_left": (255, 0, 127),   # 좌회전 - 분홍색
    "normal_driving": (0, 255, 0) # 일반 주행 - 녹색
}

# bbox 형식 변환 함수 추가
def convert_bbox_format(bbox, format_type="xyxy_to_cxcywh"):
    """
    bbox 형식 변환: 
    - xyxy_to_cxcywh: [x1, y1, x2, y2] -> [center_x, center_y, width, height]
    - cxcywh_to_xyxy: [center_x, center_y, width, height] -> [x1, y1, x2, y2]
    """
    if format_type == "xyxy_to_cxcywh":
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        return [center_x, center_y, width, height]
    elif format_type == "cxcywh_to_xyxy":
        center_x, center_y, width, height = bbox
        x1 = center_x - width / 2
        y1 = center_y - height / 2
        x2 = center_x + width / 2
        y2 = center_y + height / 2
        return [x1, y1, x2, y2]
    else:
        return bbox

# 정규화 관련 함수 추가
def is_normalized(value):
    """값이 이미 정규화되어 있는지 확인 (0~1 사이 값인지)"""
    return 0 <= value <= 1

def check_data_normalized(data_sample):
    """데이터 샘플을 확인하여 정규화 필요성 확인"""
    # 첫 번째 항목의 bbox 기준으로 판단
    if not data_sample:
        return True  # 데이터가 없으면 정규화 필요 없음
    
    item = data_sample[0]
    if "bbox" in item:
        center_x, center_y, width, height = item["bbox"]
        # bbox의 값이 1보다 크면 정규화가 필요
        if max(center_x, center_y, width, height) > 1:
            return False
    
    return True  # 이미 정규화되어 있음

def normalize_coordinates(vehicle_data, width, height):
    """좌표를 주어진 해상도로 정규화"""
    normalized_data = []
    
    # 대각선 길이 계산 (정규화 기준값)
    diagonal = DISTANCE_THRESHOLD
    
    for item in vehicle_data:
        normalized_item = item.copy()
        
        # 경계 상자(bbox) 정규화 - 새 형식: [center_x, center_y, width, height]
        if "bbox" in item:
            center_x, center_y, bbox_width, bbox_height = item["bbox"]
            normalized_item["bbox"] = [
                center_x / width, 
                center_y / height, 
                bbox_width / width, 
                bbox_height / height
            ]
        
        # 이웃 위치(neighbors_positions) 정규화
        if "neighbors_positions" in item:
            normalized_neighbors_positions = []
            for pos in item["neighbors_positions"]:
                if len(pos) == 2 and (pos[0] == 0 and pos[1] == 0):  # 기본값은 그대로 유지
                    normalized_neighbors_positions.append([0, 0])
                else:
                    normalized_neighbors_positions.append([
                        pos[0] / width, 
                        pos[1] / height
                    ])
            normalized_item["neighbors_positions"] = normalized_neighbors_positions
        
        # 이웃 거리(neighbors_distances) 정규화 추가
        if "neighbors_distances" in item:
            normalized_neighbors_distances = []
            for dist in item["neighbors_distances"]:
                # 0인 경우 그대로 유지, 나머지는 대각선 길이로 정규화
                if dist == 0:
                    normalized_neighbors_distances.append(0)
                else:
                    normalized_neighbors_distances.append(dist / diagonal)
            normalized_item["neighbors_distances"] = normalized_neighbors_distances
        
        # 이웃 크기(neighbors_sizes) 정규화 추가
        if "neighbors_sizes" in item:
            normalized_neighbors_sizes = []
            for size in item["neighbors_sizes"]:
                if len(size) == 2 and (size[0] == 0 and size[1] == 0):
                    normalized_neighbors_sizes.append([0, 0])
                else:
                    normalized_neighbors_sizes.append([
                        size[0] / width,
                        size[1] / height
                    ])
            normalized_item["neighbors_sizes"] = normalized_neighbors_sizes
        
        # 이웃 정보(neighbors_info) 정규화 추가
        if "neighbors_info" in item:
            normalized_neighbors_info = []
            for neighbor in item["neighbors_info"]:
                if len(neighbor) >= 4:  # (거리, ID, 위치, 박스) 형식 확인
                    # 거리(d) 값 정규화
                    dist = neighbor[0] / diagonal
                    nbr_id = neighbor[1]
                    nbr_pos = neighbor[2]
                    old_nbr_bbox = neighbor[3]
                    
                    # 위치 정규화
                    normalized_nbr_pos = [nbr_pos[0] / width, nbr_pos[1] / height]
                    
                    # 박스를 새 형식으로 변환 및 정규화
                    nbr_center_x = (old_nbr_bbox[0] + old_nbr_bbox[2]) / 2
                    nbr_center_y = (old_nbr_bbox[1] + old_nbr_bbox[3]) / 2
                    nbr_width = old_nbr_bbox[2] - old_nbr_bbox[0]
                    nbr_height = old_nbr_bbox[3] - old_nbr_bbox[1]
                    
                    normalized_nbr_bbox = [
                        nbr_center_x / width,
                        nbr_center_y / height,
                        nbr_width / width,
                        nbr_height / height
                    ]
                    
                    # 나머지 정보 유지하면서 새 튜플 생성
                    normalized_neighbor = list(neighbor)
                    normalized_neighbor[0] = dist  # 정규화된 거리 값으로 업데이트
                    normalized_neighbor[2] = normalized_nbr_pos
                    normalized_neighbor[3] = normalized_nbr_bbox
                    normalized_neighbors_info.append(tuple(normalized_neighbor))
                else:
                    normalized_neighbors_info.append(neighbor)  # 형식이 다르면 그대로 유지
            normalized_item["neighbors_info"] = normalized_neighbors_info
        
        # union_bbox 정규화 (centerx, centery, width, height 형식으로 변환)
        if "union_bbox" in item:
            x1, y1, x2, y2 = item["union_bbox"]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width_union = x2 - x1
            height_union = y2 - y1
            
            normalized_item["union_bbox"] = [
                center_x / width, 
                center_y / height, 
                width_union / width, 
                height_union / height
            ]
        
        normalized_data.append(normalized_item)
    
    return normalized_data

def draw_filled_rounded_rectangle(img, pt1, pt2, color, radius, alpha=0.3):
    x1, y1 = pt1
    x2, y2 = pt2
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x1 + radius, y2 - radius), color, -1)
    cv2.rectangle(overlay, (x2 - radius, y1 + radius), (x2, y2 - radius), color, -1)
    cv2.ellipse(overlay, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, -1)
    cv2.ellipse(overlay, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, -1)
    cv2.ellipse(overlay, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, -1)
    cv2.ellipse(overlay, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_rounded_rectangle(img, pt1, pt2, color, thickness, radius):
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

def get_color_by_class(class_id):
    return CLASS_COLORS.get(class_id, (255, 255, 255))

def draw_filled_arrow_line(img, pt1, pt2, color, thickness=1, tip_length=10):
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    line_length = np.sqrt(dx * dx + dy * dy)
    if line_length < 1e-5:
        return
    nx, ny = dx / line_length, dy / line_length
    arrow_base = (pt2[0] - tip_length * nx, pt2[1] - tip_length * ny)
    perp = np.array([-ny, nx])
    arrow_width = tip_length * 0.5
    base_left = (arrow_base[0] + perp[0] * arrow_width, arrow_base[1] + perp[1] * arrow_width)
    base_right = (arrow_base[0] - perp[0] * arrow_width, arrow_base[1] - perp[1] * arrow_width)
    cv2.line(img, pt1, (int(arrow_base[0]), int(arrow_base[1])), color, thickness)
    pts = np.array([[int(pt2[0]), int(pt2[1])],
                    [int(base_left[0]), int(base_left[1])],
                    [int(base_right[0]), int(base_right[1])]], np.int32)
    cv2.fillPoly(img, [pts], color)

def compute_motion_vector(prev_frame, curr_frame, patch_center, patch_size, step=5):
    x, y = patch_center
    half_size = patch_size // 2
    x_coords = np.arange(x - half_size, x + half_size, step)
    y_coords = np.arange(y - half_size, y + half_size, step)
    xx, yy = np.meshgrid(x_coords, y_coords)
    pts = np.vstack([xx.ravel(), yy.ravel()]).T.astype(np.float32)
    pts = pts.reshape(-1, 1, 2)
    lk_params = dict(winSize=(patch_size, patch_size), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    nextPts, status, err = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, pts, None, **lk_params)
    valid_mask = status.reshape(-1) == 1
    if np.sum(valid_mask) == 0:
        return (0, 0), pts, nextPts, valid_mask
    displacements = nextPts[valid_mask].reshape(-1, 2) - pts[valid_mask].reshape(-1, 2)
    avg_disp = np.mean(displacements, axis=0)
    return (avg_disp[0], avg_disp[1]), pts, nextPts, valid_mask

# 선 색상 및 Legend 라벨
line_colors = [(0, 0, 255), (255, 0, 0), (0, 165, 255), (0, 255, 0)]
legend_labels = ["1. short1", "2. short2", "3. short3", "4. short4"]
DISTANCE_THRESHOLD = 200

# 방향 벡터 보정 옵션 설정
direction_adjustment_method = "exponential"
direction_history = {}
smoothed_directions = {}
last_directions = {}

# Optical Flow 관련 변수
prev_gray = None

# 모델 로드
# model_path = "/home/lee/research/Research2025/YOLOv11/tracking_vehicle/drone_dataset_yolo11m/weights/best.pt"
model_path = "/home/lee/research/Research2025/YOLOv11/yolov9_visdrone_gongeoptap50.pt"
model = YOLO(model_path)

# 비디오 경로 설정
video_path = "/home/lee/research/Research2025/공업탑데이터/video_data/received_file_20240822_115028.avi"
dataset_name = os.path.splitext(os.path.basename(video_path))[0]

# 차선 데이터 경로 설정
lane_json_path = "/home/lee/research/Research2025/YOLOv11/lane_annotations.json"

# 차선 색상 정의 - 각 레이블마다 다른 색상 사용
CENTERLINE_COLOR = (255, 255, 255)  # 중앙선은 흰색

# 레이블별 색상 생성 함수 추가
def get_lane_color(lane_id):
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
    try:
        if "-" in lane_id:
            index = int(lane_id.split("-")[0])
        else:
            index = int(lane_id)
    except ValueError:
        index = sum(ord(c) for c in lane_id) % len(colors)
    return colors[index % len(colors)]

# 차선 데이터 로드 함수
def load_lane_data(json_path):
    print(f"차선 데이터 로드 시도: {json_path}")
    try:
        with open(json_path, 'r') as f:
            lane_data = json.load(f)
            
        lanes = {}
        centerlines = {}
        exclusion_zones = []
        
        if "lanes" in lane_data:
            for lane in lane_data["lanes"]:
                if "label" in lane and "points" in lane:
                    lane_id = lane["label"]
                    points = lane["points"]
                    lanes[lane_id] = points
                    if "centerline" in lane:
                        centerlines[lane_id] = lane["centerline"]
        if "exclusion_zones" in lane_data:
            exclusion_zones = lane_data["exclusion_zones"]
            
        print(f"차선 데이터 로드 완료: {len(lanes)}개 차선, {len(exclusion_zones)}개 제외 영역")
        return lanes, centerlines, exclusion_zones
    except Exception as e:
        print(f"차선 데이터 로드 오류: {e}")
        traceback.print_exc()
        return {}, {}, []

# 차량이 차선 영역 내에 있는지 확인하는 함수
def is_point_in_lane(point, lane_points):
    if len(lane_points) < 3:
        return False
    polygon = Polygon(lane_points)
    point_obj = Point(point[0], point[1])
    return polygon.contains(point_obj)

# 새로 추가: 검출된 차량의 중심이 exclusion zone 내에 있는지 확인하는 함수
def is_point_in_exclusion_zones(point, width, height):
    for zone in exclusion_zones:
        if len(zone) < 3:
            continue
        pixel_zone = [(int(p[0] * width), int(p[1] * height)) for p in zone]
        if is_point_in_lane(point, pixel_zone):
            return True
    return False

# 정규화된 좌표를 실제 픽셀 좌표로 변환하는 함수
def normalize_to_pixel(norm_coords, width, height):
    pixel_coords = []
    for coord in norm_coords:
        x_pixel = int(coord[0] * width)
        y_pixel = int(coord[1] * height)
        pixel_coords.append([x_pixel, y_pixel])
    return pixel_coords

# 차선 찾기 함수
def find_nearest_lane(vehicle_pos):
    min_distance = float('inf')
    nearest_lane = None
    for lane_id, centerline_points in lane_centerlines.items():
        for point in centerline_points:
            distance = math.hypot(vehicle_pos[0] - point[0], vehicle_pos[1] - point[1])
            if distance < min_distance:
                min_distance = distance
                nearest_lane = lane_id
    return nearest_lane, min_distance

# 현재 파일의 절대 경로 및 폴더 생성
current_path = os.path.abspath(__file__)
current_path = os.path.dirname(current_path)
base_dir = "visualization6_test"
dataset_dir = os.path.join(current_path, base_dir, dataset_name)
video_dir = os.path.join(current_path, dataset_dir, "video")
image_dir = os.path.join(current_path, dataset_dir, "image")
json_dir = os.path.join(current_path, dataset_dir, "json")
labeling_dir = os.path.join(dataset_dir, "labeling")
os.makedirs(video_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)
os.makedirs(json_dir, exist_ok=True)
os.makedirs(labeling_dir, exist_ok=True)

lane_centerlines = {}
lane_splines = {}
lane_curvatures = {}
vehicle_lane_assignments = {}
last_lane_assignments = {}
vehicle_directions = {}
vehicle_speed_history = {}
speed_history = {}

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

ret, frame = cap.read()
if not ret:
    cap.release()
    raise Exception("비디오를 읽을 수 없습니다.")

orig_height, orig_width = frame.shape[:2]
hd_width, hd_height = 1280, 720
scale_x = orig_width / hd_width
scale_y = orig_height / hd_height

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_out_path = os.path.join(video_dir, f"{dataset_name}_result.mp4")
video_writer = cv2.VideoWriter(video_out_path, fourcc, fps, (orig_width, orig_height))

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
output_data = []
last_positions = {}
last_speeds = {}
frame_index = 0
frame_counter = 1

def is_vehicle_stopped(vehicle_id, current_speed, threshold=0.5):
    global vehicle_speed_history
    if vehicle_id not in vehicle_speed_history:
        vehicle_speed_history[vehicle_id] = []
    vehicle_speed_history[vehicle_id].append(current_speed)
    if len(vehicle_speed_history[vehicle_id]) > 3:
        vehicle_speed_history[vehicle_id].pop(0)
    if len(vehicle_speed_history[vehicle_id]) < 3:
        return current_speed <= threshold
    avg_speed = sum(vehicle_speed_history[vehicle_id]) / 3
    return avg_speed <= threshold

def calculate_direction_change(prev_direction, curr_direction):
    if prev_direction is None or curr_direction is None:
        return 0
    def normalize(vector):
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
    prev_norm = normalize(prev_direction)
    curr_norm = normalize(curr_direction)
    dot_product = np.dot(prev_norm, curr_norm)
    dot_product = max(-1.0, min(1.0, dot_product))
    angle = math.acos(dot_product)
    angle_deg = math.degrees(angle)
    change_percent = angle_deg / 180.0 * 100
    return change_percent

def draw_vehicle_category(img, bbox, category, scale_factor=1.0):
    if not category or category == "":
        return img
    x1, y1, x2, y2 = bbox
    color = CATEGORY_COLORS.get(category, (200, 200, 200))
    badge_height = int(25 * scale_factor)
    badge_width = int(120 * scale_factor)
    badge_x = int(x1)
    badge_y = int(y1 - badge_height - 5)
    if badge_y < 0:
        badge_y = int(y1 + 5)
    overlay = img.copy()
    cv2.rectangle(overlay, (badge_x, badge_y), (badge_x + badge_width, badge_y + badge_height), color, -1)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (badge_x, badge_y), (badge_x + badge_width, badge_y + badge_height), (0, 0, 0), 1)
    text_size = cv2.getTextSize(category, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale_factor, int(1 * scale_factor))[0]
    text_x = badge_x + (badge_width - text_size[0]) // 2
    text_y = badge_y + (badge_height + text_size[1]) // 2
    cv2.putText(img, category, (text_x + 1, text_y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale_factor, (0, 0, 0), int(2 * scale_factor))
    cv2.putText(img, category, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale_factor, (255, 255, 255), int(1 * scale_factor))
    return img

# 차선 데이터 로드
lanes, centerlines, exclusion_zones = load_lane_data(lane_json_path)
lane_areas = {}
lane_centerlines = {}

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

ret, frame = cap.read()
if not ret:
    cap.release()
    raise Exception("비디오를 읽을 수 없습니다.")

orig_height, orig_width = frame.shape[:2]
hd_width, hd_height = 1280, 720
scale_x = orig_width / hd_width
scale_y = orig_height / hd_height

for lane_id, area in lanes.items():
    lane_areas[lane_id] = normalize_to_pixel(area, hd_width, hd_height)
    print(f"차선 {lane_id} 영역 첫 번째 점: {lane_areas[lane_id][0]}")

for lane_id, centerline in centerlines.items():
    lane_centerlines[lane_id] = normalize_to_pixel(centerline, hd_width, hd_height)
    print(f"차선 {lane_id} 중앙선 첫 번째 점: {lane_centerlines[lane_id][0]}")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_save = frame.copy()
    frame_hd = cv2.resize(frame, (hd_width, hd_height))
    
    frame_lanes_hd = frame_hd.copy()
    frame_result_hd = frame_hd.copy()
    frame_lanes_orig = frame_save.copy()
    
    for lane_id, points in lane_areas.items():
        if len(points) >= 3:
            lane_color = get_lane_color(lane_id)
            points_array = np.array(points, np.int32)
            overlay = frame_hd.copy()
            cv2.fillPoly(overlay, [points_array], lane_color)
            alpha = 0.3
            cv2.addWeighted(overlay, alpha, frame_hd, 1 - alpha, 0, frame_hd)
            cv2.polylines(frame_hd, [points_array], isClosed=True, color=lane_color, thickness=2)
            points_orig = [[int(p[0] * scale_x), int(p[1] * scale_y)] for p in points]
            points_array_orig = np.array(points_orig, np.int32)
            overlay_orig = frame_save.copy()
            cv2.fillPoly(overlay_orig, [points_array_orig], lane_color)
            cv2.addWeighted(overlay_orig, alpha, frame_save, 1 - alpha, 0, frame_save)
            cv2.polylines(frame_save, [points_array_orig], isClosed=True, color=lane_color, thickness=int(2 * scale_x))
            text = lane_id
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            label_pos_hd = (points_array[0][0] + 10, points_array[0][1] + 20)
            cv2.putText(frame_hd, text, label_pos_hd, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            label_pos_orig = (points_array_orig[0][0] + int(10 * scale_x), points_array_orig[0][1] + int(20 * scale_y))
            text_size_orig = (int(text_size[0] * scale_x), int(text_size[1] * scale_y))
            cv2.rectangle(frame_save, 
                         (label_pos_orig[0] - int(5 * scale_x), label_pos_orig[1] - text_size_orig[1] - int(5 * scale_y)),
                         (label_pos_orig[0] + text_size_orig[0] + int(5 * scale_x), label_pos_orig[1] + int(5 * scale_y)),
                         (0, 0, 0), -1)
            cv2.putText(frame_save, text, label_pos_orig, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale_x, (255, 255, 255), int(2 * scale_x))
    
    for lane_id, centerline in lane_centerlines.items():
        if len(centerline) >= 2:
            for i in range(len(centerline) - 1):
                pt1 = (int(centerline[i][0]), int(centerline[i][1]))
                pt2 = (int(centerline[i+1][0]), int(centerline[i+1][1]))
                cv2.line(frame_hd, pt1, pt2, CENTERLINE_COLOR, 2)
            centerline_orig = [[int(p[0] * scale_x), int(p[1] * scale_y)] for p in centerline]
            for i in range(len(centerline_orig) - 1):
                pt1 = (int(centerline_orig[i][0]), int(centerline_orig[i][1]))
                pt2 = (int(centerline_orig[i+1][0]), int(centerline_orig[i+1][1]))
                cv2.line(frame_save, pt1, pt2, CENTERLINE_COLOR, int(2 * scale_x))
    
    if frame_index == 0:
        print(f"차선 영역 수: {len(lane_areas)}, 중앙선 수: {len(lane_centerlines)}")
        print(f"HD 프레임 크기: {frame_hd.shape}, 원본 프레임 크기: {frame_save.shape}")
    
    # Optical Flow 계산
    gray = cv2.cvtColor(frame_hd, cv2.COLOR_BGR2GRAY)
    if prev_gray is not None:
        h_gray, w_gray = gray.shape
        center_patch = (w_gray // 2, h_gray // 2)
        patch_size_flow = 50
        offset_flow = patch_size_flow * 4
        patch_centers = [
            (center_patch[0] - offset_flow, center_patch[1]),
            (center_patch[0] - offset_flow, center_patch[1] - offset_flow),
            (center_patch[0] - offset_flow, center_patch[1] + offset_flow),
            center_patch,
            (center_patch[0] + offset_flow, center_patch[1]),
            (center_patch[0] + offset_flow, center_patch[1] - offset_flow),
            (center_patch[0] + offset_flow, center_patch[1] + offset_flow),
            (center_patch[0], center_patch[1] - offset_flow),
            (center_patch[0], center_patch[1] + offset_flow)
        ]
        motion_vectors = [compute_motion_vector(prev_gray, gray, center, patch_size_flow, step=5)[0] for center in patch_centers]
        patch_avg_vector = (sum(v[0] for v in motion_vectors) / len(motion_vectors),
                            sum(v[1] for v in motion_vectors) / len(motion_vectors))
        cv2.arrowedLine(frame_hd, center_patch,
                        (int(center_patch[0] + patch_avg_vector[0]), int(center_patch[1] + patch_avg_vector[1])),
                        (0, 0, 255), 2, tipLength=0.5)
    else:
        patch_avg_vector = [0, 0]
    prev_gray = gray.copy()

    results = model.track(frame_result_hd, persist=True, conf=0.1, tracker="bytetrack.yaml")
    
    frame_data = []

    for r in results:
        for i, box in enumerate(r.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(r.boxes.cls[i])
            if class_id in [0, 1, 2, 3]:
                object_id = int(r.boxes.id[i])
                color_bbox = get_color_by_class(class_id)
                
                # 경계 상자 좌표
                x1, y1, x2, y2 = map(int, box)
                
                # 새로운 방식으로 bbox 정의: [center_x, center_y, width, height]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                
                # ★ 추가: 차량 중심이 exclusion zone 내에 있다면 무시
                if is_point_in_exclusion_zones([center_x, center_y], hd_width, hd_height):
                    continue

                draw_filled_rounded_rectangle(frame_hd, (x1, y1), (x2, y2), color_bbox, radius=10, alpha=0.3)
                draw_rounded_rectangle(frame_hd, (x1, y1), (x2, y2), color_bbox, thickness=2, radius=10)
                cv2.putText(frame_hd, f"ID:{object_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bbox, 2)

                x1_o = int(x1 * scale_x)
                y1_o = int(y1 * scale_y)
                x2_o = int(x2 * scale_x)
                y2_o = int(y2 * scale_y)
                draw_filled_rounded_rectangle(frame_save, (x1_o, y1_o), (x2_o, y2_o), color_bbox, radius=int(10*scale_x), alpha=0.3)
                draw_rounded_rectangle(frame_save, (x1_o, y1_o), (x2_o, y2_o), color_bbox, thickness=2, radius=int(10*scale_x))
                cv2.putText(frame_save, f"ID:{object_id}", (x1_o, y1_o - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bbox, 2)

                # bbox 값 저장 (center_x, center_y, width, height 형식)
                bbox = [center_x, center_y, width, height]
                
                nearest_lane_id = ""
                lane_distance = -1.0  # 기본값을 -1로 변경 (속하지 않음을 의미)
                
                # 차량이 차선 내부에 있는지 확인
                is_in_lane = False
                for lane_id, lane_points in lane_areas.items():
                    if is_point_in_lane([center_x, center_y], lane_points):
                        nearest_lane_id = lane_id
                        lane_distance = 0  # 차선 내부에 있으면 거리는 0
                        is_in_lane = True
                        break
                
                # 어떤 차선에도 속하지 않는 경우 가장 가까운 차선과의 거리 계산
                if not is_in_lane:
                    min_distance = float('inf')
                    for lane_id, lane_points in lane_areas.items():
                        if len(lane_points) < 3:
                            continue
                        # 차선의 다각형 객체 생성
                        lane_polygon = Polygon(lane_points)
                        # 차량 위치의 Point 객체 생성
                        vehicle_point = Point(center_x, center_y)
                        # 최단 거리 계산
                        current_distance = lane_polygon.exterior.distance(vehicle_point)
                        if current_distance < min_distance:
                            min_distance = current_distance
                            nearest_lane_id = lane_id
                    
                    if nearest_lane_id:  # 가장 가까운 차선을 찾은 경우
                        lane_distance = min_distance
                    else:
                        lane_distance = -1.0  # 여전히 찾지 못한 경우 -1 유지
                
                if object_id in last_positions:
                    prev_center = last_positions[object_id]
                    dx = center_x - prev_center[0]
                    dy = center_y - prev_center[1]
                    displacement = math.hypot(dx, dy)
                    speed = displacement
                    raw_direction = [dx/displacement, dy/displacement] if displacement != 0 else [0, 0]
                else:
                    speed = 0
                    displacement = 0
                    raw_direction = [0, 0]
                
                acceleration = 0
                if object_id in last_speeds:
                    prev_speed = last_speeds[object_id]
                    acceleration = (speed - prev_speed) * fps
                last_speeds[object_id] = speed
                last_positions[object_id] = [center_x, center_y]
                
                if object_id not in speed_history:
                    speed_history[object_id] = []
                speed_history[object_id].append(speed)
                if len(speed_history[object_id]) > 5:
                    speed_history[object_id].pop(0)
                speeds_to_consider = speed_history[object_id][-3:] if len(speed_history[object_id]) >= 3 else speed_history[object_id]
                avg_speed = sum(speeds_to_consider) / len(speeds_to_consider)
                category = "stop" if avg_speed <= 0.8 else "normal_driving"

                disp_text = f"Disp: {displacement:.2f}"
                cv2.putText(frame_hd, disp_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame_save, disp_text, (x1_o, y2_o + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cat_text = f"Cat: {category}"
                cv2.putText(frame_hd, cat_text, (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame_save, cat_text, (x1_o, y2_o + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if direction_adjustment_method == "averaging":
                    if object_id not in direction_history:
                        direction_history[object_id] = []
                    direction_history[object_id].append(raw_direction)
                    if len(direction_history[object_id]) > 5:
                        direction_history[object_id].pop(0)
                    sum_dx = sum(d[0] for d in direction_history[object_id])
                    sum_dy = sum(d[1] for d in direction_history[object_id])
                    norm = math.hypot(sum_dx, sum_dy)
                    direction = [sum_dx/norm, sum_dy/norm] if norm != 0 else [0, 0]
                elif direction_adjustment_method == "exponential":
                    alpha = 0.3
                    if object_id in smoothed_directions:
                        prev_smoothed = smoothed_directions[object_id]
                        new_dx = alpha * raw_direction[0] + (1 - alpha) * prev_smoothed[0]
                        new_dy = alpha * raw_direction[1] + (1 - alpha) * prev_smoothed[1]
                        norm = math.hypot(new_dx, new_dy)
                        direction = [new_dx/norm, new_dy/norm] if norm != 0 else [0, 0]
                        smoothed_directions[object_id] = direction
                    else:
                        direction = raw_direction
                        smoothed_directions[object_id] = direction
                elif direction_adjustment_method == "scale_adjustment":
                    MIN_DISPLACEMENT_THRESHOLD = 2
                    if displacement < MIN_DISPLACEMENT_THRESHOLD:
                        direction = last_directions.get(object_id, raw_direction)
                    else:
                        direction = raw_direction
                        last_directions[object_id] = direction
                else:
                    direction = raw_direction
                
                if nearest_lane_id:
                    cv2.putText(frame_hd, f"Lane:{nearest_lane_id}", (x1, y1 - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bbox, 2)
                    cv2.putText(frame_save, f"Lane:{nearest_lane_id}", (x1_o, y1_o - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bbox, 2)
                else:
                    cv2.putText(frame_hd, "No Lane", (x1, y1 - 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bbox, 2)
                    cv2.putText(frame_save, "No Lane", (x1_o, y1_o - 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bbox, 2)
                
                data = {
                    "object_id": object_id,
                    "bbox": bbox,  # 새 형식: [center_x, center_y, width, height]
                    "class": class_id,
                    "speed": speed,
                    "direction": direction,
                    "acceleration": acceleration,
                    "lane_id": nearest_lane_id,
                    "lane_distance": lane_distance,
                    "frame": frame_index,
                    "category": category
                }
                frame_data.append(data)

    for vehicle in frame_data:
        # bbox에서 중심점 얻기
        center_x, center_y = vehicle["bbox"][0], vehicle["bbox"][1]
        curr_center = [center_x, center_y]
        
        neighbors = []
        for other in frame_data:
            if other["object_id"] == vehicle["object_id"]:
                continue
            other_center = [other["bbox"][0], other["bbox"][1]]
            d = np.linalg.norm(np.array(curr_center) - np.array(other_center))
            
            # 다른 차량의 경계 상자를 xyxy 형식으로 변환
            other_center_x, other_center_y, other_width, other_height = other["bbox"]
            other_x1 = other_center_x - other_width / 2
            other_y1 = other_center_y - other_height / 2
            other_x2 = other_center_x + other_width / 2
            other_y2 = other_center_y + other_height / 2
            other_bbox_xyxy = [other_x1, other_y1, other_x2, other_y2]
            
            if d <= DISTANCE_THRESHOLD:
                neighbors.append((d, other["object_id"], other_center, other_bbox_xyxy, 
                                 other["speed"], other["direction"], other["acceleration"]))
                neighbors.sort(key=lambda x: x[0])
                neighbor_ids = []
                neighbor_dists = []
                neighbor_speeds = []
                neighbor_directions = []
                neighbor_accelerations = []
                neighbor_positions = []
                neighbor_sizes = []
                for k in range(4):
                    if k < len(neighbors):
                        d, nbr_id, nbr_pos, nbr_bbox_xyxy, nbr_speed, nbr_direction, nbr_acceleration = neighbors[k]
                        nbr_width = nbr_bbox_xyxy[2] - nbr_bbox_xyxy[0]
                        nbr_height = nbr_bbox_xyxy[3] - nbr_bbox_xyxy[1]
                        
                        neighbor_ids.append(nbr_id)
                        neighbor_dists.append(round(d, 3))
                        neighbor_speeds.append(nbr_speed)
                        neighbor_directions.append(nbr_direction)
                        neighbor_accelerations.append(nbr_acceleration)
                        neighbor_positions.append(list(nbr_pos))
                        neighbor_sizes.append([nbr_width, nbr_height])
                        
                        # 현재 차량과 이웃 차량 간의 라인 그리기
                        draw_filled_arrow_line(frame_hd, (int(curr_center[0]), int(curr_center[1])),
                                            (int(nbr_pos[0]), int(nbr_pos[1])), line_colors[k], thickness=1, tip_length=5)
                        curr_center_o = (int(curr_center[0] * scale_x), int(curr_center[1] * scale_y))
                        nbr_pos_o = (int(nbr_pos[0] * scale_x), int(nbr_pos[1] * scale_y))
                        draw_filled_arrow_line(frame_save, curr_center_o, nbr_pos_o, line_colors[k],
                                            thickness=1, tip_length=int(5 * scale_x))
                        
                        # 거리 표시
                        mid_x = int((int(curr_center[0]) + int(nbr_pos[0])) // 2)
                        mid_y = int((int(curr_center[1]) + int(nbr_pos[1])) // 2)
                        mid_x_o = int(mid_x * scale_x)
                        mid_y_o = int(mid_y * scale_y)
                        cv2.putText(frame_hd, f"{int(d)}", (mid_x, mid_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, line_colors[k], 1)
                        cv2.putText(frame_save, f"{int(d)}", (mid_x_o, mid_y_o),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, line_colors[k], 1)
                    else:
                        neighbor_ids.append("")
                        neighbor_dists.append(0)
                        neighbor_speeds.append(0)
                        neighbor_directions.append([0, 0])
                        neighbor_accelerations.append(0)
                        neighbor_positions.append([0, 0])
                        neighbor_sizes.append([0, 0])
                vehicle["neighbors_ids"] = neighbor_ids
                vehicle["neighbors_distances"] = neighbor_dists
                vehicle["neighbors_speeds"] = neighbor_speeds
                vehicle["neighbors_directions"] = neighbor_directions
                vehicle["neighbors_accelerations"] = neighbor_accelerations
                vehicle["neighbors_positions"] = neighbor_positions
                vehicle["neighbors_sizes"] = neighbor_sizes
                vehicle["neighbors_info"] = neighbors
                vehicle["frame"] = frame_index

    DIRECTION_VECTOR_LENGTH = 20
    for vehicle in frame_data:
        center_x, center_y = vehicle["bbox"][0], vehicle["bbox"][1]
        center = [center_x, center_y]
        direction = vehicle["direction"]
        if direction != [0, 0]:
            arrow_end = (int(center[0] + direction[0] * DIRECTION_VECTOR_LENGTH),
                         int(center[1] + direction[1] * DIRECTION_VECTOR_LENGTH))
            draw_filled_arrow_line(frame_hd, (int(center[0]), int(center[1])), arrow_end,
                                   (0, 255, 255), thickness=2, tip_length=5)
            center_o = (int(center[0] * scale_x), int(center[1] * scale_y))
            arrow_end_o = (int(arrow_end[0] * scale_x), int(arrow_end[1] * scale_y))
            draw_filled_arrow_line(frame_save, center_o, arrow_end_o,
                                   (0, 255, 255), thickness=2, tip_length=int(5*scale_x))
    # --- [Step 3] Union BBox 계산 ---
    for vehicle in frame_data:
        center_x, center_y, width, height = vehicle["bbox"]
        # 현재 차량의 bbox를 xyxy 형식으로 변환
        x1 = center_x - width / 2
        y1 = center_y - height / 2
        x2 = center_x + width / 2
        y2 = center_y + height / 2
        
        union_bbox_xyxy = [x1, y1, x2, y2]  # 시작은 자신의 bbox (xyxy 형식)
        
        for neighbor in vehicle.get("neighbors_info", []):
            if len(neighbor) >= 4:
                nbr_bbox = neighbor[3]  # neighbors_info에서는 xyxy 형식으로 저장됨
                
                union_bbox_xyxy[0] = min(union_bbox_xyxy[0], nbr_bbox[0])
                union_bbox_xyxy[1] = min(union_bbox_xyxy[1], nbr_bbox[1])
                union_bbox_xyxy[2] = max(union_bbox_xyxy[2], nbr_bbox[2])
                union_bbox_xyxy[3] = max(union_bbox_xyxy[3], nbr_bbox[3])
        
        # union_bbox를 center_x, center_y, width, height 형식으로 변환
        union_center_x = (union_bbox_xyxy[0] + union_bbox_xyxy[2]) / 2
        union_center_y = (union_bbox_xyxy[1] + union_bbox_xyxy[3]) / 2
        union_width = union_bbox_xyxy[2] - union_bbox_xyxy[0]
        union_height = union_bbox_xyxy[3] - union_bbox_xyxy[1]
        
        vehicle["union_bbox"] = [union_center_x, union_center_y, union_width, union_height]

    cv2.putText(frame_hd, f"Frame: {frame_index} FPS: {int(fps)}", (10, frame_hd.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame_save, f"Frame: {frame_index} FPS: {int(fps)}", (10, frame_save.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    EXCLUSION_ZONE_COLOR = (100, 100, 100)
    for zone in exclusion_zones:
        zone_points_hd = np.array([[int(p[0] * hd_width), int(p[1] * hd_height)] for p in zone], np.int32)
        zone_points_hd = zone_points_hd.reshape((-1, 1, 2))
        overlay_hd = frame_hd.copy()
        cv2.fillPoly(overlay_hd, [zone_points_hd], EXCLUSION_ZONE_COLOR)
        alpha = 0.3
        cv2.addWeighted(overlay_hd, alpha, frame_hd, 1 - alpha, 0, frame_hd)
        cv2.polylines(frame_hd, [zone_points_hd], isClosed=True, color=(50, 50, 50), thickness=2)
        zone_points_orig = np.array([[int(p[0] * orig_width), int(p[1] * orig_height)] for p in zone], np.int32)
        zone_points_orig = zone_points_orig.reshape((-1, 1, 2))
        overlay_orig = frame_save.copy()
        cv2.fillPoly(overlay_orig, [zone_points_orig], EXCLUSION_ZONE_COLOR)
        cv2.addWeighted(overlay_orig, alpha, frame_save, 1 - alpha, 0, frame_save)
        cv2.polylines(frame_save, [zone_points_orig], isClosed=True, color=(50, 50, 50), thickness=int(2 * scale_x))
    # --- [Step 4] Labeling Crop 저장 ---
    for vehicle in frame_data:
        # union_bbox를 xyxy 형식으로 변환
        union_center_x, union_center_y, union_width, union_height = vehicle["union_bbox"]
        union_x1 = max(0, int(union_center_x - union_width / 2))
        union_y1 = max(0, int(union_center_y - union_height / 2))
        union_x2 = min(orig_width, int(union_center_x + union_width / 2))
        union_y2 = min(orig_height, int(union_center_y + union_height / 2))
        
        if union_x2 > union_x1 and union_y2 > union_y1:
            crop_img = frame_hd[union_y1:union_y2, union_x1:union_x2]
            vehicle_folder = os.path.join(labeling_dir, str(vehicle["object_id"]))
            os.makedirs(vehicle_folder, exist_ok=True)
            crop_filename = f"{frame_counter:08d}_{vehicle['object_id']}.jpg"
            crop_out_path = os.path.join(vehicle_folder, crop_filename)
            cv2.imwrite(crop_out_path, crop_img)
    cv2.imwrite(os.path.join(image_dir, f"{frame_counter:08d}.jpg"), frame_save)
    video_writer.write(frame_save)

    cv2.imshow("YOLO11 Tracking", frame_hd)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print(f"Saving data to JSON at frame {frame_index}...")
        print(f"Current output_data length: {len(output_data)}")
        json_out_path = os.path.join(json_dir, "vehicle_data.json")
        try:
            with open(json_out_path, "w") as f:
                json.dump(output_data, f)
            print(f"JSON successfully saved to: {json_out_path}")
        except Exception as e:
            print(f"Error saving JSON: {e}")
        break
    
    output_data.extend(frame_data)
    frame_counter += 1
    frame_index += 1

cap.release()
video_writer.release()
cv2.destroyAllWindows()

# 정규화 전 데이터 갯수 출력
print(f"정규화 전 데이터 갯수: {len(output_data)}")

# 저장하기 전에 데이터 정규화
print("좌표값 정규화 수행 중...")
output_data = normalize_coordinates(output_data, hd_width, hd_height)
print(f"좌표값 정규화 완료: {len(output_data)}개 데이터 처리됨")

# 정규화된 데이터 저장
json_out_path = os.path.join(json_dir, "vehicle_data.json")
with open(json_out_path, "w") as f:
    json.dump(output_data, f)
print(f"정규화된 JSON 데이터 저장 완료: {json_out_path}")
