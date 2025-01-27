from ultralytics import YOLO
import os
import xml.etree.ElementTree as ET
import random
import shutil
import yaml
import torch
import datetime
import cv2
import base64
import numpy as np
import os
import cv2
import csv
import numpy as np
from ultralytics import YOLO
import ollama
from collections import defaultdict
curr_dir = os.getcwd()
origin_dir = "/app"

# car_crop 이미지를 Base64로 인코딩하는 함수
def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.png', image)  # 이미지를 PNG 형식으로 인코딩
    encoded_image = base64.b64encode(buffer).decode('utf-8')  # Base64로 변환
    return encoded_image

# 모델 로드
# GPU 사용 가능 여부 확인
device = "cuda" if torch.cuda.is_available() else "cpu"

# YOLO 모델 로드 및 디바이스 설정
model = YOLO("yolo11x.pt").to(device)
print(type(model.names),len(model.names))
print(f"Using device: {model.device}")

moving_target_object_l = ['car','bus','truck']
detect_target_object_l = ['person']
final_file_name = "detecting_stop.csv"
results_dir = os.path.join(origin_dir,"yolo_structure","results")
status_csv = os.path.join(results_dir,final_file_name)

with open(status_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["datetime", "distance","event","detected", "source_name"])  # CSV 헤더 작성

# 비디오 경로 설정
# video_path = "test/traffic.mp4"
video_path = os.path.join(origin_dir,"yolo_structure","test") 
# test_video_file = os.path.join(video_path,"traffic.mp4")
test_video_file = os.path.join(video_path,"fewers.mp4")
output_path = f"{results_dir}/detecting_stop_{video_path.split('/')[-1].split('.')[0]}.mp4"

# 비디오 읽기
cap = cv2.VideoCapture(test_video_file)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 비디오 저장 설정
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 이전 위치 추적용 딕셔너리
previous_positions = {}

# 정지 상태 감지 임계값 (픽셀 이동 거리)
STOP_THRESHOLD = 20  # 픽셀
LANE_CHECK_FRAME = 5

reverse_direction = ""
# 프레임 처리
frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    frame_num +=1
    if not ret:
        break
    if frame_num == LANE_CHECK_FRAME:
        ## 레인 방향을 체크!!
        encoded_frame = encode_image_to_base64(frame) 
        res = ollama.chat(
            model="llama3.2-vision:11b",
            messages=[
                {
                    'role': 'user',
                    'content': """I want to know the correct direction in which the vehicle is moving. 
                    The answer should be simple words like "left to right," "right to left," "top to bottom," or "bottom to top."
                    """,
                    'images': [encoded_frame]
                }
            ]
        )
        reverse_direction =  res['message']['content']
        print(">>>>>>>>>>>>>>>>>>  ", reverse_direction)

    # YOLO 추적 수행
    results = model.track(frame, conf=0.5, show=False, tracker="bytetrack.yaml")

    # 추적된 객체들 그리기
    for result in results:
        for box in result.boxes:
            
            cls = int(box.cls)  # 클래스 ID
            track_id = int(box.id)   # 추적 ID
            confidence = box.conf[0]  # 신뢰도

            class_name = result.names[cls]
            if class_name in moving_target_object_l:  # 'car' 클래스만 Ollama 호출
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 경계 상자 좌표
                label = f"{class_name} ID {track_id} ({confidence:.2f})"
                final_class_name = class_name
                # 이전 위치와 비교하여 정지 여부 확인
                if track_id in previous_positions:
                    prev_x, prev_y = previous_positions[track_id]
                    # 이동 거리 계산 (Euclidean Distance)
                    distance = ((x1 - prev_x) ** 2 + (y1 - prev_y) ** 2) ** 0.5
    
                    if distance < STOP_THRESHOLD:  # 정지 상태
                        this_event = "STOP"
                        color = (255, 0, 0)  # 파란색
                        final_class_name = class_name
                        # if original_class_name == "car":  # 'car' 클래스만 Ollama 호출                        
                        #     ## DETECTING TYPE
                        #     car_crop = frame[y1:y2, x1:x2]
                        #     encoded_car_crop = encode_image_to_base64(car_crop)  # 이미지를 Base64로 인코딩
                        #     # Ollama API 호출
                        #     res = ollama.chat(
                        #         model="llama3.2-vision:11b",
                        #         messages=[
                        #             {
                        #                 'role': 'user',
                        #                 'content': "tell me what is in this image. Answer only one in [SEDAN/SUV/VAN/TRUCK/BUS] answer in one word",
                        #                 'images': [encoded_car_crop]
                        #             }
                        #         ]
                        #     )
            
                        #     final_class_name = res['message']['content'].strip().upper().replace(".", "").replace(" ", "")
                        # else:
                        #     final_class_name = class_name

                        label = f"{final_class_name} ID {track_id} ({confidence:.2f})"
                        label += f" [{this_event}]"
                        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                        with open(status_csv, mode="a", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow([current_time,00,this_event, final_class_name, os.path.basename(test_video_file)])
                                                
                    elif y1 < prev_y:  # 역주행 (y축 감소)
                        this_event = "REVERSE"
                        final_class_name = class_name
                        color = (0, 0, 255)  # 빨간색
                        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                        with open(status_csv, mode="a", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow([current_time,00,this_event, final_class_name, os.path.basename(test_video_file)])  
                        label = f"{final_class_name}  ID {track_id} ({confidence:.2f})"
                        label += f" [{this_event}]"

                    else:
                        color = (0, 255, 0)  # 초록색 (정상 이동)
                else:
                    color = (0, 0, 0)  # 검정색 - 트래킹안댐
                # 현재 좌표 저장
                previous_positions[track_id] = (x1, y1)
                # 경계 상자와 라벨 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 20),  # 텍스트 위치를 상단으로 더 띄움
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,  # 텍스트 크기를 키움
                    color,  # 텍스트 색상 (경계 상자 색상과 동일)
                    3,  # 텍스트 두께를 키움
                )
            elif class_name in detect_target_object_l: 
                this_event = class_name.upper()
                final_class_name = class_name
                color =  (255, 105, 180) # 핑크!!
                current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                with open(status_csv, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([current_time,00,this_event, final_class_name, os.path.basename(test_video_file)])                             

                label = f"{final_class_name} ({confidence:.2f})"
                label += f" [{this_event}]"


                # 경계 상자와 라벨 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 20),  # 텍스트 위치를 상단으로 더 띄움
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,  # 텍스트 크기를 키움
                    color,  # 텍스트 색상 (경계 상자 색상과 동일)
                    3,  # 텍스트 두께를 키움
                )

    # 출력 비디오에 프레임 저장
    out.write(frame)

# 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()
print(reverse_direction)
