import os
from collections import Counter
import random
import shutil
import hashlib
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
# frame_hash 생성 함수
def generate_frame_hash(frame, video_file):
    combined_string = f"{frame}_{video_file}"
    return hashlib.sha256(combined_string.encode()).hexdigest()

# 모델 로드
# GPU 사용 가능 여부 확인
device = "cuda" if torch.cuda.is_available() else "cpu"

# 중요번수 설정
STOP_THRESHOLD = 1  # 픽셀
REVERSE_THRESHOLD = 1
LANE_CHECK_FRAME = 5
FAST = True
moving_target_object_l = ['car','bus','truck','bicycle','motorcycle']
detect_target_object_l = ['person','cow']


# YOLO 모델 로드 및 디바이스 설정
model = YOLO("yolo11x.pt").to(device)
print(type(model.names),len(model.names))
print(f"Using device: {model.device}")


# 비디오 경로 설정
video_path = os.path.join(origin_dir,"yolo_structure","test") 
test_video_file = os.path.join(video_path,"traffic.mp4")
test_video_file = os.path.join(video_path,"fewers.mp4")
# test_video_file = os.path.join(video_path,"oneway.mp4")
# test_video_file = os.path.join(video_path,"korea.mp4")
# test_video_file = os.path.join(video_path,"cow.mp4")




## 최종 저장경로 설정!!
results_dir = os.path.join(origin_dir,"yolo_structure","results")
## 사건파일저장
crop_save_dir = os.path.join(results_dir,"frame_img") 

video_unique = test_video_file.split('/')[-1].split('.')[0]
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if FAST:
    final_file_name = f"{video_unique}_fast_{current_time}.csv"
    status_csv = os.path.join(results_dir,final_file_name)
    output_path = f"{results_dir}/{video_unique}_fast_{current_time}.mp4"
else:
    final_file_name = f"{video_unique}_llm_{current_time}.csv"
    status_csv = os.path.join(results_dir,final_file_name)
    output_path = f"{results_dir}/{video_unique}_llm_{current_time}.mp4"
with open(status_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["datetime", "distance","event","detected", "source_name","frame_hash"])  # CSV 헤더 작성


# 비디오 읽기
cap = cv2.VideoCapture(test_video_file)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 총 프레임 수 가져오기
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total number of frames: {total_frames}")

# 비디오 저장 설정
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 이전 위치 추적용 딕셔너리
previous_positions = {}


final_dir = ""
reverse_direction_l = []
# 프레임 처리
frame_num = 0
while cap.isOpened():
    progress_percent = int((frame_num / total_frames) * 100)
    print(f"Progress : {progress_percent} % || {frame_num} / {total_frames}")
    ret, frame = cap.read()
    frame_num +=1
    if frame_num > 10000000:
        frame_num = 0
    if not ret:
        break
    # if (frame_num >= LANE_CHECK_FRAME) and (frame_num <= LANE_CHECK_FRAME+10):
    #     ## 레인 방향을 체크!!
    #     encoded_frame = encode_image_to_base64(frame) 
    #     res = ollama.chat(
    #         model="llama3.2-vision:11b",
    #         messages=[
    #             {
    #                 'role': 'user',
    #                 'content': """I want to know the correct direction in which the vehicle is moving. 
    #                 The answer should be simple words wold be only one of these 
    #                 ["top to bottom," "bottom to top","left to right," "right to left," ]
    #                 Please make the judgment based on the arrows on the road and the direction of the vehicle's movement.
    #                 """,
    #                 'images': [encoded_frame]
    #             }
    #         ]
    #     )
    #     reverse_direction =  res['message']['content']
    #     if "RIGHTTOLEFT" in reverse_direction.replace(" ","").upper():
    #         dir_res = "RIGHTTOLEFT"
    #     elif "LEFTTORIGHT" in reverse_direction.replace(" ","").upper():
    #         dir_res = "LEFTTORIGHT"
    #     elif "TOPTOBOTTOM" in reverse_direction.replace(" ","").upper():
    #         dir_res = "TOPTOBOTTOM"
    #     elif "BOTTOMTOTOP" in reverse_direction.replace(" ","").upper():
    #         dir_res = "BOTTOMTOTOP"
    #     else:
    #         dir_res = "ERROR"

    #     reverse_direction_l.append(dir_res)
    #     print(">>>>>>>>>>>>>>>>>>  ", dir_res)

    # if frame_num == LANE_CHECK_FRAME+12:
    #     counter = Counter(reverse_direction_l)
    #     # 가장 많이 중복된 값 찾기
    #     final_dir, count = counter.most_common(1)[0]

    #     print(f"Final dir: {final_dir} (Count: {count})")
    final_dir = "TOPTOBOTTOM"
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
                # 트래킹이 될떄~!!
                if track_id in previous_positions:
                    prev_x, prev_y = previous_positions[track_id]
                    # 이동 거리 계산 (Euclidean Distance)
                    distance = ((x1 - prev_x) ** 2 + (y1 - prev_y) ** 2) ** 0.5
    
                    if distance < STOP_THRESHOLD:  # 정지 상태
                        this_event = "STOP"
                        color = (255, 0, 0)  # 파란색
                        car_crop = frame[y1:y2, x1:x2]

                        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                        combined_string = f"{video_unique}_{this_event}_{current_time}"
                        frame_unique_key = combined_string # hashlib.sha256(combined_string.encode()).hexdigest()    

                        if FAST:
                            final_class_name = class_name
                            crop_filename = os.path.join(crop_save_dir, f"{frame_unique_key}.jpg")
                            cv2.imwrite(crop_filename, car_crop)

                        else:
                            if class_name == "car":  # 'car' 클래스만 Ollama 호출                        
                                ## DETECTING TYPE
                                encoded_car_crop = encode_image_to_base64(car_crop)  # 이미지를 Base64로 인코딩
                                # Ollama API 호출
                                res = ollama.chat(
                                    model="llama3.2-vision:11b",
                                    messages=[
                                        {
                                            'role': 'user',
                                            'content': "tell me what is in this image. Answer only one in [SEDAN/SUV/VAN/TRUCK/BUS] answer in one word",
                                            'images': [encoded_car_crop]
                                        }
                                    ]
                                )
                
                                final_class_name = res['message']['content'].strip().upper().replace(".", "").replace(" ", "")
                                print(">>>>>>>>>>> ", final_class_name)
                            else:
                                final_class_name = class_name + f" [{this_event}]"

                        label = f"{final_class_name} ID {track_id} ({confidence:.2f})"
                        label += f" [{this_event}]"
                        


                        with open(status_csv, mode="a", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow([current_time,00,this_event, final_class_name, os.path.basename(test_video_file), frame_unique_key])

                    elif  len(final_dir.replace(" ","").upper()) >5:
                        # 여기..위의 조건은 별로 의미 없기는 한데 아래를 하나의 IF 로 묶기위해서 함!
                        if "RIGHTTOLEFT" in final_dir.replace(" ","").upper():
                            reverse_condition = x1 > prev_x + REVERSE_THRESHOLD  # x 좌표 증가 시 역주행
                        elif "LEFTTORIGHT" in final_dir.replace(" ","").upper():
                            reverse_condition = x1 < prev_x - REVERSE_THRESHOLD  # x 좌표 감소 시 역주행
                        elif "TOPTOBOTTOM" in final_dir.replace(" ","").upper():
                            reverse_condition = y1 < prev_y - REVERSE_THRESHOLD # y 좌표 감소 시 역주행
                        elif "BOTTOMTOTOP" in final_dir.replace(" ","").upper():
                            reverse_condition = y1 > prev_y + REVERSE_THRESHOLD # y 좌표 증가 시 역주행
                        else:
                            reverse_condition = False

                        if reverse_condition: # 조건이 맞으면!
                            this_event = "REVERSE"
                            car_crop = frame[y1:y2, x1:x2]
                            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                            combined_string = f"{video_unique}_{this_event}_{current_time}"
                            # hashed_key = hashlib.sha256(combined_string.encode()).hexdigest()
                            frame_unique_key = combined_string 

                            if FAST:
                                final_class_name = class_name
                                crop_filename = os.path.join(crop_save_dir, f"{frame_unique_key}.jpg")
                                cv2.imwrite(crop_filename, car_crop)  


                            else:
                                if class_name == "car":  # 'car' 클래스만 Ollama 호출                        
                                    ## DETECTING TYPE
                                    
                                    encoded_car_crop = encode_image_to_base64(car_crop)  # 이미지를 Base64로 인코딩
                                    # Ollama API 호출
                                    res = ollama.chat(
                                        model="llama3.2-vision:11b",
                                        messages=[
                                            {
                                                'role': 'user',
                                                'content': "tell me what is in this image. Answer only one in [SEDAN/SUV/VAN/TRUCK/BUS] answer in one word",
                                                'images': [encoded_car_crop]
                                            }
                                        ]
                                    )
                    
                                    final_class_name = res['message']['content'].strip().upper().replace(".", "").replace(" ", "")
                                else:
                                    final_class_name = class_name + f" [{this_event}]"

                            color = (0, 0, 255)  # 빨간색
                            with open(status_csv, mode="a", newline="") as file:
                                writer = csv.writer(file)
                                writer.writerow([current_time,00,this_event, final_class_name, os.path.basename(test_video_file), frame_unique_key])

                            label = f"{final_class_name}  ID {track_id} ({confidence:.2f})"
                            label += f" [{this_event}]"
                        else:
                            # 여기도 역주행이 아니면 초록색 해줘야해
                            color = (0, 255, 0)  # 초록색 (정상 이동)

                    else:
                        color = (0, 255, 0)  # 초록색 (정상 이동)
                else:
                    color = (0, 0, 0)  # 검정색 - 트래킹안댐
                # 현재 좌표 저장
                previous_positions[track_id] = (x1, y1)

            elif class_name in detect_target_object_l: 
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 경계 상자 좌표
                this_event = class_name.upper()
                final_class_name = class_name
                color =  (255, 105, 180) # 핑크!!
                current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                combined_string = f"{video_unique}_{this_event}_{current_time}"
                # hashed_key = hashlib.sha256(combined_string.encode()).hexdigest()
                frame_unique_key = combined_string 
                with open(status_csv, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([current_time,00,this_event, final_class_name, os.path.basename(test_video_file), frame_unique_key])

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
print(f"{output_path} , FRAME NUM : {frame_num}")
print(final_dir)

