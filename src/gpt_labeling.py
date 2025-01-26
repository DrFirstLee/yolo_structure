import openai
import base64
import os
import shutil

from PIL import Image
from io import BytesIO  
import sys
import os
print(os.getcwd())
# 현재 디렉토리의 상위 디렉토리 경로 추가
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)

import mykey
openai_client = openai.OpenAI(api_key = mykey.OPENAI_KEY)
using_model = 'gpt-4o-mini'
import xml.etree.ElementTree as ET
# 53개 이미지에 0.02$ , 30원 ! 이미지당 1원 안댐~!

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
# 이미지 영역을 잘라서 Base64로 인코딩하는 함수
def encode_cropped_image(image_path, coordinates):
    try:
        # 이미지 열기
        with Image.open(image_path) as img:
            # RGBA 모드일 경우 RGB로 변환
            if img.mode == "RGBA":
                img = img.convert("RGB")
            
            # 좌표 가져오기
            xtl, ytl, xbr, ybr = map(float, [
                coordinates["xtl"],
                coordinates["ytl"],
                coordinates["xbr"],
                coordinates["ybr"]
            ])
            # 이미지 자르기
            cropped_img = img.crop((xtl, ytl, xbr, ybr))
            
            # 이미지를 메모리에 저장 후 Base64로 인코딩
            buffer = BytesIO()
            cropped_img.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error processing {image_path} with coordinates {coordinates}: {e}")
        return None

origin_dir = "/app"

xml_file_dir = f"{origin_dir}/aihub/all_data/label/"
xml_area_l = []
final_img_dir =  f"{origin_dir}/yolo_structure/origin_img"
imsi_cnt = 0
for xml_file_name in os.listdir(xml_file_dir):
    imsi_cnt += 1
    available_area_l = ["Cheonan", "Hongcheon", "Gangneung"]
    
    if not any(area in xml_file_name for area in available_area_l):
        continue  # 키워드가 하나도 포함되지 않으면 반복문의 다음 단계로 넘어감
        
    xml_root = xml_file_name.replace(".xml","")

    area_name = xml_file_name.split("_")[0]
    if area_name not in xml_area_l:
        xml_area_l.append(area_name)    
    adjust_xml_file = f"{origin_dir}/yolo_structure/adjust_label_xml/{xml_file_name}"
    # print(f"{xml_file_name} >> {adjust_xml_file}")
    
    # 이미지 파일이 없다면 끝까지 ""야!!!!
    final_img_save = ""
    
    # XML 파싱
    tree = ET.parse(xml_file_dir + xml_file_name)
    root = tree.getroot()
         
    image_folder = f"{origin_dir}/aihub/all_data/img"
    # car 레이블 추출
    car_labels = []
    for image in root.findall("image"):
        image_name = image.attrib.get("name")
        image_path = f"{image_folder}/{image_name}"
        
        if not os.path.exists(image_path):
            continue
        encoded_image = encode_image(image_path)  # 이미지 인코딩
        for box in image.findall("box"):
            if box.attrib.get("label") == "car":
                coordinates = {
                    "xtl": box.attrib.get("xtl"),
                    "ytl": box.attrib.get("ytl"),
                    "xbr": box.attrib.get("xbr"),
                    "ybr": box.attrib.get("ybr"),
                }
                # Boundary 영역만 Base64 인코딩
                encoded_image = encode_cropped_image(image_path, coordinates)
                
                car = {
                    "image_name": image_name,
                    "coordinates": {
                        "xtl": box.attrib.get("xtl"),
                        "ytl": box.attrib.get("ytl"),
                        "xbr": box.attrib.get("xbr"),
                        "ybr": box.attrib.get("ybr"),
                    },
                    "z_order": box.attrib.get("z_order"),
                    "encoded_image": encoded_image,  # Base64 인코딩된 이미지
                }
                car_labels.append(car)
                messages = [
                {
                    "type": "text",
                    "text": '이미지를 바탕으로 차량이 SUV/sedan/VAN/compact_car 중 무엇인지 구분해서 SUV/sedan/VAN/compact_car/etc 중 하나의 단어로만. 해당지역에서 가장 보편적인 차량은 sedan 임을 참고해. 차가 확실하게 아닐경우는 X 로 애매하면 sedan으로해줘'
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{car['encoded_image']}"}
                }
                ]
                # OpenAI API 요청
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": messages
                        }
                    ],
                    max_tokens=300
                )
                # 결과 출력
                img_analy = response.choices[0].message.content
                print(f"{image_path} >> result : {img_analy}")            
                # XML 내 label 업데이트
                box.set("label", img_analy)    
                # 수정된 XML 저장
                tree.write(adjust_xml_file, encoding="utf-8", xml_declaration=True)
                # final_img_save = os.path.join(final_img_dir , image_name)
                # Copy the final file
            else:
                # car가없이 트럭 버스만 있음!!
                1==1
            # 최종 저장은 1번만하면되!
            # tree.write(adjust_xml_file, encoding="utf-8", xml_declaration=True)
            final_img_save = os.path.join(final_img_dir , image_name)
            shutil.copy(image_path ,final_img_save  )
    if final_img_save == "":
        continue
        
    print(f"===final resul : XML : {adjust_xml_file}, img :{final_img_save} // Total area : {xml_area_l}")