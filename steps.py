import cv2
import json
import os
from ultralytics import YOLO
import numpy as np # 한글출력을 위한 추가
from PIL import ImageFont, ImageDraw, Image # 한글 출력을 위해 추가

# ---------------------------------------------------------
# 한글 출력을 위한 함수
# 터미널 명령어 pip install pillow 실행
# ---------------------------------------------------------
def draw_korean_text(img, text, pos, font_size, color):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    # 윈도우 기본 폰트 경로 (맑은 고딕)
    font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", font_size)
    draw.text(pos, text, font=font, fill=color)
    return np.array(img_pil)
# ---------------------------------------------------------

# 1. 경로 설정
#model_path = r'C:\ARdata_Python_Server\AI_server_2\best.pt'
#json_path = r'C:\ARdata_Python_Server\AI_server_2\steps.json'
model_path = r'C:\lego_test\test\AI_server_2\best.pt'     
json_path = r'C:\lego_test\test\AI_server_2\steps.json'
# model_path = r'C:\lego_test\test\best.pt'
# json_path = r'C:\lego_test\test\steps.json'
model = YOLO(model_path)
with open(json_path, 'r', encoding='utf-8') as f:
    steps = json.load(f)

current_step_idx = 0
confirm_counter = 0
CONFIRM_THRESHOLD = 25  # 약 1~1.5초 동안 '연속'으로 감지되어야 인정

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)

    # 1. 현재 타겟 설정
    if current_step_idx < len(steps):
        target = steps[current_step_idx]["target"]
        guide_msg = steps[current_step_idx]["guide"]
    else:
        target = None
        guide_msg = "🎉 모든 조립 완료!"

    # 2. YOLO 추론 (신뢰도 높임)
    results = model.predict(frame, conf=0.7, verbose=False)
    
    detected_this_frame = False
    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 💡 핵심 로직: 현재 단계의 타겟만 노란색으로 표시, 나머지는 무시하거나 파란색
            if label == target:
                detected_this_frame = True
                color = (0, 255, 255) # 타겟은 노란색
                # 조립 가이드 화살표나 강조 표시를 여기에 추가할 수 있음
            else:
                color = (255, 0, 0) # 나머지는 파란색

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 3. 안정성 검사 (연속 인식 성공 시에만 다음 단계)
    if target and detected_this_frame:
        confirm_counter += 1
        # 화면에 진행률 표시 (게이지)
        cv2.rectangle(frame, (20, 60), (20 + (confirm_counter * 10), 75), (0, 255, 0), -1)
        
        if confirm_counter >= CONFIRM_THRESHOLD:
            print(f"✅ {target} 조립 완료 확인!")
            current_step_idx += 1
            confirm_counter = 0
    else:
        confirm_counter = 0 # 화면에서 사라지면 카운터 리셋

    # 💡 [핵심 변경점] cv2.putText 대신 한글 출력 함수 사용
    # BGR 색상 체계이므로 (0, 255, 0)은 초록색, (255, 255, 255)는 흰색입니다.
    frame = draw_korean_text(frame, guide_msg, (20, 20), 25, (0, 255, 0))
    cv2.imshow("LEGO AR Guide", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()