import asyncio
import websockets
import base64
import json
import cv2
import numpy as np
import os
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image

# ---------------------------------------------------------
# 한글 출력을 위한 함수
# ---------------------------------------------------------
def draw_korean_text(img, text, pos, font_size, color):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    # 윈도우 기본 폰트 경로 (맑은 고딕)
    font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", font_size)
    draw.text(pos, text, font=font, fill=color)
    return np.array(img_pil)
# ---------------------------------------------------------

## 1. 경로 설정
model_path = r'C:\ARdata_Python_Server\AI_server_2\best.pt'
json_path = r'C:\ARdata_Python_Server\AI_server_2\steps.json'
# model_path = r'C:\lego_test\test\AI_server_2\best.pt'     
# json_path = r'C:\lego_test\test\AI_server_2\steps.json'
# model_path = r'C:\lego_test\test\best.pt'
# json_path = r'C:\lego_test\test\steps.json'

model = YOLO(model_path)
with open(json_path, 'r', encoding='utf-8') as f:
    steps = json.load(f)

# 전역 상태 변수 (조립 단계 및 카운터)
current_step_idx = 0
confirm_counter = 0
CONFIRM_THRESHOLD = 25  # 약 1~1.5초 동안 '연속'으로 감지되어야 인정

async def handle_client(websocket):
    # 전역 변수를 함수 안에서 수정하기 위해 global 선언
    global current_step_idx, confirm_counter
    
    print(f"🔌 유니티 클라이언트 연결됨! 주소: {websocket.remote_address}")
    
    try:
        async for message in websocket:
            # 1. 유니티에서 보낸 Base64 이미지를 해독하여 OpenCV 프레임으로 변환
            image_data = base64.b64decode(message) if isinstance(message, str) else message
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                await websocket.send(json.dumps({"status": "error", "msg": "이미지 깨짐"}))
                continue
            
            # (선택) 유니티에서 회전해서 오지만, 필요시 서버에서도 회전 가능
            # frame = cv2.flip(frame, 1)

            # 2. 현재 타겟 설정
            if current_step_idx < len(steps):
                target = steps[current_step_idx]["target"]
                guide_msg = steps[current_step_idx]["guide"]
            else:
                target = None
                guide_msg = "🎉 모든 조립 완료!"

            # 3. YOLO 추론 (신뢰도 0.7)
            results = model.predict(frame, conf=0.7, verbose=False)
            
            detected_this_frame = False
            detections_list = [] # 유니티로 보낼 데이터를 담을 리스트

            for r in results:
                for box in r.boxes:
                    label = model.names[int(box.cls[0])]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

                    # 유니티로 보낼 중앙 좌표 계산
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # 유니티 전달용 리스트에 추가
                    detections_list.append({
                        "label": label, 
                        "box": [x1, y1, x2, y2], 
                        "center": [center_x, center_y], 
                        "conf": round(conf, 2)
                    })

                    # 💡 현재 단계의 타겟만 노란색으로 표시, 나머지는 파란색
                    if label == target:
                        detected_this_frame = True
                        color = (0, 255, 255) # 노란색
                    else:
                        color = (255, 0, 0) # 파란색

                    # 서버쪽 모니터링 화면에 박스 그리기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 4. 안정성 검사 (연속 인식 성공 시에만 다음 단계)
            if target and detected_this_frame:
                confirm_counter += 1
                # 서버 화면에 진행률 표시 (게이지)
                cv2.rectangle(frame, (20, 60), (20 + (confirm_counter * 10), 75), (0, 255, 0), -1)
                
                if confirm_counter >= CONFIRM_THRESHOLD:
                    print(f"✅ [{target}] 조립 완료 확인! 다음 단계로 넘어갑니다.")
                    current_step_idx += 1
                    confirm_counter = 0
            else:
                confirm_counter = 0 # 화면에서 타겟이 사라지면 카운터 리셋

            # 5. 서버 디버깅용 화면 출력 (한글 메시지 포함)
            frame = draw_korean_text(frame, guide_msg, (20, 20), 25, (0, 255, 0))
            cv2.imshow("LEGO AR Server Monitor", frame)
            
            # 서버 창 닫기 키 (q)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

            # 🚀 6. 최종 결과를 JSON으로 묶어서 유니티로 응답
            response = {
                "status": "success",
                "current_step": current_step_idx, # 현재 단계 번호
                "guide_msg": guide_msg,           # 현재 단계 가이드 텍스트 (예: "빨간 블록을 찾으세요")
                "results": detections_list        # 인식된 모든 박스 좌표들
            }
            await websocket.send(json.dumps(response))

    except Exception as e:
        print(f"⚠️ 연결 종료됨: {e}")
    finally:
        cv2.destroyAllWindows()

async def main():
    # 유니티 코드(ARImageSender.cs)에 포트가 8000으로 되어 있으므로 8000번 사용
    port = 8000
    async with websockets.serve(handle_client, "0.0.0.0", port):
        print(f"📡 AI AR 조립 가이드 서버 시작됨 (포트: {port}) - 유니티 연결 대기 중...")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())