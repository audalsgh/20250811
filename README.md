# 3일간 Roboflow 차선인식 프로젝트 진행 

## SegFormer 모델은 런파드에서 실습해볼것.
1. YOLOv11 segmentation 사물인식 19개정도를 ai studio에 물어봐서 colab에서 실행하기.
```
# 3. 라이브러리 임포트 및 모델 로드
import torch
import cv2
import time
from ultralytics import YOLO

# GPU 사용 가능 여부 확인 및 장치 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"사용 장치: {device}")

# YOLO 분할 모델 로드
# --- 여기를 수정했습니다! ---
# 'YOLO11x-seg.pt'를 실제 파일 이름인 'yolo11x-seg.pt' (소문자)로 변경했습니다.
model = YOLO('yolo11x-seg.pt').to(device)

# 4. 비디오 추론 및 결과 파일로 저장

# 입력 비디오 열기
video_path = "driving_video.mp4"
cap = cv2.VideoCapture(video_path)

# 출력 비디오 설정
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_in = cap.get(cv2.CAP_PROP_FPS)
# 결과 파일 이름을 바꿔서 이전 결과와 헷갈리지 않도록 합니다.
output_path = "driving_video_result_yolo11.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 코덱 설정
out = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))

# 진행 상황을 알리는 메시지
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"총 {total_frames} 프레임의 영상 처리를 시작합니다. 완료 후 '{output_path}' 파일이 생성됩니다.")

# 프레임 처리 루프
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 모델 추론 실행
    results = model.predict(frame, device=device, verbose=False)

    # 결과(분할 마스크, 경계 상자 등)를 프레임에 그리기
    annotated_frame = results[0].plot()

    # 처리된 프레임을 비디오 파일에 쓰기
    out.write(annotated_frame)
    
    frame_count += 1
    if frame_count % 100 == 0: # 100 프레임마다 진행 상황 출력
        print(f"{frame_count} / {total_frames} 프레임 처리 중...")

# 완료 메시지 출력
print(f"영상 처리가 완료되었습니다. 결과는 {output_path} 파일로 저장되었습니다.")

# 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()
```

2. 허깅페이스 코드는 런파드.
```
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

def process_video_all_objects(input_video, output_video):
    """모든 객체를 색상별로 표시하는 세그멘테이션"""

    print(f"🎬 {input_video} 처리 시작...")

    # 1. Hugging Face 세그멘테이션 모델 로드
    try:
        segmenter = pipeline(
            "image-segmentation",
            model="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
            device=0 if torch.cuda.is_available() else -1
        )
        print("✅ 모델 로드 완료")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return

    # 2. 객체별 색상 정의 (BGR 형식)
    colors = {
        'road': [0, 255, 0],          # 녹색
        'sidewalk': [255, 255, 0],    # 노란색
        'building': [128, 128, 128],   # 회색
        'wall': [128, 0, 0],          # 갈색
        'fence': [255, 0, 255],       # 마젠타
        'pole': [0, 255, 255],        # 시안
        'traffic light': [0, 0, 255], # 빨간색 - 신호등
        'traffic sign': [255, 255, 255], # 흰색 - 표지판
        'vegetation': [0, 128, 0],     # 어두운 녹색
        'terrain': [128, 64, 0],      # 갈색
        'sky': [255, 128, 0],         # 하늘색
        'person': [255, 0, 0],        # 파란색 - 사람
        'rider': [128, 0, 255],       # 보라색
        'car': [0, 0, 128],           # 어두운 빨간색 - 자동차
        'truck': [255, 0, 128],       # 핑크 - 트럭
        'bus': [128, 255, 0],         # 연두색 - 버스
        'train': [0, 128, 255],       # 주황색
        'motorcycle': [255, 128, 128], # 연분홍 - 오토바이
        'bicycle': [128, 255, 255]     # 연청색 - 자전거
    }

    # 3. 비디오 열기
    cap = cv2.VideoCapture(input_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📹 비디오: {width}x{height}, {fps}fps, {total_frames}프레임")

    # 4. 출력 비디오 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            # RGB 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # 세그멘테이션 실행
            results = segmenter(pil_image)

            # 오버레이 이미지 생성
            overlay = np.zeros_like(frame)
            detected_objects = []

            # 모든 세그멘테이션 결과 처리
            for result in results:
                label = result['label'].lower()
                mask = np.array(result['mask'])

                # 해당 객체의 색상 가져오기
                if label in colors:
                    color = colors[label]
                    overlay[mask] = color
                    detected_objects.append(label)
                else:
                    # 정의되지 않은 객체는 기본 색상
                    overlay[mask] = [64, 64, 64]  # 어두운 회색
                    detected_objects.append(label)

            # 원본 프레임과 오버레이 합성
            result_frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

            # 화면에 검출된 객체 목록 표시
            unique_objects = list(set(detected_objects))
            y_offset = 30

            # 배경 박스 그리기 (텍스트 가독성)
            cv2.rectangle(result_frame, (10, 10), (400, 30 + len(unique_objects) * 25), (0, 0, 0), -1)

            for i, obj in enumerate(unique_objects):
                if obj in colors:
                    color = colors[obj]
                    # 객체명과 색상 표시
                    cv2.putText(result_frame, f"{obj}", (15, y_offset + i*25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 프레임 정보 표시
            cv2.putText(result_frame, f"Frame: {frame_count}/{total_frames}",
                       (width-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            out.write(result_frame)

        except Exception as e:
            print(f"프레임 {frame_count} 처리 실패: {e}")
            # 실패 시 원본 프레임 사용
            out.write(frame)

        frame_count += 1

        # 진행률 출력
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"⏳ 진행률: {progress:.1f}% | 검출된 객체: {len(unique_objects)}개")

    cap.release()
    out.release()

    print(f"✅ 완료! 결과: {output_video}")
    print(f"총 {frame_count}프레임 처리됨")

# 디버그용: 한 프레임만 테스트
def test_single_frame(video_path):
    """한 프레임만 테스트해서 뭐가 검출되는지 확인"""

    segmenter = pipeline(
        "image-segmentation",
        model="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        device=0 if torch.cuda.is_available() else -1
    )

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        results = segmenter(pil_image)

        print("🔍 검출된 객체들:")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['label']}")

        return results
    else:
        print("❌ 프레임 읽기 실패")
        return None

# 실행 옵션들
print("=== 세그멘테이션 테스트 ===")
print("1. 한 프레임 테스트:")
print("test_single_frame('/content/2.mp4')")
print("\n2. 전체 비디오 처리:")
print("#process_video_all_objects('/content/2.mp4', '/content/2_all_objects.mp4')")

# 실행
#test_single_frame('/content/1.mp4')  # 먼저 테스트
process_video_all_objects('/content/2.mp4', '/content/2_all_objects.mp4')  # 전체 처리
```
