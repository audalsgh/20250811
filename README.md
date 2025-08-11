# 3일간 Roboflow 차선인식 프로젝트 진행예정
[Roboflow 프로젝트 계획](https://docs.google.com/document/d/1rxQHvxAIZM0pTspVIDAUx2ZEmjZRNSW-f6OrcitXFqM/edit?tab=t.0)<br>
[텐서RT와 파이토치 비교 코드모음](https://docs.google.com/document/d/179l9DqTYZJ1oGDWNA-TGtFZjs1PprWVoVlV-dTw_XrM/edit?tab=t.0)<br>
추후에 사용할 ai studio 가입하기

## 1. YOLOv11 segmentation 사물인식 코드 colab에서 실습
- YOLOv11 segmentation은 19개 정도의 사물인식이 된다고 알려져있음.<br>
[코랩에서 실행한 코드](0811_YOLOv11_Segmentation.ipynb)
<img width="2404" height="1080" alt="image" src="https://github.com/user-attachments/assets/3ac1df3b-cefc-4ee8-93b9-8600d73b37fe" /><br>
-> 차선같은거 보단 사물인식을 목표로 함.

## 2. SegFormer 모델이란?
SegFormer 모델은 런파드에서 실습해볼것.
- NVIDIA에서 제안한 Transformer 기반의 세그멘테이션(Segmentation) 모델
- 기존모델은 CNN(Convolutional Neural Network)을 백본으로 사용했지만, SegFormer는 Efficient Self-Attention 기반의 Transformer Encoder를 사용
- Hierarchical Transformer Encoder : 여러 해상도의 피처맵을 생성해 다단계(Context + Detail) 정보를 통합.
- 다양한 입력 크기와 해상도에서 강한 성능
  - Cityscapes, ADE20K 같은 고해상도 데이터셋에서 좋은 성능을 보임.
  - 모바일/엣지 디바이스에도 적용 가능할 정도로 경량화 가능.
- Lightweight MLP Decoder : 복잡한 디코더 대신 단순한 MLP(Multi-Layer Perceptron) 구조를 사용하여 속도와 메모리 효율이 높음.
  - MLP 구조란, 인접한 층의 모든 뉴런이 서로 연결됨 (Fully Connected Layer) 완전 연결층이며,<br>모든 입력을 여러 층(Layer)의 뉴런(Neuron)을 거치게 하여 비선형적으로 변환하면서 복잡한 함수를 근사하는 구조.
  - CNN처럼 지역 패턴(이미지 필터링)은 못하지만, 전역적인 관계 학습에 강함

```
[이미지 입력]
      ↓
[Hierarchical Transformer Encoder]
      ↓
[Lightweight MLP Decoder]
      ↓
[픽셀 단위 클래스 맵 출력]
```

장점
- CNN 없이도 SOTA(SOTA=State-of-the-Art) 성능 달성
- 학습 속도 빠르고 추론 효율이 높음
- 작은 모델(SegFormer-B0)부터 대형 모델(SegFormer-B5)까지 다양한 크기 제공

활용 분야 예시
- 자율주행 차량의 도로 객체 인식
- 위성/항공 이미지 분석
- 의료 영상(CT, MRI) 장기 분할
- 로봇 비전

### 허깅페이스(Hugging Face)란?
- **AI 모델 깃허브라고 할수있다.**
- 오픈소스 AI 플랫폼이자 머신러닝 커뮤니티로, 특히 **자연어 처리(NLP)**를 비롯한 다양한 AI 모델을 쉽게 사용,공유할 수 있는 환경을 제공합니다.
- 전 세계 개발자와 연구자들이 만든 사전 학습(Pre-trained) 모델 수십만개를 다운 가능.
- Transformers 라이브러리 : Hugging Face에서 제공하는 대표 Python 라이브러리.<br>텍스트, 이미지, 음성 모델을 쉽게 불러와서 추론,학습 가능 (PyTorch, TensorFlow, JAX 지원)
- Datasets 라이브러리 : 대규모 공개 데이터셋을 쉽게 다운로드·전처리할 수 있는 툴.<br>NLP뿐 아니라 이미지, 음성 데이터셋도 포함
- Trainer API : 초보자도 몇 줄 코드로 모델 fine-tuning 가능하게 지원하는 학습 인터페이스

장점
- 손쉬운 사전학습 모델 활용: 모델 로드는 한 줄이면 충분. `from transformers import pipeline`
- 멀티프레임워크 지원: PyTorch, TensorFlow, JAX
- NLP, CV(컴퓨터 비전), ASR(음성 인식) 등 다양한 도메인을 가짐.

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
