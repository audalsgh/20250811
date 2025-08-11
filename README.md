# 3일간 Roboflow 차선인식 프로젝트 진행예정
[Roboflow 프로젝트 계획](https://docs.google.com/document/d/1rxQHvxAIZM0pTspVIDAUx2ZEmjZRNSW-f6OrcitXFqM/edit?tab=t.0)<br>
https://www.ai-studio.co.kr/fo/board/dashboard#init<br>
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

## 3. 허깅페이스(Hugging Face)란?
**AI 모델 깃허브라고 할수있다.**
- 오픈소스 AI 플랫폼이자 머신러닝 커뮤니티로, 특히 **자연어 처리(NLP)**를 비롯한 다양한 AI 모델을 쉽게 사용,공유할 수 있는 환경을 제공합니다.
- 전 세계 개발자와 연구자들이 만든 사전 학습(Pre-trained) 모델 수십만개를 다운 가능.
- Transformers 라이브러리 : Hugging Face에서 제공하는 대표 Python 라이브러리.<br>텍스트, 이미지, 음성 모델을 쉽게 불러와서 추론,학습 가능 (PyTorch, TensorFlow, JAX 지원)
- Datasets 라이브러리 : 대규모 공개 데이터셋을 쉽게 다운로드·전처리할 수 있는 툴.<br>NLP뿐 아니라 이미지, 음성 데이터셋도 포함
- Trainer API : 초보자도 몇 줄 코드로 모델 fine-tuning 가능하게 지원하는 학습 인터페이스

장점
- 손쉬운 사전학습 모델 활용: 모델 로드는 한 줄이면 충분. `from transformers import pipeline`
- 멀티프레임워크 지원: PyTorch, TensorFlow, JAX
- NLP, CV(컴퓨터 비전), ASR(음성 인식) 등 다양한 도메인을 가짐.

허깅페이스에서 다운받은 BERT모델로 문장을 분류하는 간단한 예제
```python
from transformers import pipeline

# 감정 분석 파이프라인 불러오기
classifier = pipeline("sentiment-analysis")

# 예측
result = classifier("I love Hugging Face!")
print(result)

# 출력결과는 문장의 문맥을 파악하여 긍정의 범주에 99% 속한다고 나옴!
[{'label': 'POSITIVE', 'score': 0.9998}]
```

## 4. ADAS는 무엇이고, 텐서RT와 파이토치를 왜 비교할까?
[텐서RT와 파이토치를 비교하는 ADAS 코드](https://docs.google.com/document/d/179l9DqTYZJ1oGDWNA-TGtFZjs1PprWVoVlV-dTw_XrM/edit?tab=t.0)<br>
(ADAS 코드는 코랩에선 충돌이 많이 생기기때문에 런파드에서 하는게 좋지만, 주피터랩과 Connect문제로 그냥 코랩에서 진행)

**ADAS 모델을 만들때 PyTorch가 쓰이고, 만든 모델을 실제 차량에 넣어서 실시간으로 돌리는 단계에선 TensorRT가 쓰이는것!**
- 동일 파이프라인 유지 : 전처리, 입력 크기(imgsz=640, conf·NMS) 기준이 동일한 이미지 세트(1/2/3.jpg), conf=0.5로 맞춰서 공정한 비교를 하는중.
- FPS 측정 방식 : 20회 반복 + 3회 워밍업 + torch.cuda.synchronize()로 측정중.
- 시각화/ADAS 로직 : 차선 검출은 두 경로(Pytorch, TRT)에서 동일하게 적용되어 결과 비교 가능.
- 충돌 위험 영역/오브젝트 카운트 등 ADAS 지표들도 동일한 기준으로 출력됨.

-> TensorRT는 실차/엣지에서 추론 전용으로 최적화하는데 쓰임, 모델이 FP16/INT8로 변환되어 FPS↑, 지연↓, 메모리↓

<img width="1728" height="589" alt="image" src="https://github.com/user-attachments/assets/1b9f5853-87cb-4d72-acb5-3243f1065bc4" /><br>
gpt에게 원래 논리에 부합하는 코랩용 비교코드를 달라고함. 원본코드는 너무 오래걸려서 나중에 해볼것.

