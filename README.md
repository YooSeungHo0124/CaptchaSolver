# 🔬 CAPTCHA Solver - Deep Learning 기반 캡챠 인식 시스템

Deep-CAPTCHA 논문 구조를 기반으로 한 CNN 모델을 사용하여 텍스트 기반 캡챠를 자동으로 인식하는 Streamlit 웹 애플리케이션입니다.

## 📋 프로젝트 개요

이 프로젝트는 5자리 영문 대문자 + 숫자 조합의 캡챠 이미지를 인식하는 딥러닝 모델을 구현하고, 사용자 친화적인 웹 인터페이스를 제공합니다.

### 주요 특징

- 🎯 **높은 정확도**: 검증 데이터 기준 88.83% 전체 정확도, 95.17% 문자 정확도
- 🧠 **Deep-CAPTCHA 구조**: 논문 기반의 검증된 CNN 아키텍처
- 🚀 **실시간 학습**: Streamlit을 통한 모델 훈련 과정 시각화
- 💻 **사용자 친화적**: 직관적인 웹 인터페이스로 누구나 쉽게 사용 가능
- 📊 **성능 모니터링**: 훈련 과정 및 결과를 실시간으로 확인

## 🏗️ 모델 아키텍처

### Deep-CAPTCHA CNN 구조
입력 이미지 (150×60 픽셀, 흑백) ↓ Conv2D (32 filters, 5×5) + ReLU + MaxPool (2×2) ↓ Conv2D (48 filters, 5×5) + ReLU + MaxPool (2×2) ↓ Conv2D (64 filters, 5×5) + ReLU + MaxPool (2×2) ↓ Flatten + Dense (512 neurons) + ReLU + Dropout (30%) ↓ 병렬 Softmax 층 (5개 문자 위치별) ↓ 출력: 5자리 예측 결과


### 모델 사양

- **총 파라미터**: ~500K
- **학습 가능 파라미터**: ~500K
- **손실 함수**: CrossEntropyLoss (문자별)
- **옵티마이저**: Adam (lr=0.0001)
- **배치 크기**: 128
- **에포크**: 50

## 🚀 시작하기

### 필요 조건

- Python 3.8+
- pip

### 설치

1. 저장소 클론
git clone [https://github.com/yourusername/captcha-solver.git](https://github.com/yourusername/captcha-solver.git)
cd captcha-solver```
가상환경 생성 및 활성화
bash
python -m venv captcha_env
# Windows
.\captcha_env\Scripts\activate
# macOS/Linux
source captcha_env/bin/activate
필요한 패키지 설치
bash
pip install streamlit torch torchvision numpy pillow captcha scikit-learn matplotlib opencv-python
실행
bash
streamlit run app.py


브라우저에서 자동으로 http://localhost:8501이 열립니다.



📖 사용 방법  
모델 훈련: 앱 실행 시 자동으로 모델 훈련이 시작됩니다   
캡챠 생성: "새 캡챠 생성" 버튼으로 테스트용 캡챠 이미지 생성  
AI 인식: "AI로 인식하기" 버튼으로 모델의 예측 결과 확인  
직접 테스트: 사람이 직접 캡챠를 읽고 입력하여 AI와 비교  


📊 성능 지표  
최종 훈련 결과 (Epoch 50/50)  
지표	훈련 데이터	검증 데이터  
문자 정확도	90.58%	95.17%  
전체 정확도	65.21%	88.83%  
손실값	0.3192	0.3124  

✅ 과적합 없음 (검증 성능 > 훈련 성능)   
✅ 실용적 수준의 높은 정확도  
✅ 일반화 능력 우수  

🔧 기술 스택  
프레임워크: PyTorch  
웹 인터페이스: Streamlit  
이미지 처리: PIL, OpenCV  
데이터 처리: NumPy, scikit-learn  
시각화: Matplotlib  
캡챠 생성: captcha  

🎓 학습 데이터  
데이터셋 크기: 1,000개 원본 샘플  
데이터 증강: 3배 증강 (원본 + 노이즈 + 밝기 조정)  
총 학습 샘플: 3,000개  
훈련/검증 분할: 80% / 20%  
데이터 증강 기법   
원본 이미지: 왜곡 없는 깔끔한 캡챠  
노이즈 추가: 미세한 가우시안 노이즈 (σ=0.005)  
밝기 조정: ±5% 범위의 밝기 변화  

🎯 주요 기능  
1. 자동 모델 훈련  
앱 실행 시 자동으로 데이터 생성 및 모델 훈련  
실시간 진행률 표시  
최고 성능 모델 자동 저장  
2. 캡챠 생성 옵션  
기본 캡챠: 표준 난이도의 캡챠  
깔끔한 캡챠: 읽기 쉬운 고품질 캡챠  
3. 성능 시각화  
훈련/검증 손실 그래프  
실시간 정확도 지표  
모델 파라미터 정보  
4. 대화형 테스트  
AI 자동 인식  
사용자 직접 입력  
정답 비교 및 피드백  
🔬 Deep-CAPTCHA 논문 기반  
이 프로젝트는 다음 논문의 구조를 기반으로 합니다:  
  
"Deep-CAPTCHA: a deep learning based CAPTCHA solver for vulnerability assessment"

논문의 주요 개념  
병렬 Softmax 층: 각 문자 위치별로 독립적인 분류기 사용  
3단계 CNN-MaxPool: 계층적 특징 추출  
데이터 증강: 다양한 캡챠 변형에 대한 강건성 확보  

📝 라이선스  
이 프로젝트는 MIT 라이선스 하에 배포됩니다.  

📧 연락처  
프로젝트 링크: [https://github.com/yourusername/captcha-solver](https://github.com/YooSeungHo0124/CaptchaSolver)  
email : tmdgh0124@korea.ac.kr
