import streamlit as st
from captcha.image import ImageCaptcha
import string, random
import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import os
import cv2

# -----------------------------
# 1. 기본 설정 (사람이 읽을 수 있는 크기)
# -----------------------------
CAPTCHA_LENGTH = 5
CHAR_SET = string.ascii_uppercase + string.digits
IMG_WIDTH = 150  # 사람이 읽기 쉬운 크기
IMG_HEIGHT = 60  # 사람이 읽기 쉬운 크기
NUM_CLASSES = len(CHAR_SET)

# 디바이스 설정 (CPU)
device = torch.device('cpu')

# -----------------------------
# 2. 캡챠 생성 함수
# -----------------------------
def generate_captcha_text(length=CAPTCHA_LENGTH):
    return ''.join(random.choices(CHAR_SET, k=length))

def generate_captcha_image(text):
    """기본적인 깔끔한 캡챠 이미지 생성"""
    image = ImageCaptcha(width=IMG_WIDTH, height=IMG_HEIGHT)
    data = image.generate(text)
    img = Image.open(data).convert('L')  # 흑백
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    return img

def generate_clean_captcha_image(text):
    """매우 깔끔하고 읽기 쉬운 캡챠 이미지 생성 (기본 테스트용)"""
    # 더 큰 크기로 생성 후 리사이즈
    image = ImageCaptcha(width=IMG_WIDTH*2, height=IMG_HEIGHT*2)
    data = image.generate(text)
    img = Image.open(data).convert('L')
    # 부드럽게 리사이즈
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
    return img

# -----------------------------
# 3. 문자 → 인덱스 변환
# -----------------------------
char_to_index = {c:i for i,c in enumerate(CHAR_SET)}
index_to_char = {i:c for i,c in enumerate(CHAR_SET)}

def text_to_labels(text):
    return [char_to_index[c] for c in text]

def labels_to_text(labels):
    return ''.join([index_to_char[i] for i in labels])

# -----------------------------
# 4. Deep-CAPTCHA CNN 모델 (논문 구조)
# -----------------------------
class DeepCaptchaCNN(nn.Module):
    def __init__(self, num_classes, captcha_length):
        super(DeepCaptchaCNN, self).__init__()
        
        # Deep-CAPTCHA 구조: 3개의 CNN-MaxPool 쌍
        self.features = nn.Sequential(
            # 첫 번째 CNN-MaxPool 쌍
            nn.Conv2d(1, 32, 5, padding='same'),  # 5x5 커널, same 패딩
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 2x2 MaxPool
            
            # 두 번째 CNN-MaxPool 쌍
            nn.Conv2d(32, 48, 5, padding='same'),  # 48개 뉴런
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 세 번째 CNN-MaxPool 쌍
            nn.Conv2d(48, 64, 5, padding='same'),  # 64개 뉴런
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Dense 층 (512개 뉴런) - 새로운 이미지 크기에 맞게 조정
        # 150x60 -> 75x30 -> 37x15 -> 18x7 (MaxPool 후)
        self.dense = nn.Sequential(
            nn.Linear(64 * 18 * 7, 512),  # 새로운 크기에 맞게 조정
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)  # 30% 드롭아웃
        )
        
        # 병렬 Softmax 층 (각 문자 위치별)
        self.char_classifiers = nn.ModuleList([
            nn.Linear(512, num_classes) for _ in range(captcha_length)
        ])
        
    def forward(self, x):
        # 특징 추출
        x = self.features(x)
        x = x.view(x.size(0), -1)
        
        # Dense 층
        x = self.dense(x)
        
        # 각 문자 위치별로 병렬 Softmax 층 적용
        outputs = []
        for i in range(len(self.char_classifiers)):
            char_output = self.char_classifiers[i](x)
            outputs.append(char_output)
        
        return torch.stack(outputs, dim=1)  # [batch_size, captcha_length, num_classes]

# -----------------------------
# 5. 데이터셋 클래스
# -----------------------------
class CaptchaDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = torch.FloatTensor(self.images[idx]).unsqueeze(0)  # 채널 차원 추가
        label = torch.LongTensor(self.labels[idx])
        return image, label

# -----------------------------
# 6. 데이터 증강
# -----------------------------
def augment_image(img_array):
    """기본적인 데이터 증강 (과도한 왜곡 제거)"""
    augmented_images = []
    
    # 원본 (왜곡 없음)
    augmented_images.append(img_array)
    
    # 약간의 노이즈만 추가 (매우 미세한 수준)
    noise = np.random.normal(0, 0.005, img_array.shape)  # 노이즈 강도 감소
    img_noisy = np.clip(img_array + noise, 0, 1)
    augmented_images.append(img_noisy)
    
    # 약간의 밝기 조정 (미세한 수준)
    brightness_factor = np.random.uniform(0.95, 1.05)  # 범위 축소
    img_bright = np.clip(img_array * brightness_factor, 0, 1)
    augmented_images.append(img_bright)
    
    return augmented_images

def generate_dataset(num_samples=1000):
    X = []
    y = []
    
    for _ in range(num_samples):
        text = generate_captcha_text()
        img = generate_captcha_image(text)
        img_array = np.array(img)/255.0
        
        # 데이터 증강 적용
        augmented_images = augment_image(img_array)
        for aug_img in augmented_images:
            X.append(aug_img)
            y.append(text_to_labels(text))
    
    return np.array(X), np.array(y)

# -----------------------------
# 7. 모델 생성 함수
# -----------------------------
def create_model(num_classes, captcha_length):
    """Deep-CAPTCHA CNN 모델 생성"""
    model = DeepCaptchaCNN(num_classes, captcha_length)
    return model

# -----------------------------
# 8. 훈련 함수
# -----------------------------

# --- Replacement train_model (CrossEntropy per character) ---
def train_model(model, train_loader, val_loader, epochs=50):
    model = model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")

    # Use CrossEntropy per-character (expects integer class labels)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_chars = 0
        correct_chars = 0
        total_samples = 0
        correct_samples = 0

        for images, labels in train_loader:
            images = images.to(device)                     # [B,1,H,W]
            labels = labels.to(device)                     # [B, L] long

            optimizer.zero_grad()
            outputs = model(images)                        # [B, L, C]
            B, L, C = outputs.shape

            # reshape: (B*L, C) and (B*L,)
            outputs_flat = outputs.view(B * L, C)
            labels_flat = labels.view(B * L)

            loss = criterion(outputs_flat, labels_flat)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * B
            # predictions
            preds = torch.argmax(outputs, dim=2)           # [B, L]
            # per-char correct
            correct_chars += (preds == labels).sum().item()
            total_chars += B * L
            # whole-sample correct (exact match)
            for i in range(B):
                if torch.equal(preds[i], labels[i]):
                    correct_samples += 1
            total_samples += B

        train_loss = running_loss / len(train_loader.dataset)
        train_char_acc = 100.0 * correct_chars / total_chars
        train_full_acc = 100.0 * correct_samples / total_samples
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct_chars = 0
        val_total_chars = 0
        val_correct_samples = 0
        val_total_samples = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                B, L, C = outputs.shape
                outputs_flat = outputs.view(B * L, C)
                labels_flat = labels.view(B * L)
                loss = criterion(outputs_flat, labels_flat)
                val_running_loss += loss.item() * B

                preds = torch.argmax(outputs, dim=2)
                val_correct_chars += (preds == labels).sum().item()
                val_total_chars += B * L
                for i in range(B):
                    if torch.equal(preds[i], labels[i]):
                        val_correct_samples += 1
                val_total_samples += B

        val_loss = val_running_loss / len(val_loader.dataset)
        val_char_acc = 100.0 * val_correct_chars / val_total_chars
        val_full_acc = 100.0 * val_correct_samples / val_total_samples
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Char Acc: {train_char_acc:.2f}%, Train Full Acc: {train_full_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Char Acc: {val_char_acc:.2f}%, Val Full Acc: {val_full_acc:.2f}%")

        # save best by val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch+1
            torch.save(model.state_dict(), 'best_deep_captcha_model.pth')

    print(f"Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
    return train_losses, val_losses, train_char_acc, val_char_acc, train_full_acc, val_full_acc


# -----------------------------
# 9. 예측 함수
# -----------------------------
# --- Replacement predict_captcha (no softmax confusion) ---
def predict_captcha(model, img_array):
    model.eval()
    with torch.no_grad():
        img_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
        outputs = model(img_tensor)  # [1, L, C]
        preds = torch.argmax(outputs, dim=2).squeeze(0).cpu().numpy()  # [L]
        return labels_to_text(preds)


# -----------------------------
# 10. Streamlit 앱
# -----------------------------
st.set_page_config(page_title="Basic CAPTCHA Solver", page_icon="🔬", layout="wide")

st.title("🔬 Basic CAPTCHA Solver")
st.markdown("**기본적인 깔끔한 캡챠부터 시작하는 CNN 모델**")

# 사이드바 정보
with st.sidebar:
    st.header("🔬 Basic CAPTCHA 특징")
    st.markdown("""
    **기본적인 접근 방법:**
    - 🚀 **큰 이미지 크기**: 150×60 픽셀 (읽기 쉬움)
    - ⚡ **깔끔한 캡챠**: 과도한 왜곡 제거
    - 💻 **점진적 학습**: 기본부터 시작
    - 🎯 **최소 전처리**: 원본에 가까운 이미지
    - 🔧 **사람 검증**: 사람이 읽을 수 있는 수준
    """)
    
    st.markdown("---")
    st.markdown("### 📊 모델 정보")
    if 'model_trained' in st.session_state and st.session_state['model_trained']:
        model = st.session_state['model']
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        st.metric("총 파라미터", f"{total_params:,}")
        st.metric("학습 가능 파라미터", f"{trainable_params:,}")
        st.metric("학습률", f"{100 * trainable_params / total_params:.1f}%")

# 모델 상태 관리
if 'model_trained' not in st.session_state:
    st.session_state['model_trained'] = False

# 모델 훈련 섹션
if not st.session_state['model_trained']:
    st.info("🔄 Deep-CAPTCHA 모델을 훈련하는 중입니다. 잠시만 기다려주세요...")
    
    # 진행률 표시
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 1. 데이터 생성
    status_text.text("📊 데이터셋 생성 중...")
    progress_bar.progress(10)
    X, y = generate_dataset(num_samples=1000)  # 데이터 증가
    
    # 2. 데이터 분할
    status_text.text("🔄 데이터 분할 중...")
    progress_bar.progress(20)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. 데이터로더 생성
    status_text.text("📦 데이터로더 준비 중...")
    progress_bar.progress(30)
    train_dataset = CaptchaDataset(X_train, y_train)
    val_dataset = CaptchaDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)  # 논문의 배치 크기
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    # 4. 모델 생성
    status_text.text("🏗️ Deep-CAPTCHA 모델 생성 중...")
    progress_bar.progress(40)
    model = create_model(NUM_CLASSES, CAPTCHA_LENGTH)
    
    # 5. 모델 훈련
    status_text.text("🎯 Deep-CAPTCHA 모델 훈련 중... (논문 매개변수)")
    progress_bar.progress(50)
    
    # 훈련 과정을 실시간으로 표시
    train_losses, val_losses, final_train_char_acc, final_val_char_acc, final_train_full_acc, final_val_full_acc = train_model(model, train_loader, val_loader, epochs=50)
    
    # 6. 최고 모델 로드
    status_text.text("💾 최고 성능 모델 로드 중...")
    progress_bar.progress(90)
    if os.path.exists('best_deep_captcha_model.pth'):
        model.load_state_dict(torch.load('best_deep_captcha_model.pth', map_location='cpu'))
    
    # 7. 완료
    status_text.text("✅ Deep-CAPTCHA 모델 훈련 완료!")
    progress_bar.progress(100)
    
    st.session_state['model'] = model
    st.session_state['model_trained'] = True
    st.session_state['train_losses'] = train_losses
    st.session_state['val_losses'] = val_losses
    st.session_state['final_train_char_acc'] = final_train_char_acc
    st.session_state['final_val_char_acc'] = final_val_char_acc
    st.session_state['final_train_full_acc'] = final_train_full_acc
    st.session_state['final_val_full_acc'] = final_val_full_acc
    
    st.success(f"🎉 Deep-CAPTCHA 모델 훈련 완료! 최종 정확도: {final_val_full_acc:.2f}%")

# 훈련 결과 시각화
if st.session_state['model_trained']:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 훈련 과정")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(st.session_state['train_losses'], label='Train Loss', color='blue', linewidth=2)
        ax.plot(st.session_state['val_losses'], label='Validation Loss', color='red', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Deep-CAPTCHA Model Training Progress', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.markdown("### 📊 성능 지표")
        st.metric("최종 훈련 정확도 (Full)", f"{st.session_state['final_train_full_acc']:.2f}%")
        st.metric("최종 검증 정확도 (Full)", f"{st.session_state['final_val_full_acc']:.2f}%")
        st.metric("최종 훈련 정확도 (Char)", f"{st.session_state['final_train_char_acc']:.2f}%")
        st.metric("최종 검증 정확도 (Char)", f"{st.session_state['final_val_char_acc']:.2f}%")
        st.metric("훈련 데이터", "1,000 samples")
        st.metric("에포크", "50")

# 캡챠 테스트 섹션
st.markdown("---")
st.markdown("### 🧪 캡챠 테스트")

# 캡챠 생성 옵션
col_gen1, col_gen2 = st.columns(2)
with col_gen1:
    if st.button("🔄 새 캡챠 생성 (기본)"):
        st.session_state['captcha_text'] = generate_captcha_text()
        st.session_state['captcha_img'] = generate_captcha_image(st.session_state['captcha_text'])
        st.session_state['captcha_type'] = "기본"

with col_gen2:
    if st.button("✨ 새 캡챠 생성 (깔끔)"):
        st.session_state['captcha_text'] = generate_captcha_text()
        st.session_state['captcha_img'] = generate_clean_captcha_image(st.session_state['captcha_text'])
        st.session_state['captcha_type'] = "깔끔"

# 초기 캡챠 생성
if 'captcha_text' not in st.session_state:
    st.session_state['captcha_text'] = generate_captcha_text()
    st.session_state['captcha_img'] = generate_clean_captcha_image(st.session_state['captcha_text'])
    st.session_state['captcha_type'] = "깔끔"

col1, col2 = st.columns([1, 1])

with col1:
    st.image(st.session_state['captcha_img'], caption=f"인식할 캡챠 ({st.session_state.get('captcha_type', '기본')} 타입)", use_container_width=False)
    st.markdown(f"**실제 정답: `{st.session_state['captcha_text']}`**")
    st.markdown(f"**이미지 크기: {IMG_WIDTH}×{IMG_HEIGHT} 픽셀**")

with col2:
    st.markdown("### 🤖 Deep-CAPTCHA 모델 예측")
    
    if st.button("🔍 AI로 인식하기"):
        with st.spinner("Deep-CAPTCHA 모델로 텍스트를 인식하는 중..."):
            img_array = np.array(st.session_state['captcha_img'])/255.0
            predicted_text = predict_captcha(st.session_state['model'], img_array)
        
        st.success(f"**인식된 텍스트: `{predicted_text}`**")
        
        # 정확도 확인
        if predicted_text == st.session_state['captcha_text']:
            st.balloons()
            st.success("🎉 **완벽하게 맞췄습니다!**")
        else:
            st.warning(f"❌ **틀렸습니다** (정답: `{st.session_state['captcha_text']}`)")

# 사용자 입력 테스트
st.markdown("---")
st.markdown("### 👤 직접 입력해보기")

user_input = st.text_input("직접 입력:", "", max_chars=CAPTCHA_LENGTH)

if st.button("✅ 검증"):
    if user_input.upper() == st.session_state['captcha_text']:
        st.success("🎉 **정답입니다!**")
    else:
        st.error(f"❌ **틀렸습니다** (정답: `{st.session_state['captcha_text']}`)")

# Deep-CAPTCHA 모델 설명
st.markdown("---")
st.markdown("### 📖 Deep-CAPTCHA 모델 상세 설명")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Deep-CAPTCHA 논문 구조:**
    
    논문에서 제안한 **3개의 CNN-MaxPool 쌍**과 **병렬 Softmax 층**을 사용합니다.
    
    **전처리 단계:**
    - 이미지 크기: 67×25 픽셀
    - 그레이스케일 변환
    - 중앙값 필터 (노이즈 감소)
    
    **네트워크 구조:**
    - 3개 CNN-MaxPool 쌍 (32→48→64 뉴런)
    - 512개 Dense 층
    - 30% Dropout
    - 병렬 Softmax 층
    """)

with col2:
    st.markdown("""
    **학습 매개변수:**
    
    - **손실 함수**: Binary Cross Entropy
    - **옵티마이저**: Adam (lr=0.0001)
    - **에포크**: 50
    - **배치 크기**: 128
    - **패딩**: Same padding
    
    **논문 성능:**
    - 정확도: 98.94%
    - 기존 방식 대비 8.9% 향상
    - 단일 시그모이드: 90.04%
    - 병렬 Softmax: 98.94%
    """)

# 성능 비교
st.markdown("---")
st.markdown("### 📊 성능 비교")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 🔴 기존 방식")
    st.markdown("""
    - 전체 모델 학습
    - 파라미터: ~1M+
    - 학습 시간: 길음
    - 메모리: 많이 필요
    """)

with col2:
    st.markdown("#### 🟡 EasyOCR")
    st.markdown("""
    - 사전 훈련 모델
    - 파라미터: 고정
    - 학습 시간: 0초
    - 메모리: 중간
    """)

with col3:
    st.markdown("#### 🟢 Deep-CAPTCHA")
    st.markdown("""
    - 논문 기반 모델
    - 파라미터: ~500K
    - 정확도: 98.94%
    - 논문 검증된 구조
    """)

# 사용법 안내
st.markdown("---")
st.markdown("### 🚀 사용법")
st.markdown("""
1. **🔄 새 캡챠 생성**: 새로운 캡챠 이미지를 생성합니다
2. **🔍 AI로 인식하기**: Deep-CAPTCHA 모델이 자동으로 텍스트를 인식합니다
3. **직접 입력**: 본인이 보는 문자를 직접 입력해보세요
4. **성능 비교**: AI와 사람의 정확도를 비교해보세요

**특징:**
- 논문 기반 검증된 구조
- 중앙값 필터 전처리
- 병렬 Softmax 층
- Binary Cross Entropy 손실 함수
""")