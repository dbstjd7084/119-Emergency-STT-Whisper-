# 119-Emergency-STT-Whisper

기존 배포 119 긴급 음성 모델(STT) ESPnet은 긴 추론 시간과 함께 저조한 성능을 보임에 따라 Whisper의 DoRA 기반 파인튜닝을 통해 추론 시간 단축 및 더 높은 정확도를 달성을 목적으로 진행했으며,

"1 Epoch" 만 학습했습니다.

# 모델 실행 결과
## 기존 ESPnet 배포 모델
<img src="https://github.com/dbstjd7084/119-Emergency-STT-Whisper-/blob/1c2bc67ee3af3dcf46973f12610f27946af323a8/images/espnet.png" alt="ESPnet"/>

CER 10 미만

## Whisper-Base(72.6M params)
<img src="https://github.com/dbstjd7084/119-Emergency-STT-Whisper-/blob/1c2bc67ee3af3dcf46973f12610f27946af323a8/images/whisper-base.png" alt="Whisper-Base"/>

## Whisper-Small(0.2B params)
<img src="https://github.com/dbstjd7084/119-Emergency-STT-Whisper-/blob/1c2bc67ee3af3dcf46973f12610f27946af323a8/images/whisper-small.png" alt="Whisper-Small"/>

## Whisper-Large-v3-Turbo(0.8B params)
<img src="https://github.com/dbstjd7084/119-Emergency-STT-Whisper-/blob/1c2bc67ee3af3dcf46973f12610f27946af323a8/images/whisper-large-v3-turbo.png" alt="Whisper-Large-v3-Turbo"/>

## 실행 방법 (How to Run)

Python 3.12 에서 구동하는 것을 권장합니다. 16GB VRAM을 갖춘 GPU 환경에서 테스트했습니다.

### 1. 환경 설정 및 패키지 설치
먼저 저장소를 복제하고 루트 폴더로 이동한 뒤, 필수 의존성 패키지를 설치합니다.

1. 이 저장소를 복제합니다:
   ```bash
   git clone https://github.com/dbstjd7084/119-Emergency-STT-Whisper-.git
   ```
2. 디렉토리로 이동합니다:
   ```bash
   cd 119-Emergency-STT-Whisper-
   ```
3. 필수 패키지를 설치합니다:
   ```bash
   pip install -r requirements.txt
   ```

### 2. 모델 다운로드
[드라이브 경로](https://drive.google.com/file/d/1oTNxn8cR9LFOZEmn6hCfZg0epCMTulxy/view?usp=sharing, "모델 다운로드")에서 모델을 다운로드 후 루트 폴더에 압축을 풉니다.

ESPnet [(AI 허브 공식 배포 모델)](https://aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOneDataTy=DATA004&srchOptnCnd=OPTNCND001&searchKeyword=&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=71768, "AI 허브 이동")

Whisper-Base(72.6M params)

Whisper-Small(0.2B params)

Whisper-Large-v3-Turbo(0.8B params)

### 3. 웹 애플리케이션(Streamlit) 실행

패키지 설치가 완료되면, 아래 명령어를 입력하여 STT 모델 추론 데모 웹페이지를 실행합니다.

```bash
streamlit run app.py

```

* 명령어를 실행하면 기본 브라우저가 자동으로 열리며, **119 긴급신고 음성 자동 기록 시스템** 데모를 확인할 수 있습니다.
* `test_data` 폴더에 포함된 `.wav` 샘플 파일을 업로드하여 모델의 성능을 직접 테스트할 수 있습니다.
