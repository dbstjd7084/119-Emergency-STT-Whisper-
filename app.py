import streamlit as st
import tempfile
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# --- 경로 설정 ---
ASR_CONFIG_PATH = os.path.join(BASE_DIR, "espnet", "train_asr_conformer10_hop_length160.yaml")
ASR_MODEL_PATH = os.path.join(BASE_DIR, "espnet", "model.pth")
BPE_MODEL_PATH = os.path.join(BASE_DIR, "espnet", "kr_token_list", "bpe_unigram2309", "bpe.model")
TOKEN_LIST_PATH = os.path.join(BASE_DIR, "espnet", "kr_token_list", "bpe_unigram2309", "tokens.txt")
FEATS_STATS_PATH = os.path.join(BASE_DIR, "espnet", "feats_stats.npz")

# 💡 학습된 커스텀 모델 경로 지정 (폴더명이 다르면 여기서 수정하세요)
CUSTOM_SMALL_LORA_PATH = os.path.join(BASE_DIR, "whisper-119-emergency-small-model", "final_lora_model")
CUSTOM_BASE_DORA_PATH = os.path.join(BASE_DIR, "whisper-119-emergency-base-dora", "final_dora_model")
CUSTOM_LARGE_DORA_PATH = os.path.join(BASE_DIR, "whisper-119-emergency-large-v3-turbo-dora", "final_dora_model")

st.set_page_config(
    page_title="119 긴급신고 음성 자동 기록 시스템",
    page_icon="🚨",
    layout="wide"
)

st.markdown(
    """
    <style>
        .block-container { max-width: 980px; padding-top: 3rem; padding-bottom: 4rem; }
        [data-testid="stAppViewContainer"] { background: #f6f8fb; }
        [data-testid="stHeader"] { background: transparent; }
        .intro-panel { background: #ffffff; border: 1px solid #e4e8ef; border-radius: 8px; padding: 28px; margin-bottom: 18px; box-shadow: 0 16px 40px rgba(20, 35, 60, 0.08); }
        .page-title { margin: 0 0 8px; color: #182230; font-size: 2rem; font-weight: 800; }
        .page-subtitle { margin: 0 0 28px; color: #667085; font-size: 1rem; line-height: 1.6; }
        div[data-testid="stFileUploader"] { padding: 18px; border: 1px dashed #b8c2d6; border-radius: 8px; background: #fbfcff; }
        div[data-testid="stTextArea"] textarea { border-radius: 8px; border-color: #d0d5dd; font-size: 1rem; line-height: 1.6; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="intro-panel">
    <h1 class="page-title">🚨 119 긴급신고 음성 자동 기록 시스템</h1>
    <p class="page-subtitle">통화 음성 파일을 분석하여 대화 내역을 생성합니다.<br>
    음성 엔진을 선택하고 파일을 업로드하세요.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# 1. 모델 선택 컴포넌트에 커스텀 모델 3종으로 업데이트
selected_model = st.selectbox(
    "사용할 STT 엔진 모델 선택",
    [
        "Whisper Large V3 Turbo (119 맞춤형 DoRA)",
        "Whisper Base (119 맞춤형 DoRA)",     
        "Whisper Small (119 맞춤형 LoRA)",    
        "ESPnet (배포 학습 모델)", 
        "Whisper Large V3 Turbo", 
        "Qwen3-ASR-0.6B"
    ]
)

uploaded_file = st.file_uploader("음성 파일 업로드", type=["wav"])

# --- 캐싱 로더 함수들 ---
@st.cache_resource
def get_runtime_asset_paths():
    asset_dir = os.path.join(tempfile.gettempdir(), "espnet_asr_assets")
    os.makedirs(asset_dir, exist_ok=True)
    assets = {
        "bpemodel": (BPE_MODEL_PATH, os.path.join(asset_dir, "bpe.model")),
        "token_list": (TOKEN_LIST_PATH, os.path.join(asset_dir, "tokens.txt")),
        "stats_file": (FEATS_STATS_PATH, os.path.join(asset_dir, "feats_stats.npz")),
    }
    runtime_paths = {}
    for key, (source_path, runtime_path) in assets.items():
        if not os.path.exists(runtime_path) or os.path.getsize(runtime_path) != os.path.getsize(source_path):
            shutil.copy2(source_path, runtime_path)
        runtime_paths[key] = runtime_path
    return runtime_paths

@st.cache_resource
def load_bpe_tokenizer():
    import sentencepiece as spm
    runtime_paths = get_runtime_asset_paths()
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(runtime_paths["bpemodel"])
    return tokenizer

@st.cache_resource
def load_speech2text():
    from espnet2.bin.asr_inference import Speech2Text
    runtime_paths = get_runtime_asset_paths()
    return Speech2Text(
        asr_train_config=get_asr_config_path(),
        asr_model_file=ASR_MODEL_PATH,
        token_type="bpe",
        bpemodel=runtime_paths["bpemodel"],
        device="cuda",
        beam_size=5, # 5로 하향 통일
        ctc_weight=0.3
    )

@st.cache_resource
def load_faster_whisper():
    import torch
    import os
    
    # [강력한 버그 픽스] Python 3.8+ Windows 환경 DLL 인식 오류 해결
    if os.name == 'nt':
        try:
            torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
            if os.path.exists(torch_lib_path):
                os.add_dll_directory(torch_lib_path)
            
            import site
            for site_pkg in site.getsitepackages():
                cublas_path = os.path.join(site_pkg, "nvidia", "cublas", "bin")
                cudnn_path = os.path.join(site_pkg, "nvidia", "cudnn", "bin")
                if os.path.exists(cublas_path):
                    os.add_dll_directory(cublas_path)
                if os.path.exists(cudnn_path):
                    os.add_dll_directory(cudnn_path)
                    
            # 시스템 환경변수에 등록된 기본 CUDA 패스도 추가 탐색
            cuda_path = os.environ.get("CUDA_PATH")
            if cuda_path and os.path.exists(os.path.join(cuda_path, "bin")):
                os.add_dll_directory(os.path.join(cuda_path, "bin"))
                
        except Exception as e:
            print(f"DLL 경로 등록 중 경고 (무시 가능): {e}")

    try:
        from faster_whisper import WhisperModel
        return WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    except Exception as e:
        # 💡 faster-whisper (CTranslate2)의 고질적인 버전 불일치 에러를 우아하게 캐치
        if "cublas" in str(e).lower() or "cudnn" in str(e).lower() or "not found" in str(e).lower():
            st.error("""
            **[Faster-Whisper 실행 오류] CUDA 12 호환성 라이브러리가 필요합니다.**
            
            현재 사용 중인 파이썬 환경의 PyTorch가 `cu130` 기반이어서, 
            Faster-Whisper가 요구하는 **CUDA 12 전용 DLL(`cublas64_12.dll`)**을 찾지 못했습니다.
            
            해결을 위해 터미널을 열고 아래 명령어를 실행하여 호환성 패키지를 추가로 설치해 주세요:
            ```bash
            pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
            ```
            설치가 완료된 후 Streamlit을 재시작하시면 정상적으로 구동됩니다.
            """)
            st.stop()
        else:
            st.error(f"Faster-Whisper 로드 중 알 수 없는 오류 발생: {e}")
            st.stop()

@st.cache_resource
def load_qwen_asr():
    import torch
    from qwen_asr import Qwen3ASRModel
    return Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-0.6B", dtype=torch.bfloat16, device_map="cuda:0")

# 2. FFmpeg 에러를 피하기 위해 Pipeline 대신 순수 모델만 로드
@st.cache_resource
def load_custom_whisper(base_model_name, peft_model_path):
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from peft import PeftModel
    import torch

    processor = WhisperProcessor.from_pretrained(peft_model_path, language="Korean", task="transcribe")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        base_model_name, 
        device_map="cuda", 
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"
    )
    
    model = PeftModel.from_pretrained(base_model, peft_model_path)
    model.eval()
    
    return processor, model


def tokenize_text(text):
    tokenizer = load_bpe_tokenizer()
    return tokenizer.encode(text, out_type=str)

@st.cache_resource
def get_asr_config_path():
    with open(ASR_CONFIG_PATH, "r", encoding="utf-8") as config_file:
        config_text = config_file.read()
    config_text = config_text.replace("\ninit: none\n", "\ninit: null\n")
    runtime_paths = get_runtime_asset_paths()
    def yaml_path(path): return "'" + path.replace("\\", "/").replace("'", "''") + "'"
    required_config = {
        "token_list": yaml_path(runtime_paths["token_list"]),
        "token_type": "bpe",
        "bpemodel": yaml_path(runtime_paths["bpemodel"]),
        "input_size": "null",
        "frontend": "default",
        "normalize": "global_mvn",
        "normalize_conf": "\n    stats_file: " + yaml_path(runtime_paths["stats_file"]),
        "ctc_conf": "{}",
    }
    missing_config = [(k, v) for k, v in required_config.items() if not (f"\n{k}:" in config_text or config_text.startswith(f"{k}:"))]
    if not missing_config: return ASR_CONFIG_PATH
    patched_config_path = os.path.join(tempfile.gettempdir(), "espnet_asr_config_with_token_list.yaml")
    patched_config = config_text.rstrip() + "\n\n" + "\n".join(f"{k}: {v}" for k, v in missing_config) + "\n"
    with open(patched_config_path, "w", encoding="utf-8") as config_file:
        config_file.write(patched_config)
    return patched_config_path


# 3. run_stt 함수 추론 로직
def run_stt(audio_path, model_type, asr_config_path=None, bpemodel_path=None):
    if model_type == "ESPnet (배포 학습 모델)":
        from espnet2.bin.asr_inference import Speech2Text
        import librosa
        import numpy as np
        import soundfile as sf
        speech, sample_rate = sf.read(audio_path, dtype="float32")
        if speech.ndim > 1: speech = np.mean(speech, axis=1)
        if sample_rate != 16000: speech = librosa.resample(speech, orig_sr=sample_rate, target_sr=16000)
        speech2text = load_speech2text()
        result = speech2text(speech)
        metadata = {"engine": "ESPnet", "beam_size": 5, "ctc_weight": 0.3} # 💡 5로 하향 통일
        return result[0][0], metadata

    elif model_type == "Whisper Large V3 Turbo":
        model = load_faster_whisper()
        segments, info = model.transcribe(audio_path, beam_size=5, language="ko", vad_filter=True) # 💡 5로 하향 통일
        result_text = " ".join([segment.text for segment in segments])
        metadata = {"detected_language": info.language, "audio_duration_sec": round(info.duration, 2)}
        return result_text.strip(), metadata

    elif model_type == "Qwen3-ASR-0.6B":
        model = load_qwen_asr()
        results = model.transcribe(audio=audio_path, language="Korean")
        result_text = results[0].text
        metadata = {"engine": "Qwen3-ASR-0.6B"}
        return result_text, metadata

    # 💡 커스텀 학습 모델 분기 (수동 30초 청킹 적용)
    elif "119 맞춤형" in model_type:
        import librosa
        import soundfile as sf
        import torch

        if "Small" in model_type:
            processor, model = load_custom_whisper("openai/whisper-small", CUSTOM_SMALL_LORA_PATH)
        elif "Base" in model_type:
            processor, model = load_custom_whisper("openai/whisper-base", CUSTOM_BASE_DORA_PATH)
        elif "Large" in model_type:
            processor, model = load_custom_whisper("openai/whisper-large-v3-turbo", CUSTOM_LARGE_DORA_PATH)

        # 1. 오디오 로드 및 16kHz 강제 리샘플링
        speech, sample_rate = sf.read(audio_path, dtype="float32")
        if speech.ndim > 1:
            speech = speech.mean(axis=1)
        if sample_rate != 16000:
            speech = librosa.resample(speech, orig_sr=sample_rate, target_sr=16000)

        # 2. FFmpeg 에러 원천 차단: 수동 30초 단위 자르기
        chunk_length_samples = 30 * 16000  # 30초 = 480,000 샘플
        result_texts = []
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="ko", task="transcribe")

        # 오디오를 30초씩 잘라서 순차적으로 번역 후 합치기
        for i in range(0, len(speech), chunk_length_samples):
            chunk = speech[i : i + chunk_length_samples]
            
            inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(model.device, dtype=model.dtype)

            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features,
                    forced_decoder_ids=forced_decoder_ids,
                    max_length=225,
                    num_beams=5,             # 5로 모두 통일
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3   # 동일 어구 무한 반복(할루시네이션) 방지
                )

            chunk_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            if chunk_text.strip():
                result_texts.append(chunk_text.strip())

        result_text = " ".join(result_texts)
        metadata = {"engine": model_type, "sampling_rate": 16000, "num_beams": 5, "chunk_length_s": 30}
        
        return result_text.strip(), metadata


if uploaded_file is not None:
    st.audio(uploaded_file)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_audio_path = tmp_file.name

    with st.spinner(f"[{selected_model}] 음성을 분석하는 중..."):
        try:
            runtime_paths = get_runtime_asset_paths()
            asr_config_path = get_asr_config_path()
            elapsed_placeholder = st.empty()
            start_time = time.perf_counter()

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    run_stt,
                    temp_audio_path,
                    selected_model,
                    asr_config_path,
                    runtime_paths["bpemodel"]
                )

                while not future.done():
                    elapsed_time = time.perf_counter() - start_time
                    elapsed_placeholder.info(f"분석 중... {elapsed_time:.1f}초 경과")
                    time.sleep(0.2)

                result = future.result()
                if isinstance(result, tuple):
                    result_text, metadata = result
                else:
                    result_text = result

            elapsed_time = time.perf_counter() - start_time
            elapsed_placeholder.empty()

            if selected_model == "ESPnet (배포 학습 모델)":
                try:
                    tokenized_text = tokenize_text(result_text)
                except Exception as e:
                    st.warning(f"BPE 토큰화 오류: {e}")

            st.success("분석 완료")
            st.info(f"분석 소요 시간: {elapsed_time:.2f}초")

            st.subheader("대화 내용")
            st.text_area("", value=result_text, height=300)

        except Exception as e:
            st.error(f"오류 발생: {e}")

        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)