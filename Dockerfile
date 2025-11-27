# ===== HuggingFace Space Docker =====
FROM python:3.12-slim

# 환경변수
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    cmake \
    wget \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /app

# ===== 환경변수 설정 =====
ENV HOME=/app
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV HF_HOME=/app/.cache/huggingface
ENV HF_HUB_CACHE=/app/.cache/huggingface/hub
ENV HUGGINGFACE_HUB_CACHE=/app/.cache/huggingface/hub
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV CHROMA_DB_PATH=/app/.cache/chroma_db

# 캐시 디렉토리 생성
RUN mkdir -p /app/.cache/huggingface /app/.streamlit && \
    chmod -R 777 /app/.cache /app/.streamlit

# 의존성 파일 복사
COPY requirements.txt .

# pip 업그레이드
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch 설치 (CUDA 지원)
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    torchvision==0.18.0 \
    torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121

# llama-cpp-python 설치 (사전 빌드 CUDA 버전)
RUN pip install --no-cache-dir \
    llama-cpp-python==0.2.90 \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# 나머지 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 파일 복사 (start.sh 포함)
COPY . .

# start.sh 실행 권한 부여 (확실하게)
RUN chmod +x /app/start.sh || echo "start.sh not found, will create"

# start.sh가 없으면 직접 생성 (Fallback)
RUN if [ ! -f /app/start.sh ]; then \
    echo '#!/bin/bash' > /app/start.sh && \
    echo 'streamlit run chatbot_app.py --server.port=7860 --server.address=0.0.0.0 --server.headless=true' >> /app/start.sh && \
    chmod +x /app/start.sh; \
    fi

# 권한 확인 (디버깅)
RUN ls -la /app/start.sh

# Streamlit 포트
EXPOSE 7860

# 시작 (안전하게)
CMD ["/bin/bash", "/app/start.sh"]
