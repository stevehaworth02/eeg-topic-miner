FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git gcc libopenblas-dev libomp-dev wget curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/
COPY config/ config/
COPY run_pipeline.sh run_pipeline.sh
COPY run_pipeline.ps1 run_pipeline.ps1
COPY .env.example ./

# Dear users, please don't copy your real .env here.

CMD bash -c "bash run_pipeline.sh && python src/query_faiss_index.py"
