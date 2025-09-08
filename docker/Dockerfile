FROM nvcr.io/nvidia/pytorch:25.06-py3

WORKDIR /app

COPY ../requirements.txt .
RUN sed -i '/^torch/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

COPY main.py config.py ./
COPY src/ ./src/

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["python", "main.py"]