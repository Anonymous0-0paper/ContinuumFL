FROM nvcr.io/nvidia/pytorch:25.08-py3

WORKDIR /app

COPY ../requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "main.py"]