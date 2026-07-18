FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY gateway/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip==25.1.1 \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && useradd --create-home --uid 10001 gateway
COPY gateway /app/gateway
USER gateway
EXPOSE 8080
CMD ["uvicorn", "gateway.app:app", "--host", "0.0.0.0", "--port", "8080"]
