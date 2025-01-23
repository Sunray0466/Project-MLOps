FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /app

COPY deployment/ app/

WORKDIR /app
RUN pip install -r requirements_backend.txt --no-cache-dir --verbose

EXPOSE $PORT
CMD exec uvicorn --port $PORT --host 0.0.0.0 backend:app