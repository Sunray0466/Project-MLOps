FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /app

COPY deployment/requirements_frontend.txt app/requirements_frontend.txt
COPY deployment/frontend.py app/frontend.py

WORKDIR /app

RUN pip install -r requirements_frontend.txt --no-cache-dir --verbose

EXPOSE 8080

ENTRYPOINT ["streamlit", "run", "frontend.py", "--server.port=8080", "--server.address=0.0.0.0"]