# Imagen unica para la API, el worker y el indexador del RAG (cambia solo el comando, ver docker-compose.yml).
FROM python:3.13-slim  # misma version mayor con la que se desarrollo y probo el proyecto

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# Primero las dependencias, para que Docker reutilice esta capa si el codigo cambia.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY orquestador ./orquestador
COPY rag ./rag
COPY app ./app
COPY data ./data

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
