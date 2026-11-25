# Atajos. Todos asumen el entorno virtual activado (source .venv/bin/activate).
.PHONY: evidencia instalar test tipos todo infra api worker up detener verificar carga diagrama indexar

instalar:
	pip install -r requirements.txt

test:
	pytest

tipos:
	mypy .

todo: instalar test tipos

# Crea el indice de Pinecone (si falta) y sincroniza data/politicas (necesita el .env)
indexar:
	python -m rag.setup_index
	python -m rag.ingest

# Redis + Phoenix en Docker; la API y el worker corren en tu terminal (ver README)
infra:
	docker compose up -d redis phoenix

api:
	uvicorn app.main:app --port 8000

worker:
	python -m app.worker

# Todo en contenedores, con una instruccion (valida el .env, indexa el RAG y espera a la API)
up:
	./run.sh

detener:
	./run.sh --detener

# Comprueba contra el Redis REAL: modulos, checkpoints, pausa y reanudacion (no usa la API de Anthropic)
verificar:
	python scripts/verificar_entorno.py

# Las 6 corridas de evidencia (RAG, aprobación, rechazo, fuera de alcance) contra el sistema levantado con ./run.sh
evidencia:
	python scripts/corridas_evidencia.py

# Prueba de carga: 5 peticiones concurrentes contra la API (necesita la API y el worker andando)
carga:
	python scripts/carga.py

diagrama:
	python scripts/generar_diagrama.py
