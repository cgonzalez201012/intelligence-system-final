# Atajos. Todos asumen el entorno virtual activado (source .venv/bin/activate).
.PHONY: instalar test tipos todo infra api worker up verificar carga diagrama

instalar:
	pip install -r requirements.txt

test:
	pytest

tipos:
	mypy .

todo: instalar test tipos

# Redis + Phoenix en Docker; la API y el worker corren en tu terminal (ver README)
infra:
	docker compose up -d redis phoenix

api:
	uvicorn app.main:app --port 8000

worker:
	python -m app.worker

# Todo en contenedores
up:
	docker compose up --build

# Comprueba contra el Redis REAL: modulos, checkpoints, pausa y reanudacion (no usa la API de Anthropic)
verificar:
	python scripts/verificar_entorno.py

# Prueba de carga: 5 peticiones concurrentes contra la API (necesita la API y el worker andando)
carga:
	python scripts/carga.py

diagrama:
	python scripts/generar_diagrama.py
