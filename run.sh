#!/usr/bin/env bash
# Despliegue local con UNA instruccion:  ./run.sh
#
#   1. comprueba Docker y el archivo .env (si no existe lo crea desde .env.example y te avisa)
#   2. construye las imagenes y levanta Redis, Phoenix, el indexador del RAG, la API y el worker
#   3. espera a que el RAG este indexado y a que la API responda
#
# Para apagar todo:  ./run.sh --detener
set -euo pipefail
cd "$(dirname "$0")"

COLOR_OK=$'\033[32m'; COLOR_ERR=$'\033[31m'; SIN_COLOR=$'\033[0m'
ok()  { echo "${COLOR_OK}✔${SIN_COLOR} $*"; }
fallar() { echo "${COLOR_ERR}✘ $*${SIN_COLOR}" >&2; exit 1; }

if [[ "${1:-}" == "--detener" ]]; then
  docker compose down
  ok "Todo detenido (los datos de Redis y Phoenix se conservan; para borrarlos: docker compose down -v)."
  exit 0
fi

command -v docker >/dev/null 2>&1 || fallar "No encuentro Docker. Instalá Docker Desktop: https://www.docker.com/products/docker-desktop/"
docker info >/dev/null 2>&1 || fallar "Docker está instalado pero no está corriendo. Abrí Docker Desktop y volvé a ejecutar ./run.sh"
docker compose version >/dev/null 2>&1 || fallar "Falta el plugin 'docker compose' (viene con Docker Desktop reciente)."
ok "Docker listo"

if [[ ! -f .env ]]; then
  cp .env.example .env
  fallar "Creé el archivo .env a partir de .env.example. Completá ANTHROPIC_API_KEY, PINECONE_API_KEY y GOOGLE_API_KEY y volvé a ejecutar ./run.sh"
fi

valor_de() {  # lee VARIABLE=valor del .env sin ejecutarlo
  # quita saltos de linea de Windows, comentarios al final de la linea, comillas y espacios (sin xargs: en macOS difiere)
  grep -E "^$1=" .env | tail -n1 | cut -d= -f2- | tr -d '\r' \
    | sed -E 's/[[:space:]]+#.*$//; s/^[[:space:]]+//; s/[[:space:]]+$//; s/^"(.*)"$/\1/' \
    | sed -E "s/^'(.*)'\$/\\1/" || true
}
faltan=()
for clave in ANTHROPIC_API_KEY PINECONE_API_KEY GOOGLE_API_KEY; do
  valor="$(valor_de "$clave")"
  if [[ -z "$valor" || "$valor" == pega-aca* ]]; then faltan+=("$clave"); fi
done
if [[ -z "$(valor_de INDEX_NAME)" ]]; then faltan+=("INDEX_NAME"); fi
if (( ${#faltan[@]} > 0 )); then
  fallar "Faltan valores en .env: ${faltan[*]}  (copiá .env.example y pegá tus claves; INDEX_NAME es el índice de Pinecone, p. ej. rag-gobierno-de-datos)"
fi
ok "Archivo .env completo"

echo "Construyendo y levantando los servicios (la primera vez tarda unos minutos)..."
# El worker depende del indexador (service_completed_successfully): si el indexador falla, `up` termina con error.
if ! docker compose up -d --build; then
  docker compose logs indexador 2>/dev/null | tail -n 25
  fallar "No se pudo levantar todo. Si el indexador del RAG falló, revisá PINECONE_API_KEY, GOOGLE_API_KEY e INDEX_NAME en .env."
fi

ok "RAG indexado en Pinecone"

echo "Esperando a que la API responda..."
for _ in $(seq 1 60); do
  if curl -fsS http://localhost:8000/health >/dev/null 2>&1; then
    ok "API lista"
    cat <<TXT

  API y documentación interactiva   http://localhost:8000/docs
  Trazas (Arize Phoenix)            http://localhost:6006
  Logs                              docker compose logs -f worker
  Apagar                            ./run.sh --detener

  Probalo con una tarea:
    curl -s -X POST http://localhost:8000/tasks -H 'content-type: application/json' \\
      -d '{"tarea": "Priorizá los incidentes críticos abiertos del maestro de artículos y citá la política que fija sus plazos."}'
TXT
    exit 0
  fi
  sleep 2
done
docker compose logs api | tail -n 25
fallar "La API no respondió en 2 minutos. Mirá los logs de arriba (docker compose logs)."
