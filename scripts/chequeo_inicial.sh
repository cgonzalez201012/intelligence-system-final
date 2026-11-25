#!/usr/bin/env bash
# Prepara el entorno y comprueba todo lo que no necesita la clave de Anthropic.
# Deja el resultado completo en docs/evidencia/chequeo-inicial.log
# Uso (desde la raiz del proyecto):  bash scripts/chequeo_inicial.sh
cd "$(dirname "$0")/.." || exit 1
mkdir -p docs/evidencia
LOG=docs/evidencia/chequeo-inicial.log
exec > >(tee "$LOG") 2>&1

paso() { printf '\n=== %s ===\n' "$1"; }

paso "Python"
PY=""
for candidato in python3.13 python3.12 python3; do
  if command -v "$candidato" >/dev/null 2>&1 && "$candidato" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)'; then PY="$candidato"; break; fi
done
if [ -z "$PY" ]; then echo "ERROR: hace falta Python 3.12 o superior."; python3 --version; exit 1; fi
"$PY" --version

paso "Entorno virtual e instalacion"
[ -d .venv ] || "$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt && echo "dependencias instaladas"

paso "Tests"
pytest -q 2>&1 | tail -8
paso "Tipos"
mypy . 2>&1 | tail -3

paso "Docker"
docker --version && docker compose version
docker compose up -d redis phoenix 2>&1 | grep -v -E 'Pulling fs layer|Waiting|Downloading|Extracting|Download complete|Verifying Checksum|Pull complete'
echo "esperando a que Redis quede sano..."
for _ in $(seq 1 30); do
  [ "$(docker compose ps redis --format '{{.Health}}' 2>/dev/null)" = "healthy" ] && break
  sleep 2
done
docker compose ps
docker compose exec -T redis redis-server --version

paso "Verificacion contra el Redis real (modulos, checkpoints, pausa y reanudacion)"
python scripts/verificar_entorno.py
echo "codigo de salida: $?"

paso "Phoenix"
curl -s -o /dev/null -w "Phoenix responde en http://localhost:6006 con HTTP %{http_code}\n" http://localhost:6006/ || echo "Phoenix no responde"
echo
echo "Listo. Log guardado en $LOG"
