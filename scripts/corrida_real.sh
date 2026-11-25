#!/usr/bin/env bash
# Corrida real de punta a punta: levanta todo en Docker, prueba una tarea con aprobacion humana
# y despues dispara las 5 peticiones concurrentes. Gasta API de Anthropic (centavos).
# Deja el resultado en docs/evidencia/corrida-real.log y los resumenes JSON de cada prueba.
# Uso (desde la raiz del proyecto, con la clave ya pegada en .env):
#   bash scripts/corrida_real.sh          # las dos pruebas, una detras de otra
#   bash scripts/corrida_real.sh carga    # solo las 5 peticiones concurrentes (dashboard limpio, con 5 trazas)
#   bash scripts/corrida_real.sh hitl     # solo la tarea con aprobacion humana
cd "$(dirname "$0")/.." || exit 1
MODO="${1:-todo}"
case "$MODO" in todo|carga|hitl) ;; *) echo "Modo desconocido: $MODO (usar todo, carga o hitl)"; exit 2 ;; esac
mkdir -p docs/evidencia
LOG=docs/evidencia/corrida-real-$MODO.log
exec > >(tee "$LOG") 2>&1
paso() { printf '\n=== %s ===\n' "$1"; }

paso "Clave de Anthropic"
if grep -Eq '^ANTHROPIC_API_KEY=sk-ant-[A-Za-z0-9_-]{20,}' .env 2>/dev/null; then
  echo "hay una clave cargada en .env (no se muestra)"
else
  echo "ERROR: falta la clave. Abri .env y pegala en la linea ANTHROPIC_API_KEY=..."; exit 1
fi

paso "Levantar Redis, Phoenix, API y worker"
docker compose up -d --build 2>&1 | tail -25
echo "esperando a la API..."
for _ in $(seq 1 60); do curl -sf http://localhost:8000/health >/dev/null && break; sleep 2; done
curl -s http://localhost:8000/health || { echo "ERROR: la API no responde"; docker compose logs --tail 40 api worker; exit 1; }
echo
docker compose ps --format 'table {{.Service}}\t{{.Status}}'

# shellcheck disable=SC1091
source .venv/bin/activate

if [ "$MODO" != "carga" ]; then
paso "Prueba 1: una tarea con incidentes criticos (se pausa, el script la aprueba, termina)"
python scripts/carga.py --n 1 --perfil con-aprobacion --precio-entrada 1 --precio-salida 5 \
  --salida docs/evidencia/hitl-1-job.json
echo "Avisos 'enviados' (simulados) en Redis:"
docker compose exec -T redis redis-cli LRANGE escalaciones:enviadas 0 -1 | cut -c1-200
fi
[ "$MODO" = "todo" ] && sleep 5

if [ "$MODO" != "hitl" ]; then
paso "Prueba 2: 5 peticiones concurrentes"
date '+Inicio: %H:%M:%S'
python scripts/carga.py --n 5 --precio-entrada 1 --precio-salida 5 \
  --salida docs/evidencia/carga-5-concurrentes.json
date '+Fin: %H:%M:%S'
fi

paso "Ultimas lineas del worker"
docker compose logs worker --tail 25 --no-log-prefix 2>&1 | cut -c1-220

paso "Control de trazas"
RECHAZADAS=$(docker compose logs worker --no-log-prefix 2>&1 | grep -c "Failed to export spans")
if [ "$RECHAZADAS" -eq 0 ]; then echo "ok: Phoenix recibio las trazas sin errores"; else echo "ATENCION: $RECHAZADAS errores al enviar trazas a Phoenix"; fi

echo
echo "Listo. Ahora abri http://localhost:6006 (Phoenix) para sacar las capturas."
