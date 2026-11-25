"""Guarda el contrato OpenAPI de la API en docs/openapi.json (sin levantar el servidor ni Redis).

Uso:  python scripts/exportar_openapi.py          (regenera el archivo)
      python scripts/exportar_openapi.py --check  (falla si el archivo no coincide con el codigo)
"""

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from app.main import crear_app  # noqa: E402

RUTA = RAIZ / "docs" / "openapi.json"


def contrato() -> str:
    return json.dumps(crear_app().openapi(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    texto = contrato()
    if "--check" in sys.argv:
        if not RUTA.exists() or RUTA.read_text(encoding="utf-8") != texto:
            sys.exit("docs/openapi.json está desactualizado: corré `python scripts/exportar_openapi.py`.")
        print("docs/openapi.json está al día.")
    else:
        RUTA.parent.mkdir(exist_ok=True)
        RUTA.write_text(texto, encoding="utf-8")
        print(f"Contrato guardado en {RUTA.relative_to(RAIZ)}")
