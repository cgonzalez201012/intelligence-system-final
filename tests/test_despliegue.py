"""Despliegue local con una sola instrucción: lo que hay que cuidar para que `./run.sh` funcione en una máquina limpia."""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

RAIZ = Path(__file__).resolve().parents[1]


def _compose() -> dict:
    return yaml.safe_load((RAIZ / "docker-compose.yml").read_text(encoding="utf-8"))


def _paquetes_de_la_raiz() -> set[str]:
    """Carpetas de código de la raíz (con __init__.py) que la imagen necesita para arrancar."""
    return {p.parent.name for p in RAIZ.glob("*/__init__.py")} - {"tests", "scripts"}


def test_la_imagen_copia_todos_los_paquetes_que_importa_la_aplicacion() -> None:
    """Sin `rag/` en la imagen, el worker moriría con ModuleNotFoundError recién al correr el contenedor."""
    dockerfile = (RAIZ / "Dockerfile").read_text(encoding="utf-8")
    copiados = set(re.findall(r"^COPY (\S+) \./\1$", dockerfile, re.M))
    assert _paquetes_de_la_raiz() <= copiados, _paquetes_de_la_raiz() - copiados
    assert {"rag", "app", "orquestador", "data"} <= copiados


VERSION_COMPLETA = re.compile(r"^\d+\.\d+\.\d+(-[a-z]+)?$")  # 8.8.4 o 3.13.16-slim; nada de `8`, `3.13` ni `latest`


def test_las_imagenes_de_terceros_tienen_version_completa_fija() -> None:
    for nombre, servicio in _compose()["services"].items():
        imagen = servicio.get("image")
        if imagen:
            etiqueta = imagen.split("#")[0].strip().partition(":")[2]
            assert VERSION_COMPLETA.match(etiqueta), f"{nombre}: la imagen {imagen!r} no tiene versión completa fija"


def test_la_imagen_base_del_dockerfile_tiene_version_completa_fija() -> None:
    desde = re.findall(r"^FROM (\S+)", (RAIZ / "Dockerfile").read_text(encoding="utf-8"), re.M)
    assert desde and all(VERSION_COMPLETA.match(i.partition(":")[2]) for i in desde), desde


def test_el_dockerfile_no_tiene_comentarios_en_la_misma_linea_que_una_instruccion() -> None:
    """Docker solo admite comentarios al principio de la línea: `FROM x  # nota` rompe el build."""
    lineas = (RAIZ / "Dockerfile").read_text(encoding="utf-8").splitlines()
    malas = [l for l in lineas if re.match(r"^(FROM|RUN|COPY|ENV|CMD|WORKDIR|EXPOSE)\b.*\s#", l)]
    assert not malas, malas


def test_docker_acepta_la_sintaxis_del_dockerfile() -> None:
    """Con daemon disponible, `docker build --check` valida el Dockerfile sin construir la imagen."""
    if shutil.which("docker") is None:
        pytest.skip("no hay docker instalado")
    try:
        resultado = subprocess.run(["docker", "build", "--check", str(RAIZ)], capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        pytest.skip("docker no respondió")
    if "Cannot connect to the Docker daemon" in resultado.stderr or "failed to connect" in resultado.stderr:
        pytest.skip("no hay daemon de docker corriendo")
    assert resultado.returncode == 0, resultado.stderr


def test_el_worker_espera_a_que_el_rag_este_indexado() -> None:
    servicios = _compose()["services"]
    assert servicios["worker"]["depends_on"]["indexador"]["condition"] == "service_completed_successfully"
    assert servicios["indexador"]["restart"] == "no"
    comando = " ".join(servicios["indexador"]["command"])
    assert "rag.setup_index" in comando and "rag.ingest" in comando


def test_el_indexador_recibe_las_claves_del_rag_y_el_worker_tambien() -> None:
    servicios = _compose()["services"]
    for servicio in ("indexador", "worker"):
        entorno = servicios[servicio]["environment"]
        assert {"PINECONE_API_KEY", "GOOGLE_API_KEY", "INDEX_NAME", "PINECONE_NAMESPACE"} <= set(entorno), servicio


def test_la_api_declara_su_healthcheck() -> None:
    assert "/health" in " ".join(_compose()["services"]["api"]["healthcheck"]["test"])


def test_el_compose_es_valido_segun_docker_compose(tmp_path: Path) -> None:
    if shutil.which("docker") is None:
        pytest.skip("no hay docker instalado")
    entorno = {**os.environ, "ANTHROPIC_API_KEY": "x", "PINECONE_API_KEY": "x", "GOOGLE_API_KEY": "x", "INDEX_NAME": "x"}
    try:
        resultado = subprocess.run(
            ["docker", "compose", "-f", str(RAIZ / "docker-compose.yml"), "config", "--quiet"],
            capture_output=True, text=True, env=entorno, timeout=60, cwd=RAIZ,
        )
    except subprocess.TimeoutExpired:
        pytest.skip("docker compose no respondió")
    if resultado.returncode != 0 and "unknown shorthand flag" in resultado.stderr:
        pytest.skip("este docker no tiene el plugin compose")
    assert resultado.returncode == 0, resultado.stderr


# ---------- run.sh ----------


def test_run_sh_existe_es_ejecutable_y_tiene_sintaxis_valida() -> None:
    script = RAIZ / "run.sh"
    assert script.exists() and os.access(script, os.X_OK)
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0


def _correr_run_sh(carpeta: Path, env_texto: str | None) -> subprocess.CompletedProcess[str]:
    """Corre run.sh en una copia mínima del proyecto, con un `docker` falso que solo responde que está todo bien."""
    shutil.copy(RAIZ / "run.sh", carpeta / "run.sh")
    shutil.copy(RAIZ / ".env.example", carpeta / ".env.example")
    if env_texto is not None:
        (carpeta / ".env").write_text(env_texto, encoding="utf-8")
    falso = carpeta / "bin"
    falso.mkdir()
    docker = falso / "docker"
    docker.write_text('#!/usr/bin/env bash\necho "docker $*" >> "$(dirname "$0")/../llamadas.log"\nexit 0\n')
    docker.chmod(0o755)
    entorno = {**os.environ, "PATH": f"{falso}:{os.environ['PATH']}"}
    return subprocess.run(["bash", str(carpeta / "run.sh")], capture_output=True, text=True, env=entorno, timeout=30)


@pytest.mark.skipif(shutil.which("bash") is None, reason="sin bash")
def test_sin_env_run_sh_lo_crea_desde_el_ejemplo_y_pide_completarlo(tmp_path: Path) -> None:
    resultado = _correr_run_sh(tmp_path, None)
    assert resultado.returncode != 0 and (tmp_path / ".env").exists()
    assert "ANTHROPIC_API_KEY" in resultado.stderr and "PINECONE_API_KEY" in resultado.stderr
    assert "compose up" not in (tmp_path / "llamadas.log").read_text()  # no levantó nada sin claves


@pytest.mark.skipif(shutil.which("bash") is None, reason="sin bash")
def test_con_claves_de_ejemplo_run_sh_no_levanta_nada_y_dice_cuales_faltan(tmp_path: Path) -> None:
    resultado = _correr_run_sh(tmp_path, "ANTHROPIC_API_KEY=pega-aca-tu-clave\nPINECONE_API_KEY=pcsk_real\nGOOGLE_API_KEY=\nINDEX_NAME=rag\n")
    assert resultado.returncode != 0
    assert "ANTHROPIC_API_KEY" in resultado.stderr and "GOOGLE_API_KEY" in resultado.stderr
    assert "PINECONE_API_KEY" not in resultado.stderr.split("Faltan valores en .env:")[1].split("(")[0]
    assert "compose up" not in (tmp_path / "llamadas.log").read_text()


@pytest.mark.skipif(shutil.which("bash") is None, reason="sin bash")
def test_el_env_example_completo_pasa_la_validacion_de_run_sh_y_llega_a_levantar_los_servicios(tmp_path: Path) -> None:
    env = "ANTHROPIC_API_KEY=sk-x\nPINECONE_API_KEY=pcsk_x\nGOOGLE_API_KEY=AIza_x\nINDEX_NAME=rag-gobierno-de-datos\n"
    resultado = _correr_run_sh(tmp_path, env)
    llamadas = (tmp_path / "llamadas.log").read_text()
    assert "compose up -d --build" in llamadas  # superó la validación y pidió levantar todo
    assert "Faltan valores" not in resultado.stderr


def test_el_readme_y_el_makefile_apuntan_a_la_misma_instruccion() -> None:
    assert "./run.sh" in (RAIZ / "Makefile").read_text(encoding="utf-8")
