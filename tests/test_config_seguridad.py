"""Configuracion, higiene del repositorio (sin secretos) y reproducibilidad."""

import ast
import re
from pathlib import Path

import pytest

from orquestador.config import BASE_DIR, Configuracion
from orquestador.errores import ErrorConfiguracion

RAIZ = Path(BASE_DIR)
VARIABLES = (
    "ANTHROPIC_API_KEY", "MODELO_LLM", "MODELO_SUPERVISOR", "MODELO_ESPECIALISTAS",
    "MAX_PASOS", "MAX_INTENTOS_POR_AGENTE", "RECURSION_ESPECIALISTA",
    "REDIS_URL", "OBSERVABILIDAD", "PHOENIX_COLLECTOR_ENDPOINT", "PHOENIX_PROJECT_NAME",
    "VERSION_PROMPTS", "WORKER_CONCURRENCIA", "TTL_JOBS_SEGUNDOS", "TTL_CHECKPOINTS_MINUTOS",
)


@pytest.fixture(autouse=True)
def _entorno_limpio(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in VARIABLES:
        monkeypatch.delenv(variable, raising=False)


def _archivos_del_repo() -> list[Path]:
    ignorados = {".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".git", ".ipynb_checkpoints"}
    return [
        ruta for ruta in RAIZ.rglob("*")
        if ruta.is_file() and not (set(ruta.relative_to(RAIZ).parts) & ignorados) and ruta.name != ".env"
    ]


# ---------- Configuracion ----------


def test_valores_por_defecto() -> None:
    config = Configuracion.desde_entorno(cargar_dotenv=False, requerir_clave=False)
    assert (config.modelo_supervisor, config.modelo_especialistas) == ("claude-haiku-4-5", "claude-haiku-4-5")
    assert (config.max_pasos, config.max_intentos_por_agente, config.recursion_especialista) == (10, 3, 12)


def test_sin_clave_falla_con_un_mensaje_que_dice_que_hacer() -> None:
    with pytest.raises(ErrorConfiguracion, match=r"Copiá \.env\.example"):
        Configuracion.desde_entorno(cargar_dotenv=False)


def test_la_clave_se_lee_del_entorno_y_se_limpia(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "  clave  ")
    assert Configuracion.desde_entorno(cargar_dotenv=False).anthropic_api_key == "clave"


def test_un_modelo_por_agente_pisa_al_modelo_general(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODELO_LLM", "modelo-general")
    monkeypatch.setenv("MODELO_SUPERVISOR", "modelo-del-supervisor")
    config = Configuracion.desde_entorno(cargar_dotenv=False, requerir_clave=False)
    assert (config.modelo_supervisor, config.modelo_especialistas) == ("modelo-del-supervisor", "modelo-general")


@pytest.mark.parametrize(
    ("variable", "valor", "fragmento"),
    [
        ("MAX_PASOS", "dos", "numero entero"),
        ("MAX_PASOS", "2", "al menos 3"),
        ("MAX_INTENTOS_POR_AGENTE", "0", "al menos 1"),
        ("RECURSION_ESPECIALISTA", "3", "al menos 4"),
    ],
)
def test_los_limites_absurdos_se_rechazan(monkeypatch: pytest.MonkeyPatch, variable: str, valor: str, fragmento: str) -> None:
    monkeypatch.setenv(variable, valor)
    with pytest.raises(ErrorConfiguracion, match=fragmento):
        Configuracion.desde_entorno(cargar_dotenv=False, requerir_clave=False)


def test_el_recursion_limit_del_grafo_deja_margen_sobre_el_maximo_teorico() -> None:
    """Con N delegaciones hacen falta 3N + 1 pasos del grafo (supervisor, especialista, validacion y el cierre)."""
    for max_pasos in (3, 10, 25):
        config = Configuracion("k", "m", "m", max_pasos, 3, 12)
        assert config.recursion_limit_grafo > 3 * max_pasos + 1


def test_la_configuracion_es_inmutable() -> None:
    config = Configuracion("k", "m", "m", 10, 3, 12)
    with pytest.raises(AttributeError):
        config.max_pasos = 99  # type: ignore[misc]


# ---------- Secretos ----------


def test_ningun_archivo_del_repo_contiene_una_clave_de_api() -> None:
    patron = re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}|AKIA[0-9A-Z]{16}|tvly-[A-Za-z0-9]{10,}|ghp_[A-Za-z0-9]{20,}")
    for ruta in _archivos_del_repo():
        if ruta.suffix in {".png", ".pyc"}:
            continue
        assert not patron.search(ruta.read_text(encoding="utf-8", errors="ignore")), f"posible clave en {ruta}"


def test_el_gitignore_excluye_el_env_pero_no_el_ejemplo() -> None:
    reglas = (RAIZ / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in reglas and "!.env.example" in reglas
    assert ".venv/" in reglas and "__pycache__/" in reglas


def test_el_env_example_tiene_un_marcador_y_documenta_todas_las_variables() -> None:
    texto = (RAIZ / ".env.example").read_text(encoding="utf-8")
    assert "ANTHROPIC_API_KEY=pega-aca-tu-clave" in texto
    for variable in VARIABLES:
        assert variable in texto, variable


def test_no_hay_un_env_con_secretos_dentro_del_repo_publicado() -> None:
    """El .env puede existir en la maquina de Cris (esta ignorado); aca se verifica que git lo ignora."""
    assert ".env" in (RAIZ / ".gitignore").read_text(encoding="utf-8").splitlines()


# ---------- Reproducibilidad ----------


def test_las_dependencias_estan_fijadas_a_una_version_exacta() -> None:
    for linea in (RAIZ / "requirements.txt").read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith(("#", "-r ")):
            continue
        assert re.fullmatch(r"[A-Za-z0-9_.\-]+(\[[a-z,]+\])?==\d+(\.\d+)*", linea), linea


def test_requirements_incluye_lo_necesario_para_levantar_y_probar_con_un_solo_comando() -> None:
    texto = (RAIZ / "requirements.txt").read_text(encoding="utf-8")
    for paquete in (
        "langgraph==", "langchain-anthropic==", "pydantic==", "fastapi==", "uvicorn", "redis==",
        "langgraph-checkpoint-redis==", "arize-phoenix-otel==", "openinference-instrumentation-langchain==",
        "pytest==", "pytest-asyncio==", "fakeredis==", "httpx==",
    ):
        assert paquete in texto, paquete


def test_docker_compose_levanta_redis_phoenix_api_y_worker() -> None:
    compose = (RAIZ / "docker-compose.yml").read_text(encoding="utf-8")
    for servicio in ("redis:", "phoenix:", "api:", "worker:"):
        assert f"\n  {servicio}" in compose, servicio
    assert "image: redis:8" in compose, "los checkpoints necesitan RediSearch y RedisJSON (Redis 8+ o Stack)"
    assert "app.worker" in compose and "app.main:app" in compose
    assert "sk-ant" not in compose


def test_el_docker_compose_es_yaml_valido_y_define_los_servicios() -> None:
    import yaml

    compose = yaml.safe_load((RAIZ / "docker-compose.yml").read_text(encoding="utf-8"))
    assert set(compose["services"]) == {"redis", "phoenix", "indexador", "api", "worker"}
    assert compose["services"]["worker"]["command"] == ["python", "-m", "app.worker"]
    assert "ANTHROPIC_API_KEY" in compose["services"]["worker"]["environment"]
    assert compose["services"]["phoenix"]["environment"]["PHOENIX_ENABLE_AUTH"] == "false", "con login, Phoenix rechaza las trazas (401)"


def test_el_dockerfile_copia_solo_el_codigo_y_los_datos() -> None:
    dockerfile = (RAIZ / "Dockerfile").read_text(encoding="utf-8")
    for linea in ("COPY orquestador ./orquestador", "COPY rag ./rag", "COPY app ./app", "COPY data ./data"):
        assert linea in dockerfile
    assert ".env" not in dockerfile.replace(".env.example", "")


# ---------- Calidad del codigo ----------


def _modulos_del_proyecto() -> list[Path]:
    carpetas = [RAIZ / "orquestador", RAIZ / "orquestador" / "agents", RAIZ / "app", RAIZ / "scripts"]
    return sorted(ruta for carpeta in carpetas for ruta in carpeta.glob("*.py"))


def test_todas_las_funciones_del_proyecto_tienen_anotaciones_de_tipo() -> None:
    sin_tipos: list[str] = []
    for ruta in _modulos_del_proyecto():
        for nodo in ast.walk(ast.parse(ruta.read_text(encoding="utf-8"))):
            if isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef)):
                argumentos = [a for a in [*nodo.args.args, *nodo.args.kwonlyargs] if a.arg not in {"self", "cls"}]
                if nodo.returns is None or any(a.annotation is None for a in argumentos):
                    sin_tipos.append(f"{ruta.relative_to(RAIZ)}:{nodo.name}")
    assert sin_tipos == []


def test_todos_los_modulos_tienen_docstring() -> None:
    for ruta in _modulos_del_proyecto():
        if ruta.name == "__init__.py" and not ruta.read_text(encoding="utf-8").strip():
            continue
        assert ast.get_docstring(ast.parse(ruta.read_text(encoding="utf-8"))), str(ruta)


def test_ningun_modulo_manda_temperature_al_modelo() -> None:
    """La API rechaza `temperature` con este modelo (400): no tiene que aparecer en el codigo que arma el modelo."""
    for ruta in _modulos_del_proyecto():
        arbol = ast.parse(ruta.read_text(encoding="utf-8"))
        llamadas = [n for n in ast.walk(arbol) if isinstance(n, ast.keyword) and n.arg == "temperature"]
        assert llamadas == [], str(ruta)


# ---------- Documentacion ----------


def _mermaid_del_grafo_de_produccion() -> str:
    from langgraph.checkpoint.memory import InMemorySaver

    from app.graph import construir_grafo_produccion
    from tests.fakes import SupervisorFalso, decidir, especialistas_falsos

    class _Nulo:
        async def notificar(self, job_id: str, accion: object) -> object:
            raise NotImplementedError

    grafo = construir_grafo_produccion(
        SupervisorFalso(guion=[decidir("FINISH")]), especialistas_falsos(),
        Configuracion("k", "m", "m", 10, 3, 12), checkpointer=InMemorySaver(), notificador=_Nulo(),  # type: ignore[arg-type]
    )
    return grafo.get_graph().draw_mermaid().rstrip("\n")


def test_el_diagrama_del_readme_es_el_que_genera_langgraph() -> None:
    """Si cambia el grafo y nadie regenera el diagrama (`make diagrama`), este test avisa."""
    generado = _mermaid_del_grafo_de_produccion()
    assert (RAIZ / "docs" / "grafo.mmd").read_text(encoding="utf-8").rstrip("\n") == generado
    assert f"```mermaid\n{generado}\n```" in (RAIZ / "README.md").read_text(encoding="utf-8")


def test_el_readme_no_deja_marcadores_sin_completar() -> None:
    texto = (RAIZ / "README.md").read_text(encoding="utf-8")
    assert "{{" not in texto and "(pendiente)" not in texto and "TODO" not in texto and "COMPLETAR" not in texto


def test_el_readme_referencia_archivos_que_existen() -> None:
    texto = (RAIZ / "README.md").read_text(encoding="utf-8")
    rutas = re.findall(r"\(((?:docs|screenshots|scripts|app|orquestador|tests)/[^)\s#]+)\)", texto)
    assert rutas, "el README deberia enlazar a la evidencia"
    for ruta in rutas:
        assert (RAIZ / ruta).exists(), ruta


def test_el_readme_explica_como_levantar_y_como_reproducir_la_prueba_de_carga() -> None:
    texto = (RAIZ / "README.md").read_text(encoding="utf-8")
    for fragmento in (
        "docker compose up -d redis phoenix", "python -m app.worker", "uvicorn app.main:app",
        "scripts/carga.py", "/approve", "p95", "costo por ejecución",
    ):
        assert fragmento in texto, fragmento


def test_la_estructura_pedida_por_la_consigna_existe() -> None:
    for ruta in (
        "app/main.py", "app/graph.py", "app/worker.py", "app/observability.py", "app/hitl.py",
        "requirements.txt", "docker-compose.yml", ".env.example", "README.md", "screenshots",
    ):
        assert (RAIZ / ruta).exists(), ruta


# ---------- Evidencia ----------


def test_las_capturas_pedidas_estan_en_la_carpeta_screenshots() -> None:
    capturas = {ruta.name.lower() for ruta in (RAIZ / "screenshots").glob("*") if ruta.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    assert capturas, "falta la evidencia: capturas de las trazas y del dashboard en screenshots/"
    for tema in ("traza", "costo", "p95"):
        assert any(tema in nombre for nombre in capturas), f"falta una captura de {tema}"
