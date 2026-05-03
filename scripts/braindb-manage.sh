#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$REPO_ROOT/docker-compose.yml"
ENV_FILE="$REPO_ROOT/.env"
ENV_EXAMPLE="$REPO_ROOT/.env.example"
DOCKER_BIN="${DOCKER_BIN:-docker}"

log() { printf '%s\n' "$*"; }
warn() { printf 'warn: %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

compose() {
  "$DOCKER_BIN" compose -f "$COMPOSE_FILE" "$@"
}

ensure_env_file() {
  if [[ ! -f "$ENV_FILE" ]]; then
    [[ -f "$ENV_EXAMPLE" ]] || die "missing .env.example; cannot create .env"
    cp "$ENV_EXAMPLE" "$ENV_FILE"
    warn "created .env from .env.example"
  fi
}

env_value() {
  local key="$1"
  local default_value="${2:-}"
  [[ -f "$ENV_FILE" ]] || { printf '%s' "$default_value"; return; }

  python - "$ENV_FILE" "$key" "$default_value" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
default = sys.argv[3]

for line in path.read_text().splitlines():
    if not line or line.lstrip().startswith('#') or '=' not in line:
        continue
    k, v = line.split('=', 1)
    if k.strip() == key:
        print(v.strip())
        raise SystemExit(0)

print(default)
PY
}

env_set() {
  local key="$1"
  local value="$2"

  python - "$ENV_FILE" "$key" "$value" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
value = sys.argv[3]

lines = path.read_text().splitlines()
updated = False
out = []

for line in lines:
    if line.startswith(f"{key}="):
        out.append(f"{key}={value}")
        updated = True
    else:
        out.append(line)

if not updated:
    out.append(f"{key}={value}")

path.write_text("\n".join(out) + "\n")
PY
}

database_url() {
  env_value DATABASE_URL
}

ensure_database_url() {
  local url
  url="$(database_url)"
  [[ -n "$url" ]] || die ".env must set DATABASE_URL"
}

ensure_network() {
  "$DOCKER_BIN" network inspect local-network >/dev/null 2>&1 || "$DOCKER_BIN" network create local-network >/dev/null
}

health_port() {
  env_value API_PORT 8100
}

health_url() {
  printf 'http://localhost:%s/health' "$(health_port)"
}

wait_for_health() {
  require_cmd curl
  local url attempts sleep_s response
  url="$(health_url)"
  attempts=30
  sleep_s=2

  while (( attempts > 0 )); do
    response="$(curl -fsS "$url" 2>/dev/null || true)"
    if [[ "$response" == *'"status":"ok"'* ]]; then
      log "health: ok ($url)"
      return 0
    fi
    sleep "$sleep_s"
    ((attempts--))
  done

  warn "health check failed after waiting; try: curl -s $(health_url)"
  return 1
}

openai_compatible_base_url() {
  env_value AGENT_BASE_URL
}

openai_compatible_root_url() {
  local base
  base="$(openai_compatible_base_url)"
  [[ -n "$base" ]] || return 1
  base="${base%/}"
  case "$base" in
    */v1)
      printf '%s\n' "${base%/v1}"
      ;;
    *)
      printf '%s\n' "$base"
      ;;
  esac
}

fetch_openai_compatible_models() {
  require_cmd curl
  local base root payload
  base="$(openai_compatible_base_url)"
  [[ -n "$base" ]] || return 1

  root="$(openai_compatible_root_url)"

  payload="$(
    curl -fsS --max-time 4 "$base/models" 2>/dev/null || \
    curl -fsS --max-time 4 "$root/api/tags" 2>/dev/null || \
    curl -fsS --max-time 4 "$root/v1/models" 2>/dev/null || true
  )"

  [[ -n "$payload" ]] || return 1

  python - "$payload" <<'PY'
import json
import sys

raw = sys.argv[1]
try:
    data = json.loads(raw)
except Exception:
    raise SystemExit(1)

models = []
if isinstance(data, dict):
    if isinstance(data.get('models'), list):
        for item in data['models']:
            if isinstance(item, dict):
                name = item.get('name') or item.get('model') or item.get('id')
                if name:
                    models.append(name)
    if isinstance(data.get('data'), list):
        for item in data['data']:
            if isinstance(item, dict):
                name = item.get('id') or item.get('name')
                if name:
                    models.append(name)

seen = set()
for model in models:
    model = model.strip()
    if not model:
        continue
    if not model.startswith('openai/'):
        model = f'openai/{model}'
    if model not in seen:
        seen.add(model)
        print(model)
PY
}

maybe_set_openai_compatible_model() {
  local existing models count model
  existing="$(env_value AGENT_MODEL)"
  [[ -n "$existing" ]] && return 0

  if ! models="$(fetch_openai_compatible_models)"; then
    models=""
  fi
  if [[ -z "$models" ]]; then
    die "LLM_PROFILE=openai_compatible/local_ollama needs AGENT_MODEL, and auto-discovery failed. Run: ./scripts/braindb-manage.sh models"
  fi

  count=0
  while IFS= read -r model; do
    [[ -n "$model" ]] && ((count++))
  done <<<"$models"
  if [[ "$count" == "1" ]]; then
    model="$(printf '%s\n' "$models")"
    env_set AGENT_MODEL "$model"
    log "set AGENT_MODEL=$model"
    return 0
  fi

  die "LLM_PROFILE=openai_compatible/local_ollama needs AGENT_MODEL and discovery found multiple models. Run: ./scripts/braindb-manage.sh models; then set AGENT_MODEL=openai/<model-id> in .env"
}

warn_if_unconfigured() {
  local database_url llm_profile deepinfra_key nim_key openai_key agent_model agent_base_url
  database_url="$(env_value DATABASE_URL)"
  llm_profile="$(env_value LLM_PROFILE deepinfra)"
  deepinfra_key="$(env_value DEEPINFRA_API_KEY)"
  nim_key="$(env_value NVIDIA_NIM_API_KEY)"
  openai_key="$(env_value OPENAI_API_KEY)"
  agent_model="$(env_value AGENT_MODEL)"
  agent_base_url="$(env_value AGENT_BASE_URL)"

  case "$database_url" in
    ""|postgresql://user:password@host:5432/braindb)
      warn "DATABASE_URL still looks like the example; update .env before expecting a successful start"
      ;;
  esac

  case "$llm_profile" in
    deepinfra)
      [[ -n "$deepinfra_key" ]] || warn "LLM_PROFILE=deepinfra but DEEPINFRA_API_KEY is empty"
      ;;
    nim)
      [[ -n "$nim_key" ]] || warn "LLM_PROFILE=nim but NVIDIA_NIM_API_KEY is empty"
      ;;
    codex)
      [[ -n "$openai_key" ]] || warn "LLM_PROFILE=codex but OPENAI_API_KEY is empty"
      ;;
    openai_compatible|local_ollama)
      [[ -n "$agent_base_url" ]] || die "LLM_PROFILE=openai_compatible/local_ollama requires AGENT_BASE_URL"
      if [[ -z "$agent_model" ]]; then
        maybe_set_openai_compatible_model
      fi
      ;;
  esac
}

print_openai_compatible_models() {
  local models
  if ! models="$(fetch_openai_compatible_models)"; then
    die "could not reach OpenAI-compatible models endpoint from AGENT_BASE_URL"
  fi
  [[ -n "$models" ]] || die "no OpenAI-compatible models found at AGENT_BASE_URL"
  printf '%s\n' "$models"
}

start_stack() {
  ensure_env_file
  ensure_database_url
  warn_if_unconfigured
  ensure_network
  compose up -d --build
  wait_for_health
}

update_stack() {
  ensure_env_file
  ensure_database_url
  warn_if_unconfigured
  ensure_network
  compose up -d --build --force-recreate
  wait_for_health
}

status_stack() {
  ensure_env_file
  ensure_database_url
  ensure_network
  compose ps
  if command -v curl >/dev/null 2>&1; then
    curl -fsS "$(health_url)" || true
    printf '\n'
  fi
}

logs_stack() {
  ensure_env_file
  ensure_database_url
  ensure_network
  compose logs -f --tail="${TAIL_LINES:-200}" "$@"
}

usage() {
  cat <<'EOF'
Usage: braindb-manage.sh <command>

Commands:
  start, bootstrap, up  Ensure .env/network and start the stack
  update, upgrade       Recreate services
  status                Show compose status and health
  logs [service...]     Follow service logs (default tail=200)
  models                List models from AGENT_BASE_URL
  help                  Show this help

Env overrides:
  DOCKER_BIN=docker|podman  Docker-compatible CLI to use
  TAIL_LINES=200            Lines shown by logs
EOF
}

main() {
  require_cmd "$DOCKER_BIN"
  local cmd="${1:-help}"
  shift || true

  case "$cmd" in
    start|bootstrap|up)
      start_stack "$@"
      ;;
    update|upgrade)
      update_stack "$@"
      ;;
    status)
      status_stack "$@"
      ;;
    logs)
      logs_stack "$@"
      ;;
    models)
      ensure_env_file
      print_openai_compatible_models
      ;;
    help|-h|--help)
      usage
      ;;
    *)
      die "unknown command: $cmd (try: help)"
      ;;
  esac
}

main "$@"
