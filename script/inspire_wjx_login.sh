#!/usr/bin/env bash
set -euo pipefail
INSPIRE_LOGIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSPIRE_LOGIN_ENTRY="$(command -v inspire)"
IFS= read -r INSPIRE_LOGIN_SHEBANG < "${INSPIRE_LOGIN_ENTRY}"
INSPIRE_LOGIN_PYTHON="${INSPIRE_LOGIN_SHEBANG#\#!}"
if [[ ! -x "${INSPIRE_LOGIN_PYTHON}" ]]; then
  echo "Cannot resolve the installed Inspire Python runtime from ${INSPIRE_LOGIN_ENTRY}" >&2
  exit 1
fi
exec "${INSPIRE_LOGIN_PYTHON}" "${INSPIRE_LOGIN_ROOT}/scripts/refresh_inspire_login.py" "$@"
