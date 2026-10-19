#!/usr/bin/env bash
set -euo pipefail

FUSIONRL_PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${FUSIONRL_VERL_ROOT:?Set FUSIONRL_VERL_ROOT to the pin-checked, patched XiaomiMiMo/verl checkout}"
export PYTHONPATH="${FUSIONRL_PACKAGE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec python -m fusionrl.integrations.verl.launcher --verl-root "${FUSIONRL_VERL_ROOT}" "$@"
