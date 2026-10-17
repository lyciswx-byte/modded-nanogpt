#!/usr/bin/env bash
# GB10 / DGX Spark entrypoint; H100 configuration remains on 92-1xH100.
exec bash "$(dirname "$0")/run_gb10.sh"
