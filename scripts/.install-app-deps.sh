#!/bin/bash
# Call scripts with bash -i ./scripts/<scriptname>.sh

source scripts/.before-script.sh

read -p "Install deps for dev or prod (write dev or write prod, default to dev): " ENV_TARGET
ENV_TARGET=${ENV_TARGET:-dev}
uv sync --extra $ENV_TARGET