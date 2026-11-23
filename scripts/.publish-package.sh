#!/bin/bash
# Call scripts with bash -i ./scripts/<scriptname>.sh

source scripts/.before-script.sh

DEFAULT_NEXUS_URL=https://nexus.framatome.corp/repository/dpitdatascience/
read -p "Nexus URL (default to $DEFAULT_NEXUS_URL): " NEXUS_URL
NEXUS_URL=${NEXUS_URL:-$DEFAULT_NEXUS_URL}

uv publish --publish-url $NEXUS_URL