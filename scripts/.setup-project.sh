#!/bin/bash
## Use variables defined in before scripts to setup project based on templates

source scripts/.before-script.sh
source scripts/.functions.sh

## Create structure
source scripts/.make-dirs.sh

## Copy templates
source scripts/.copy-pyproject.sh
source scripts/.copy-license.sh
source scripts/.copy-app.sh
source scripts/.copy-dotenv.sh
source scripts/.copy-gitignore.sh
source scripts/.copy-logs.sh
source scripts/.copy-quality-cfg.sh
source scripts/.copy-readme.sh
source scripts/.copy-python-version.sh

