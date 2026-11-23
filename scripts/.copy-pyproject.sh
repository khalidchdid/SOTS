#!/bin/bash
## Use variables defined in before scripts to setup project based on templates

source scripts/.before-script.sh
source scripts/.functions.sh

copy_template_to_target $PYPROJECT_FILE_TEMPLATE $PYPROJECT_FILE
copy_template_to_target $PYPROJECT_SETUP_FILE_TEMPLATE $PYPROJECT_SETUP_FILE