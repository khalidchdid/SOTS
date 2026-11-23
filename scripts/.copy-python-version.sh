#!/bin/bash
## Use variables defined in before scripts to setup project based on templates

source scripts/.before-script.sh
source scripts/.functions.sh

copy_template_to_target $PYTHON_VERSION_FILE_TEMPLATE $PYTHON_VERSION_FILE