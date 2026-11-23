#!/bin/bash
## Use variables defined in before scripts to setup project based on templates

source scripts/.before-script.sh
source scripts/.functions.sh

# copy_template_to_target $FLAKE8_FILE_TEMPLATE $FLAKE8_FILE
copy_template_to_target $PRECOMMIT_FILE_TEMPLATE $PRECOMMIT_FILE
# copy_template_to_target $PYLINT_FILE_TEMPLATE $PYLINT_FILE