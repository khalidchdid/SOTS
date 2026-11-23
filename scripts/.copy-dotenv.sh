#!/bin/bash
## Use variables defined in before scripts to setup project based on templates

source scripts/.before-script.sh
source scripts/.functions.sh

copy_template_to_target $DOTENV_FILE_TEMPLATE $DOTENV_FILE