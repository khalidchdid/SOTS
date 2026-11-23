#!/bin/bash
## Use variables defined in before scripts to setup project based on templates

source scripts/.before-script.sh
source scripts/.functions.sh

copy_template_to_target $LOGS_CONFIG_TEMPLATE $LOGS_CONFIG_FILE