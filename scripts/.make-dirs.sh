#!/bin/bash
## Use variables defined in before scripts to setup project based on templates

source scripts/.before-script.sh
source scripts/.functions.sh

## Create structure
make_a_dir $LOGS_FOLDER
add_gitkeep_in_folder $LOGS_FOLDER
make_a_dir $LOGS_CONFIG_FOLDER
make_a_dir $SRC_DIR
add_init_in_folder $SRC_DIR
make_a_dir $SRC_DIR_PKG
add_init_in_folder $SRC_DIR_PKG
make_a_dir $PACKAGES_DIR
add_gitkeep_in_folder $PACKAGES_DIR