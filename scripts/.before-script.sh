#!/bin/bash
## Source / include this script before anu other script to pre-perfom some actions
##  and make env variables accessibles

# Path
export PATH="$HOME/.local/bin:$PATH"

# Templates
export SCRIPTS_FOLDER="scripts"
export TEMPLATES_FOLDER="$SCRIPTS_FOLDER/.templates"

# Get versions and project name from toml file or from user input if toml file is missing
if [ -f "pyproject.toml" ]; then 
    export PROJECT_NAME=$(grep -m 1 name pyproject.toml | tr -s ' ' | tr -d '"' | tr -d "'" | cut -d' ' -f3)
    export VERSION=$(grep -m 1 version pyproject.toml | tr -s ' ' | tr -d '"' | tr -d "'" | cut -d' ' -f3)
    export PYTHON_VERSION=$(grep -m 1 requires-python pyproject.toml | tr -s ' ' | tr -d '">=' | tr -d "'" | cut -d' ' -f3)
elif [ -z "$PROJECT_NAME" ]; then
    export DEFAULT_PYTHON_VERSION="3.12.7"
    read -p "Firstname: " FIRST_NAME
    export FIRST_NAME=$FIRST_NAME
    read -p "Surname: " SURNAME
    export SURNAME=$SURNAME
    read -p "email@framatome.com: " EMAIL
    export EMAIL=$EMAIL
    read -p "Project name: " PROJECT_NAME
    export PROJECT_NAME=$PROJECT_NAME
    read -p "Version: " VERSION
    export VERSION=$VERSION
    read -p "Python version (default to $DEFAULT_PYTHON_VERSION): " PYTHON_VERSION
    PYTHON_VERSION=${PYTHON_VERSION:-$DEFAULT_PYTHON_VERSION}
    export PYTHON_VERSION=$PYTHON_VERSION
fi
export SMALL_PYTHON_VERSION=$(echo $PYTHON_VERSION | sed -e 's/\.//g' | head -c 3)
export CONDA_VENV_NAME="py$PYTHON_VERSION"

## Logs
export LOGS_FILE_EXTENSTION="log"
export LOGS_PARENT_FOLDER="logs"
export LOGS_FOLDER="$LOGS_PARENT_FOLDER/$PROJECT_NAME"
export LOGS_FILE="$LOGS_FOLDER/$PROJECT_NAME.$LOGS_FILE_EXTENSTION"

export LOGS_CONFIG_FOLDER="logs"
export LOGS_CONFIG_TEMPLATE="$TEMPLATES_FOLDER/.logs/.config.toml.template"
export LOGS_CONFIG_FILE="$LOGS_CONFIG_FOLDER/$PROJECT_NAME.config.toml"

## Dotenv
export DOTENV_FILE_TEMPLATE="$TEMPLATES_FOLDER/.secrets/.env.template"
export DOTENV_FILE=".env"

## Readme
export README_FILE_TEMPLATE="$TEMPLATES_FOLDER/.readme/.README.md.template"
export README_FILE="README.md"

## Pyproject
export PYPROJECT_FILE_TEMPLATE="$TEMPLATES_FOLDER/.pyproject/.pyproject.toml.template"
export PYPROJECT_FILE="pyproject.toml"

export PYPROJECT_SETUP_FILE_TEMPLATE="$TEMPLATES_FOLDER/.pyproject/.setup.cfg.template"
export PYPROJECT_SETUP_FILE="setup.cfg"

## Build
export BUILD_DIR="dist"
export BUILD_LIB_DIR="$BUILD_DIR/lib"
export BUILD_WHEEL_DIR="$BUILD_DIR/wheel"

## Src
export SRC_DIR="src"
export SRC_DIR_MODULE="$SRC_DIR"
export SRC_DIR_PKG="$SRC_DIR/$PROJECT_NAME"

## App
export APP_FILE_TEMPLATE="$TEMPLATES_FOLDER/.src/.app.py.template"
export APP_FILE="$SRC_DIR/app.py"
export APP_FILE_MODULE="$SRC_DIR_MODULE.app"

## 3rdparty packages
export PACKAGES_DIR="packages"

# License
export LICENSE_FILE="LICENSE"
if [ -f $LICENSE_FILE ]; then 
    echo "License file exists, not asking for its level..." > /dev/null
elif [ -z "$LICENSE_LEVEL" ]; then
    export DEFAULT_LICENSE_LEVEL="c1"
    read -p "License level (c1 or c2, default to $DEFAULT_LICENSE_LEVEL): " LICENSE_LEVEL
fi
export LICENSE_LEVEL=${LICENSE_LEVEL:-$DEFAULT_LICENSE_LEVEL}
export LICENSE_FILE_TEMPLATE="$TEMPLATES_FOLDER/.license/.$LICENSE_LEVEL.license.template"

## Gitignore
export GITIGNORE_FILE_TEMPLATE="$TEMPLATES_FOLDER/.gitignore/.gitignore"
export GITIGNORE_FILE=".gitignore"

## Python version
export PYTHON_VERSION_FILE_TEMPLATE="$TEMPLATES_FOLDER/.python/.python-version.template"
export PYTHON_VERSION_FILE=".python-version"


## Quality
export FLAKE8_FILE_TEMPLATE="$TEMPLATES_FOLDER/.quality/.flake8.template"
export FLAKE8_FILE=".flake8"

export PRECOMMIT_FILE_TEMPLATE="$TEMPLATES_FOLDER/.quality/.pre-commit-config.yaml.template"
export PRECOMMIT_FILE=".pre-commit-config.yaml"

export PYLINT_FILE_TEMPLATE="$TEMPLATES_FOLDER/.quality/.pylintrc.template"
export PYLINT_FILE=".pylintrc"

# Generic
export DO_USE_AZURE_OPENAI=1
