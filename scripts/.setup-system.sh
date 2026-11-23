#!/bin/bash
## Use variables defined in before scripts to setup project based on templates

# Retrive firstname, surname and emails
read -p "Firstname: " FIRST_NAME
read -p "Surname: " SURNAME
read -p "email@framatome.com: " EMAIL

# Generic
export SCRIPTS_FOLDER="scripts"
export CERTS_FOLDER="$SCRIPTS_FOLDER/.certs"
export SYSTEM_CONFIG_FOLDER="$SCRIPTS_FOLDER/.systemconfig"

## Certs
export SSL_CERT_FILE_CA_BUNDLE_CRT_TEMPLATE="$CERTS_FOLDER/.ca-bundle.crt"
export SSL_CERT_FILE_CA_BUNDLE_CRT="$HOME/ca-bundle.crt"

export CERT_FILE_CA_BUNDLE_PEM_TEMPLATE="$CERTS_FOLDER/.ca-bundle.pem"
export CERT_FILE_CA_BUNDLE_PEM="$HOME/ca-bundle.pem"

## System
export BASH_PROFILE_FILE_TEMPLATE="$SYSTEM_CONFIG_FOLDER/.bash_profile.template"
export BASH_PROFILE_FILE="$HOME/.bash_profile"

export BASHRC_FILE_TEMPLATE="$SYSTEM_CONFIG_FOLDER/.bashrc.template"
export BASHRC_FILE="$HOME/.bashrc"

export CONDARC_FILE_TEMPLATE="$SYSTEM_CONFIG_FOLDER/.condarc.template"
export CONDARC_FILE="$HOME/.condarc"

export GITCONFIG_FILE_TEMPLATE="$SYSTEM_CONFIG_FOLDER/.gitconfig.template"
export GITCONFIG_FILE="$HOME/.gitconfig"

export PIPINI_FILE_TEMPLATE="$SYSTEM_CONFIG_FOLDER/.pip.ini.template"
export PIPINI_FILE="$HOME/AppData/Roaming/pip/pip.ini"

# Back-up an existing file
do_backup_file(){
    file=$1
    timestamp=$(date +"%T")
    cp $file $file.bakup.$timestamp
}

# Replacing existing certificates
maybe_replace_file_with_backup(){
    template=$1
    file=$2
    read -p "Replace $file (default to no) ? (y/n): " DO_REPLACE_FILE
    DO_REPLACE_FILE=${DO_REPLACE_FILE:-n}

    if [ $DO_REPLACE_FILE == "y" ]; then
        if [ -f $file ]; then
            echo "Do backup $file..."
            do_backup_file $file
        fi
        echo "Replacing $file with $template..."
        cp $template $file
    fi
    if [ $DO_REPLACE_FILE == "n" ]; then
        echo "Not replacing $file with $template..."
    fi
}       

maybe_replace_file_with_backup $SSL_CERT_FILE_CA_BUNDLE_CRT_TEMPLATE $SSL_CERT_FILE_CA_BUNDLE_CRT
maybe_replace_file_with_backup $CERT_FILE_CA_BUNDLE_PEM_TEMPLATE $CERT_FILE_CA_BUNDLE_PEM
maybe_replace_file_with_backup $BASH_PROFILE_FILE_TEMPLATE $BASH_PROFILE_FILE
maybe_replace_file_with_backup $BASHRC_FILE_TEMPLATE $BASHRC_FILE
sed -i "s|__SSL_CERT_FILE_CA_BUNDLE_CRT__|$SSL_CERT_FILE_CA_BUNDLE_CRT|g" $BASHRC_FILE
read -p "Keep UV_CACHE_DIR in .bashrc ? Current is $(cat $BASHRC_FILE | grep UV_CACHE_DIR). (y/n): " KEEP_UV_CACHE_DIR
KEEP_UV_CACHE_DIR=${KEEP_UV_CACHE_DIR:-n}
if [ $KEEP_UV_CACHE_DIR == "n" ]; then
    sed -i "/\b\(UV_CACHE_DIR\)\b/d" $BASHRC_FILE
fi
read -p "Keep UV_PYTHON_PREFERENCE in .bashrc ? Current is $(cat $BASHRC_FILE | grep UV_PYTHON_PREFERENCE). (y/n): " KEEP_UV_PYTHON_REFERENCE
KEEP_UV_PYTHON_REFERENCE=${KEEP_UV_PYTHON_REFERENCE:-n}
if [ $KEEP_UV_PYTHON_REFERENCE == "n" ]; then
    sed -i "/\b\(UV_PYTHON_PREFERENCE\)\b/d" $BASHRC_FILE
fi


maybe_replace_file_with_backup $CONDARC_FILE_TEMPLATE $CONDARC_FILE

maybe_replace_file_with_backup $GITCONFIG_FILE_TEMPLATE $GITCONFIG_FILE

sed -i "s|__SSL_CERT_FILE_CA_BUNDLE_CRT__|$SSL_CERT_FILE_CA_BUNDLE_CRT|g" $GITCONFIG_FILE
sed -i "s|__FIRST_NAME__|$FIRST_NAME|g" $GITCONFIG_FILE
sed -i "s|__SURNAME__|$SURNAME|g" $GITCONFIG_FILE
sed -i "s|__EMAIL__|$EMAIL|g" $GITCONFIG_FILE

maybe_replace_file_with_backup $PIPINI_FILE_TEMPLATE $PIPINI_FILE
sed -i "s|__CERT_FILE_CA_BUNDLE_PEM__|$CERT_FILE_CA_BUNDLE_PEM|g" $PIPINI_FILE

# Change windows/linux unique syntax
if [ "$(uname)" == "Darwin" ]; then
    echo "Mac OS X platform..."
    # Do something under Mac OS X platform
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    echo "Linux platform..."
    # Do something under GNU/Linux platform
elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW32_NT" ]; then
    echo "WIN32 platform..."  
    # Do something under 32 bits Windows NT platform
elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW64_NT" ]; then
    echo "WIN64 platform..."  
    sed -i "s|/c/|C:/|g" $GITCONFIG_FILE $PIPINI_FILE
    # Do something under 64 bits Windows NT platform
fi