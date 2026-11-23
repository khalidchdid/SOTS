#!/bin/bash
# Export proxy variables in current context

source scripts/.before-script.sh

source $DOTENV_FILE # access $HTTP_PROXY

export HTTP_PROXY=$HTTP_PROXY
export http_proxy=$http_proxy

export HTTPS_PROXY=$HTTPS_PROXY
export https_proxy=$https_proxy

export FTP_PROXY=$FTP_PROXY
export ftp_proxy=$ftp_proxy

export no_proxy=$no_proxy