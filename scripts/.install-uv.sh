#!/bin/bash

# Preliminary: define some variables and function
BASE_URL="https://nexus.framatome.corp/service/rest/v1/search?repository=uv"
continuationToken=""
format="^[0-9]+\.[0-9]+\.[0-9]+$"
UV_VERSION="0.5.2"

download_uv_version(){
  uv_version=$1
  curl -LsSf "https://nexus.framatome.corp/repository/uv/$1/uv-installer.sh" | env INSTALLER_DOWNLOAD_URL="https://nexus.framatome.corp/repository/uv/$1" sh
}

# Introduction
echo
echo "You are about to download uv from Framatome Nexus..."
echo
echo "Below are the different options:"
echo
echo -e "Choice\tuv version\t\t\tComments"
echo -e "------\t----------\t\t\t--------"
echo -e "#1\t0.5.2\t\t\t\tKnown stable version for 'scripts'"
echo -e "#2\tUser specific uv version\tExpected format: x.y.z"
echo -e "#3\tLatest version available\t0.7.4 (2025-05-16)"
echo

# Choose an option
while true; do
  read -p "Make your choice: " answer
  case "$answer" in
    [123])
      echo
      break
      ;;
    *)
      echo "Invalid choice: please enter a valid choice."
      echo
      ;;
  esac
done

# User specific uv version
if [[ "$answer" == "2" ]]; then
  while true; do
    read -p "Enter the desired version (format x.y.z): " UV_VERSION
    if [[ $UV_VERSION =~ $format ]]; then
      echo
      break
    else
      echo "Invalid format: please enter a valid version format."
      echo
    fi
  done
fi

# Latest version available
if [[ "$answer" == "3" ]]; then
  while :; do
    # Build the URL
    if [[ -n "$continuationToken" ]]; then
      URL="${BASE_URL}&continuationToken=${continuationToken}"
    else
      URL="${BASE_URL}"
    fi

    # Fetch JSON
    response=$(curl -s -L -X GET "$URL")

    # Extract the token
    rawToken=$(echo "$response" | awk -F'"' '/continuationToken/{print $(NF-1)}')

    # Guard: only keep the result if it's not the keyword itself
    if [[ -n "$rawToken" && "$rawToken" != "continuationToken" ]]; then
      continuationToken="$rawToken"
    else
      continuationToken=""
    fi

    # If empty, we're done
    if [[ -z "$continuationToken" ]]; then
      break
    fi
  done

  # Get all lines containing "group"
  lines=$(printf '%s' "$response" | grep '"group"')

  # Extract version (x.y.z), sort numerically, and keep the highest one
  UV_VERSION=$(printf '%s\n' "$lines" \
    | sed -E 's/.*"\/([0-9]+\.[0-9]+\.[0-9]+)".*/\1/' \
    | sort -V \
    | tail -n1)

  echo "Latest detected version: $UV_VERSION"
fi

# Install uv
download_uv_version $UV_VERSION