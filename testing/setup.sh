#!/usr/bin/env bash

# setup.sh – choose system environment and export SYSTEM_ENVIRONMENT

# Ensure the script is being sourced, not executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Error: please source this script: 'source setup.sh'"
  return 1 2>/dev/null || exit 1
fi

options=("LOCAL" "POLARIS" "AURORA")

echo "Select system environment to set up:"
PS3="Enter choice (1-${#options[@]}): "
select opt in "${options[@]}"; do
  case $opt in
    LOCAL|POLARIS|AURORA)
      export SYSTEM_ENVIRONMENT="$opt"
      echo "SYSTEM_ENVIRONMENT set to '$opt'"
      break
      ;;
    *)
      echo "Invalid choice. Please try again."
      ;;
  esac
done