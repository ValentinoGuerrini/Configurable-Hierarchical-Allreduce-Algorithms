#!/usr/bin/env bash
# setup.sh – choose system environment and export SYSTEM_ENVIRONMENT

# 1) Ensure we're being sourced, not executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Error: please source this script: 'source setup.sh'"
  return 1 2>/dev/null || exit 1
fi

# 2) Define the options (just as text, no arrays)
echo "Select system environment to set up:"
echo "  1) LOCAL"
echo "  2) POLARIS"
echo "  3) AURORA"

# 3) Prompt
printf "Enter choice (1-3): "
read choice

# 4) Map numeric choice to name
case "$choice" in
  1) env="LOCAL"   ;;
  2) env="POLARIS" ;;
  3) env="AURORA"  ;;
  *)
    echo "Invalid choice: '$choice'. Please run 'source setup.sh' again."
    return 1
    ;;
esac

# 5) Export and confirm
export SYSTEM_ENVIRONMENT="$env"
echo "SYSTEM_ENVIRONMENT set to '$env'"