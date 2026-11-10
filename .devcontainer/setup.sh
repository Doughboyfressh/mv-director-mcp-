#!/usr/bin/env bash
# One-time setup when the Codespace is created.
set -e
sudo apt-get update -qq
sudo apt-get install -y -qq ffmpeg fonts-dejavu-core >/dev/null
pip install -q -e .
echo "Setup complete."
