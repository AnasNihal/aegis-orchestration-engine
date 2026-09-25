#!/bin/sh
set -eu
git config --local user.name "AnasNihal"
git config --local user.email "108085694+AnasNihal@users.noreply.github.com"
git config --local core.hooksPath .githooks
echo "Aegis Git identity guard installed."
python3 scripts/check_git_identity.py
