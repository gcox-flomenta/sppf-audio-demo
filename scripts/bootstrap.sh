#!/bin/bash
# bootstrap.sh — Entry point fetched by RunPod pod via curl
# Clones the repo using GH_TOKEN from env, then runs the full training script.
set -e
git clone "https://${GH_TOKEN}@github.com/${GH_REPO}.git" /workspace/repo
cd /workspace/repo
bash scripts/runpod_train.sh
