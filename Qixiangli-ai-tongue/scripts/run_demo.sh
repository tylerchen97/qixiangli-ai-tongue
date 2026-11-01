#!/usr/bin/env bash
set -e
python -m tcm_tongue demo --out outputs
echo "结果见 outputs/summary.json 与 outputs/figures/"
