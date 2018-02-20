#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

echo "Test lshknn"
python test/test_small.py
python test/test_small2.py
