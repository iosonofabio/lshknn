#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

if [ $TRAVIS_OS_NAME == 'osx' ]; then
  export PATH="$HOME/miniconda/bin:$PATH"
  source $HOME/miniconda/bin/activate
fi

echo "Test lshknn"
python test/test_small.py
python test/test_small2.py
