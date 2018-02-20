#!/bin/bash
if [ $TRAVIS_OS_NAME == 'osx' ]; then
  export PATH="$HOME/miniconda/bin:$PATH"
  source $HOME/miniconda/bin/activate
fi

# Install requirements
pip install -r requirements.txt

# old setuptools also has a bug for extras, but it still compiles
pip install -v '.'
if [ $? != 0 ]; then
    exit 1
fi
