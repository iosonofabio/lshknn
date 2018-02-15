#!/bin/bash
if [ $TRAVIS_OS_NAME == 'linux' ]; then
  echo "Installing deps for linux"
  sudo apt-get -qq update
  sudo apt-get install -y cmake pkg-config
  ./.travis_install_eigen3.sh
elif [ $TRAVIS_OS_NAME == 'osx' ]; then
  echo "Installing deps for OSX"
  if [ $PYTHON_VERSION == "2.7" ]; then
    CONDA_VER='2'
  elif [ $PYTHON_VERSION == "3.6" ]; then
    CONDA_VER='3'
  else
    echo "Conda only supports 2.7 and 3.6"
  fi
  curl "https://repo.continuum.io/miniconda/Miniconda${CONDA_VER}-latest-MacOSX-x86_64.sh" -o "miniconda.sh"
  bash "miniconda.sh" -b -p $HOME/miniconda
  echo "$PATH"
  export PATH="$HOME/miniconda/bin:$PATH"
  source $HOME/miniconda/bin/activate
  # Use pip from conda
  conda install -y pip
  pip --version
  echo "FIXME: eigen3 and pkg-config come here!"
else
  echo "OS not recognized: $TRAVIS_OS_NAME"
  exit 1
fi

