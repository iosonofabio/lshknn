#!/bin/bash
case $1 in
 testpypi) 
  export PYPI=https://test.pypi.org/legacy/
  export PYPY_IDX=https://test.pypi.org/simple/
  ;;
 pypi) 
  export PYPI=https://upload.pypi.org/legacy/
  export PYPY_IDX=https://pypi.python.org/simple/
  export TWINE_PASSWORD=$TWINE_PASSWORD_PYPI
  ;;
 *)
  echo "Deploy script arg not understood: $1"
  exit 1
  ;;
esac
 
echo "Deploy onto: $PYPI"

pip install twine
python setup.py sdist
distfile=$(ls dist/*.tar.gz)
echo "Twine username: $TWINE_USERNAME"
echo "Twine repository url: $PYPI"
echo "Twine distfile: $distfile"
twine upload --username "${TWINE_USERNAME}" --password "${TWINE_PASSWORD}" --repository-url "${PYPI}" --skip-existing "${distfile}"
