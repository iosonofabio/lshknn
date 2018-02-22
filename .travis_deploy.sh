#!/bin/bash
echo "Deploy onto: $PYPI"

pip install twine
python setup.py sdist
distfile=$(ls dist/*.tar.gz)
twine upload --repository testpypi --repository-url $PYPI --skip-existing $distfile

echo "Test install from pip"
sudo pip uninstall lshknn
sudo pip install --index-url $PYPY_IDX lshknn
