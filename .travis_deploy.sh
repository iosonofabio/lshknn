#!/bin/bash
echo "Deploy onto: $PYPI"

pip install twine
python setup.py sdist
distfile=$(ls dist/*.tar.gz)
twine upload --repository testpypi --repository-url $PYPI --skip-existing
