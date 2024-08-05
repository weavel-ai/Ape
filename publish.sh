#!/bin/bash

# Exit on first error
set -e

# Clean up old distribution archives
echo "Cleaning up old builds..."
rm -rf dist/
rm -rf build/
rm -rf *.egg-info

# Generate distribution archives
echo "Building distribution..."
python setup.py sdist bdist_wheel

# Upload using twine
echo "Uploading to PyPI..."
twine upload dist/*

echo "Publishing complete!"

# If you want to automatically open a web browser to check your library on PyPI
# Uncomment the following line and replace 'your_package_name' with the name of your library
# xdg-open "https://pypi.org/project/your_package_name/"

