#!/bin/sh

set -e

# spandrel
cp README.md libs/spandrel/README.md
cd libs/spandrel
python3 -m build . --wheel
rm README.md
cd ../..

# spandrel_nc
cd libs/spandrel_nc
python3 -m build . --wheel
cd ../..

# spandrel_nc_cl
cd libs/spandrel_nc_cl
python3 -m build . --wheel
cd ../..
