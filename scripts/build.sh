#!/bin/sh

set -e

# spandrel
cp README.md libs/spandrel/README.md
cd libs/spandrel
python3 -m build .
rm README.md
cd ../..

# spandrel_extra_arches
cd libs/spandrel_extra_arches
python3 -m build .
cd ../..
