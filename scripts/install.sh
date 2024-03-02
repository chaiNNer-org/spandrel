# dev dependencies
pip install -r requirements-dev.txt

# editable installs
pip install -e src/spandrel -e src/spandrel_nc -e src/spandrel_nc_cl --extra-index-url https://download.pytorch.org/whl/cpu
