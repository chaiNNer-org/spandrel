
## Development

### pip

pip is a Python package manager. You already have pip if you use Python 3.4 and later version which include it by default. Read [this](https://pip.pypa.io/en/stable/installing/#do-i-need-to-install-pip) to know how to check whether pip is installed. Read [this](https://pip.pypa.io/en/stable/installing/#installing-with-get-pip-py) if you need to install it.

### Use VS Code

Visual Studio Code is the most popular code editor today, our example package is configured to work with VS Code.

Install VS Code extension "[Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)".

"Python" VS Code extension will suggest you install pylint. Also, the example package is configured to use pytest with VS Code + Python extensions, so, install pylint and pytest:

```bash
pip install pylint pytest
```

(It's likely you will be prompted to install them, if that's the case, you don't need to type and execute the command)

**vscode.env**'s content is now `PYTHONPATH=/;src/;${PYTHONPATH}` which is good for Windows. If you use Linux or MacOS, you need to change it to `PYTHONPATH=/:src/:${PYTHONPATH}` (replacing `;` with `:`). If the PATH is not properly set, you'll see linting errors in test files and pytest won't be able to run **tests/test\_\*.py** files correctly.

Close and reopen VS Code. You can now click the lab flask icon in the left menu and run all tests there, with pytest. pytest seems better than the standard unittest framework, it supports `unittest` thus you can keep using `import unittest` in your test files.

The example package also has a **.editorconfig** file. You may install VS Code extension "[EditorConfig for VS Code](https://marketplace.visualstudio.com/items?itemName=EditorConfig.EditorConfig)" that uses the file. With current configuration, the EditorConfig tool can automatically use spaces (4 spaces for .py, 2 for others) for indentation, set `UTF-8` encoding, `LF` end of lines, trim trailing whitespaces in non Markdown files, etc.

In VS Code, you can go to File -> Preferences -> Settings, type "Python Formatting Provider" in the search box, and choose one of the three Python code formatting tools (autopep8, black and yapf), you'll be prompted to install it. The shortcuts for formatting of a code file are <kbd>Shift</kbd> + <kbd>Alt</kbd> + <kbd>F</kbd> (Windows); <kbd>Shift</kbd> + <kbd>Option (Alt)</kbd> + <kbd>F</kbd> (MacOS); <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>I</kbd> (Linux).

### Write your package

In **src/examplepy/** (`examplepy` should have been replaced by your module name) folder, rename **module1.py** and write your code in it. Add more module .py files if you need to.

### Write your tests

In **tests/** folder, rename **test_module1.py** (to **test\_\*.py**) and write your unit test code (with [unittest](https://docs.python.org/3/library/unittest.html)) in it. Add more **test\_\*.py** files if you need to.

<details><summary><strong>The testing tool `tox` will be used in the automation with GitHub Actions CI/CD. If you want to use `tox` locally, click to read the "Use tox locally" section</strong></summary>

### Use tox locally

Install tox and run it:

```bash
pip install tox
tox
```

In our configuration, tox runs a check of source distribution using [check-manifest](https://pypi.org/project/check-manifest/) (which requires your repo to be git-initialized (`git init`) and added (`git add .`) at least), setuptools's check, and unit tests using pytest. You don't need to install check-manifest and pytest though, tox will install them in a separate environment.

The automated tests are run against several Python versions, but on your machine, you might be using only one version of Python, if that is Python 3.9, then run:

```bash
tox -e py39
```

</details>

If you add more files to the root directory (**example_pypi_package/**), you'll need to add your file to `check-manifest --ignore` list in **tox.ini**.

<details><summary><strong>Thanks to GitHub Actions' automated process, you don't need to generate distribution files locally. But if you insist, click to read the "Generate distribution files" section</strong></summary>

## Generate distribution files

### Install tools

Install or upgrade `setuptools` and `wheel`:

```bash
python -m pip install --user --upgrade setuptools wheel
```

(If `python3` is the command on your machine, change `python` to `python3` in the above command, or add a line `alias python=python3` to **~/.bashrc** or **~/.bash_aliases** file if you use bash on Linux)

### Generate `dist`

From `example_pypi_package` directory, run the following command, in order to generate production version for source distribution (sdist) in `dist` folder:

```bash
python setup.py sdist bdist_wheel
```

### Install locally

Optionally, you can install dist version of your package locally before uploading to [PyPI](https://pypi.org/) or [TestPyPI](https://test.pypi.org/):

```bash
pip install dist/example_pypi_package-0.1.0.tar.gz
```

(You may need to uninstall existing package first:

```bash
pip uninstall example_pypi_package
```

There may be several installed packages with the same name, so run `pip uninstall` multiple times until it says no more package to remove.)

</details>

## Upload to PyPI

### Register on PyPI and get token

Register an account on [PyPI](https://pypi.org/), go to [Account settings § API tokens](https://pypi.org/manage/account/#api-tokens), "Add API token". The PyPI token only appears once, copy it somewhere. If you missed it, delete the old and add a new token.

(Register a [TestPyPI](https://test.pypi.org/) account if you are uploading to TestPyPI)

### Set secret in GitHub repo

On the page of your newly created or existing GitHub repo, click **Settings** -> **Secrets** -> **New repository secret**, the **Name** should be `PYPI_API_TOKEN` and the **Value** should be your PyPI token (which starts with `pypi-`).

### Push or release

The example package has automated tests and upload (publishing) already set up with GitHub Actions:

- Every time you `git push` or a pull request is submitted on your `master` or `main` branch, the package is automatically tested against the desired Python versions with GitHub Actions.
- Every time a new release (either the initial version or an updated version) is created, the latest version of the package is automatically uploaded to PyPI with GitHub Actions.

### View it on pypi.org

After your package is published on PyPI, go to [https://pypi.org/project/example-pypi-package/](https://pypi.org/project/example-pypi-package/) (`_` becomes `-`). Copy the command on the page, execute it to download and install your package from PyPI. (or test.pypi.org if you use that)

If you want to modify the description / README of your package on pypi.org, you have to publish a new version.

<details><summary><strong>If you publish your package to PyPI manually, click to read</strong></summary>

### Install Twine

Install or upgrade Twine:

```bash
python -m pip install --user --upgrade twine
```

Create a **.pypirc** file in your **$HOME** (**~**) directory, its content should be:

```ini
[pypi]
username = __token__
password = <PyPI token>
```

(Use `[testpypi]` instead of `[pypi]` if you are uploading to [TestPyPI](https://test.pypi.org/))

Replace `<PyPI token>` with your real PyPI token (which starts with `pypi-`).

(if you don't manually create **$HOME/.pypirc**, you will be prompted for a username (which should be `__token__`) and password (which should be your PyPI token) when you run Twine)

### Upload

Run Twine to upload all of the archives under **dist** folder:

```bash
python -m twine upload --repository pypi dist/*
```

(use `testpypi` instead of `pypi` if you are uploading to [TestPyPI](https://test.pypi.org/))

### Update

When you finished developing a newer version of your package, do the following things.

Modify the version number `__version__` in **src\examplepy\_\_init\_\_.py**.

Delete all old versions in **dist**.

Run the following command again to regenerate **dist**:

```bash
python setup.py sdist bdist_wheel
```

Run the following command again to upload **dist**:

```bash
python -m twine upload --repository pypi dist/*
```

(use `testpypi` instead of `pypi` if needed)

</details>

## References

- [Python Packaging Authority (PyPA)'s sample project](https://github.com/pypa/sampleproject)
- [PyPA's Python Packaging User Guide](https://packaging.python.org/tutorials/packaging-projects/)
- [Stackoverflow questions and answers](https://stackoverflow.com/questions/41093648/how-to-test-that-pypi-install-will-work-before-pushing-to-pypi-python)
- [GitHub Actions Guides: Building and testing Python](https://docs.github.com/en/free-pro-team@latest/actions/guides/building-and-testing-python)
