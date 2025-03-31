# openai-tests
OpenAI tests

To run tests, you have to run virtual environment for this repository.

## Running venv
Set python version for repo and set venv and activate.

```bash
$ pyenv install 3.12.7
$ pyenv local 3.12.7
$ python -m venv env
$ source env/bin/activate
```

## Install python dependencies in repo directory
```bash
pip install -U setuptools pip
pip install -e .
```

## Run tests

```bash
$ pytest -sv
```
