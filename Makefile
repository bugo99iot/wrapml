########################################################################################################################
# UTILS
########################################################################################################################
UNAME := $(shell uname)

ifeq ($(UNAME), Linux)
	B64_FLAG := w
endif

ifeq ($(UNAME), Darwin)
	B64_FLAG := b
endif

.PHONY: help
help:
	@awk 'BEGIN {FS = ":.*?## "} /^[0-9a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

########################################################################################################################
# DEV TOOLS
########################################################################################################################

.PHONY: pip-freeze
pip-freeze: ## Freeze requirements
	. venv/bin/activate && pip freeze > requirements.txt

.PHONY: test
test: ## Run tests
	make clear-pytest-cache
	. venv/bin/activate && cd src && python -m pytest -s -vv
	make clear-pytest-cache

.PHONY: clear-pytest-cache
clear-pytest-cache: ## Clear pytest cache
	find . -path "*/*.pyc"  -delete
	find . -path "*/*.pyo"  -delete
	find . -type d -name  "__pycache__" -exec rm -r {} +
	find . -path "*/__pycache__" -type d -exec rm -r {} ';'

.PHONY: clear-pytest-git-history
clear-pytest-git-history: ## Clear pytest git history
	git rm --cached */__pycache__/* || true

.PHONY: install-xgboost-mac
install-xgboost-mac: ## Install XGBoost on a mac
	brew install gcc
	brew info gcc
	export CC=gcc-8; export CXX=g++-8; pip install xgboost

.PHONY: deploy-to-pypi
deploy-to-pypi: ## Deploy to PyPi
	rm -r dist || true
	rm -r WrapML.egg-info || true
	python setup.py sdist
	twine upload dist/* --verbose
