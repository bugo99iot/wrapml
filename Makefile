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
	. ml_venv/bin/activate && pip freeze > requirements.txt

.PHONY: test
test: ## Run tests
	make clear-pytest-cache
	. ml_venv/bin/activate && cd core && python -m pytest -s -vv
	make clear-pytest-cache

.PHONY: clear-pytest-cache
clear-pytest-cache: ## Clear pytest cache
	rm -r .pytest_cache || true
