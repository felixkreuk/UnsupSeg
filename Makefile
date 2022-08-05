.PHONY: lint black test

COMMIT=$(shell git log --pretty=format:'%h' -n 1)

lint:
	flake8 src

black:
	black --check src
