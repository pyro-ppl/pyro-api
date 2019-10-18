.PHONY: all install docs lint  test clean FORCE

all: docs test

install:
	pip install -e .[dev,test]

docs: FORCE
	make -C docs html

lint: FORCE
	flake8

test: lint FORCE
	pytest -vx test

clean: FORCE
	git clean -dfx -e pyroapi-egg.info

FORCE:
