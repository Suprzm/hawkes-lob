.PHONY: test lint install

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

lint:
	black src/ tests/