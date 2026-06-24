.PHONY: validate test env smoke

validate:
	python scripts/validate_repo.py
	python scripts/render_literature.py --check

test:
	python -m pytest -q

env:
	python scripts/verify_environment.py

smoke: validate test env
