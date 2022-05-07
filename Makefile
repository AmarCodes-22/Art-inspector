all: black validate_docs mypy isort

black:
	black --experimental-string-processing src/ scripts/ deploy/

validate_docs:
	-pydocstyle --convention=numpy src/ --add-ignore=D105

mypy:
	-mypy src/ scripts/

isort:
	isort src/ scripts/ deploy/
