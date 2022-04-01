all: black validate_docs mypy

black:
	black --experimental-string-processing src/ scripts/

validate_docs:
	-pydocstyle --convention=numpy src/ --add-ignore=D105

mypy:
	-mypy src/ scripts/
