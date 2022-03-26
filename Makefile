black:
	black --experimental-string-processing src/ scripts/

validate_docs:
	pydocstyle --convention=numpy src/ --add-ignore=D105