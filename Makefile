##################################################
#
# Neurallambda
#
##################################################

.PHONY: run
run:
	PYTHONPATH=. python example/t01_sandbox.py

.PHONY: test
test:
	PYTHONPATH=. pytest

.PHONY: watch
watch:
	inotifywait -r -e modify,move,create,delete "neurallambda/" "test/" --format '%w%f' --quiet --timefmt "%H:%M:%S" --include "$$({*.py}|{*_test.py})$$" | \
	while read DIRECTORY FILE; do \
		echo "File $$FILE modified in $$DIRECTORY at $$TIME"; \
		PYTHONPATH=. pytest; \
	done
