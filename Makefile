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
	inotifywait -r -m -e modify,move,create,delete "neurallambda/" "test/" --format '%w%f' --quiet | \
	while read FILE; do \
		if echo "$$FILE" | grep -qE '\.py$$' && ! echo "$$FILE" | grep -qE '\.#'; then \
			echo "Python file changed: $$FILE"; \
			PYTHONPATH=. pytest; \
		fi; \
	done
