##################################################
#
# Neurallambda
#
##################################################


.PHONY: test
test:
	pytest


# USAGE:
#   make watch
#   make watch test/test_stack.py
.PHONY: watch
watch:
	@TEST_FILE=$(filter-out $@,$(MAKECMDGOALS)); \
	LAST_FILE=""; \
	LAST_TIME=$$(date +%s); \
	inotifywait -r -m -e modify,move,create,delete "src/" "test/" --format '%w%f' --quiet | \
	while read FILE; do \
		CURRENT_TIME=$$(date +%s); \
		if echo "$$FILE" | grep -qE '\.py$$' && ! echo "$$FILE" | grep -qE '\.#'; then \
			if [ $$(($$CURRENT_TIME - $$LAST_TIME)) -lt 2 ]; then \
				echo "Skipping due to debounce: $$FILE"; \
			else \
				echo "Python file changed: $$FILE"; \
				if [ -z "$$TEST_FILE" ]; then \
					pytest; \
				else \
					pytest $$TEST_FILE; \
				fi; \
			fi; \
			LAST_FILE="$$FILE"; \
			LAST_TIME=$$CURRENT_TIME; \
		fi; \
	done
