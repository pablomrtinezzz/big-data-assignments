# ==============================================================================
# Top-Level Orchestration Makefile for Big Data Individual Assignments
# ==============================================================================

.PHONY: all help assignment-1 assignment-2 test clean

help:
	@echo "Big Data Individual Assignments Suite (2026-2027)"
	@echo "Available commands:"
	@echo "  make assignment-1  - Build, test, benchmark, and compile Assignment 1 report"
	@echo "  make assignment-2  - Build, test, benchmark, and compile Assignment 2 report"
	@echo "  make test          - Run correctness test suites for all implemented assignments"
	@echo "  make clean         - Clean temporary build and cache artifacts across all assignments"

assignment-1:
	$(MAKE) -C assignment_1 all

assignment-2:
	$(MAKE) -C assignment_2 all

test:
	$(MAKE) -C assignment_1 test
	@if [ -f assignment_2/Makefile ]; then $(MAKE) -C assignment_2 test; fi

clean:
	$(MAKE) -C assignment_1 clean
	@if [ -f assignment_2/Makefile ]; then $(MAKE) -C assignment_2 clean; fi
	find . -name ".DS_Store" -delete 2>/dev/null || true
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
