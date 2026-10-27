# ==============================================================================
# Makefile for Assignment 2: Dense Optimisations & Sparse Matrices
# ==============================================================================

.PHONY: all build test benchmark plots report package clean

PYTHON = python3
TYPST = typst

all: build test

build:
	@if [ -d src/dense ] && [ -f src/dense/Makefile ]; then $(MAKE) -C src/dense; fi
	@if [ -d src/sparse ] && [ -f src/sparse/Makefile ]; then $(MAKE) -C src/sparse; fi

test: build
	$(PYTHON) tests/run_all_tests.py

benchmark: build
	@if [ -f benchmarks/run_benchmarks_dense.py ]; then $(PYTHON) benchmarks/run_benchmarks_dense.py; fi
	@if [ -f benchmarks/run_benchmarks_sparse.py ]; then $(PYTHON) benchmarks/run_benchmarks_sparse.py; fi

plots:
	@if [ -f analysis/generate_plots_and_tables.py ]; then $(PYTHON) analysis/generate_plots_and_tables.py; fi

report: plots
	@if [ -f report/report.typ ]; then $(TYPST) compile report/report.typ ./Assignment02_Martinez_Report.pdf; fi

package: report
	rm -f Assignment02_Martinez_Materials.zip
	zip -r Assignment02_Martinez_Materials.zip \
		README.md requirements.txt Makefile Assignment02_Martinez_Report.pdf \
		common src tests benchmarks analysis report \
		-x "*/.DS_Store" ".DS_Store" "*__MACOSX*" "*__pycache__*" "*.pyc" "*.o" "*.out"

clean:
	@if [ -d src/dense ] && [ -f src/dense/Makefile ]; then $(MAKE) -C src/dense clean; fi
	@if [ -d src/sparse ] && [ -f src/sparse/Makefile ]; then $(MAKE) -C src/sparse clean; fi
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name ".DS_Store" -delete 2>/dev/null || true
	rm -f Assignment02_Martinez_Materials.zip
