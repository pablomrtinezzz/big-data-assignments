# ==============================================================================
# Makefile for Assignment 3: Shared-Memory Parallelism & ARM NEON Vectorisation
# ==============================================================================

.PHONY: all build test benchmark plots report presentation package clean

PYTHON = python3
TYPST = typst

all: build test report presentation package

build:
	@if [ -d src ] && [ -f src/Makefile ]; then $(MAKE) -C src; fi

test: build
	$(PYTHON) tests/run_all_tests.py

benchmark: build
	@if [ -f benchmarks/run_benchmarks_parallel.py ]; then $(PYTHON) benchmarks/run_benchmarks_parallel.py; fi
	@if [ -f benchmarks/run_benchmarks_vector.py ]; then $(PYTHON) benchmarks/run_benchmarks_vector.py; fi

plots:
	@if [ -f analysis/generate_plots_and_tables.py ]; then $(PYTHON) analysis/generate_plots_and_tables.py; fi

report: plots
	@if [ -f report/report.typ ]; then $(TYPST) compile report/report.typ ./Assignment03_Martinez_Report.pdf; fi

presentation: plots
	@if [ -f presentation/presentation.typ ]; then \
		$(TYPST) compile presentation/presentation.typ presentation/Assignment03_Presentation.pdf; \
		$(TYPST) compile --format png --ppi 200 presentation/presentation.typ presentation/slides/slide_{p}.png; \
	fi
	@if [ -f presentation/guia_conocimiento_presentacion.typ ]; then \
		$(TYPST) compile presentation/guia_conocimiento_presentacion.typ presentation/GUIA_PRESENTACION_ES.pdf; \
	fi

package: report presentation
	rm -f Assignment03_Martinez_Materials.zip
	zip -r Assignment03_Martinez_Materials.zip \
		README.md requirements.txt Makefile Assignment03_Martinez_Report.pdf \
		common src tests benchmarks analysis report presentation \
		-x "*/.DS_Store" ".DS_Store" "*__MACOSX*" "*__pycache__*" "*.pyc" "*.o" "*.out"

clean:
	@if [ -d src ] && [ -f src/Makefile ]; then $(MAKE) -C src clean; fi
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name ".DS_Store" -delete 2>/dev/null || true
	rm -f Assignment03_Martinez_Materials.zip
