.PHONY: all build test benchmark plots report presentation package clean

PYTHON = python3
TYPST = typst

all: build test benchmark plots report presentation package

build:
	$(MAKE) -C src/c
	$(MAKE) -C src/java

test: build
	$(PYTHON) tests/run_correctness_tests.py

benchmark: build
	$(PYTHON) benchmarks/run_benchmarks.py

plots:
	$(PYTHON) analysis/generate_plots_and_tables.py

report: plots
	$(TYPST) compile report/report.typ ./Assignment01_Martinez_Report.pdf

presentation: plots
	mkdir -p presentation/figures presentation/slides
	cp -f report/figures/*.png presentation/figures/ 2>/dev/null || true
	$(TYPST) compile presentation/presentation.typ presentation/Assignment01_Presentation.pdf
	$(TYPST) compile --format png --ppi 144 presentation/presentation.typ presentation/slides/slide_{p}.png

package: report
	rm -f Assignment01_Martinez_Materials.zip
	zip -r Assignment01_Martinez_Materials.zip \
		README.md requirements.txt Makefile Assignment01_Martinez_Report.pdf \
		common src tests benchmarks analysis report \
		-x "*/.DS_Store" ".DS_Store" "*__MACOSX*" "*__pycache__*" "*.pyc" "src/java/build/*" "*.o" "src/c/matmul_opt" "src/c/matmul_o0"

clean:
	$(MAKE) -C src/c clean
	$(MAKE) -C src/java clean
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name ".DS_Store" -delete 2>/dev/null || true
	rm -f report/Assignment01_Martinez_Report.pdf Assignment01_Martinez_Materials.zip

distclean: clean
	rm -rf tests/correctness_summary.json
	rm -rf benchmarks/results/*.csv
	rm -rf report/figures/*.png report/figures/*.pdf
	rm -rf report/tables_autogen.typ
	rm -f Assignment01_Martinez_Report.pdf

