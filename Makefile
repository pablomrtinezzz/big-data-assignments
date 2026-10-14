.PHONY: all build test benchmark plots report package clean

PYTHON = python3
TYPST = typst

all: build test benchmark plots report package

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
	$(TYPST) compile report/report.typ report/Assignment01_Martinez_Report.pdf
	cp report/Assignment01_Martinez_Report.pdf ./Assignment01_Martinez_Report.pdf

package: report
	rm -f Assignment01_Martinez_Materials.zip
	zip -r Assignment01_Martinez_Materials.zip \
		README.md requirements.txt Makefile BD_Individual_Assignments_2026_2027.pdf Assignment01_Martinez_Report.pdf \
		common src tests benchmarks analysis report \
		-x "*/.DS_Store" ".DS_Store" "*__MACOSX*" "*__pycache__*" "*.pyc" "src/java/build/*" "*.o" "src/c/matmul_opt" "src/c/matmul_o0"


clean:
	$(MAKE) -C src/c clean
	$(MAKE) -C src/java clean
	rm -rf tests/correctness_summary.json
	rm -rf benchmarks/results/*.csv
	rm -rf report/figures/*.png report/figures/*.pdf
	rm -rf report/tables_autogen.typ
	rm -rf report/Assignment01_Martinez_Report.pdf Assignment01_Martinez_Report.pdf
	rm -rf Assignment01_Martinez_Materials.zip
