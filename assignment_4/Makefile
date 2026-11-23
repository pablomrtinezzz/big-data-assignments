# ==============================================================================
# Assignment 4: Distributed Matrix Multiplication (Apache Spark)
# ==============================================================================
.PHONY: all build baselines-bin test env benchmark plots report package clean

PYTHON = python3
TYPST  = typst
SPARK_FILTER = grep -v -E "WARN|FutureWarning|require_minimum|log4j|setLogLevel" || true

all: build test report package

build:
	$(MAKE) -C kernels
	$(MAKE) -C ../assignment_3/src

test: build
	$(PYTHON) tests/test_correctness.py 2>&1 | $(SPARK_FILTER)

env:
	$(PYTHON) benchmarks/environment.py

benchmark: build env
	rm -f benchmarks/results/raw_spark.csv benchmarks/results/sessions.json
	$(PYTHON) benchmarks/run_baselines.py
	$(PYTHON) benchmarks/run_spark.py 2>&1 | $(SPARK_FILTER)

plots:
	$(PYTHON) analysis/make_figures.py

report: plots
	$(TYPST) compile report/report.typ Assignment04_Martinez_Report.pdf

package: report
	rm -f Assignment04_Martinez_Materials.zip
	zip -qr Assignment04_Martinez_Materials.zip README.md requirements.txt Makefile \
		Assignment04_Martinez_Report.pdf kernels src tests benchmarks analysis report \
		-x "*/.DS_Store" "*__pycache__*" "*.pyc" "*.dylib" "*.so"

clean:
	$(MAKE) -C kernels clean
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf spark-warehouse metastore_db
