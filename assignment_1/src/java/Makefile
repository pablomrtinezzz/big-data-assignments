JAVAC = javac
JAVA = java

all: compile

compile:
	mkdir -p build
	$(JAVAC) -d build ../../common/PRNG.java MatrixMultiplication.java

clean:
	rm -rf build
