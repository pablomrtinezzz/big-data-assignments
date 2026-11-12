CC ?= clang
SDKROOT ?= $(shell if [ -d "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk" ]; then echo "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"; elif [ -d "/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk" ]; then echo "/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk"; else xcrun --show-sdk-path 2>/dev/null; fi)
ifneq ($(SDKROOT),)
  SYSROOT_FLAG = -isysroot $(SDKROOT)
else
  SYSROOT_FLAG =
endif

OMP_PREFIX ?= $(shell brew --prefix libomp 2>/dev/null || echo "/opt/homebrew/opt/libomp")
OMP_CFLAGS ?= -Xclang -fopenmp -I$(OMP_PREFIX)/include
OMP_LDFLAGS ?= -L$(OMP_PREFIX)/lib -lomp

CFLAGS = -O3 -Wall -Wextra -std=c99 $(SYSROOT_FLAG) $(OMP_CFLAGS)
LDFLAGS = $(OMP_LDFLAGS)

TARGET = matmul_parallel

all: $(TARGET)

$(TARGET): matmul_parallel.c ../common/prng.h
	$(CC) $(CFLAGS) matmul_parallel.c $(LDFLAGS) -o $(TARGET)

asm: matmul_parallel.c
	$(CC) $(CFLAGS) -S matmul_parallel.c -o matmul_parallel.s

clean:
	rm -f $(TARGET) *.o *.s
