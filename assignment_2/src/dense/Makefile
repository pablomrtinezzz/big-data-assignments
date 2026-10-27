CC ?= clang
SDKROOT ?= $(shell if [ -d "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk" ]; then echo "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"; elif [ -d "/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk" ]; then echo "/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk"; else xcrun --show-sdk-path 2>/dev/null; fi)
ifneq ($(SDKROOT),)
  SYSROOT_FLAG = -isysroot $(SDKROOT)
else
  SYSROOT_FLAG =
endif

CFLAGS = -O2 -Wall -Wextra -std=c99 $(SYSROOT_FLAG)
TARGET = matmul_dense

all: $(TARGET)

$(TARGET): matmul_dense.c ../../common/prng.h
	$(CC) $(CFLAGS) -o $(TARGET) matmul_dense.c

clean:
	rm -f $(TARGET) *.o
