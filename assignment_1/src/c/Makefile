CC ?= clang
SDKROOT ?= $(shell if [ -d "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk" ]; then echo "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"; elif [ -d "/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk" ]; then echo "/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk"; else xcrun --show-sdk-path 2>/dev/null; fi)
ifneq ($(SDKROOT),)
  SYSROOT_FLAG = -isysroot $(SDKROOT)
else
  SYSROOT_FLAG =
endif
CFLAGS_COMMON = -Wall -Wextra -std=c99 $(SYSROOT_FLAG)

all: matmul_opt matmul_o0

matmul_opt: matmul.c ../../common/prng.h
	$(CC) $(CFLAGS_COMMON) -O2 matmul.c -o matmul_opt

matmul_o0: matmul.c ../../common/prng.h
	$(CC) $(CFLAGS_COMMON) -O0 matmul.c -o matmul_o0

clean:
	rm -f matmul_opt matmul_o0
