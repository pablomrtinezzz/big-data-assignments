CC ?= clang
SDKROOT ?= $(shell if [ -d "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk" ]; then echo "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"; else xcrun --show-sdk-path 2>/dev/null; fi)
ifneq ($(SDKROOT),)
  SYSROOT_FLAG = -isysroot $(SDKROOT)
endif
CFLAGS = -O3 -Wall -Wextra -std=c99 -fPIC $(SYSROOT_FLAG)

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  LIB = libblockmm.dylib
  SHARED = -dynamiclib
else
  LIB = libblockmm.so
  SHARED = -shared
endif

all: $(LIB)

$(LIB): blockmm.c
	$(CC) $(CFLAGS) $(SHARED) blockmm.c -o $(LIB)

clean:
	rm -f libblockmm.dylib libblockmm.so
