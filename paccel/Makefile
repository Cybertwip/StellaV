# Convenience targets for the bare-minimum PAccel port.
# Go targets need Go 1.21+; cpp targets need a C++20 compiler.

GO ?= go

.PHONY: all go-build go-test cpp-test chat clean

all: go-build cpp-test

go-build:
	$(GO) build ./...

go-test:
	$(GO) test ./libpaccel/

cpp-test:
	g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -Icpp/include \
		cpp/src/accelerator.cpp cpp/test/selftest.cpp -o /tmp/paccel_selftest
	/tmp/paccel_selftest

chat: go-build
	$(GO) run ./paccelchat chat

clean:
	rm -f /tmp/paccel_selftest
	rm -rf cpp/build
