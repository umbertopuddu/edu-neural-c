# Compiler and flags
CC = gcc
CFLAGS = -Wall -O2 -std=c99
LDFLAGS = -lm

# Test names
TESTS = parab_test xor_test abs_test complex_root_test

# Target executables
TARGETS = $(addprefix nn_,$(TESTS))

# Default target: build all tests
all: $(TARGETS)

# Rule to build each test
nn_%: nn.c %.c nn.h
	$(CC) $(CFLAGS) nn.c $*.c -o $@ $(LDFLAGS)

# Test target: build and run each test
test: $(TARGETS)
	@for target in $(TARGETS); do \
		echo "Running $$target..."; \
		./$$target; \
	done

# Clean rule
clean:
	rm -f $(TARGETS) *.o *.params *.bin
