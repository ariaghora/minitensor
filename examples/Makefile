CC = gcc 
CFLAGS = -std=c99 -Wall -g -O3 -Werror -Wstrict-prototypes
SOURCES = ../minitensor.c 
TEST_SOURCE = ./*.c
EXAMPLE_SOURCE = $(wildcard *.c)
EXAMPLE_OUT = $(patsubst %.c, %.out, $(EXAMPLE_SOURCE))

$(EXAMPLE_OUT): $(EXAMPLE_SOURCE)
	$(CC) -o $@ $^ $(SOURCES) $(CFLAGS)

.PHONY: clean
clean:
	rm -rf *.out