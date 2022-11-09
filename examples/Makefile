CC = gcc 
CFLAGS = -std=c99 -Wall -g -O3 -Werror -Wstrict-prototypes
SOURCES = ../minitensor.c 
TEST_SOURCE = ./*.c
EXAMPLE_SOURCE = $(wildcard *.c)
EXAMPLE_OUT = $(patsubst %.c, %, $(EXAMPLE_SOURCE))

all: $(EXAMPLE_OUT)

$(EXAMPLE_OUT): %: %.c
	$(CC) -o $@.out $< $(SOURCES) $(CFLAGS)

.PHONY: clean
clean:
	rm -rf *.out