CC = gcc 
CFLAGS = -std=c99 -Wall -g -O3 -Werror -Wstrict-prototypes
SOURCES = ../minitensor.c 
TEST_SOURCE = ./*.c
EXAMPLE_SOURCE = minimize_quadratic_func.c
EXAMPLE_OUT = minimize_quadratic_func.out

$(EXAMPLE_OUT): $(EXAMPLE_SOURCE)
	$(CC) -o $@ $^ $(SOURCES) $(CFLAGS)

.PHONY: clean
clean:
	rm -rf *.out