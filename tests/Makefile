CC = gcc 
CFLAGS = -std=c99 -Wall -g -O3 -Werror -Wstrict-prototypes -lm
SOURCES = ../minitensor.c 
TEST_SOURCE = ./*.c
VGFLAGS = --track-origins=yes --leak-check=full --show-leak-kinds=all -s

.PHONY: minitensor_test
minitensor_test:
	@$(CC) -o minitensor_test $(TEST_SOURCE) $(SOURCES) $(CFLAGS)

.PHONY: test 
test: minitensor_test
	@make clean && make && ./minitensor_test && rm -f ./minitensor_test

.PHONY: test/dbgmem
test/dbgmem: minitensor_test
	@make clean && make && valgrind $(VGFLAGS) ./minitensor_test && rm -f minitensor_test

clean:
	@rm -f minitensor_test