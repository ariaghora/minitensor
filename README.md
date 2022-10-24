## Usage

Include `minitensor.h` in your source files and compile them along with `tensor.c` and `math.c`.

## Running tests

`cd` into `tests` directory and invoke one of the following commands:

- `make test` : compile and run unit tests
- `make test/dbgmem` : compile and run unit tests and also perform memory check with valgrind.
  ensure that valgrind is already installed.