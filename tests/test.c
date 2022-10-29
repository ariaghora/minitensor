#include "test.h"

#include <stdio.h>

void run_tests(void) {
        Test t = new_test();

#ifndef SKIP_BASIC_TESTS
        run_tensor_creation_tests(&t);
        run_tensor_slice_tests(&t);
        run_tensor_access_tests(&t);
        run_context_tests(&t);
#endif

#ifndef SKIP_MATH_TESTS
        run_tensor_addition_tests(&t);
        run_tensor_subtraction_tests(&t);
        run_tensor_sum_tests(&t);
#endif

#ifndef SKIP_AUTOGRAD_TESTS
        run_simple_autograd_tests(&t);
        run_autograd_backward_tests(&t);
        run_tensor_reduce_tests(&t);
        run_autograd_add_tests(&t);
#endif

        printf("========================================================================\n");
        printf("tests completed\n");
        printf("PASS: %d\n", t.npass);
        printf("FAIL: %d\n\n", t.nfail);
}

int main(void) {
        run_tests();
}