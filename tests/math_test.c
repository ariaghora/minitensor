#include <stdio.h>

#include "../minitensor.h"
#include "test.h"

void run_tensor_addition_tests(Test *t) {
        MTContext *ctx       = mt_new_context();
        MTTensor  *x         = mt_new_tensor(ctx, Arr(float, 1, 2, 3, 4), Arr(int, 2, 2), 2);
        MTTensor  *y         = mt_new_tensor(ctx, Arr(float, 1, 2, 3, 4), Arr(int, 2, 2), 2);
        MTTensor  *z         = mt_new_tensor(ctx, Arr(float, 0, 0, 0, 0), Arr(int, 2, 2), 2);
        MTTensor  *expected1 = mt_new_tensor(ctx, Arr(float, 2, 4, 6, 8), Arr(int, 2, 2), 2);
        MTTensor  *expected2 = mt_new_tensor(ctx, Arr(float, 1, 2, 3, 4), Arr(int, 2, 2), 2);

        MTTensor *result1 = mt_tensor_add(x, y);
        MTTensor *result2 = mt_tensor_add(x, z);
        assert_true(t, mt_is_tensor_eq(expected1, result1), "test simple addition 1", "addition result should be 2, 4, 6, 7");
        assert_true(t, mt_is_tensor_eq(expected2, result2), "test simple addition 2", "addition result should be x original values: 1, 2, 3, 4");

        MTTensor *sc1       = mt_new_scalar(ctx, 2);
        MTTensor *sc2       = mt_new_scalar(ctx, 3);
        MTTensor *result3   = mt_tensor_add(x, sc1);
        MTTensor *expected3 = mt_new_tensor(ctx, Arr(float, 3, 4, 5, 6), Arr(int, 2, 2), 2);

        assert_true(t, mt_is_tensor_eq(result3, expected3), "test tensor-scalar addition", "tensor-scalar addition result is wrog");
        assert_true(t, mt_is_tensor_eq(mt_tensor_add(sc1, sc2), mt_new_scalar(ctx, 5)), "test scalar-scalar addition", "scalar-scalar addition result is wrog");

        mt_free(ctx);
}

void run_tensor_subtraction_tests(Test *t) {
        MTContext *ctx       = mt_new_context();
        MTTensor  *x         = mt_new_tensor(ctx, Arr(float, 1, 2, 3, 4), Arr(int, 2, 2), 2);
        MTTensor  *y         = mt_new_tensor(ctx, Arr(float, 1, 2, 3, 4), Arr(int, 2, 2), 2);
        MTTensor  *z         = mt_new_tensor(ctx, Arr(float, 0, 0, 0, 0), Arr(int, 2, 2), 2);
        MTTensor  *expected1 = mt_new_tensor(ctx, Arr(float, 2, 4, 6, 8), Arr(int, 2, 2), 2);
        MTTensor  *expected2 = mt_new_tensor(ctx, Arr(float, 1, 2, 3, 4), Arr(int, 2, 2), 2);

        MTTensor *result1 = mt_tensor_sub(x, y);
        MTTensor *result2 = mt_tensor_sub(x, z);
        assert_true(t, mt_is_tensor_eq(expected1, result1), "simple subtraction 1 is correct", "subtraction result should be all-zeros");
        assert_true(t, mt_is_tensor_eq(expected2, result2), "simple subtraction 2 is correct", "subtraction result should be x original values: 1, 2, 3, 4");

        mt_free(ctx);
}

void run_tensor_reduce_tests(Test *t) {
        MTContext *ctx     = mt_new_context();
        MTTensor  *x       = mt_new_tensor(ctx, Arr(float, 1, 2, 3, 4, 5, 6), Arr(int, 3, 2), 2);
        MTTensor  *xslice1 = mt_tensor_reduce(x, 0, mt_tensor_add);
        MTTensor  *xslice2 = mt_tensor_reduce(x, 1, mt_tensor_add);
        MTTensor  *exp1    = mt_new_tensor(ctx, Arr(float, 9, 12), Arr(int, 1, 2), 2);
        MTTensor  *exp2    = mt_new_tensor(ctx, Arr(float, 3, 7, 11), Arr(int, 3), 1);

        assert_true(t, mt_is_tensor_eq(xslice1, exp1), "test tensor reduce along axis 0", "the result should be {{9, 12}}");
        assert_true(t, mt_is_tensor_eq(xslice2, exp2), "test tensor reduce along axis 1 with squeeze", "the result should be {{3}, {7}, {11}}");

        mt_free(ctx);
}