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
        mt_assert_true(t, mt_is_tensor_eq(expected1, result1), "test simple addition 1", "addition result should be 2, 4, 6, 7");
        mt_assert_true(t, mt_is_tensor_eq(expected2, result2), "test simple addition 2", "addition result should be x original values: 1, 2, 3, 4");

        MTTensor *sc1       = mt_new_scalar(ctx, 2);
        MTTensor *sc2       = mt_new_scalar(ctx, 3);
        MTTensor *result3   = mt_tensor_add(x, sc1);
        MTTensor *expected3 = mt_new_tensor(ctx, Arr(float, 3, 4, 5, 6), Arr(int, 2, 2), 2);

        mt_assert_true(t, mt_is_tensor_eq(result3, expected3), "test tensor-scalar addition", "tensor-scalar addition result is wrog");
        mt_assert_true(t, mt_is_tensor_eq(mt_tensor_add(sc1, sc2), mt_new_scalar(ctx, 5)), "test scalar-scalar addition", "scalar-scalar addition result is wrog");

        MTTensor *big   = mt_new_tensor(ctx, Arr(float, 1, 2, 3, 4), Arr(int, 2, 2), 2);
        MTTensor *small = mt_new_tensor(ctx, Arr(float, 2, 2), Arr(int, 2), 1);
        mt_assert_true(t,
                       mt_is_tensor_eq(
                           mt_tensor_add(big, small),
                           mt_new_tensor(ctx, Arr(float, 3, 4, 5, 6), Arr(int, 2, 2), 2)),
                       "test tensor-tensor addition with broadcasting (1)",
                       "should be {3, 4, 5, 6}");

        /* Now the small is also 2 dimension, but size of 1 in the leading dim */
        small = mt_new_tensor(ctx, Arr(float, 1, 2), Arr(int, 2), 1);
        mt_assert_true(t,
                       mt_is_tensor_eq(
                           mt_tensor_add(big, small),
                           mt_new_tensor(ctx, Arr(float, 2, 4, 4, 6), Arr(int, 2, 2), 2)),
                       "test tensor-tensor addition with broadcasting (1)",
                       "should be {3, 4, 5, 6}");
        mt_context_free(ctx);
}

void run_tensor_sum_tests(Test *t) {
        MTContext *ctx  = mt_new_context();
        MTTensor  *x    = mt_new_tensor(ctx, Arr(float, 1, 2, 3, 4, 5, 6), Arr(int, 3, 2), 2);
        MTTensor  *res1 = mt_tensor_sum(x, 0, 1);
        MTTensor  *res2 = mt_tensor_sum(x, 0, 0);
        MTTensor  *res3 = mt_tensor_sum(x, -1, 1);
        MTTensor  *res4 = mt_tensor_sum(x, -1, 0);

        mt_assert_true(t, mt_is_tensor_eq(res1, mt_new_tensor(ctx, Arr(float, 9, 12), Arr(int, 1, 2), 2)), "test sum dim 0 keep dim", "must be {{1, 2}}");
        mt_assert_true(t, mt_is_tensor_eq(res2, mt_new_tensor(ctx, Arr(float, 9, 12), Arr(int, 2), 1)), "test sum dim 0 no keep dim", "must be {9, 12}");
        mt_assert_true(t, mt_is_tensor_eq(res3, mt_new_tensor(ctx, Arr(float, 21), Arr(int, 1, 1), 2)), "test sum all dims keep dim", "must be {{21}}");
        mt_assert_true(t, mt_is_tensor_eq(res4, mt_new_scalar(ctx, 21)), "test sum all dims no keep dim", "must be 21");

        // sum consecutively at dims 0 and 1, retaining the dimension
        MTTensor *res5 = mt_tensor_sum(x, 0, 1);
        res5           = mt_tensor_sum(res5, 1, 1);
        mt_assert_true(t, mt_is_tensor_eq(res5, mt_new_tensor(ctx, Arr(float, 21), Arr(int, 1, 1), 2)), "test sum dim 0 then 1 keep dim", "must be {{ 21 }}");

        // sum consecutively at dim 0 twice, withohut retaining the dimension
        MTTensor *res6 = mt_tensor_sum(x, 0, 0);
        res6           = mt_tensor_sum(res6, 0, 0);
        mt_assert_true(t, mt_is_tensor_eq(res6, mt_new_scalar(ctx, 21)), "test sum dim 0 twice, without keeping dim", "must be {{ 21 }}");

        mt_context_free(ctx);
}

void run_tensor_subtraction_tests(Test *t) {
        MTContext *ctx       = mt_new_context();
        MTTensor  *x         = mt_new_tensor(ctx, Arr(float, 1, 2, 3, 4), Arr(int, 2, 2), 2);
        MTTensor  *y         = mt_new_tensor(ctx, Arr(float, 1, 2, 3, 4), Arr(int, 2, 2), 2);
        MTTensor  *z         = mt_new_tensor(ctx, Arr(float, 0, 0, 0, 0), Arr(int, 2, 2), 2);
        MTTensor  *expected1 = mt_new_tensor(ctx, Arr(float, 0, 0, 0, 0), Arr(int, 2, 2), 2);
        MTTensor  *expected2 = mt_new_tensor(ctx, Arr(float, 1, 2, 3, 4), Arr(int, 2, 2), 2);

        MTTensor *result1 = mt_tensor_sub(x, y);
        MTTensor *result2 = mt_tensor_sub(x, z);
        mt_assert_true(t, mt_is_tensor_eq(expected1, result1), "simple subtraction 1 is correct", "subtraction result should be all-zeros");
        mt_assert_true(t, mt_is_tensor_eq(expected2, result2), "simple subtraction 2 is correct", "subtraction result should be x original values: 1, 2, 3, 4");

        MTTensor *big   = mt_new_tensor(ctx, Arr(float, 1, 2, 3, 4), Arr(int, 2, 2), 2);
        MTTensor *small = mt_new_tensor(ctx, Arr(float, 2, 2, 2, 2), Arr(int, 2), 1);
        mt_assert_true(t,
                       mt_is_tensor_eq(
                           mt_tensor_sub(big, small),
                           mt_new_tensor(ctx, Arr(float, -1, 0, 1, 2), Arr(int, 2, 2), 2)),
                       "test tensor-tensor subtraction with broadcasting",
                       "should be {-1, 0, 1, 2}");
        mt_context_free(ctx);
}

void run_tensor_el_multiplication_tests(Test *t) {
        MTContext *ctx = mt_new_context();
        MTTensor  *x   = mt_new_tensor(ctx, Arr(float, 1, 2, 3), Arr(int, 3), 1);
        MTTensor  *y   = mt_new_tensor(ctx, Arr(float, 4, 5, 6), Arr(int, 3), 1);
        mt_tensor_enable_grad(x), mt_tensor_enable_grad(y);

        MTTensor *z = mt_tensor_mul(x, y);
        mt_tensor_backward(
            z,
            mt_new_tensor(ctx, Arr(float, -1, -2, -3), Arr(int, 3), 1));

        mt_assert_true(
            t,
            mt_is_tensor_eq(
                z,
                mt_new_tensor(ctx, Arr(float, 4, 10, 18), Arr(int, 3), 1)),
            "test element-wise mul correct",
            "should be {4, 10, 18}"

        );

        mt_assert_true(
            t,
            mt_is_tensor_eq(
                x->grad,
                mt_new_tensor(ctx, Arr(float, -4, -10, -18), Arr(int, 3), 1)),
            "test element-wise mul x grad correct",
            "should be {-4, -10, -18}"

        );

        mt_assert_true(
            t,
            mt_is_tensor_eq(
                y->grad,
                mt_new_tensor(ctx, Arr(float, -1, -4, -9), Arr(int, 3), 1)),
            "test element-wise mul y grad correct",
            "should be {-1, -4, -9}");

        mt_context_free(ctx);
}

void run_tensor_negation_tests(Test *t) {
        MTContext *ctx = mt_new_context();
        MTTensor  *x   = mt_new_tensor(ctx, Arr(float, 1, 2, 3, 4), Arr(int, 2, 2), 2);
        mt_assert_true(t,
                       mt_is_tensor_eq(
                           mt_tensor_neg(x),
                           mt_new_tensor(ctx, Arr(float, -1, -2, -3, -4), Arr(int, 2, 2), 2)),
                       "test negation",
                       "should be {-1,-2,-3,-4}");
        mt_context_free(ctx);
}

void run_tensor_transpose_tests(Test *t) {
        MTContext *ctx = mt_new_context();
        MTTensor  *x   = mt_new_tensor(ctx, Arr(float, 1, 2, 3, 4, 5, 6), Arr(int, 3, 2), 2);
        MTTensor  *x_t = mt_new_tensor(ctx, Arr(float, 1, 3, 5, 2, 4, 6), Arr(int, 2, 3), 2);
        mt_assert_true(t,
                       mt_is_tensor_eq(mt_tensor_transpose(x), x_t),
                       "test tensor transpose",
                       "should be {{1, 3, 5}, {2, 4, 6}}");

        mt_context_free(ctx);
}

float add(float x, float y) { return x + y; }
void  run_tensor_reduce_tests(Test *t) {
         MTContext *ctx     = mt_new_context();
         MTTensor  *x       = mt_new_tensor(ctx, Arr(float, 1, 2, 3, 4, 5, 6), Arr(int, 3, 2), 2);
         MTTensor  *xslice1 = mt_tensor_reduce(x, 0, add, 1);
         MTTensor  *xslice2 = mt_tensor_reduce(x, 1, add, 0);
         MTTensor  *exp1    = mt_new_tensor(ctx, Arr(float, 9, 12), Arr(int, 1, 2), 2);
         MTTensor  *exp2    = mt_new_tensor(ctx, Arr(float, 3, 7, 11), Arr(int, 3), 1);

         mt_assert_true(t, mt_is_tensor_eq(xslice1, exp1), "test tensor reduce along axis 0", "the result should be {{9, 12}}");
         mt_assert_true(t, mt_is_tensor_eq(xslice2, exp2), "test tensor reduce along axis 1 with squeeze", "the result should be {3, 7, 11}");

         mt_context_free(ctx);
}

void run_tensor_matrix_multiplication_tests(Test *t) {
        MTContext *ctx = mt_new_context();
        MTTensor  *x   = mt_new_tensor(ctx, Arr(float, 1, 2, 3, 4, 5, 6), Arr(int, 3, 2), 2);
        MTTensor  *y   = mt_new_tensor(ctx, Arr(float, 1, 1, 1, 1), Arr(int, 2, 2), 2);
        MTTensor  *res = mt_new_tensor(ctx, Arr(float, 3, 3, 7, 7, 11, 11), Arr(int, 3, 2), 2);
        mt_assert_true(
            t,
            mt_is_tensor_eq(mt_tensor_matmul(x, y), res),
            "test matmul 1",
            "should be {{3, 3}, {7, 7}, {11, 11}}");
        mt_context_free(ctx);
}