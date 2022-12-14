#include <math.h>
#include <stdio.h>

#include "../minitensor.h"
#include "test.h"

void run_simple_autograd_tests(Test *t) {
        MTContext *ctx = mt_new_context();

        MTTensor *x = mt_new_tensor(ctx, Arr(float, 1, 2, 3, 4), Arr(int, 2, 2), 2);

        mt_tensor_enable_grad(x);
        mt_assert_true(t, x->req_grad, "test enabling grad", "x should require grad");
        mt_assert_true(t, mt_is_tensor_eq(x->grad, mt_new_tensor_full(ctx, 0., Arr(int, 2, 2), 2)), "test initial grad", "intial grad should be all-zero");
        mt_tensor_disable_grad(x);
        mt_assert_true(t, !x->req_grad, "test disabling grad", "x should NOT require grad");
        mt_assert_true(t, x->grad == NULL, "test grad val after disabling grad", "grad val should be NULL");

        MTTensor *y = mt_new_tensor(ctx, Arr(float, 2, 4, 6, 8), Arr(int, 2, 2), 2);
        mt_tensor_enable_grad(y);

        /* test if at least one of operand requires grad, then the the result also requires grad */
        MTTensor *res = mt_tensor_add(x, y);
        mt_assert_true(t, res->req_grad, "test binop with grad", "res should require grad");
        mt_assert_true(t, res->deps[1]->tensor == y, "test binop resulting rchild with grad", "res rchild should be y");

        mt_tensor_disable_grad(y);
        res = mt_tensor_add(x, y);
        mt_assert_true(t, !res->req_grad, "test binop without grad", "res should NOT require grad");

        mt_context_free(ctx);
}

void run_autograd_backward_tests(Test *t) {
        MTContext *ctx = mt_new_context();

        MTTensor *x = mt_new_tensor(ctx, Arr(float, 1, 2, 3), Arr(int, 3, 1), 2);
        MTTensor *y = mt_new_tensor(ctx, Arr(float, 1, 2, 3), Arr(int, 3, 1), 2);
        mt_tensor_enable_grad(x);
        mt_tensor_enable_grad(y);

        MTTensor *expgrad1 = mt_new_tensor(ctx, Arr(float, 1, 1, 1), Arr(int, 3, 1), 2);

        MTTensor *sum = mt_tensor_sum(x, -1, 0);
        mt_assert_true(t, sum->data[0] == 6, "test tensor sum", "sum should be 6");
        mt_tensor_backward(sum, NULL);
        mt_assert_true(t, x->grad != NULL, "test dependent grad non-null after backward", "dependent grad should not be null");
        mt_assert_true(t, mt_is_tensor_eq(x->grad, expgrad1), "test dependent grad value", "grad value should be all 1");
        mt_assert_true(t, sum->deps[0]->grad_fn != NULL, "test grad_fn not null", "grad_fn should not be NULL");

        MTTensor *expgrad2 = mt_new_tensor(ctx, Arr(float, 3, 3, 3), Arr(int, 3, 1), 2);
        sum                = mt_tensor_sum(y, -1, 0);
        mt_tensor_backward(sum, mt_new_scalar(ctx, 3.0));
        mt_assert_true(t, mt_is_tensor_eq(y->grad, expgrad2), "test dependent grad value", "grad value should be all 3");
        mt_context_free(ctx);
}

void run_autograd_add_tests(Test *t) {
        MTContext *ctx = mt_new_context();
        /* test backward when one of operand does not require grad */
        MTTensor *a = mt_new_scalar(ctx, 1);
        MTTensor *b = mt_new_scalar(ctx, 1);
        mt_tensor_enable_grad(a);
        mt_tensor_backward(mt_tensor_add(a, b), NULL);

        MTTensor *x = mt_new_tensor(ctx, Arr(float, 1, 2, 3), Arr(int, 3), 1);
        MTTensor *y = mt_new_tensor(ctx, Arr(float, 4, 5, 6), Arr(int, 3), 1);
        mt_tensor_enable_grad(x);
        mt_tensor_enable_grad(y);

        /* simple add */
        MTTensor *z = mt_tensor_add(x, y);
        mt_assert_true(t, mt_is_tensor_eq(z, mt_new_tensor(ctx, Arr(float, 5, 7, 9), Arr(int, 3), 1)), "test simple add correct", "should be {5, 7, 9}");

        mt_tensor_backward(z, mt_new_tensor(ctx, Arr(float, -1, -2, -3), Arr(int, 3), 1));
        mt_assert_true(t, mt_is_tensor_eq(x->grad, mt_new_tensor(ctx, Arr(float, -1, -2, -3), Arr(int, 3), 1)), "test grad A+B wrt A", "should be {-1,-2,-3}");
        mt_assert_true(t, mt_is_tensor_eq(y->grad, mt_new_tensor(ctx, Arr(float, -1, -2, -3), Arr(int, 3), 1)), "test grad A+B wrt B", "should be {-1,-2,-3}");

        /* add with broadcasting */
        MTTensor *big   = mt_new_tensor(ctx, Arr(float, 1, 2, 3, 4, 5, 6), Arr(int, 2, 3), 2);
        MTTensor *small = mt_new_tensor(ctx, Arr(float, 7, 8, 9), Arr(int, 3), 1);
        mt_tensor_enable_grad(big);
        mt_tensor_enable_grad(small);
        MTTensor *res = mt_tensor_add(big, small);
        mt_tensor_backward(res, mt_new_tensor(ctx, Arr(float, 1, 1, 1, 1, 1, 1), Arr(int, 2, 3), 2));

        mt_assert_true(t,
                       mt_is_tensor_eq(
                           big->grad,
                           mt_new_tensor(ctx, Arr(float, 1, 1, 1, 1, 1, 1), Arr(int, 2, 3), 2)),
                       "test A+B grad of bigger tensor (1)", "should be {{1, 1, 1}, {1, 1, 1}}");
        mt_assert_true(t,
                       mt_is_tensor_eq(
                           small->grad,
                           mt_new_tensor(ctx, Arr(float, 2, 2, 2), Arr(int, 3), 1)),
                       "test A+B grad of smaller tensor (1)", "should be {2, 2, 2}");

        big   = mt_new_tensor(ctx, Arr(float, 1, 2, 3, 4, 5, 6), Arr(int, 2, 3), 2);
        small = mt_new_tensor(ctx, Arr(float, 7, 8, 9), Arr(int, 1, 3), 2);
        mt_tensor_enable_grad(small);
        mt_tensor_enable_grad(big);

        res = mt_tensor_add(big, small);

        mt_assert_true(t,
                       mt_is_tensor_eq(
                           res,
                           mt_new_tensor(ctx, Arr(float, 8, 10, 12, 11, 13, 15), Arr(int, 2, 3), 2)),
                       "test A+B result, broadcasted, with grad ", "should be {{8, 10, 12}, {11, 13, 15}}");

        mt_tensor_backward(res, mt_new_tensor(ctx, Arr(float, 1, 1, 1, 1, 1, 1), Arr(int, 2, 3), 2));
        mt_assert_true(t,
                       mt_is_tensor_eq(
                           big->grad,
                           mt_new_tensor(ctx, Arr(float, 1, 1, 1, 1, 1, 1), Arr(int, 2, 3), 2)),
                       "test A+B grad of bigger tensor (2)", "should be {{1, 1, 1}, {1, 1, 1}}");
        mt_assert_true(t,
                       mt_is_tensor_eq(
                           small->grad,
                           mt_new_tensor(ctx, Arr(float, 2, 2, 2), Arr(int, 1, 3), 2)),
                       "test A+B grad of smaller tensor (2)", "should be {{2, 2, 2}}}");
        mt_context_free(ctx);
}

void run_autograd_add_same_tensors_tests(Test *t) {
        MTContext *ctx = mt_new_context();
        MTTensor  *x   = mt_new_tensor(ctx, Arr(float, 1, 2, 3), Arr(int, 3), 1);
        mt_tensor_enable_grad(x);
        MTTensor *res = mt_tensor_add(x, x);
        mt_tensor_backward(res, mt_new_tensor(ctx, Arr(float, 2, 2, 2), Arr(int, 3), 1));
        mt_assert_true(t, mt_is_tensor_eq(res, mt_new_tensor(ctx, Arr(float, 2, 4, 6), Arr(int, 3), 1)), "test simple addition on same tensors requiring grad", "should be {2, 4, 6}");
        mt_assert_true(t, mt_is_tensor_eq(x->grad, mt_new_tensor(ctx, Arr(float, 4, 4, 4), Arr(int, 3), 1)), "test simple grad of addition on same tensors", "should be {4, 4, 4}");
        res = mt_tensor_add(res, x);
        mt_tensor_backward(res, mt_new_tensor(ctx, Arr(float, 2, 2, 2), Arr(int, 3), 1));
        mt_assert_true(t, mt_is_tensor_eq(x->grad, mt_new_tensor(ctx, Arr(float, 10, 10, 10), Arr(int, 3), 1)), "test simple grad of addition on same tensors once more", "should be {10,10,10}");
        mt_context_free(ctx);
}

void run_autograd_division_tests(Test *t) {
        MTContext *ctx = mt_new_context();

        MTTensor *x = mt_new_tensor(ctx, Arr(float, 2, 4, 6), Arr(int, 3), 1);
        MTTensor *y = mt_new_scalar(ctx, 2);
        mt_tensor_enable_grad(x), mt_tensor_enable_grad(y);

        MTTensor *z    = mt_tensor_div(x, y);
        MTTensor *grad = mt_new_tensor(ctx, Arr(float, 1, 1, 1), Arr(int, 3), 1);
        mt_tensor_backward(z, grad);

        MTTensor *x_grad = mt_new_tensor(ctx, Arr(float, 0.5, 0.5, 0.5), Arr(int, 3), 1);
        MTTensor *y_grad = mt_new_scalar(ctx, -3);

        mt_assert_true(
            t,
            mt_is_tensor_eq(z, mt_new_tensor(ctx, Arr(float, 1, 2, 3), Arr(int, 3), 1)),
            "test div result",
            "it should be {1, 2 ,3}");

        mt_assert_true(
            t,
            mt_is_tensor_eq(x->grad, x_grad),
            "test div grad 1",
            "it should be {.5, .5, .5}");

        mt_assert_true(
            t,
            mt_is_tensor_eq(y->grad, y_grad),
            "test div grad 2",
            "it should be -3");

        mt_context_free(ctx);
}

void run_autograd_matmul_tests(Test *t) {
        MTContext *ctx = mt_new_context();

        MTTensor *x = mt_new_tensor(ctx, Arr(float, 1, 2, 3, 4, 5, 6), Arr(int, 3, 2), 2);
        MTTensor *y = mt_new_tensor(ctx, Arr(float, 10, 20), Arr(int, 2, 1), 2);
        mt_tensor_enable_grad(x), mt_tensor_enable_grad(y);
        MTTensor *z    = mt_tensor_matmul(x, y);
        MTTensor *grad = mt_new_tensor(ctx, Arr(float, -1, -2, -3), Arr(int, 3, 1), 2);
        mt_tensor_backward(z, grad);

        mt_assert_true(
            t,
            mt_is_tensor_eq(x->grad, mt_tensor_matmul(grad, mt_tensor_transpose(y))),
            "test matmul grad 1",
            "-");

        mt_assert_true(
            t,
            mt_is_tensor_eq(y->grad, mt_tensor_matmul(mt_tensor_transpose(x), grad)),
            "test matmul grad 1",
            "-");

        mt_context_free(ctx);
}

void run_autograd_exp_tests(Test *t) {
        MTContext *ctx    = mt_new_context();
        MTTensor  *x      = mt_new_tensor(ctx, Arr(float, 0, 2, 4, 6), Arr(int, 4), 1);
        MTTensor  *x_grad = mt_new_tensor(ctx,
                                          Arr(float, 1.0, 7.389056205749512, 54.598148345947266, 403.4288024902344),
                                          Arr(int, 4), 1);
        mt_tensor_enable_grad(x);

        MTTensor *res  = mt_tensor_exp(x);
        MTTensor *grad = mt_new_tensor_full(ctx, 1, Arr(int, 4), 1);
        mt_tensor_backward(res, grad);

        /* test x.grad after exp(x) backward */
        mt_assert_true(
            t,
            mt_is_tensor_almost_eq(x->grad, x_grad),
            "test exp grad 1",
            "Should be {1.0, 7.39, 54.60, 403.43}");

        /* test x.grad after exp(exp(x)) backward */
        x      = mt_new_tensor(ctx, Arr(float, 1.0, 0.1, 0.01, 0.001), Arr(int, 4), 1);
        x_grad = mt_new_tensor(ctx,
                               Arr(float, 41.19355010986328, 3.337329864501953, 2.773333787918091, 2.7237253189086914),
                               Arr(int, 4), 1);
        mt_tensor_enable_grad(x);
        res = mt_tensor_exp(mt_tensor_exp(x));
        mt_tensor_backward(res, grad);

        mt_assert_true(
            t,
            mt_is_tensor_almost_eq(x->grad, x_grad),
            "test exp grad 1",
            "Should be equals to {41.19, 3.34, 2.77, 2.72}");
        mt_tensor_print_debug(x->grad);

        mt_context_free(ctx);
}

void run_autograd_neg_tests(Test *t) {
        MTContext *ctx = mt_new_context();
        MTTensor  *x   = mt_new_tensor(ctx, Arr(float, 1, 2, 3), Arr(int, 3), 1);
        mt_tensor_enable_grad(x);

        MTTensor *res = mt_tensor_neg(x);

        /* negation */
        mt_tensor_backward(res, mt_new_tensor(ctx, Arr(float, 2, 2, 2), Arr(int, 3), 1));
        mt_assert_true(
            t,
            mt_is_tensor_eq(x->grad, mt_new_tensor(ctx, Arr(float, -2, -2, -2), Arr(int, 3), 1)),
            "test grad negation",
            "should be {-2, -2, -2}");

        /* sum of negation */
        x = mt_new_tensor(ctx, Arr(float, 1, 2, 3), Arr(int, 3), 1);
        mt_tensor_enable_grad(x);
        res = mt_tensor_sum(mt_tensor_neg(x), -1, 0);
        mt_tensor_backward(res, NULL);
        mt_assert_true(
            t,
            mt_is_tensor_eq(x->grad, mt_new_tensor(ctx, Arr(float, -1, -1, -1), Arr(int, 3), 1)),
            "test grad sum of negation",
            "should be {-1, -1, -1}");

        mt_context_free(ctx);
}

void run_autograd_log_tests(Test *t) {
        MTContext *ctx = mt_new_context();
        MTTensor  *x   = mt_new_tensor(ctx, Arr(float, 1, 2, 3), Arr(int, 3), 1);
        mt_tensor_enable_grad(x);

        MTTensor *res = mt_tensor_log(x);

        mt_tensor_backward(res, mt_new_tensor(ctx, Arr(float, -1, -1, -1), Arr(int, 3), 1));
        mt_assert_true(
            t,
            mt_is_tensor_eq(x->grad, mt_new_tensor(ctx,
                                                   Arr(float, -1.0 / 1, -1.0 / 2, -1.0 / 3),
                                                   Arr(int, 3), 1)),
            "test grad log",
            "should be {-1.00..., -0.50..., -0.33...}");

        mt_context_free(ctx);
}

void run_autograd_relu_tests(Test *t) {
        MTContext *ctx = mt_new_context();
        MTTensor  *x   = mt_new_tensor(ctx, Arr(float, -1, 2, 3), Arr(int, 3), 1);
        mt_tensor_enable_grad(x);

        MTTensor *res = mt_tensor_relu(x);

        mt_tensor_backward(res, mt_new_tensor(ctx, Arr(float, 1, 1, 1), Arr(int, 3), 1));
        mt_assert_true(
            t,
            mt_is_tensor_eq(x->grad, mt_new_tensor(ctx,
                                                   Arr(float, 0, 1, 1),
                                                   Arr(int, 3), 1)),
            "test grad relu",
            "should be {0, 1, 1}");

        /* relu(x) + x */
        mt_tensor_zero_grad(x);
        res = mt_tensor_add(mt_tensor_relu(x), x);
        mt_tensor_backward(res, mt_new_tensor(ctx, Arr(float, 1, 1, 1), Arr(int, 3), 1));
        mt_assert_true(
            t,
            mt_is_tensor_eq(x->grad, mt_new_tensor(ctx,
                                                   Arr(float, 1, 2, 2),
                                                   Arr(int, 3), 1)),
            "test grad relu(x)+x",
            "should be {1, 2, 2}");

        /* sum(relu(x) + x) */
        mt_tensor_zero_grad(x);
        res = mt_tensor_sum(mt_tensor_add(mt_tensor_relu(x), x), -1, 0);
        mt_tensor_backward(res, mt_new_scalar(ctx, 0.5));
        mt_assert_true(
            t,
            mt_is_tensor_eq(x->grad, mt_new_tensor(ctx,
                                                   Arr(float, 0.5, 1, 1),
                                                   Arr(int, 3), 1)),
            "test grad sum(relu(x)+x)",
            "should be {0.5, 1, 1}");

        mt_context_free(ctx);
}