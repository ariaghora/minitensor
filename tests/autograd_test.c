#include <stdio.h>

#include "../minitensor.h"
#include "test.h"

void run_simple_autograd_tests(Test *t) {
        MTContext *ctx = mt_new_context();

        MTTensor *x = mt_new_tensor(ctx, Arr(float, 1, 2, 3, 4), Arr(int, 2, 2), 2);

        mt_tensor_enable_grad(x);
        assert_true(t, x->req_grad, "test enabling grad", "x should require grad");
        assert_true(t, mt_is_tensor_eq(x->grad, mt_new_tensor_full(ctx, 0., Arr(int, 2, 2), 2)), "test initial grad", "intial grad should be all-zero");
        mt_tensor_disable_grad(x);
        assert_true(t, !x->req_grad, "test disabling grad", "x should NOT require grad");
        assert_true(t, x->grad == NULL, "test grad val after disabling grad", "grad val should be NULL");

        MTTensor *y = mt_new_tensor(ctx, Arr(float, 2, 4, 6, 8), Arr(int, 2, 2), 2);
        mt_tensor_enable_grad(y);

        // test if at least one of operand requires grad, then the the result
        // also requires grad
        MTTensor *res = mt_tensor_add(x, y);
        assert_true(t, res->req_grad, "test binop with grad", "res should require grad");
        assert_true(t, res->deps[0] == x, "test binop resulting lchild with grad", "res lchild should be x");
        assert_true(t, res->deps[1] == y, "test binop resulting rchild with grad", "res rchild should be y");

        mt_tensor_disable_grad(y);
        res = mt_tensor_add(x, y);
        assert_true(t, !res->req_grad, "test binop without grad", "res should NOT require grad");
        assert_true(t, res->deps[0] == NULL, "test binop resulting lchild without grad", "res lchild should be NULL");
        assert_true(t, res->deps[1] == NULL, "test binop resulting rchild without grad", "res rchild should be NULL");

        mt_free(ctx);
}

void run_autograd_backward_tests(Test *t) {
        MTContext *ctx = mt_new_context();
        MTTensor  *x   = mt_new_tensor(ctx, Arr(float, 1, 2, 3), Arr(int, 3, 1), 2);
        MTTensor  *y   = mt_new_tensor(ctx, Arr(float, 1, 2, 3), Arr(int, 3, 1), 2);
        mt_tensor_enable_grad(x);
        mt_tensor_enable_grad(y);

        MTTensor *expgrad1 = mt_new_tensor(ctx, Arr(float, 1, 1, 1), Arr(int, 3, 1), 2);

        MTTensor *sum = mt_tensor_sum(x, -1, 0);
        assert_true(t, sum->data[0] == 6, "test tensor sum", "sum should be 6");
        mt_tensor_backward(sum, NULL);
        assert_true(t, x->grad != NULL, "test dependent grad non-null after backward", "dependent grad should not be null");
        assert_true(t, mt_is_tensor_eq(x->grad, expgrad1), "test dependent grad value", "grad value should be all 1");
        assert_true(t, x->grad_fn != NULL, "test grad_fn not null", "grad_fn should not be NULL");

        MTTensor *expgrad2 = mt_new_tensor(ctx, Arr(float, 3, 3, 3), Arr(int, 3, 1), 2);
        sum                = mt_tensor_sum(y, -1, 0);
        mt_tensor_backward(sum, mt_new_scalar(ctx, 3.0));
        assert_true(t, mt_is_tensor_eq(y->grad, expgrad2), "test dependent grad value", "grad value should be all 3");
        mt_free(ctx);
}