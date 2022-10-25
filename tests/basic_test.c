#include <stdio.h>

#include "../minitensor.h"
#include "test.h"

void run_tensor_creation_tests(Test *t) {
        MTContext *ctx = mt_new_context();
        MTTensor  *x   = mt_new_tensor(ctx, Arr(float, 1, 2, 3, 4), Arr(int, 2, 2), 2);
        MTTensor  *y   = mt_new_tensor(ctx, Arr(float, 1, 2, 3, 4), Arr(int, 2, 2), 2);
        MTTensor  *z   = mt_new_tensor(ctx, Arr(float, 3, 3, 3, 3), Arr(int, 2, 2), 2);
        assert_true(t, x->datalen == 4, "test initial data length", "the data length should be 4");
        assert_true(t, mt_is_tensor_eq(x, y), "test identical tensors 1", "content of x and y should be the same");
        assert_true(t, !mt_is_tensor_eq(x, z), "test identical tensors 2", "content of x and y should be different");
        assert_true(t, mt_arrsame(x->indices[0], Arr(int, 0, 1), 2), "test initial 1st axis indices", "indices must be [0, 1]");
        assert_true(t, mt_arrsame(x->indices[1], Arr(int, 0, 1), 2), "test initial 2nd axis indices", "indices must be [0, 1]");

        // scalar
        MTTensor *sc = mt_new_scalar(ctx, 42);
        assert_true(t, sc->ndims == 0, "test scalar dims", "scalar dimension must be 0");

        // full tensor
        MTTensor *full          = mt_new_tensor_full(ctx, 3, Arr(int, 2, 2), 2);
        MTTensor *expected_full = mt_new_tensor(ctx, Arr(float, 3, 3, 3, 3), Arr(int, 2, 2), 2);
        assert_true(t, mt_is_tensor_eq(full, expected_full), "test creating full tensor", "tensor must be all 3");

        // test stride initialization
        x = mt_new_tensor(ctx, Arr(float, 1, 2, 3, 4, 5, 6, 7, 8, 9), Arr(int, 3, 3), 2);
        assert_true(t, x->strides[0] == 3, "test 1st stride of (3,3) tensor", "1st stride must be 3");
        assert_true(t, x->strides[1] == 1, "test 2nd stride of (3,3) tensor", "2nd stride must be 1");

        mt_free(ctx);
}

void run_tensor_access_tests(Test *t) {
        MTContext *ctx  = mt_new_context();
        MTTensor  *scal = mt_new_scalar(ctx, 42);
        MTTensor  *x    = mt_new_tensor(ctx, Arr(float, 1, 2), Arr(int, 2), 1);
        MTTensor  *y    = mt_new_tensor(ctx, Arr(float, 1, 2, 3, 4), Arr(int, 2, 2), 2);
        MTTensor  *z    = mt_new_tensor(ctx, Arr(float, 1, 2, 3, 4, 5, 6, 7, 8), Arr(int, 2, 2, 2), 3);

        assert_true(t, mt_tensor_get_v(scal) == 42, "test getval from scalar", "the value should be 42");
        assert_true(t, mt_tensor_get_1(x, 0) == 1, "test 1st getval from 1-tensor", "the value should be 1");
        assert_true(t, mt_tensor_get_1(x, 1) == 2, "test 2nd getval from 1-tensor", "the value should be 2");
        assert_true(t, mt_tensor_get_2(y, 0, 0) == 1, "test 1st getval from 2-tensor", "the value should be 1");
        assert_true(t, mt_tensor_get_2(y, 1, 1) == 4, "test 2nd getval from 2-tensor", "the value should be 4");
        assert_true(t, mt_tensor_get_3(z, 0, 1, 1) == 4, "test 1st getval from 3-tensor", "the value should be 4");
        assert_true(t, mt_tensor_get_3(z, 1, 0, 1) == 6, "test 1st getval from 3-tensor", "the value should be 6");
        mt_free(ctx);
}

void run_tensor_slice_tests(Test *t) {
        MTContext *ctx    = mt_new_context();
        MTTensor  *x      = mt_new_tensor(ctx, Arr(float, 1, 2, 3, 4, 5, 6), Arr(int, 3, 2), 2);
        MTTensor  *y      = mt_new_tensor(ctx, Arr(float, 1, 2, 3, 4, 5, 6, 7, 8), Arr(int, 2, 2, 2), 3);
        MTTensor  *slice1 = mt_tensor_slice(ctx, x, 0, Arr(int, 1), 1);
        MTTensor  *slice2 = mt_tensor_slice(ctx, x, 0, Arr(int, 0, 2), 2);
        MTTensor  *slice3 = mt_tensor_slice(ctx, y, 1, Arr(int, 0), 1);
        MTTensor  *slice4 = mt_tensor_slice(ctx, y, 2, Arr(int, 1), 1);
        MTTensor  *exp1   = mt_new_tensor(ctx, Arr(float, 3, 4), Arr(int, 1, 2), 2);
        MTTensor  *exp2   = mt_new_tensor(ctx, Arr(float, 1, 2, 5, 6), Arr(int, 2, 2), 2);
        MTTensor  *exp3   = mt_new_tensor(ctx, Arr(float, 1, 2, 5, 6), Arr(int, 2, 1, 2), 3);
        MTTensor  *exp4   = mt_new_tensor(ctx, Arr(float, 2, 4, 6, 8), Arr(int, 2, 2, 1), 3);

        assert_true(t, mt_is_tensor_eq(slice1, exp1), "test slice 2-tensor 1", "the value should be [3, 4]");
        assert_true(t, mt_is_tensor_eq(slice2, exp2), "test slice 2-tensor 2", "the value should be [1, 2, 5, 6]");
        assert_true(t, mt_is_tensor_eq(slice3, exp3), "test slice 3-tensor 1", "the value should be [1, 2, 5, 6]");
        assert_true(t, mt_is_tensor_eq(slice4, exp4), "test slice 3-tensor 2", "the value should be [1, 2, 5, 6]");

        mt_free(ctx);
}

void f(int *arr) {
}

void run_context_tests(Test *t) {
        MTContext *ctx = mt_new_context();
        MTTensor  *a   = mt_new_scalar(ctx, 1.0);
        MTTensor  *b   = mt_new_scalar(ctx, 10.0);
        MTTensor  *c   = mt_new_tensor(ctx, Arr(float, 1, 2), Arr(int, 2, 1), 2);
        MTTensor  *d   = mt_new_tensor(ctx, Arr(float, 2, 4), Arr(int, 2, 1), 2);
        (void)(a), (void)(b), (void)(c), (void)(d);

        mt_tensor_free(a);
        mt_tensor_free(c);

        mt_context_defrag(ctx);
        assert_true(t, ctx->ntracked == 2, "test tracker defrag", "defragged context should have 2 ntracked");
        assert_true(t, ctx->tracked[0]->ndims == 0, "test 1st remaining tracked ndims", "1st remaining ndims should be 0");
        assert_true(t, ctx->tracked[1]->ndims == 2, "test 2nd remaining tracked ndims", "2nd remaining ndims should be 2");

        MTTensor *w    = mt_new_scalar(ctx, 3.0);
        MTTensor *x    = mt_new_scalar(ctx, 2.0);
        MTTensor *y    = mt_new_scalar(ctx, 1.0);
        MTTensor *loss = mt_tensor_sub(mt_tensor_mul(w, x), y);
        assert_true(t, x->isleaf, "test if x is a leaf", "x should be a leaf");
        assert_true(t, !loss->isleaf, "test if binop result not a leaf", "result of binop should not be a leaf");

        mt_free(ctx);
}