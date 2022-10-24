#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "minitensor.h"

#define mt_push_deps(t, dep) ({ \
    dep->parent         = t;    \
    t->deps[t->ndeps++] = dep;  \
})

MTTensor *tensor_bfunc(MTTensor *a, MTTensor *b, float (*bfunc)(float, float)) {
    if (a->datalen != b->datalen) {
        if (a->ndims != 0 && b->ndims != 0) {
            EXIT_WITH_ERROR("a and b have incompatible sizes");
        }
    }
    if (a->context != b->context)
        EXIT_WITH_ERROR("a and b cannot be in different context");

    // TODO: following bloack of consecutive statements is very dirty. consider
    //       resolving the target datalen, ndims, shape, etc according to broadcasting
    //       rule.
    MTTensor *res = mt_alloc_empty_tensor(a->context);
    res->data     = newptr(float, a->ndims > b->ndims ? a->datalen : b->datalen);
    res->datalen  = a->ndims > b->ndims ? a->datalen : b->datalen;
    res->ndims    = a->ndims > b->ndims ? a->ndims : b->ndims;
    res->shape    = a->ndims > b->ndims ? newptr(int, a->ndims) : newptr(int, b->ndims);
    res->strides  = a->ndims > b->ndims ? newptr(int, a->ndims) : newptr(int, b->ndims);
    res->isleaf   = 0;

    if (a->ndims > b->ndims) {
        mt_memcpy(res->shape, a->shape, a->ndims);
        mt_memcpy(res->strides, a->strides, a->ndims);
    } else {
        mt_memcpy(res->shape, b->shape, b->ndims);
        mt_memcpy(res->strides, b->strides, b->ndims);
    }

    if (a->ndims == 0) {
        float val = a->data[0];
        for (long i = 0; i < b->datalen; i++)
            res->data[i] = bfunc(val, b->data[i]);
    } else if (b->ndims == 0) {
        float val = b->data[0];
        for (long i = 0; i < a->datalen; i++)
            res->data[i] = bfunc(a->data[i], val);
    } else {
        for (long i = 0; i < a->datalen; i++)
            res->data[i] = bfunc(a->data[i], b->data[i]);
    }

    if (a->req_grad || b->req_grad) {
        res->req_grad = 1;
        mt_push_deps(res, a);
        mt_push_deps(res, b);
    }
    return res;
}

// addition operation
float     __add(float a, float b) { return a + b; }
void      __add_backward(MTTensor *grad);
MTTensor *mt_tensor_add(MTTensor *a, MTTensor *b) {
    return tensor_bfunc(a, b, __add);
}

// subtraction operation
float     __sub(float a, float b) { return a - b; }
void      __sub_backward(MTTensor *grad);
MTTensor *mt_tensor_sub(MTTensor *a, MTTensor *b) {
    return tensor_bfunc(a, b, __sub);
}

// element-wise multiplication operation
float     __mul(float a, float b) { return a * b; }
MTTensor *mt_tensor_mul(MTTensor *a, MTTensor *b) {
    return tensor_bfunc(a, b, __mul);
}

// sum operation
MTTensor *__sum_backward(MTTensor *self, MTTensor *grad) {
    MTTensor *ones = mt_new_tensor_full(grad->context, 1.0, self->shape,
                                        self->ndims);
    return mt_tensor_mul(grad, ones);
}
MTTensor *mt_tensor_sum(MTTensor *t, int dim) {
    float sum = 0;
    for (long i = 0; i < t->datalen; i++)
        sum += t->data[i];
    MTTensor *res = new_scalar(t->context, sum);
    res->isleaf   = 0;

    if (t->req_grad) {
        mt_tensor_enable_grad(res);
        t->grad_fn = __sum_backward;
        mt_push_deps(res, t);
    }
    return res;
}

MTTensor *mt_tensor_reduce(MTTensor *t, int dim, TensorBFunc bfunc) {
    if (t->shape[dim] <= 1)
        EXIT_WITH_ERROR("cannot reduce at index with size <= 1");

    MTTensor *res = mt_tensor_slice(t->context, t, dim, Arr(int, 0), 1);
    for (long i = 1; i < t->shape[dim]; i++) {
        MTTensor *sl  = mt_tensor_slice(t->context, t, dim, Arr(int, i), 1);
        MTTensor *tmp = res;

        res = bfunc(res, sl);

        tensor_free(tmp);
        tensor_free(sl);
    }
    mt_context_defrag(t->context);
    return res;
}