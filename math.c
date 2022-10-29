#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "minitensor.h"

#define mt_push_deps(t, dep) ({    \
        dep->parent         = t;   \
        t->deps[t->ndeps++] = dep; \
})

MTTensor *tensor_bfunc(MTTensor *a, MTTensor *b, BFunc bfunc) {
        if (a->datalen != b->datalen) {
                if (a->ndims != 0 && b->ndims != 0) {
                        EXIT_WITH_ERROR("a and b have incompatible sizes");
                }
        }
        if (a->context != b->context)
                EXIT_WITH_ERROR("a and b cannot be in different context");

        float *resdata = mt_newptr(
            float,
            a->ndims > b->ndims ? a->datalen : b->datalen);

        if (a->ndims == 0) {
                float val = a->data[0];
                for (long i = 0; i < b->datalen; i++)
                        resdata[i] = bfunc(val, b->data[i]);
        } else if (b->ndims == 0) {
                float val = b->data[0];
                for (long i = 0; i < a->datalen; i++)
                        resdata[i] = bfunc(a->data[i], val);
        } else {
                for (long i = 0; i < a->datalen; i++)
                        resdata[i] = bfunc(a->data[i], b->data[i]);
        }
        MTTensor *res = mt_new_tensor(
            a->context,
            resdata,
            a->ndims > b->ndims ? a->shape : b->shape,
            a->ndims > b->ndims ? a->ndims : b->ndims);
        res->isleaf = 0;

        if (a->req_grad || b->req_grad) {
                res->req_grad = 1;
                mt_push_deps(res, a);
                mt_push_deps(res, b);
        }
        free(resdata);
        return res;
}

/* addition operation */
float     __add(float a, float b) { return a + b; }
MTTensor *__add_backward_a(MTTensor **prtdeps, MTTensor *grad) {
        MTTensor *self = prtdeps[0];
        /* sum out added dims */
        int ndims_added = grad->ndims - self->ndims;
        for (int i = 0; i < ndims_added; i++)
                grad = mt_tensor_sum(grad, 0, 0);
        /* sum across broadcasted (but non-added dims) */
        for (int i = 0; i < self->ndims; i++) {
                if (self->shape[i] == 1)
                        grad = mt_tensor_sum(grad, i, 1);
        }
        return grad;
}
MTTensor *__add_backward_b(MTTensor **prtdeps, MTTensor *grad) {
        MTTensor *self = prtdeps[1];
        /* sum out added dims */
        int ndims_added = grad->ndims - self->ndims;
        for (int i = 0; i < ndims_added; i++)
                grad = mt_tensor_sum(grad, 0, 0);
        /* sum across broadcasted (but non-added dims) */
        for (int i = 0; i < self->ndims; i++) {
                if (self->shape[i] == 1)
                        grad = mt_tensor_sum(grad, i, 1);
        }
        return grad;
}
MTTensor *mt_tensor_add(MTTensor *a, MTTensor *b) {
        MTTensor *res = tensor_bfunc(a, b, __add);
        if (a->req_grad || b->req_grad) {
                mt_tensor_enable_grad(res);
                res->deps[0]          = a;
                res->deps[1]          = b;
                res->deps[0]->grad_fn = __add_backward_a;
                res->deps[1]->grad_fn = __add_backward_b;
        }
        return res;
}

/* subtraction operation */
float     __sub(float a, float b) { return a - b; }
void      __sub_backward(MTTensor *grad);
MTTensor *mt_tensor_sub(MTTensor *a, MTTensor *b) {
        return tensor_bfunc(a, b, __sub);
}

/* element-wise multiplication operation */
float     __mul(float a, float b) { return a * b; }
MTTensor *mt_tensor_mul(MTTensor *a, MTTensor *b) {
        return tensor_bfunc(a, b, __mul);
}

/* sum operation */
MTTensor *__sum_backward(MTTensor **prtdeps, MTTensor *grad) {
        MTTensor *self = prtdeps[0];
        MTTensor *ones = mt_new_tensor_full(grad->context, 1.0, self->shape,
                                            self->ndims);
        return mt_tensor_mul(grad, ones);
}
MTTensor *mt_tensor_sum(MTTensor *t, int dim, int keepdim) {
        if (dim > -1) return mt_tensor_reduce(t, dim, mt_tensor_add, keepdim);

        float sum = 0;
        for (long i = 0; i < t->datalen; i++)
                sum += t->data[i];

        MTTensor *res = NULL;
        if (!keepdim) {
                res = mt_new_scalar(t->context, sum);
        } else {
                int shape[t->ndims];
                for (int i = 0; i < t->ndims; i++) shape[i] = 1;
                res = mt_new_tensor(t->context, Arr(float, sum), shape, t->ndims);
        }

        res->isleaf = 0;
        if (t->req_grad) {
                mt_tensor_enable_grad(res);
                t->grad_fn = __sum_backward;
                mt_push_deps(res, t);
        }
        return res;
}

MTTensor *mt_tensor_reduce(MTTensor *t, int dim, TensorBFunc bfunc, int keepdims) {
        if (t->shape[dim] <= 1)
                EXIT_WITH_ERROR("cannot reduce at index with size <= 1");

        MTTensor *res = mt_tensor_slice(t->context, t, dim, Arr(int, 0), 1);
        for (long i = 1; i < t->shape[dim]; i++) {
                MTTensor *sl  = mt_tensor_slice(t->context, t,
                                                dim, Arr(int, i), 1);
                MTTensor *tmp = res;

                res = bfunc(res, sl);

                mt_tensor_free(tmp);
                mt_tensor_free(sl);
        }
        if (!keepdims) {
                mt_squeeze_aspects_at_dim(dim, res->shape, res->strides,
                                          res->indices, res->ndims);
                res->ndims--;
        }
        mt_context_defrag(t->context);

        return res;
}